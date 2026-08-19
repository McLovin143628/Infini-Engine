//! The **geo-anchor**: where on Earth world `(0, 0, 0)` is (Wave G).
//!
//! A level that imports real-world data needs one place that says "the world
//! origin is *here*, in *this* coordinate reference system". Without it a DEM, a
//! road centreline and a coastline each arrive in their own numbers and nothing
//! can put them on the same ground.
//!
//! # Why the anchor is a *projected* CRS, and why that is forced
//!
//! Architecture rule 6 is `1 world unit = 1 metre, no unit-scale factors, ever`
//! (`docs/memos/units-doctrine.md`). Architecture rule 3 is f64 world
//! coordinates. Together they decide this design outright:
//!
//! > **The world is a plane in exactly one projected, metric CRS.**
//!
//! A *geographic* CRS (EPSG:4326 and friends) is in **degrees**, and a degree is
//! not a metre — nor is it a constant number of metres, since a degree of
//! longitude shrinks to nothing at the poles. There is no scale factor allowed to
//! bridge that, so a geographic CRS can never be world coordinates here. A
//! project picks one projected CRS (a UTM zone, a state plane, an Albers equal-
//! area…) and **every import is transformed into it at the import door**.
//! Degrees survive only as an input format and as a readout in the wizard.
//!
//! # The axis convention is not a free choice
//!
//! It would be easy to believe the sign of the northing is ours to pick. It is
//! not — [`crate::solar`] pinned it in P17.1, because the sun is placed from a
//! latitude and a longitude in the East/North/Up topocentric frame:
//!
//! | engine axis | compass |
//! |-------------|---------|
//! | `+X`        | East    |
//! | `+Y`        | Up      |
//! | `+Z`        | South   |
//!
//! North is `-Z`. So the anchor's mapping is fixed:
//!
//! ```text
//! world.x = easting  − origin_easting_m         // +X is East
//! world.z = origin_northing_m − northing        // +Z is South, so this NEGATES
//! world.y = elevation − origin_height_m         // +Y is Up
//! ```
//!
//! Getting the `z` sign backwards would mirror every imported map east-for-west
//! while leaving the sun where it was — a defect with no visual signature until
//! somebody notices the shadows fall the wrong way at noon. `anchor_axes_match_the_solar_frame`
//! pins it against the solar module's own constants rather than against a
//! restatement of them.
//!
//! It also happens to be the convention a north-up raster already wants: a
//! GeoTIFF's first row is its **northern** edge and its rows march south, which
//! is exactly `+Z` increasing. A north-up DEM therefore feeds the existing
//! row-at-a-time tiler with no flip anywhere.
//!
//! # Precision is a non-issue, and here is the arithmetic
//!
//! UTM eastings run ~1.6 × 10⁵ … 8.3 × 10⁵ m and northings up to ~10⁷ m. `f64`
//! carries ~15–16 significant decimal digits, so a raw northing is already
//! resolved to well under a micrometre before anything is subtracted; after
//! subtracting the origin the numbers are small. [`crate::FloatingOrigin`] then
//! does the f64 → f32 render hand-off exactly as it does for any other world
//! coordinate. There is no precision motivation for a scale factor, which is
//! fortunate, because rule 6 forbids one.
//!
//! # Grid convergence, and the second-source-of-truth hazard
//!
//! In any projected CRS, **grid north is not true north** except along the
//! central meridian; the angle between them is the *grid convergence*, and it
//! reaches a degree or two at the edge of a UTM zone. The sun is placed from
//! *true* north ([`crate::solar`]), so a world whose `+Z` follows *grid* south
//! carries that error into its shadows unless somebody accounts for it.
//!
//! This module therefore stores the convergence with the anchor, alongside the
//! origin's geodetic latitude and longitude. Both are **derived once at the
//! import door** — `inf-gis` inverts the projection and computes them — and then
//! carried as plain numbers, so nothing downstream needs a projection library to
//! know where on Earth it is. That matters for the ring split: `inf-math` and
//! `inf-scene` must not depend on `proj4rs`.
//!
//! **The ruling this encodes** (Wave G): when a level has an anchor, the anchor
//! is *authoritative* for where on Earth the world is, and the sky's
//! `TimeOfDay::latitude_deg`/`longitude_deg` should be **derived** from it. A
//! mismatch is a **cook advisory**, not a silent correction — the P16 law that a
//! silent hazard gets named rather than quietly fixed. This house has been
//! bitten by a second source of truth at least twice before (the `inf-scene`
//! mirror law, the `GpuLight` triplication law), and a wrong sun that nothing
//! complains about is precisely that shape.
//!
//! **What is built and what is not** (Wave-G audit A5, stated so the ruling is
//! not mistaken for the implementation): [`GeoAnchor::solar_place`] and
//! [`GeoAnchor::disagrees_with_place`] are the *predicates* that ruling needs,
//! and they are tested — but **no cook calls them yet**, and
//! [`GeoAnchor::grid_convergence_deg`] is derived at the import door, persisted,
//! and read by nothing. The anchor therefore records where on Earth the world is
//! and does not yet move the sun or correct a shadow for grid convergence. The
//! landing site is the cook's advisory pass beside the other `.inf_lvl`
//! advisories; until it exists, an author with an anchor and a hand-set
//! `TimeOfDay` has two places and no complaint.
//!
//! # Units
//!
//! Per the units doctrine: the four origin components are **metres**; latitude,
//! longitude and convergence are **degrees**, which is legal here for the same
//! reason euler angles are — they are a data/UI boundary quantity, never used as
//! a world coordinate.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Where on Earth world `(0, 0, 0)` is, and in what CRS the world plane lives.
///
/// See the [module docs](self) for the axis convention (which is pinned by
/// [`crate::solar`], not chosen here) and for why the CRS must be projected.
///
/// # Persistence
///
/// This is a **settings block** on the scene (`.inf_lvl` schema v24), shaped like
/// the world-partition block that preceded it: `#[serde(default)]`, additive, and
/// meaningless-but-harmless on a level that never imported real-world data
/// ([`enabled`](Self::enabled) is `false` and every number is zero).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GeoAnchor {
    /// Whether this level is georeferenced at all.
    ///
    /// A level with `enabled == false` is an ordinary made-up world: importers
    /// refuse to reproject into it (there is nothing to reproject *to*), and the
    /// sky keeps whatever latitude the author set by hand.
    pub enabled: bool,
    /// The projected CRS the world plane lives in, as an authority code
    /// (`"EPSG:32610"`) or a full proj4 string (`"+proj=utm +zone=10 …"`).
    ///
    /// Resolved by `inf_gis::epsg` against a curated table; an unsupported code
    /// is refused **by name**, never guessed. Empty when
    /// [`enabled`](Self::enabled) is `false`.
    pub crs: String,
    /// The CRS easting (metres) that world `X = 0` means.
    pub origin_easting_m: f64,
    /// The CRS northing (metres) that world `Z = 0` means. Remember `+Z` is
    /// **south**, so world Z *decreases* as northing increases.
    pub origin_northing_m: f64,
    /// The elevation (metres above the vertical datum) that world `Y = 0` means.
    pub origin_height_m: f64,
    /// Geodetic latitude of the origin, degrees, `+` north.
    ///
    /// Derived at the import door by inverting the projection once, then carried
    /// as a plain number. This is what the sky's latitude is derived *from*.
    pub origin_latitude_deg: f64,
    /// Geodetic longitude of the origin, degrees, `+` east.
    pub origin_longitude_deg: f64,
    /// **Grid convergence** at the origin: degrees from grid north to true north,
    /// `+` when true north lies clockwise (east) of grid north.
    ///
    /// Zero on the central meridian and growing toward a zone edge. See the
    /// module docs for why the sun needs it.
    pub grid_convergence_deg: f64,
    /// The vertical datum the heights are referenced to (`"EGM2008"`,
    /// `"NAVD88"`, `"WGS84 ellipsoid"`, …), recorded for the author's benefit.
    ///
    /// **Informational only** — this engine applies no geoid model, so a source
    /// in orthometric heights and a source in ellipsoidal heights differ by the
    /// geoid undulation (tens of metres in places) and nothing here corrects it.
    /// Recording the string is what lets an import that mixes two datums be
    /// *noticed*; see the disposition memo's named gaps.
    pub vertical_datum: String,
}

impl Default for GeoAnchor {
    /// An un-georeferenced world: exactly what every level authored before
    /// schema v24 meant, which is what makes the block additive.
    fn default() -> Self {
        Self {
            enabled: false,
            crs: String::new(),
            origin_easting_m: 0.0,
            origin_northing_m: 0.0,
            origin_height_m: 0.0,
            origin_latitude_deg: 0.0,
            origin_longitude_deg: 0.0,
            grid_convergence_deg: 0.0,
            vertical_datum: String::new(),
        }
    }
}

/// What is wrong with a [`GeoAnchor`] that cannot be used.
///
/// Refusals are **values** (the house law), so a caller can put the reason in
/// front of the author instead of unwrapping.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GeoAnchorError {
    /// The anchor is switched off — there is nothing to transform into.
    Disabled,
    /// The CRS string is empty on an enabled anchor.
    NoCrs,
    /// A component is NaN or infinite. Carries the field name.
    NotFinite(&'static str),
    /// Latitude outside `[-90, 90]` or longitude outside `[-180, 180]`.
    OutOfRange(&'static str),
}

impl std::fmt::Display for GeoAnchorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeoAnchorError::Disabled => write!(
                f,
                "this level has no geo-anchor — set one in World Settings before \
                 importing georeferenced data, so the importer knows where on \
                 Earth the world origin is"
            ),
            GeoAnchorError::NoCrs => write!(
                f,
                "the geo-anchor is enabled but names no CRS; give it an authority \
                 code (for example \"EPSG:32610\") or a full proj4 string"
            ),
            GeoAnchorError::NotFinite(field) => write!(
                f,
                "the geo-anchor's `{field}` is not a finite number (NaN or \
                 infinity) — an anchor with a non-finite component would make \
                 every imported coordinate non-finite too"
            ),
            GeoAnchorError::OutOfRange(field) => write!(
                f,
                "the geo-anchor's `{field}` is outside the range a geodetic \
                 coordinate can take"
            ),
        }
    }
}

impl std::error::Error for GeoAnchorError {}

impl GeoAnchor {
    /// Check that this anchor can actually be transformed through.
    ///
    /// **The NaN door.** Every component is checked for finiteness here, once, so
    /// no import path has to remember to. A single NaN in the origin would make
    /// every transformed coordinate NaN — and a NaN elevation survives
    /// `f32::min`/`max`, so a terrain tile built from one reports perfectly
    /// healthy height bounds while being unusable. That is the exact mechanism
    /// the terrain importer's own finiteness door exists to stop, met one layer
    /// earlier.
    pub fn validate(&self) -> Result<(), GeoAnchorError> {
        if !self.enabled {
            return Err(GeoAnchorError::Disabled);
        }
        if self.crs.trim().is_empty() {
            return Err(GeoAnchorError::NoCrs);
        }
        for (name, v) in [
            ("origin_easting_m", self.origin_easting_m),
            ("origin_northing_m", self.origin_northing_m),
            ("origin_height_m", self.origin_height_m),
            ("origin_latitude_deg", self.origin_latitude_deg),
            ("origin_longitude_deg", self.origin_longitude_deg),
            ("grid_convergence_deg", self.grid_convergence_deg),
        ] {
            if !v.is_finite() {
                return Err(GeoAnchorError::NotFinite(name));
            }
        }
        if !(-90.0..=90.0).contains(&self.origin_latitude_deg) {
            return Err(GeoAnchorError::OutOfRange("origin_latitude_deg"));
        }
        if !(-180.0..=180.0).contains(&self.origin_longitude_deg) {
            return Err(GeoAnchorError::OutOfRange("origin_longitude_deg"));
        }
        Ok(())
    }

    /// Projected CRS coordinates → world metres.
    ///
    /// `easting`/`northing` are in the anchor's own CRS; `elevation` is metres
    /// above its vertical datum. See the [module docs](self) for why `z` negates.
    ///
    /// Deliberately **not** validating: this runs per-coordinate on imports of
    /// millions of points, so the anchor is validated once at the door by
    /// [`validate`](Self::validate) and this stays three subtractions.
    #[inline]
    pub fn world_from_projected(&self, easting: f64, northing: f64, elevation: f64) -> DVec3 {
        DVec3::new(
            easting - self.origin_easting_m,
            elevation - self.origin_height_m,
            // +Z is SOUTH (see the module docs) — north is -Z, so an increasing
            // northing walks toward -Z.
            self.origin_northing_m - northing,
        )
    }

    /// World metres → projected CRS coordinates, as `(easting, northing,
    /// elevation)`. The exact inverse of [`world_from_projected`](Self::world_from_projected).
    #[inline]
    pub fn projected_from_world(&self, world: DVec3) -> (f64, f64, f64) {
        (
            world.x + self.origin_easting_m,
            self.origin_northing_m - world.z,
            world.y + self.origin_height_m,
        )
    }

    /// The latitude/longitude the sky should use, when this anchor is enabled.
    ///
    /// Returns `None` for a disabled anchor, which is the signal to keep whatever
    /// the author set on `TimeOfDay` by hand.
    #[inline]
    pub fn solar_place(&self) -> Option<(f64, f64)> {
        self.enabled
            .then_some((self.origin_latitude_deg, self.origin_longitude_deg))
    }

    /// Whether `TimeOfDay`'s hand-authored place disagrees with this anchor by
    /// more than `tolerance_deg`.
    ///
    /// The **advisory** predicate, not a fixer: per the P16 law a silent hazard
    /// gets named. A cook that finds this `true` emits an advisory naming both
    /// places; it does not quietly move the sun.
    pub fn disagrees_with_place(&self, lat_deg: f64, lon_deg: f64, tolerance_deg: f64) -> bool {
        let Some((alat, alon)) = self.solar_place() else {
            return false;
        };
        if !(lat_deg.is_finite() && lon_deg.is_finite()) {
            return true;
        }
        // Longitude difference the short way round, so 179.9° E vs 179.9° W is
        // 0.2° apart rather than 359.8°.
        let dlon = {
            let d = (alon - lon_deg).abs() % 360.0;
            if d > 180.0 {
                360.0 - d
            } else {
                d
            }
        };
        (alat - lat_deg).abs() > tolerance_deg || dlon > tolerance_deg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn anchor() -> GeoAnchor {
        GeoAnchor {
            enabled: true,
            crs: "EPSG:32610".into(),
            // A plausible UTM zone-10N origin near Vancouver.
            origin_easting_m: 491_000.0,
            origin_northing_m: 5_459_000.0,
            origin_height_m: 0.0,
            origin_latitude_deg: 49.28,
            origin_longitude_deg: -123.12,
            grid_convergence_deg: 0.09,
            vertical_datum: "EGM2008".into(),
        }
    }

    /// **The convention gate.** The anchor's axes are pinned against
    /// [`crate::solar`]'s own frame rather than against a restatement of it, so
    /// the two can never drift apart: if somebody re-decides the compass in the
    /// solar module, this fails.
    ///
    /// Un-fix mutation: flip the `z` sign in `world_from_projected` and the
    /// "north is -Z" assertion fails.
    #[test]
    fn anchor_axes_match_the_solar_frame() {
        let a = anchor();

        // Walking NORTH (increasing northing) must walk toward -Z, because the
        // solar module places the sun in a frame where +Z is south.
        let here = a.world_from_projected(a.origin_easting_m, a.origin_northing_m, 0.0);
        let north = a.world_from_projected(a.origin_easting_m, a.origin_northing_m + 1000.0, 0.0);
        assert_eq!(
            here,
            DVec3::ZERO,
            "the origin must land on the world origin"
        );
        assert_eq!(
            north.z, -1000.0,
            "north must be -Z (the solar frame's +Z is south)"
        );
        assert_eq!(north.x, 0.0);

        // Walking EAST walks toward +X.
        let east = a.world_from_projected(a.origin_easting_m + 250.0, a.origin_northing_m, 0.0);
        assert_eq!(east.x, 250.0, "east must be +X");
        assert_eq!(east.z, 0.0);

        // Elevation is +Y.
        let up = a.world_from_projected(a.origin_easting_m, a.origin_northing_m, 87.5);
        assert_eq!(up.y, 87.5, "elevation must be +Y");

        // And the frame is right-handed: East × Up = South = +Z.
        let e = DVec3::X;
        let u = DVec3::Y;
        assert_eq!(
            e.cross(u),
            DVec3::Z,
            "East × Up must be +Z for a right-handed frame"
        );

        // Cross-check against the solar module's own sun direction: at the June
        // solstice, local solar noon in the northern hemisphere puts the sun in
        // the SOUTHERN sky, i.e. its direction has a +Z component in this frame.
        // (`sun_direction` returns the direction TO the sun.)
        let noon = crate::solar::SolarInput {
            // Local solar noon at this longitude, in UTC seconds.
            seconds: (12.0 - a.origin_longitude_deg / 15.0) * 3600.0,
            day_of_year: 172,
            latitude_deg: a.origin_latitude_deg,
            longitude_deg: a.origin_longitude_deg,
        };
        let sun = crate::solar::sun_direction(&noon);
        assert!(
            sun.y > 0.0,
            "the sun must be above the horizon at local noon (got {sun:?})"
        );
        assert!(
            sun.z > 0.0,
            "at northern-hemisphere solar noon the sun is in the SOUTHERN sky, \
             which is +Z in this frame — if this fails the anchor and the sky \
             disagree about which way north is (got {sun:?})"
        );
    }

    /// The transform is exactly invertible — the property a write-back relies on.
    #[test]
    fn projected_and_world_round_trip_exactly() {
        let a = anchor();
        for (e, n, h) in [
            (491_000.0, 5_459_000.0, 0.0),
            (512_345.678, 5_441_002.5, 1234.5),
            (400_000.0, 5_500_000.0, -37.25),
        ] {
            let w = a.world_from_projected(e, n, h);
            let (e2, n2, h2) = a.projected_from_world(w);
            assert_eq!((e, n, h), (e2, n2, h2), "round trip moved the coordinate");
        }
    }

    /// **The NaN door.** Every component is guarded, by name, and a disabled or
    /// CRS-less anchor is refused as a value rather than panicking.
    #[test]
    fn a_non_finite_anchor_is_refused_by_field_name() {
        assert_eq!(
            GeoAnchor::default().validate(),
            Err(GeoAnchorError::Disabled),
            "the default anchor is an ordinary made-up world"
        );
        let mut a = anchor();
        assert_eq!(a.validate(), Ok(()));

        a.crs = "   ".into();
        assert_eq!(a.validate(), Err(GeoAnchorError::NoCrs));

        for (name, set) in [
            (
                "origin_easting_m",
                (|a: &mut GeoAnchor| a.origin_easting_m = f64::NAN) as fn(&mut GeoAnchor),
            ),
            ("origin_northing_m", |a| a.origin_northing_m = f64::INFINITY),
            ("origin_height_m", |a| a.origin_height_m = f64::NEG_INFINITY),
            ("origin_latitude_deg", |a| a.origin_latitude_deg = f64::NAN),
            ("origin_longitude_deg", |a| {
                a.origin_longitude_deg = f64::NAN
            }),
            ("grid_convergence_deg", |a| {
                a.grid_convergence_deg = f64::NAN
            }),
        ] {
            let mut a = anchor();
            set(&mut a);
            assert_eq!(
                a.validate(),
                Err(GeoAnchorError::NotFinite(name)),
                "a non-finite {name} must be named"
            );
            // …and the message says which field, so the author can fix it.
            assert!(a.validate().unwrap_err().to_string().contains(name));
        }

        // Out-of-range geodetic values are refused too.
        let mut a = anchor();
        a.origin_latitude_deg = 91.0;
        assert_eq!(
            a.validate(),
            Err(GeoAnchorError::OutOfRange("origin_latitude_deg"))
        );
        let mut a = anchor();
        a.origin_longitude_deg = -181.0;
        assert_eq!(
            a.validate(),
            Err(GeoAnchorError::OutOfRange("origin_longitude_deg"))
        );
    }

    /// The advisory predicate: it *reports*, and it wraps longitude the short way
    /// round so the antimeridian is not a false positive.
    #[test]
    fn the_place_advisory_reports_rather_than_fixes() {
        let a = anchor();
        assert_eq!(a.solar_place(), Some((49.28, -123.12)));
        assert!(!a.disagrees_with_place(49.28, -123.12, 0.5));
        assert!(!a.disagrees_with_place(49.3, -123.1, 0.5));
        // The engine's default place (48.9 N, 0.0 E) is a whole ocean away.
        assert!(a.disagrees_with_place(48.9, 0.0, 0.5));

        // A disabled anchor never advises — it has no opinion to disagree with.
        assert!(!GeoAnchor::default().disagrees_with_place(0.0, 0.0, 0.5));

        // The antimeridian is 0.2° wide here, not 359.8°.
        let mut fiji = anchor();
        fiji.origin_longitude_deg = 179.9;
        assert!(
            !fiji.disagrees_with_place(fiji.origin_latitude_deg, -179.9, 0.5),
            "179.9E and 179.9W are 0.2 degrees apart, not 359.8"
        );

        // A non-finite hand-authored place always disagrees (it cannot be trusted).
        assert!(a.disagrees_with_place(f64::NAN, -123.12, 0.5));
    }
}
