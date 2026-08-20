//! **A building footprint's attributes, read** (IB-5b).
//!
//! The Wave-G disposition memo's G11 row, in full: *"Height limits do not map
//! onto `BuildingParams::floors` — there is **no code between a GIS attribute
//! and that field in either direction**. 'Maps onto' described a shape, not a
//! wire."* This module is the first half of the wire; the second half is
//! `inf_editor_core::gisbuild`, which is where the grammar lives.
//!
//! # Why the storey count is derived rather than defaulted
//!
//! Real footprint layers state height far more often than they state floors —
//! a LiDAR-derived building layer has `HEIGHT` on every record and `LEVELS` on
//! none. Reading only the floor count would default a whole downtown to two
//! storeys while the data to build it correctly sat in the next column. So:
//! the stated floor count wins; a stated height divided by the storey height is
//! next; the default is last, and **which of the three answered is carried on
//! every feature** ([`FloorSource`]) so the import can say "4 831 of 5 002
//! buildings took the default" instead of looking like it knew.
//!
//! # Defaults are typed, and their counts are advisories
//!
//! A missing attribute is not an error and must not be a silent zero. Every
//! fallback here is a named constant with a reason, and [`AttrCoverage`] counts
//! how often each was reached — the P16 law about silent hazards, applied to an
//! attribute table.

use crate::feature::GeoFeature;

/// The spellings a storey count is stored under, most specific first.
///
/// `building:levels` is OSM's; `NUM_FLOORS`, `STOREYS` and `NUM_STORY` are what
/// North-American municipal footprint layers use. Lookup folds case and
/// separators, so each of these covers several real column names.
pub const FLOOR_FIELDS: [&str; 10] = [
    "building:levels",
    "levels",
    "num_floors",
    "numfloors",
    "floors",
    "storeys",
    "stories",
    "num_story",
    "num_stories",
    "nfloors",
];

/// The spellings a building height (in metres) is stored under.
///
/// **Metres, and this is where that assumption lives.** A layer in feet states
/// so in its `.prj`'s vertical unit, which
/// [`crate::crs::Transform::with_vertical_unit`] applies to *geometry*; an
/// attribute column carries no unit at all. `HEIGHT_FT` is therefore read
/// separately — see [`HEIGHT_FT_FIELDS`] — rather than being read as metres and
/// building a downtown a third of its real height.
pub const HEIGHT_FIELDS: [&str; 9] = [
    "building:height",
    "height_m",
    "heightm",
    "bldg_height",
    "bldgheight",
    "roof_height",
    "max_height",
    "height",
    "hgt",
];

/// Height columns that state **feet** in their own name.
pub const HEIGHT_FT_FIELDS: [&str; 4] = ["height_ft", "heightft", "hgt_ft", "bldg_ht_ft"];

/// The spellings a building's use/type is stored under.
pub const TYPE_FIELDS: [&str; 9] = [
    "building",
    "building:use",
    "bldg_type",
    "bldgtype",
    "struct_type",
    "occupancy",
    "landuse",
    "zoning",
    "use",
];

/// Metres per storey when a layer states height but not floors.
///
/// Three metres is the residential figure and the median across mixed stock;
/// an office storey is 3.5–4 m and a warehouse one is 6+, so a tower derived
/// from height alone comes out a little tall in floors and exactly right in
/// height — which is the error that matters, because the height is the datum
/// the source actually stated.
pub const DEFAULT_STOREY_HEIGHT_M: f64 = 3.0;

/// The storey count for a footprint that states neither floors nor height.
pub const DEFAULT_FLOORS: u32 = 2;

/// The tallest storey count an attribute may assert.
///
/// A hostile-input door, not a preference: a `HEIGHT` column with a sentinel of
/// `9999` in it — which real layers carry — is a 3 333-storey building, and the
/// grammar would try to plan it. `inf_pcg::MAX_FLOORS` is 400; this is lower
/// because a *derived* count is a guess and a guess of 400 storeys is wrong in a
/// way that costs minutes.
pub const MAX_ATTR_FLOORS: u32 = 200;

/// Feet, in metres — the exact conversion, spelled once.
pub const FOOT_M: f64 = 0.3048;

/// Where a footprint's storey count came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FloorSource {
    /// The source stated a floor count.
    Attribute,
    /// Derived from a stated height.
    DerivedFromHeight,
    /// Neither was stated; [`DEFAULT_FLOORS`].
    Default,
}

impl FloorSource {
    pub const fn label(self) -> &'static str {
        match self {
            FloorSource::Attribute => "attribute",
            FloorSource::DerivedFromHeight => "height",
            FloorSource::Default => "default",
        }
    }
}

/// What a footprint's attribute row says about the building on it.
#[derive(Clone, Debug, PartialEq)]
pub struct FootprintAttrs {
    /// Storeys, always at least 1.
    pub floors: u32,
    pub floor_source: FloorSource,
    /// The source asked for more than [`MAX_ATTR_FLOORS`] and was cut down to
    /// it.
    ///
    /// **Carried because a clamp is a silent hazard otherwise** (I2 audit). A
    /// `HEIGHT` column full of the `9999` sentinel asks for 3 333 storeys on
    /// every record; clamping each one produces a downtown of 200-storey towers
    /// with nothing to see but the skyline. Counted per layer by
    /// [`AttrCoverage`] and raised as an advisory there.
    pub clamped: bool,
    /// The stated height in metres, when the source had one.
    pub height_m: Option<f64>,
    /// The stated use/type, verbatim, when the source had one.
    pub kind: Option<String>,
}

/// Fallbacks for a footprint that states nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FootprintDefaults {
    pub floors: u32,
    pub storey_height_m: f64,
}

impl Default for FootprintDefaults {
    fn default() -> Self {
        Self {
            floors: DEFAULT_FLOORS,
            storey_height_m: DEFAULT_STOREY_HEIGHT_M,
        }
    }
}

/// **Read one footprint's attributes.** The wire's first half.
pub fn footprint_attrs(f: &GeoFeature, d: &FootprintDefaults) -> FootprintAttrs {
    let storey = if d.storey_height_m.is_finite() && d.storey_height_m > 0.1 {
        d.storey_height_m
    } else {
        DEFAULT_STOREY_HEIGHT_M
    };
    // Height first, because it is also an output — a caller that wants the real
    // height rather than a rounded storey count needs it either way.
    let height_m = f
        .attr_number(&HEIGHT_FIELDS)
        .or_else(|| f.attr_number(&HEIGHT_FT_FIELDS).map(|v| v * FOOT_M))
        .filter(|v| v.is_finite() && *v > 0.0);

    // Not clamped here: the ceiling is applied at ONE place below, so the
    // difference between what the source asked for and what it got is visible
    // (`FootprintAttrs::clamped`) rather than absorbed on the way past.
    let stated = f
        .attr_number(&FLOOR_FIELDS)
        .filter(|v| v.is_finite() && *v >= 1.0)
        .map(|v| v as u32);

    let (asked, floor_source) = match stated {
        Some(n) => (n, FloorSource::Attribute),
        None => match height_m {
            Some(h) => (
                (h / storey).round().max(0.0) as u32,
                FloorSource::DerivedFromHeight,
            ),
            None => (d.floors, FloorSource::Default),
        },
    };
    let floors = asked.clamp(1, MAX_ATTR_FLOORS);

    FootprintAttrs {
        floors,
        floor_source,
        clamped: asked > MAX_ATTR_FLOORS,
        height_m,
        kind: f
            .attr_text(&TYPE_FIELDS)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty()),
    }
}

/// How often each fallback was reached across a whole layer — the advisory
/// counts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttrCoverage {
    pub features: usize,
    pub floors_from_attribute: usize,
    pub floors_from_height: usize,
    pub floors_defaulted: usize,
    /// Features whose stated storey count or height asked for more than
    /// [`MAX_ATTR_FLOORS`]. See [`FootprintAttrs::clamped`].
    pub floors_clamped: usize,
    pub typed: usize,
    pub untyped: usize,
}

impl AttrCoverage {
    pub fn observe(&mut self, a: &FootprintAttrs) {
        self.features += 1;
        match a.floor_source {
            FloorSource::Attribute => self.floors_from_attribute += 1,
            FloorSource::DerivedFromHeight => self.floors_from_height += 1,
            FloorSource::Default => self.floors_defaulted += 1,
        }
        if a.clamped {
            self.floors_clamped += 1;
        }
        if a.kind.is_some() {
            self.typed += 1;
        } else {
            self.untyped += 1;
        }
    }

    /// The advisories a layer's coverage deserves — one sentence each, and only
    /// when the number is worth an author's attention.
    pub fn advisories(&self) -> Vec<crate::Advisory> {
        let mut out = Vec::new();
        if self.features == 0 {
            return out;
        }
        if self.floors_defaulted * 2 > self.features {
            out.push(crate::Advisory::new(
                "attrs.floors_defaulted",
                format!(
                    "{} of {} footprints state neither a floor count nor a height, so \
                     they took the {DEFAULT_FLOORS}-storey default. The fields this \
                     engine reads are {} and {}; if the layer spells it differently, \
                     name the column at the import door.",
                    self.floors_defaulted,
                    self.features,
                    FLOOR_FIELDS.join("/"),
                    HEIGHT_FIELDS.join("/")
                ),
            ));
        }
        if self.floors_from_height > 0 {
            out.push(crate::Advisory::new(
                "attrs.floors_from_height",
                format!(
                    "{} of {} footprints state a height but no floor count; their \
                     storeys are height / {DEFAULT_STOREY_HEIGHT_M} m rounded, which \
                     is right for residential stock and runs one or two storeys high \
                     for offices and warehouses.",
                    self.floors_from_height, self.features
                ),
            ));
        }
        if self.floors_clamped > 0 {
            out.push(crate::Advisory::new(
                "attrs.floors_clamped",
                format!(
                    "{} of {} footprints asked for more than {MAX_ATTR_FLOORS} \
                     storeys and were cut down to it. A height column full of a \
                     sentinel — 9999 is the common one, and is 3 333 storeys at \
                     {DEFAULT_STOREY_HEIGHT_M} m — reads exactly like this. Check \
                     the column before building the layer.",
                    self.floors_clamped, self.features
                ),
            ));
        }
        if self.untyped * 2 > self.features {
            out.push(crate::Advisory::new(
                "attrs.untyped",
                format!(
                    "{} of {} footprints state no building type, so they all take the \
                     same archetype. The fields this engine reads are {}.",
                    self.untyped,
                    self.features,
                    TYPE_FIELDS.join("/")
                ),
            ));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feature::{Attr, GeoGeometry};
    use glam::DVec3;

    fn feature(pairs: &[(&str, Attr)]) -> GeoFeature {
        let mut f = GeoFeature::new(GeoGeometry::Point(DVec3::ZERO));
        for (k, v) in pairs {
            f.attributes.insert((*k).to_string(), v.clone());
        }
        f
    }

    /// **The stated floor count wins, a stated height is next, and the default is
    /// last — and every feature says which.**
    #[test]
    fn a_floor_count_comes_from_the_attribute_then_the_height_then_the_default() {
        let d = FootprintDefaults::default();

        // 1. Stated outright, under a spelling with a separator in it.
        let a = footprint_attrs(&feature(&[("NUM_FLOORS", Attr::Number(11.0))]), &d);
        assert_eq!((a.floors, a.floor_source), (11, FloorSource::Attribute));

        // …and it wins over a height that disagrees.
        let a = footprint_attrs(
            &feature(&[
                ("building:levels", Attr::Text("4".into())),
                ("HEIGHT", Attr::Number(60.0)),
            ]),
            &d,
        );
        assert_eq!(
            (a.floors, a.floor_source),
            (4, FloorSource::Attribute),
            "a counted floor beats a measured height"
        );
        assert_eq!(a.height_m, Some(60.0), "the height is still carried");

        // 2. Derived from a height.
        let a = footprint_attrs(&feature(&[("BLDG_HEIGHT", Attr::Number(30.0))]), &d);
        assert_eq!(
            (a.floors, a.floor_source),
            (10, FloorSource::DerivedFromHeight),
            "30 m at 3 m a storey is 10 floors"
        );

        // A FEET column is feet. Reading it as metres builds a downtown at a
        // third of its height, which is the exact defect the terrain importer's
        // own feet conversion exists to stop.
        let a = footprint_attrs(&feature(&[("HEIGHT_FT", Attr::Number(100.0))]), &d);
        assert!((a.height_m.unwrap() - 30.48).abs() < 1e-9, "{a:?}");
        assert_eq!(a.floors, 10);

        // 3. Nothing stated.
        let a = footprint_attrs(&feature(&[("NAME", Attr::Text("Shed".into()))]), &d);
        assert_eq!(
            (a.floors, a.floor_source),
            (DEFAULT_FLOORS, FloorSource::Default)
        );
        assert_eq!(a.height_m, None);
    }

    /// Hostile and absent values become defaults, never zero-storey buildings or
    /// three-thousand-storey ones.
    #[test]
    fn sentinels_and_non_finite_attributes_are_refused_at_this_door() {
        let d = FootprintDefaults::default();
        for bad in [
            Attr::Number(0.0),
            Attr::Number(-3.0),
            Attr::Number(f64::NAN),
            Attr::Number(f64::INFINITY),
            Attr::Null,
        ] {
            let a = footprint_attrs(&feature(&[("floors", bad.clone())]), &d);
            assert_eq!(
                a.floor_source,
                FloorSource::Default,
                "{bad:?} is not a floor count"
            );
            assert!(a.floors >= 1);
        }
        // The 9999 sentinel a real height column carries.
        let a = footprint_attrs(&feature(&[("HEIGHT", Attr::Number(9999.0))]), &d);
        assert_eq!(
            a.floors, MAX_ATTR_FLOORS,
            "a sentinel height is clamped, not planned"
        );
        assert!(
            a.clamped,
            "…and the clamp SAYS SO — a whole layer of sentinels is a downtown of \
             200-storey towers with nothing to see (I2 audit)"
        );
        // A stated count past the ceiling clamps too, and also says so.
        let a = footprint_attrs(&feature(&[("floors", Attr::Number(5000.0))]), &d);
        assert_eq!(a.floors, MAX_ATTR_FLOORS);
        assert!(a.clamped);
        // A count the ceiling did not touch does NOT say so.
        assert!(!footprint_attrs(&feature(&[("floors", Attr::Number(12.0))]), &d).clamped);
        assert!(!footprint_attrs(&feature(&[]), &d).clamped);
        // A degenerate storey height falls back rather than dividing by ~zero.
        let a = footprint_attrs(
            &feature(&[("HEIGHT", Attr::Number(30.0))]),
            &FootprintDefaults {
                floors: 2,
                storey_height_m: 0.0,
            },
        );
        assert_eq!(a.floors, 10);
    }

    /// The coverage counts turn "this import guessed" into a sentence with a
    /// number in it.
    #[test]
    fn coverage_counts_become_advisories_with_the_remedy_named() {
        let d = FootprintDefaults::default();
        let mut cov = AttrCoverage::default();
        for _ in 0..7 {
            cov.observe(&footprint_attrs(&feature(&[]), &d));
        }
        for _ in 0..2 {
            cov.observe(&footprint_attrs(
                &feature(&[("HEIGHT", Attr::Number(12.0))]),
                &d,
            ));
        }
        cov.observe(&footprint_attrs(
            &feature(&[
                ("levels", Attr::Number(3.0)),
                ("building", Attr::Text("office".into())),
            ]),
            &d,
        ));
        assert_eq!(cov.features, 10);
        assert_eq!(cov.floors_defaulted, 7);
        assert_eq!(cov.floors_from_height, 2);
        assert_eq!(cov.floors_from_attribute, 1);
        assert_eq!((cov.typed, cov.untyped), (1, 9));

        let a = cov.advisories();
        assert_eq!(a.len(), 3, "{a:?}");
        assert!(a.iter().any(|x| x.code == "attrs.floors_defaulted"
            && x.message.contains("7 of 10")
            && x.message.contains("num_floors")));
        assert!(a
            .iter()
            .any(|x| x.code == "attrs.floors_from_height" && x.message.contains("2 of 10")));
        assert!(a.iter().any(|x| x.code == "attrs.untyped"));

        // A layer that states everything raises nothing.
        let mut good = AttrCoverage::default();
        for _ in 0..5 {
            good.observe(&footprint_attrs(
                &feature(&[
                    ("levels", Attr::Number(3.0)),
                    ("building", Attr::Text("house".into())),
                ]),
                &d,
            ));
        }
        assert!(good.advisories().is_empty(), "{:?}", good.advisories());
        assert_eq!(good.floors_clamped, 0);

        // **A SENTINEL COLUMN IS AN ADVISORY, NOT A SKYLINE** (I2 audit). The
        // clamp is per feature and was silent: a `HEIGHT` of 9999 everywhere
        // gave a whole downtown of `MAX_ATTR_FLOORS` towers with nothing that
        // said the data had been overruled.
        let mut sentinel = AttrCoverage::default();
        for _ in 0..40 {
            sentinel.observe(&footprint_attrs(
                &feature(&[
                    ("HEIGHT", Attr::Number(9999.0)),
                    ("building", Attr::Text("office".into())),
                ]),
                &d,
            ));
        }
        assert_eq!(sentinel.floors_clamped, 40);
        let a = sentinel.advisories();
        assert!(
            a.iter().any(|x| x.code == "attrs.floors_clamped"
                && x.message.contains("40 of 40")
                && x.message.contains("9999")),
            "the clamp has to name the number and the sentinel: {a:?}"
        );
    }
}
