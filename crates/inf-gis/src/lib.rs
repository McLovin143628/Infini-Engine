//! GIS import: CRS transforms, vector features, tile math, classification (Wave G).
//!
//! This crate is the engine's **one door for real-world geospatial data**. It
//! turns the coordinates governments publish — eastings and northings in a named
//! projection, degrees in a geographic one — into the engine's world metres, and
//! it turns the shapes they publish (polylines, polygons, attribute tables) into
//! primitives the PCG grammar, the hydrology tools and the biome painter already
//! understand.
//!
//! # The five things a game-engine GIS path actually needs
//!
//! In dependency order, because none of them mean anything without the one
//! before:
//!
//! 1. **A world geo-anchor** — one place that says "world `(0,0,0)` is *here* on
//!    Earth, in *this* CRS". That is [`inf_math::geo::GeoAnchor`], and it lives
//!    in `inf-math` rather than here so that `inf-scene` can persist it without
//!    the shipped player linking a projection library.
//! 2. **Elevation in** — a DEM raster becomes an `.inf_terrain`. That is
//!    `inf_terrain::import` (the GeoTIFF arm), not this crate.
//! 3. **CRS → world metres** — [`crs`], so a road at UTM easting 491 234 lands on
//!    the terrain it belongs on.
//! 4. **Vector in** — [`vector`], polylines and polygons with attributes,
//!    normalised into one [`GeoFeature`] type.
//! 5. **Generators** — the things that turn a polyline into a road and a polygon
//!    into a lot. [`roads`], [`triangulate`].
//!
//! # Host-only, by rule
//!
//! Everything here runs at **import and cook time**, never at play time. The
//! shipped player loads `.inf_terrain`, `.inf_pack` and `.inf_lvl`; it never
//! parses a GeoTIFF, a Shapefile or a GeoJSON, exactly as `inf-mesh` gates
//! `meshopt`/`gltf`/`image` out of the runtime. That is the doc's own G13
//! prescription and it was already this engine's cook doctrine.
//!
//! # Two hazards this crate exists to make loud
//!
//! **Axis order is a silent killer.** EPSG:4326 is authority-defined as
//! *latitude, longitude* — and virtually every file on Earth stores it
//! *longitude, latitude*. GeoCanvas's own notes record that skipping the
//! correction "produces silently transposed coordinates with no error". A
//! transposed import puts Vancouver in the Indian Ocean and nothing complains.
//! [`crs`] therefore pins one explicit convention and tests it.
//!
//! **`proj4rs` returns NaN on failure by default.** Its relaxed mode is
//! deliberate (it targets browser mapping apps, where a failed reprojection
//! should not abort a pan), and it is exactly wrong for an import door: a NaN
//! easting becomes a NaN world coordinate becomes a NaN vertex, and `f32::min`
//! /`f32::max` ignore NaN, so the bounds of the thing built from it look
//! perfectly healthy. Every transform in [`crs`] is therefore finiteness-checked
//! **at the door**, and a non-finite result is a refusal with the offending
//! coordinate named. This is the same law the terrain importer's own finiteness
//! door encodes, met one layer earlier.
//!
//! # What we gave up by refusing GDAL, stated plainly
//!
//! `gdal`, `proj-sys`/`proj` and `geos` were refused: native C on a three-OS CI
//! with no vcpkg or brew step, and `geos` is LGPL-2.1, which is not on the deny
//! allow-list at all. The bounded, named cost is: the exotic raster drivers
//! (JPEG2000, HDF, netCDF, MrSID, ECW), LERC-compressed COGs, on-the-fly warping
//! between CRSs, OGR's universal vector driver set, GEOS geometry predicates and
//! buffering, and PROJ's full EPSG database with its NTv2 datum grids. Each of
//! those has a stated author-side workaround in
//! `docs/memos/wave-g-terrain-gis-disposition.md`. Nothing here pretends to be
//! PROJ; where we are less accurate than PROJ would be, we **say so** (see
//! [`crs::Transform::datum_advisory`]).

pub mod buildings;
pub mod classify;
pub mod crs;
pub mod epsg;
pub mod feature;
pub mod import;
pub mod roads;
pub mod terrarium;
pub mod tilemath;
pub mod triangulate;
pub mod vector;

// The crate root re-exports the doors a caller outside this crate would reach
// for. `classify_to_ids` and the terrarium tile readers were reachable only as
// `inf_gis::classify::…` / `inf_gis::terrarium::…` — which made "callable"
// narrowly true and practically false for the two paths the disposition memo
// names most often.
pub use buildings::{
    footprint_attrs, AttrCoverage, FloorSource, FootprintAttrs, FootprintDefaults,
};
pub use classify::{class_of, classify_breaks, classify_to_ids, ClassifyMethod};
pub use crs::{anchor_at, Transform, MAX_LATITUDE_DEG};
pub use epsg::{proj4_for, suggested_utm_epsg, utm_zone_for, EPSG_TABLE};
pub use feature::{Attr, GeoFeature, GeoGeometry, GeoLayer, LayerKind};
pub use import::{
    import_layer, plan_spawn, probe, resolve_crs, CrsChoice, CrsOrigin, GisProbe, ImportOptions,
    ImportRequest, LayerImport, PlannedEntity, PlannedKind, SpawnPlan, DEFAULT_MAX_ENTITIES,
    ISLAND_MAX_ENTITIES,
};
pub use roads::{
    build_all_ribbons, build_ribbon, build_surface, densify_spine, surface_to_mesh, Intersection,
    MeshBuildReport, RoadGraph, RoadKind, RoadRibbon, RoadSegment, RoadSurface, SurfaceOptions,
    DEFAULT_GROUND_STEP_M, DEFAULT_ROAD_LIFT_M, LANE_WIDTH_M,
};
pub use terrarium::{
    decode_elevation, decode_tile_png, decode_tile_rgb, encode_elevation, TerrariumTile,
};
pub use tilemath::{
    plan_zoom, tile_mercator_bounds, tile_of_lonlat, MERC_HALF_WORLD, MERC_WORLD, TILE_PX,
};
pub use triangulate::{triangulate_polygon, Triangulation};
pub use vector::{read_geojson, read_shapefile, read_vector};

/// Everything that can go wrong importing geospatial data.
///
/// **Refusals are values** (the house law): a caller puts the reason in front of
/// the author rather than unwrapping, and every variant carries enough to act on.
#[derive(Debug, thiserror::Error)]
pub enum GisError {
    /// The level has no usable geo-anchor. Carries the anchor's own reason.
    #[error("{0}")]
    Anchor(#[from] inf_math::geo::GeoAnchorError),
    /// A CRS string could not be resolved to a projection.
    #[error("unsupported coordinate reference system: {0}")]
    Crs(String),
    /// A transform produced a non-finite coordinate — see the module docs on
    /// `proj4rs`'s relaxed mode.
    #[error("{0}")]
    NotFinite(String),
    /// The file is not the format its extension claims, or is malformed.
    #[error("{0}")]
    Format(String),
    /// The geometry is present but cannot be used as asked (a polygon with two
    /// points, a ring that does not close, a self-intersecting boundary).
    #[error("{0}")]
    Geometry(String),
    /// Filesystem trouble.
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for GisError {
    fn from(e: std::io::Error) -> Self {
        GisError::Io(e.to_string())
    }
}

/// A named, non-fatal hazard found while importing.
///
/// The P16 law: **a silent hazard gets an advisory, not a silent fix.** An
/// assumed axis order, a datum shift with no grid available, a vertical datum we
/// cannot convert, a CRS we had to guess — each of those produces a usable import
/// and a wrong one, and the difference between the two is whether anybody was
/// told. GeoCanvas's divergence check is the precedent: it warns past ~1 m and
/// deliberately does *not* refuse, because "a 30 m datum offset is still a usable
/// working canvas".
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Advisory {
    /// A stable machine-readable tag (`"datum.no_grid"`, `"axis.assumed"`, …).
    pub code: String,
    /// One sentence an author can act on.
    pub message: String,
}

impl Advisory {
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for Advisory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}
