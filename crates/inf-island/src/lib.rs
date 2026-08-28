//! The island generator — a **recipe**, not a committed world (wave I7).
//!
//! # Why a recipe
//!
//! Fifty square kilometres of one-metre terrain is a quarter of a gigabyte of
//! heights before a pyramid, a road mesh or a biome map. Committing that would
//! put a build artifact in a source tree and make every future change to the
//! sampler a quarter-gigabyte diff. So this repository commits **the generator**
//! — the recipe, the designed coastline, the road network, the derived stream
//! and lake layers, the biome masks and the level — and the heavy halves are
//! built by one command on the machine that wants to run the island.
//!
//! That is the samples law read at a scale it has not been read at before:
//! *everything committed regenerates through its generator*, and where the
//! output is too large to commit, the **generator** is what is committed and the
//! CI-scale fixture is what proves it works.
//!
//! # The steps, in order
//!
//! [`BuildStep`] is the ordered list, and it is an enum with no wildcard match
//! anywhere that consumes it, so a step added later is a compile error rather
//! than a silently skipped stage (the `phase29_gate` A3 lesson, applied to a
//! pipeline):
//!
//! 1. [`BuildStep::Plan`] — which source tiles the recipe's extent needs.
//!    Pure: no I/O, no network.
//! 2. [`BuildStep::Fetch`] — fill the cache. **The only step that touches the
//!    network, and it is not in this crate**: Ring 0 decides *which* tiles
//!    ([`source::TilePlan`]) and *where they live*
//!    ([`source::cache_path`]/[`source::tile_url`]); the `inf` CLI does the
//!    transfer. CI never runs it — its fixture's bytes are committed.
//! 3. [`BuildStep::Sample`] — real elevation onto the world grid, by inverse
//!    mapping (every destination sample asks the source where it came from), so
//!    nothing is resampled twice.
//! 4. [`BuildStep::Carve`] — the designed coastline. The map is an ISLAND: the
//!    sea outside the shore is carved down a shelf, a beach band is flattened
//!    inside it, and nodata becomes ocean rather than a flat plain.
//! 5. [`BuildStep::Hydrology`] — flow accumulation over the carved ground →
//!    streams; filled pits → lakes; steep stream segments → waterfall sites.
//! 6. [`BuildStep::Biomes`] — Jenks natural breaks over height and slope into
//!    the island's palette, then the design masks stamped over the result.
//! 7. [`BuildStep::Roads`] — the designed network draped and **graded**: a road
//!    that climbs faster than the recipe allows is reported with the number.
//! 8. [`BuildStep::Pyramid`] — the LOD ladder, through `inf_terrain::build_pyramid`.
//! 9. [`BuildStep::Write`] — the `.inf_terrain`, the road mesh, and the layers.
//!
//! # What is portable and what is not
//!
//! **The sampling step is not bit-portable and everything after it is.** The
//! inverse map goes through `inf_gis::crs` and `inf_gis::tilemath`, the two
//! modules Wave G's portability gate exempts **by name** because a projection is
//! `tan`/`ln`/`atan` all the way down and `std`'s are not bit-identical across
//! platforms (the P14 law). Two consequences, both stated rather than papered
//! over:
//!
//! * the `.inf_terrain` is a **build artifact of one machine**, which is why it
//!   is not committed and why nothing compares it across operating systems;
//! * the **derived layers are committed design artifacts**, derived once and
//!   thereafter *verified* rather than re-derived — [`report::LayerDrift`]
//!   carries the comparison and raises an advisory instead of failing a build on
//!   somebody else's libm.
//!
//! Everything this crate computes *itself* — the carve, the flow accumulation,
//! the classifier, the grade audit — uses only arithmetic and
//! `inf_math::portable`, and `tests/portable_math_law.rs` is what keeps it that
//! way.

#![forbid(unsafe_code)]

pub mod biome;
pub mod build;
pub mod detail;
pub mod hydro;
pub mod layers;
pub mod recipe;
pub mod report;
pub mod roads;
pub mod shape;
pub mod source;
pub mod splat;
pub mod terrain;

pub use biome::{biome_set, classify_biomes, BiomeClassification, BiomeMask, IslandBiome};
pub use build::{
    biome_set_guid, build_island, cover_pcg_guid, level_guid, player_start, read_design,
    road_mesh_guid, slug, terrain_guid, write_content, BuildOptions, IslandBuild, IslandDesign,
    StepLog, DERIVATION_PITCH_M,
};
pub use detail::{apply_detail, DetailBand, DetailPlan, DetailStats};
pub use hydro::{FlowField, HydroParams, Lake, Stream, StreamNetwork, Waterfall};
pub use recipe::{
    AnchorSpec, BiomeSpec, GridSpec, HydroSpec, IslandRecipe, RoadSpec, SeaSpec, Site, SiteKind,
    SourceSpec, ANCHOR_AGREEMENT_DEG, RECIPE_SCHEMA_VERSION,
};
pub use report::{IslandReport, LayerDrift};
pub use roads::{
    grade_audit, nearest_route_vertex, plan_network, road_graph, GradeAudit, RoadReport, Route,
};

/// The routable road network [`road_graph`] answers with.
///
/// **Re-exported on purpose** (island wave VEH1a). `inf-gis`'s transcendental
/// exemption rests on nothing that cooks or ships re-deriving a coordinate
/// through it, and `inf-gis/tests/portable_math_law.rs` keeps that true with a
/// **manifest** ban: `runtime/inf-player/Cargo.toml` may not name the crate in
/// any dependency section, dev-dependencies included. A gate that needs the
/// island's own graph therefore reaches it through this crate — which is
/// dev-only to the player already, and whose linkage the same law governs — and
/// never spells `inf_gis` itself.
pub use inf_gis::RoadGraph;

/// How far above the ground the island draws its road ribbon, metres.
///
/// Re-exported for the same reason as [`RoadGraph`]: a gate that prices the
/// **sink** — the gap between the tarmac a driver sees and the heightfield the
/// wheels are actually cast into, since roads carry no colliders — has to read
/// the builder's own number rather than restate it, and it cannot name
/// `inf-gis` to do so.
pub use inf_gis::roads::DEFAULT_ROAD_LIFT_M;
pub use shape::{
    carve_sample, flatten_sample, smooth01, Coastline, Field, SegmentIndex, ShapeStats, Vertex3,
};
pub use source::{
    cache_path, plan_tiles, tile_url, TileId, TileMosaic, TilePlan, PLAUSIBLE_ELEVATION_M,
};
pub use splat::{
    biome_mix, stamp_splat, SplatField, SplatRules, SplatStats, LAYER_FOREST_FLOOR, LAYER_GRASS,
    LAYER_ROCK, LAYER_SAND,
};
pub use terrain::{sample_terrain, CoarseHeights, IslandGrid, ProjectionLattice, SampleStats};

/// Every step of the recipe, in the order the build runs them.
///
/// Frozen order: [`BuildStep::ALL`] is what the CI fixture's coverage arm
/// enumerates, so a step that stops running is a red test rather than a quiet
/// omission. The gate reads a *count* per step, not a substring — the I1 audit's
/// "a `contains` needle that is a prefix of a declaration can never fail".
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BuildStep {
    /// Which source tiles the extent needs (pure).
    Plan,
    /// Fill the tile cache from the network. Local only; never in CI.
    Fetch,
    /// Real elevation onto the world grid by inverse mapping.
    Sample,
    /// The designed coastline: sea shelf, beach band, nodata → ocean.
    Carve,
    /// Flow accumulation → streams; filled pits → lakes; waterfall sites.
    Hydrology,
    /// Jenks over height and slope, then the design masks.
    Biomes,
    /// The designed network draped and graded.
    Roads,
    /// The fBm band below the source's own Nyquist — designed relief, in the one
    /// slot where nothing measured earlier can move (wave TER2b).
    Detail,
    /// The LOD ladder.
    Pyramid,
    /// The `.inf_terrain`, the road mesh and the layers.
    Write,
}

impl BuildStep {
    /// Every step, in build order. The CI fixture asserts it covers all of these.
    pub const ALL: [BuildStep; 10] = [
        BuildStep::Plan,
        BuildStep::Fetch,
        BuildStep::Sample,
        BuildStep::Carve,
        BuildStep::Hydrology,
        BuildStep::Biomes,
        BuildStep::Roads,
        BuildStep::Detail,
        BuildStep::Pyramid,
        BuildStep::Write,
    ];

    /// The step's stable name — what a report prints and a gate matches on.
    pub const fn label(self) -> &'static str {
        match self {
            BuildStep::Plan => "plan",
            BuildStep::Fetch => "fetch",
            BuildStep::Sample => "sample",
            BuildStep::Carve => "carve",
            BuildStep::Hydrology => "hydrology",
            BuildStep::Biomes => "biomes",
            BuildStep::Roads => "roads",
            BuildStep::Detail => "detail",
            BuildStep::Pyramid => "pyramid",
            BuildStep::Write => "write",
        }
    }

    /// `true` for the one step that needs a network.
    ///
    /// Read by the CLI (which refuses it under `--offline`) and by the fixture's
    /// coverage arm, which asserts the *offline* build covers every step whose
    /// answer is `false` and reports the fetch as satisfied-from-cache.
    pub const fn needs_network(self) -> bool {
        matches!(self, BuildStep::Fetch)
    }
}

impl std::fmt::Display for BuildStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// What went wrong, as a value.
///
/// Refusals are values (the house law): every variant names the remedy, because
/// an island build is a twenty-minute operation and "it failed" is not a thing an
/// author can act on twenty minutes in.
#[derive(Debug, thiserror::Error)]
pub enum IslandError {
    /// The recipe file could not be read or parsed.
    #[error("the island recipe at {path}: {message}")]
    Recipe { path: String, message: String },
    /// A recipe value is outside what the generator can honour.
    #[error("{0}")]
    Settings(String),
    /// A source tile the plan named is missing from the cache.
    #[error(
        "the source tile {z}/{x}/{y} is not in the cache at {cache} — run \
         `inf island build` without `--offline` to fetch it, or point `[source] \
         cache` at a directory that has it"
    )]
    MissingTile {
        z: u8,
        x: u32,
        y: u32,
        cache: String,
    },
    /// A cached tile is not a terrarium PNG.
    #[error(
        "the cached tile {z}/{x}/{y} at {path} is not a terrarium PNG ({message}) \
         — delete it and re-fetch; a 404 body cached as a tile decodes as \
         nothing and would build a flat plain where a mountain is"
    )]
    BadTile {
        z: u8,
        x: u32,
        y: u32,
        path: String,
        message: String,
    },
    /// A GIS door refused.
    #[error("{0}")]
    Gis(#[from] inf_gis::GisError),
    /// A terrain door refused.
    #[error("terrain: {0}")]
    Terrain(String),
    /// Plain I/O.
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for IslandError {
    fn from(e: std::io::Error) -> Self {
        IslandError::Io(e.to_string())
    }
}

/// A named, non-fatal finding.
///
/// The same shape as [`inf_gis::Advisory`] and for the same reason: a hazard the
/// author cannot see is a defect, and a build that silently smooths one over is
/// worse than one that stops.
pub type Advisory = inf_gis::Advisory;
