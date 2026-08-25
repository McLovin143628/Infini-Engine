//! The recipe — the committed document an island is built from.
//!
//! # What belongs in here and what does not
//!
//! The recipe carries **decisions**: where on Earth the world is, how fine its
//! grid is, which elevation source to ask, where the sea is, which layer files
//! are the design, and what the settlement sites are. It does **not** carry
//! anything derived — no heights, no stream vertices, no biome ids. A recipe
//! that carried a derived number would be a second authority on it, and the
//! first thing that changes about a generator is the derivation.
//!
//! # The one restatement, and what it cost to learn it belongs here
//!
//! [`AnchorSpec`] states the origin **twice**: once as an easting/northing in a
//! projected CRS, and once as a latitude/longitude/convergence in degrees. That
//! is a restatement rather than a derivation, and it is here because the
//! alternative shipped a red CI on two platforms. The degrees used to be
//! inverted out of the easting/northing by `inf_gis::anchor_at` every time a
//! design was read — and that inversion is `proj4rs`, which is a series over
//! `sin`/`cos`/`atan2` and therefore a fact about the host's libm. The island's
//! `.inf_lvl` **commits** the anchor, so macOS produced
//! `origin_latitude_deg = 49.34307562364772` where Windows had blessed
//! `…773`: one ulp, one byte, one red gate.
//!
//! A restatement needs a check, and it has one:
//! `crates/inf-island/tests/stated_anchor.rs` inverts each committed recipe's
//! easting/northing through `proj4rs` and asserts the stated degrees agree
//! within [`ANCHOR_AGREEMENT_DEG`]. That is the crate's own "the derived layers
//! are **verified**, never re-derived" rule, applied to three numbers.
//!
//! # The seed rule
//!
//! Every stochastic step reads [`IslandRecipe::seed`] through a **named salt**
//! (`seed_for`), never the raw number. Two steps that both read `seed` directly
//! would move together the day one of them changed how many numbers it draws —
//! the class of defect P19's `pass_seed`/`building::pass_seed` split exists to
//! prevent, one layer up.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::IslandError;

/// The recipe's own schema version. Bumping it is a recipe change, not an engine
/// schema change — nothing on disk in a cooked pack carries this number.
///
/// **v2** (the I7 CI-red): `[anchor]` states its geodetic origin. See the module
/// docs for the byte that made it necessary.
pub const RECIPE_SCHEMA_VERSION: u32 = 2;

/// How far a stated geodetic origin may sit from the projection's own answer,
/// in degrees, before `stated_anchor.rs` calls it wrong.
///
/// **1e-8° is 1.1 mm of latitude**, and the two numbers it separates are six
/// orders of magnitude apart: a recipe states its degrees to 1e-9° (0.11 mm,
/// which is already finer than any consumer — the sun, the zone suggestion, the
/// wizard readout — can tell), so the rounding residual is at most 5e-10°, while
/// the last-ulp disagreement two libms are entitled to have at this latitude is
/// ~7e-15°. The tolerance is therefore loose enough that no platform can trip it
/// and tight enough that a transposed digit, a wrong hemisphere or a wrong zone
/// is a mile outside it.
pub const ANCHOR_AGREEMENT_DEG: f64 = 1e-8;

/// Where on Earth world `(0, 0, 0)` is.
///
/// A projected, metric CRS only. `inf_gis::require_projected_crs` is the door
/// that refuses a geographic one (degrees are not metres) and Web Mercator
/// (whose "metres" are inflated 1.53× at Vancouver's latitude, which would build
/// the island half again too large with no symptom other than everything being
/// wrong).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorSpec {
    /// An authority code (`"EPSG:32610"`) or a full proj4 string.
    pub crs: String,
    /// The CRS easting world `X = 0` means.
    pub easting_m: f64,
    /// The CRS northing world `Z = 0` means. **`+Z` is south.**
    pub northing_m: f64,
    /// The elevation world `Y = 0` means.
    #[serde(default)]
    pub height_m: f64,
    /// The origin's geodetic latitude, degrees, `+` north.
    ///
    /// **Stated, not inverted** — see the module docs. Required, with no serde
    /// default, because a default would be a silent zero: an island whose
    /// recipe forgot this line would sit at the equator, its sun would be wrong
    /// all year, and nothing would say so.
    pub latitude_deg: f64,
    /// The origin's geodetic longitude, degrees, `+` east.
    pub longitude_deg: f64,
    /// Grid convergence at the origin, degrees from grid north to true north.
    ///
    /// Zero on the CRS's central meridian and growing toward a zone edge. Stated
    /// for the same reason as the two above: it comes out of the same inversion.
    pub convergence_deg: f64,
    /// Recorded for the author; this engine applies no geoid model.
    #[serde(default = "default_vertical_datum")]
    pub vertical_datum: String,
}

fn default_vertical_datum() -> String {
    "EGM2008".to_string()
}

/// The world grid.
///
/// `tiles` is the count **per axis**, so the world is a square of
/// `tiles × (tile_resolution − 1) × meters_per_sample` metres, centred on the
/// anchor. Centred rather than corner-anchored on purpose: a player start near
/// the world origin is a player start near where the floating origin starts, and
/// a 50 km² world hung off one corner spends its whole east half at seven
/// kilometres of f32 exponent before the first rebase.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GridSpec {
    /// Samples per tile edge. A tile's last row **is** its neighbour's first.
    pub tile_resolution: u32,
    /// World metres between samples.
    pub meters_per_sample: f64,
    /// Level-0 tiles per axis.
    pub tiles: u32,
}

impl GridSpec {
    /// One tile's world span in metres — `(resolution − 1) × mps`.
    pub fn tile_span_m(&self) -> f64 {
        f64::from(self.tile_resolution.saturating_sub(1)) * self.meters_per_sample
    }

    /// The world square's edge in metres.
    pub fn extent_m(&self) -> f64 {
        f64::from(self.tiles) * self.tile_span_m()
    }

    /// Half the world square's edge.
    ///
    /// **Not the coordinate of its east and south edges when `tiles` is odd.**
    /// The world is laid out on integer level-0 tile coordinates starting at
    /// `IslandGrid::tile0 = -(tiles / 2)`, and an odd count cannot be centred on
    /// an integer boundary — so the world's real corners are
    /// [`IslandGrid::bounds`](crate::IslandGrid::bounds), which sit half a tile
    /// span east and south of `±half_extent_m`. Everything that decides *where
    /// the world is* — the site check, the source plan, the carve — reads
    /// `bounds()`; this is the plain arithmetic, for a report.
    pub fn half_extent_m(&self) -> f64 {
        self.extent_m() * 0.5
    }

    /// Level-0 tile count.
    pub fn tile_count(&self) -> u64 {
        u64::from(self.tiles) * u64::from(self.tiles)
    }

    /// Level-0 samples — the number an author should look at before starting a
    /// build, because it is what the memory bound is a function of.
    pub fn sample_count(&self) -> u64 {
        self.tile_count() * u64::from(self.tile_resolution) * u64::from(self.tile_resolution)
    }
}

/// The elevation source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceSpec {
    /// Only `"terrarium"` today — the keyless, worldwide XYZ DEM Wave G ported
    /// the codec for. Named rather than assumed so a second source is a new
    /// value here instead of a new branch everywhere.
    pub kind: String,
    /// XYZ zoom. **The dataset's own ceiling is a fact about the dataset**: AWS
    /// terrain-tiles serves terrarium to z15 and answers `NoSuchKey` above it,
    /// which is a 299-byte XML body that a cache would happily keep. See
    /// [`crate::IslandError::BadTile`].
    pub zoom: u8,
    /// The tile URL template. `{z}`, `{x}` and `{y}` are substituted.
    pub url: String,
    /// The cache directory, **relative to the recipe file**. Keep it outside the
    /// repository: the whole point of a recipe is that the bytes it fetches are
    /// not source.
    pub cache: String,
    /// Extra tiles fetched beyond the extent's own footprint, so the bilinear
    /// sampler never asks for a pixel off the mosaic's edge.
    #[serde(default = "default_tile_margin")]
    pub tile_margin: u32,
}

fn default_tile_margin() -> u32 {
    1
}

/// The sea, and the shape of the shore.
///
/// These four numbers are what turn a piece of the North Shore into an island.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeaSpec {
    /// Sea level in world metres.
    pub level_m: f64,
    /// How far below sea level the carved sea floor settles.
    pub shelf_depth_m: f64,
    /// The distance over which the sea floor falls from the shore to the shelf.
    pub shelf_width_m: f64,
    /// The band **inside** the shore that is flattened toward sea level — the
    /// beach. Zero means a cliff coast everywhere.
    pub beach_width_m: f64,
    /// How high above sea level the beach's inner edge is allowed to sit before
    /// the flattening stops. A shore under a mountain gets a cliff, which is
    /// what a shore under a mountain is.
    #[serde(default = "default_beach_rise")]
    pub beach_rise_m: f64,
}

fn default_beach_rise() -> f64 {
    6.0
}

/// A settlement site — a place the roads connect and I8's generator fills.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Site {
    pub name: String,
    pub kind: SiteKind,
    /// World X (east).
    pub x: f64,
    /// World Z (**south** — north is negative).
    pub z: f64,
    /// The radius the ground is flattened over, and the radius the biome map
    /// reserves for urban use.
    pub radius_m: f64,
}

/// What a site is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SiteKind {
    /// One of the two city sites. Flattened hardest, reserved widest.
    City,
    /// One of the five town sites.
    Town,
    /// A junction or viewpoint the circuit passes through but nobody lives at.
    Waypoint,
}

impl SiteKind {
    /// The stable name a report prints.
    pub const fn label(self) -> &'static str {
        match self {
            SiteKind::City => "city",
            SiteKind::Town => "town",
            SiteKind::Waypoint => "waypoint",
        }
    }

    /// `true` when a site of this kind reserves urban biome around itself.
    pub const fn reserves_urban(self) -> bool {
        matches!(self, SiteKind::City | SiteKind::Town)
    }
}

/// The road network's design rules.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoadSpec {
    /// The committed GeoJSON, relative to the recipe.
    pub layer: String,
    /// The steepest grade a road of any class may hold, as a **rise over run
    /// fraction** — 0.08 is 8 %, which is about the steepest a highway is built
    /// at anywhere. A segment above it is reported with its own number and its
    /// position, because "switchbacks where the terrain demands" is a design
    /// instruction and the audit is how an author knows where it was ignored.
    pub max_grade: f64,
    /// How far apart the grade audit samples along a centreline.
    #[serde(default = "default_grade_step")]
    pub grade_step_m: f64,
    /// The corridor either side of a centreline the ground is levelled across,
    /// as a multiple of the road's own half width. `0` leaves the ground alone.
    #[serde(default = "default_road_shoulder")]
    pub shoulder_mult: f64,
}

fn default_grade_step() -> f64 {
    20.0
}

fn default_road_shoulder() -> f64 {
    1.6
}

/// The biome classifier's design.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BiomeSpec {
    /// The committed design-mask GeoJSON, relative to the recipe. Each feature
    /// is a polygon carrying a `biome` attribute naming one of
    /// [`crate::IslandBiome`]'s labels; the masks are stamped **over** the
    /// classifier's answer.
    pub masks: String,
    /// Elevation above which the classifier answers alpine, in metres. Chosen by
    /// the author rather than by Jenks because a treeline is a fact about a
    /// place, not about a histogram.
    pub alpine_m: f64,
    /// Slope above which the classifier answers rock, in degrees.
    pub rock_deg: f64,
    /// How far inland from the water line beach reaches, in metres.
    pub beach_m: f64,
    /// How many natural-break classes the vegetated band is cut into.
    #[serde(default = "default_biome_classes")]
    pub classes: usize,
    /// What each natural-break class MEANS, lowest first.
    ///
    /// # Why the mapping is the author's and not the classifier's
    ///
    /// Jenks finds where the histogram's gaps are. It has no opinion about what
    /// grows in each band, and a hard-coded "lowest is plain, top is forest"
    /// carries an opinion about *somewhere else*: on this coast the forest is the
    /// bulk of the island and the open ground is the exception, so the first
    /// build's ladder put **9.8 % of the land under canopy** and called a third
    /// of a rain-forest island a grassy plain.
    ///
    /// Shorter than `classes` is legal — the last entry repeats — so
    /// `["plain", "meadow", "forest"]` over four classes merges the top two.
    #[serde(default = "default_class_biomes")]
    pub class_biomes: Vec<String>,
}

fn default_biome_classes() -> usize {
    3
}

fn default_class_biomes() -> Vec<String> {
    vec!["plain".into(), "meadow".into(), "forest".into()]
}

/// How the water is derived.
///
/// Every one of these is a threshold, and a threshold is a design decision about
/// what counts — a catchment sized for a continent finds one river on an island,
/// and one sized for a hillside finds a hundred rivulets that are really the
/// source's own noise.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HydroSpec {
    /// The catchment a cell needs before it is called a stream, square metres.
    #[serde(default = "default_catchment")]
    pub stream_catchment_m2: f64,
    /// The shallowest depression fill that counts as a lake, metres.
    #[serde(default = "default_lake_depth")]
    pub lake_depth_m: f64,
    /// The smallest lake worth a water body, square metres.
    #[serde(default = "default_lake_area")]
    pub lake_area_m2: f64,
    /// The bed gradient at which a stream segment is called a waterfall.
    #[serde(default = "default_waterfall")]
    pub waterfall_grade: f64,
    /// Derivation cells per committed stream vertex.
    #[serde(default = "default_stride")]
    pub vertex_stride: usize,
}

fn default_catchment() -> f64 {
    1.0e6
}
fn default_lake_depth() -> f64 {
    1.5
}
fn default_lake_area() -> f64 {
    2_500.0
}
fn default_waterfall() -> f64 {
    0.5
}
fn default_stride() -> usize {
    8
}

impl Default for HydroSpec {
    fn default() -> Self {
        Self {
            stream_catchment_m2: default_catchment(),
            lake_depth_m: default_lake_depth(),
            lake_area_m2: default_lake_area(),
            waterfall_grade: default_waterfall(),
            vertex_stride: default_stride(),
        }
    }
}

/// The recipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IslandRecipe {
    /// This document's schema version.
    pub schema_version: u32,
    /// The island's display name.
    pub name: String,
    /// The one number every stochastic step salts.
    #[serde(default)]
    pub seed: u64,
    pub anchor: AnchorSpec,
    pub grid: GridSpec,
    pub source: SourceSpec,
    pub sea: SeaSpec,
    /// The designed coastline: one closed polygon, in world metres, relative to
    /// the recipe.
    pub coast: String,
    pub roads: RoadSpec,
    pub biomes: BiomeSpec,
    #[serde(default)]
    pub hydro: HydroSpec,
    /// Where the derived stream layer is written and read from.
    pub streams: String,
    /// Where the derived lake layer is written and read from.
    pub lakes: String,
    /// The settlement sites.
    #[serde(default)]
    pub sites: Vec<Site>,
    /// Committed files beside the recipe that a build copies into the project's
    /// `Content` verbatim — the level, its blueprint assets, the `.inf_pcg` the
    /// biome set binds.
    ///
    /// # Why the build copies rather than the author
    ///
    /// `inf island build` is *the one documented command*. A command that
    /// produced a terrain and then required a human to copy two more files
    /// beside it before anything could be cooked would be a command whose
    /// documentation is longer than itself — and the file that gets forgotten is
    /// always the small one.
    #[serde(default)]
    pub content: Vec<String>,
    /// The directory of the recipe file itself. Filled by [`IslandRecipe::load`]
    /// and **not** serialized — a recipe that recorded its own location would
    /// stop being true the moment it moved.
    #[serde(skip)]
    pub base_dir: PathBuf,
}

impl IslandRecipe {
    /// Read a recipe from disk, remembering where it came from.
    pub fn load(path: &Path) -> Result<Self, IslandError> {
        let text = std::fs::read_to_string(path).map_err(|e| IslandError::Recipe {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        let mut r: IslandRecipe = toml::from_str(&text).map_err(|e| IslandError::Recipe {
            path: path.display().to_string(),
            message: e.to_string(),
        })?;
        r.base_dir = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        r.validate()?;
        Ok(r)
    }

    /// Parse a recipe from text, with an explicit base directory.
    pub fn parse(text: &str, base_dir: &Path) -> Result<Self, IslandError> {
        let mut r: IslandRecipe = toml::from_str(text).map_err(|e| IslandError::Recipe {
            path: base_dir.display().to_string(),
            message: e.to_string(),
        })?;
        r.base_dir = base_dir.to_path_buf();
        r.validate()?;
        Ok(r)
    }

    /// Resolve a recipe-relative path.
    pub fn resolve(&self, rel: &str) -> PathBuf {
        self.base_dir.join(rel)
    }

    /// The tile cache directory.
    pub fn cache_dir(&self) -> PathBuf {
        self.resolve(&self.source.cache)
    }

    /// A named salt over the recipe's seed.
    ///
    /// Every stochastic step calls this with its own name, so two steps cannot
    /// share a stream by accident and adding a draw to one never moves the
    /// other. FNV-1a over the name, folded into the seed — integer-only, so it
    /// is the same number on every machine.
    pub fn seed_for(&self, what: &str) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ self.seed;
        for b in what.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// The geo-anchor this recipe describes — **assembled from stated numbers,
    /// never inverted.**
    ///
    /// The CRS is still checked (a geographic one is refused by name), because
    /// that check reads a string and a table and returns a `bool`. What it does
    /// *not* do is ask `proj4rs` where the easting/northing is: that answer goes
    /// into a committed `.inf_lvl` and would make one of its bytes a fact about
    /// the machine that blessed it. See the module docs, and
    /// `tests/stated_anchor.rs` for the arm that keeps the stated degrees true.
    pub fn anchor(&self) -> Result<inf_math::geo::GeoAnchor, IslandError> {
        inf_gis::require_projected_crs(&self.anchor.crs)?;
        let anchor = inf_math::geo::GeoAnchor {
            enabled: true,
            crs: self.anchor.crs.trim().to_string(),
            origin_easting_m: self.anchor.easting_m,
            origin_northing_m: self.anchor.northing_m,
            origin_height_m: self.anchor.height_m,
            origin_latitude_deg: self.anchor.latitude_deg,
            origin_longitude_deg: self.anchor.longitude_deg,
            grid_convergence_deg: self.anchor.convergence_deg,
            vertical_datum: self.anchor.vertical_datum.trim().to_string(),
        };
        anchor.validate().map_err(|e| {
            IslandError::Settings(format!("[anchor] cannot be used as a world frame: {e}"))
        })?;
        Ok(anchor)
    }

    /// Check every value the build would otherwise discover the hard way.
    ///
    /// A twenty-minute build that refuses in the nineteenth minute for something
    /// readable in the first is a defect in the door, not in the recipe.
    pub fn validate(&self) -> Result<(), IslandError> {
        if self.schema_version != RECIPE_SCHEMA_VERSION {
            return Err(IslandError::Settings(format!(
                "this recipe declares schema_version {}, and this build \
                 understands {RECIPE_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.grid.tile_resolution < 3 {
            return Err(IslandError::Settings(format!(
                "[grid] tile_resolution is {} — a tile needs at least 3 samples \
                 an edge for a centred difference to have a neighbour",
                self.grid.tile_resolution
            )));
        }
        if self.grid.tiles == 0 {
            return Err(IslandError::Settings(
                "[grid] tiles is 0 — the island would have no ground".to_string(),
            ));
        }
        for (name, v) in [
            ("[grid] meters_per_sample", self.grid.meters_per_sample),
            ("[sea] level_m", self.sea.level_m),
            ("[sea] shelf_depth_m", self.sea.shelf_depth_m),
            ("[sea] shelf_width_m", self.sea.shelf_width_m),
            ("[sea] beach_width_m", self.sea.beach_width_m),
            ("[sea] beach_rise_m", self.sea.beach_rise_m),
            ("[roads] max_grade", self.roads.max_grade),
            ("[roads] grade_step_m", self.roads.grade_step_m),
            ("[biomes] alpine_m", self.biomes.alpine_m),
            ("[biomes] rock_deg", self.biomes.rock_deg),
            ("[biomes] beach_m", self.biomes.beach_m),
            ("[anchor] easting_m", self.anchor.easting_m),
            ("[anchor] northing_m", self.anchor.northing_m),
            ("[anchor] height_m", self.anchor.height_m),
            ("[anchor] latitude_deg", self.anchor.latitude_deg),
            ("[anchor] longitude_deg", self.anchor.longitude_deg),
            ("[anchor] convergence_deg", self.anchor.convergence_deg),
        ] {
            if !v.is_finite() {
                return Err(IslandError::Settings(format!(
                    "{name} is not a finite number — one NaN here makes every \
                     sample of a fifty-square-kilometre terrain NaN, and a NaN \
                     height survives `f32::min`/`max`, so the asset would report \
                     perfectly healthy bounds while being unusable"
                )));
            }
        }
        if self.grid.meters_per_sample <= 0.0 {
            return Err(IslandError::Settings(format!(
                "[grid] meters_per_sample is {} — it must be positive",
                self.grid.meters_per_sample
            )));
        }
        if self.roads.grade_step_m <= 0.0 {
            return Err(IslandError::Settings(
                "[roads] grade_step_m must be positive — a zero step is a \
                 division by zero per sample"
                    .to_string(),
            ));
        }
        // Finiteness is already established above, so a plain comparison here
        // cannot be tripped by a NaN.
        if self.roads.max_grade <= 0.0 {
            return Err(IslandError::Settings(format!(
                "[roads] max_grade is {} — a non-positive grade ceiling refuses \
                 every road including a level one",
                self.roads.max_grade
            )));
        }
        if self.source.kind != "terrarium" {
            return Err(IslandError::Settings(format!(
                "[source] kind is {:?}; this build knows only \"terrarium\" — the \
                 keyless worldwide XYZ DEM whose codec `inf_gis::terrarium` \
                 carries. A GeoTIFF source imports through the terrain wizard \
                 instead.",
                self.source.kind
            )));
        }
        if self.source.zoom > inf_gis::tilemath::MAX_NATIVE_ZOOM {
            return Err(IslandError::Settings(format!(
                "[source] zoom is {}, past the {} the tile math admits",
                self.source.zoom,
                inf_gis::tilemath::MAX_NATIVE_ZOOM
            )));
        }
        if self.biomes.classes == 0 {
            return Err(IslandError::Settings(
                "[biomes] classes is 0 — the vegetated band would have no class \
                 to fall into"
                    .to_string(),
            ));
        }
        if self.biomes.class_biomes.is_empty() {
            return Err(IslandError::Settings(
                "[biomes] class_biomes is empty — Jenks would find the breaks and \
                 nothing would say what they mean"
                    .to_string(),
            ));
        }
        for n in &self.biomes.class_biomes {
            if crate::biome::IslandBiome::from_label(n).is_none() {
                return Err(IslandError::Settings(format!(
                    "[biomes] class_biomes names {n:?}, which is not one of the \
                     island's biomes"
                )));
            }
        }
        for (name, v) in [
            (
                "[hydro] stream_catchment_m2",
                self.hydro.stream_catchment_m2,
            ),
            ("[hydro] lake_depth_m", self.hydro.lake_depth_m),
            ("[hydro] lake_area_m2", self.hydro.lake_area_m2),
            ("[hydro] waterfall_grade", self.hydro.waterfall_grade),
        ] {
            if !(v.is_finite() && v > 0.0) {
                return Err(IslandError::Settings(format!(
                    "{name} is {v} — every hydrology threshold must be a positive \
                     finite number"
                )));
            }
        }
        // **The world's REAL corners**, not `±half_extent_m`. The two differ by
        // half a tile span whenever `tiles` is odd, because the grid is laid out
        // on integer tile coordinates from `-(tiles / 2)` — so a recipe with an
        // odd tile count admitted sites on ground the build never makes, and the
        // source plan asked for tiles the world does not reach.
        let (lo, hi) = crate::IslandGrid::of(self).bounds();
        for s in &self.sites {
            if !(s.x.is_finite() && s.z.is_finite() && s.radius_m.is_finite()) {
                return Err(IslandError::Settings(format!(
                    "site {:?} has a non-finite position or radius",
                    s.name
                )));
            }
            if s.x < lo.x || s.x > hi.x || s.z < lo.y || s.z > hi.y {
                return Err(IslandError::Settings(format!(
                    "site {:?} is at ({:.1}, {:.1}) which is outside the world's \
                     own [{:.1}, {:.1}] — a road to it would leave the terrain and \
                     the drape would take the centreline's own elevation",
                    s.name, s.x, s.z, lo.x, hi.x
                )));
            }
        }
        Ok(())
    }

    /// The sites of one kind, in recipe order.
    pub fn sites_of(&self, kind: SiteKind) -> impl Iterator<Item = &Site> {
        self.sites.iter().filter(move |s| s.kind == kind)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A minimal well-formed recipe, used by every test in the crate.
    pub(crate) fn tiny_recipe_text() -> String {
        r#"
schema_version = 2
name = "Tiny"
seed = 7
coast = "layers/coast.geojson"
streams = "layers/streams.geojson"
lakes = "layers/lakes.geojson"

[anchor]
crs = "EPSG:32610"
easting_m = 492600.0
northing_m = 5465600.0
latitude_deg = 49.343075624
longitude_deg = -123.101873876
convergence_deg = 0.077289249

[grid]
tile_resolution = 129
meters_per_sample = 2.0
tiles = 2

[source]
kind = "terrarium"
zoom = 14
url = "https://example.invalid/{z}/{x}/{y}.png"
cache = "cache"

[sea]
level_m = 0.0
shelf_depth_m = 40.0
shelf_width_m = 200.0
beach_width_m = 20.0

[roads]
layer = "layers/roads.geojson"
max_grade = 0.08

[biomes]
masks = "layers/biomes.geojson"
alpine_m = 700.0
rock_deg = 38.0
beach_m = 25.0
"#
        .to_string()
    }

    #[test]
    fn a_recipe_round_trips_and_derives_its_own_geometry() {
        let r = IslandRecipe::parse(&tiny_recipe_text(), Path::new("/tmp/x")).unwrap();
        assert_eq!(r.name, "Tiny");
        assert_eq!(r.grid.tile_span_m(), 256.0, "128 cells x 2 m");
        assert_eq!(r.grid.extent_m(), 512.0);
        assert_eq!(r.grid.half_extent_m(), 256.0);
        assert_eq!(r.grid.tile_count(), 4);
        assert_eq!(r.grid.sample_count(), 4 * 129 * 129);
        assert_eq!(r.source.tile_margin, 1, "the default margin is one tile");
        assert_eq!(r.biomes.classes, 3);
        assert_eq!(r.anchor.vertical_datum, "EGM2008");

        // The anchor builds and lands where the CRS says it does.
        let a = r.anchor().unwrap();
        assert!(a.enabled);
        assert!(
            (a.origin_latitude_deg - 49.34).abs() < 0.05,
            "the pinned island anchor is on the North Shore, got {}",
            a.origin_latitude_deg
        );
        assert!((a.origin_longitude_deg + 123.10).abs() < 0.05);
        // …and the degrees are the recipe's OWN numbers, carried across bit for
        // bit rather than inverted out of the easting/northing. This is the
        // assertion the I7 CI-red bought: a door that re-derived them would land
        // within the tolerance above on any machine and still write a different
        // byte on each.
        assert_eq!(a.origin_latitude_deg, r.anchor.latitude_deg);
        assert_eq!(a.origin_longitude_deg, r.anchor.longitude_deg);
        assert_eq!(a.grid_convergence_deg, r.anchor.convergence_deg);

        // Re-serializing and re-parsing preserves every decision.
        let text = toml::to_string(&r).unwrap();
        let back = IslandRecipe::parse(&text, Path::new("/tmp/x")).unwrap();
        assert_eq!(back.grid, r.grid);
        assert_eq!(back.sea, r.sea);
        assert_eq!(back.anchor, r.anchor);
    }

    /// Every named salt is distinct, and the salts move with the seed.
    ///
    /// Un-fix mutation: return `self.seed` and the first assertion fails.
    #[test]
    fn every_step_gets_its_own_stream_from_one_seed() {
        let r = IslandRecipe::parse(&tiny_recipe_text(), Path::new("/tmp/x")).unwrap();
        let names = ["coast", "hydrology", "biomes", "roads", "scatter"];
        let mut seen = std::collections::BTreeSet::new();
        for n in names {
            assert!(seen.insert(r.seed_for(n)), "{n} collides with another step");
        }
        assert_eq!(seen.len(), names.len());

        let mut r2 = r.clone();
        r2.seed = r.seed + 1;
        for n in names {
            assert_ne!(
                r.seed_for(n),
                r2.seed_for(n),
                "{n} did not move with the recipe seed"
            );
        }
        // And it is the same number on every run of every machine.
        assert_eq!(r.seed_for("coast"), r.seed_for("coast"));
    }

    #[test]
    fn the_door_refuses_what_a_twenty_minute_build_would_discover_late() {
        let base = Path::new("/tmp/x");
        let bad = |edit: &dyn Fn(&mut IslandRecipe)| {
            let mut r = IslandRecipe::parse(&tiny_recipe_text(), base).unwrap();
            edit(&mut r);
            r.validate().unwrap_err().to_string()
        };

        assert!(bad(&|r| r.schema_version = 99).contains("schema_version"));
        assert!(bad(&|r| r.grid.tiles = 0).contains("no ground"));
        assert!(bad(&|r| r.grid.tile_resolution = 2).contains("centred difference"));
        let nan = bad(&|r| r.sea.level_m = f64::NAN);
        assert!(
            nan.contains("level_m") && nan.contains("healthy bounds"),
            "{nan}"
        );
        assert!(bad(&|r| r.grid.meters_per_sample = -1.0).contains("positive"));
        assert!(bad(&|r| r.roads.max_grade = 0.0).contains("level one"));
        assert!(bad(&|r| r.roads.grade_step_m = 0.0).contains("division by zero"));
        assert!(bad(&|r| r.source.kind = "geotiff".into()).contains("terrain wizard"));
        assert!(bad(&|r| r.source.zoom = 30).contains("tile math admits"));
        assert!(bad(&|r| r.biomes.classes = 0).contains("no class"));

        // **A KEY NOTHING READS IS A LOUD ERROR.** In TOML a bare key written
        // after a `[table]` header belongs to that table, so `content = [...]`
        // placed below `[source]` becomes `source.content` — a key the recipe
        // does not have, which serde's default is to ignore. It did, and the
        // copy step quietly did nothing for a build. `deny_unknown_fields` is
        // what turns that into a message.
        let misplaced = tiny_recipe_text().replace(
            "[source]\nkind = \"terrarium\"",
            "[source]\ncontent = [\"Level.inf_lvl\"]\nkind = \"terrarium\"",
        );
        let e = IslandRecipe::parse(&misplaced, base)
            .unwrap_err()
            .to_string();
        assert!(e.contains("content"), "{e}");
        // …and at the top level the same key parses.
        let ok = tiny_recipe_text().replace(
            "lakes = \"layers/lakes.geojson\"",
            "lakes = \"layers/lakes.geojson\"\ncontent = [\"Level.inf_lvl\"]",
        );
        let r = IslandRecipe::parse(&ok, base).expect("a top-level `content` parses");
        assert_eq!(r.content, vec!["Level.inf_lvl".to_string()]);
        // A misspelled key anywhere is refused, not silently defaulted.
        let typo = tiny_recipe_text().replace("beach_width_m", "beach_wdith_m");
        assert!(IslandRecipe::parse(&typo, base)
            .unwrap_err()
            .to_string()
            .contains("beach_wdith_m"));

        // A site outside the world is refused with the world's own corners.
        let outside = bad(&|r| {
            r.sites.push(Site {
                name: "Nowhere".into(),
                kind: SiteKind::Town,
                x: 5_000.0,
                z: 0.0,
                radius_m: 10.0,
            })
        });
        assert!(
            outside.contains("256.0") && outside.contains("Nowhere"),
            "{outside}"
        );
    }

    /// **AN ODD TILE COUNT IS NOT CENTRED, AND THE DOORS THAT DECIDE WHERE THE
    /// WORLD IS NOW SAY SO.**
    ///
    /// `IslandGrid` lays the world out on integer level-0 tile coordinates from
    /// `-(tiles / 2)`, which cannot be centred when `tiles` is odd — the world
    /// sits half a tile span east and south of `±half_extent_m`. The site check
    /// and the source plan both measured the *centred* square, so an odd recipe
    /// admitted sites on ground the build never makes and asked the provider for
    /// tiles the world does not reach.
    ///
    /// Un-fix mutation: put `self.grid.half_extent_m()` back in `validate` and
    /// the west site below is admitted; put it back in `plan_tiles` and the
    /// planned longitude band no longer reaches the world's own east edge.
    #[test]
    fn an_odd_tile_count_is_measured_where_the_world_actually_is() {
        let base = Path::new("/tmp/x");
        let mut r = IslandRecipe::parse(&tiny_recipe_text(), base).unwrap();
        r.grid.tiles = 5; // 5 x 256 m; tile0 = -2, so the world is [-512, 768]
        let (lo, hi) = crate::IslandGrid::of(&r).bounds();
        assert_eq!((lo.x, hi.x), (-512.0, 768.0));
        assert_eq!(
            r.grid.half_extent_m(),
            640.0,
            "the plain arithmetic is still the plain arithmetic"
        );

        // West of the world's own edge, INSIDE the centred square: admitted
        // before, refused now, and the message names the real interval.
        let mut west = r.clone();
        west.sites.push(Site {
            name: "West".into(),
            kind: SiteKind::Town,
            x: -600.0,
            z: 0.0,
            radius_m: 10.0,
        });
        let e = west.validate().unwrap_err().to_string();
        assert!(
            e.contains("West") && e.contains("-512.0") && e.contains("768.0"),
            "{e}"
        );
        // East of the centred square but inside the world: refused before,
        // admitted now.
        let mut east = r.clone();
        east.sites.push(Site {
            name: "East".into(),
            kind: SiteKind::Town,
            x: 700.0,
            z: 0.0,
            radius_m: 10.0,
        });
        east.validate().expect("700 m east is inside [-512, 768]");

        // …and the source plan covers the world rather than the centred square:
        // the east edge's own longitude is inside the planned band.
        let plan = crate::plan_tiles(&r).expect("an odd recipe plans");
        let anchor = r.anchor().unwrap();
        let tf = inf_gis::Transform::new("EPSG:4326", &anchor).unwrap();
        let (east_lon, _, _) = tf.to_source(glam::DVec3::new(hi.x, 0.0, 0.0)).unwrap();
        let (west_lon, _, _) = tf.to_source(glam::DVec3::new(lo.x, 0.0, 0.0)).unwrap();
        println!(
            "ODD GRID: world [{:.0}, {:.0}], plan lon {:.6}..{:.6}, edges \
             {west_lon:.6} / {east_lon:.6}",
            lo.x, hi.x, plan.lon.0, plan.lon.1
        );
        assert!(
            plan.lon.0 <= west_lon && plan.lon.1 >= east_lon,
            "the plan's longitude band {:.6}..{:.6} does not reach the world's \
             own edges {west_lon:.6}..{east_lon:.6}",
            plan.lon.0,
            plan.lon.1
        );
    }

    #[test]
    fn sites_are_selected_by_kind_in_recipe_order() {
        let mut r = IslandRecipe::parse(&tiny_recipe_text(), Path::new("/tmp/x")).unwrap();
        for (n, k) in [
            ("A", SiteKind::Town),
            ("B", SiteKind::City),
            ("C", SiteKind::Town),
            ("D", SiteKind::Waypoint),
        ] {
            r.sites.push(Site {
                name: n.into(),
                kind: k,
                x: 0.0,
                z: 0.0,
                radius_m: 1.0,
            });
        }
        let towns: Vec<&str> = r
            .sites_of(SiteKind::Town)
            .map(|s| s.name.as_str())
            .collect();
        assert_eq!(towns, ["A", "C"]);
        assert_eq!(r.sites_of(SiteKind::City).count(), 1);
        assert!(SiteKind::City.reserves_urban() && SiteKind::Town.reserves_urban());
        assert!(!SiteKind::Waypoint.reserves_urban());
        assert_eq!(SiteKind::Waypoint.label(), "waypoint");
    }
}
