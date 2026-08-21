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
pub const RECIPE_SCHEMA_VERSION: u32 = 1;

/// Where on Earth world `(0, 0, 0)` is.
///
/// A projected, metric CRS only. `inf_gis::anchor_at` is the door that refuses a
/// geographic one (degrees are not metres) and Web Mercator (whose "metres" are
/// inflated 1.53× at Vancouver's latitude, which would build the island half
/// again too large with no symptom other than everything being wrong).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

    /// Half the world square's edge — the coordinate of its east and south edges.
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
    /// How many natural-break classes the vegetated band is cut into. The
    /// lowest becomes plain, the middle meadow, the top forest.
    #[serde(default = "default_biome_classes")]
    pub classes: usize,
}

fn default_biome_classes() -> usize {
    3
}

/// The recipe.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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
    /// Where the derived stream layer is written and read from.
    pub streams: String,
    /// Where the derived lake layer is written and read from.
    pub lakes: String,
    /// The settlement sites.
    #[serde(default)]
    pub sites: Vec<Site>,
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

    /// The geo-anchor this recipe describes.
    pub fn anchor(&self) -> Result<inf_math::geo::GeoAnchor, IslandError> {
        Ok(inf_gis::anchor_at(
            &self.anchor.crs,
            self.anchor.easting_m,
            self.anchor.northing_m,
            self.anchor.height_m,
            &self.anchor.vertical_datum,
        )?)
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
        if !(self.roads.max_grade > 0.0) {
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
        for s in &self.sites {
            if !(s.x.is_finite() && s.z.is_finite() && s.radius_m.is_finite()) {
                return Err(IslandError::Settings(format!(
                    "site {:?} has a non-finite position or radius",
                    s.name
                )));
            }
            let half = self.grid.half_extent_m();
            if s.x.abs() > half || s.z.abs() > half {
                return Err(IslandError::Settings(format!(
                    "site {:?} is at ({:.1}, {:.1}) which is outside the world's \
                     own ±{half:.1} m — a road to it would leave the terrain and \
                     the drape would take the centreline's own elevation",
                    s.name, s.x, s.z
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
schema_version = 1
name = "Tiny"
seed = 7
coast = "layers/coast.geojson"
streams = "layers/streams.geojson"
lakes = "layers/lakes.geojson"

[anchor]
crs = "EPSG:32610"
easting_m = 492600.0
northing_m = 5465600.0

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

        // A site outside the world is refused with the world's own half extent.
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
