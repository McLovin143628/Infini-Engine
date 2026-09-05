//! The pipeline: every recipe step, in order, once.
//!
//! # The step log is the coverage proof
//!
//! [`StepLog`] records which steps ran and what each one did. The CI-scale
//! fixture asserts it names **every** member of [`BuildStep::ALL`] — a count per
//! step, not a substring, because "a `contains` needle that is a prefix of a
//! declaration can never fail" (the I1 audit). A step that stops running is a red
//! test rather than a quiet omission.
//!
//! # Fetch is not in here
//!
//! [`BuildStep::Fetch`] is logged and satisfied, never performed: Ring 0 decides
//! *which* tiles ([`crate::plan_tiles`]) and *where they live*
//! ([`crate::cache_path`]), and the `inf` CLI does the transfer. That split is
//! what lets CI run every other step against committed bytes and never touch a
//! network.

use std::path::Path;

use glam::{DVec2, DVec3};

use crate::biome::{biome_set, classify_biomes, BiomeClassification, IslandBiome};
use crate::hydro::{self, FlowField, HydroParams, Stream, StreamNetwork};
use crate::layers;
use crate::recipe::IslandRecipe;
use crate::report::{IslandReport, LayerDrift};
use crate::roads::{self, RoadReport, Route};
use crate::shape::{Coastline, SegmentIndex, Vertex3};
use crate::source::{plan_tiles, TileMosaic, TilePlan};
use crate::terrain::{self, CarvePlan, CoarseHeights, IslandGrid, ProjectionLattice};
use crate::{Advisory, BuildStep, IslandError};

/// The pitch the derivations run at, world metres.
///
/// See [`crate::terrain::CoarseHeights`] for why they get their own lattice at
/// all. Eight metres is a stream's own width: finer finds the terrain's noise at
/// a 3 m source, coarser cannot find a channel.
pub const DERIVATION_PITCH_M: f64 = 8.0;

/// What each step did.
#[derive(Clone, Debug, PartialEq)]
pub struct StepLog {
    pub step: BuildStep,
    /// One line an author reads.
    pub note: String,
}

/// How a build is asked for.
///
/// **`Default` is hand-written since wave ROAD1**, and that is not tidying:
/// `graded_roads` defaults to **true**, and a derived `Default` would have made
/// it false — so `BuildOptions::default()`, which is what `inf island build`
/// and every gate in the tree use, would have quietly built the pre-ROAD1
/// island. It did, for one measurement, and the tell was the control and the
/// subject agreeing to four decimal places.
#[derive(Clone, Debug, PartialEq)]
pub struct BuildOptions {
    /// Re-derive the streams and lakes and **overwrite** the committed layers.
    ///
    /// Off by default and it matters: the committed layers are the design, and a
    /// build that silently rewrote them every run would make an author's edit
    /// last exactly until the next build.
    pub rederive_layers: bool,
    /// Plan the road network from the sites and write the layer.
    ///
    /// Also off by default, and the same reason. This is the switch `inf island
    /// route` throws.
    pub replan_roads: bool,
    /// Skip the **heavy** halves — the terrain asset, its pyramid and the road
    /// mesh. A fast pass for the report alone.
    ///
    /// # It does NOT suppress the layer write, and that distinction is load-bearing
    ///
    /// `inf island route` is two passes: the first plans the network against the
    /// ground as it stands, the second reads that plan back and levels its
    /// corridor into the terrain before auditing it. The first pass has no use
    /// for a 342 MB terrain, so it is a dry run — and while `dry_run` also
    /// suppressed the layer write, the second pass found no committed routes,
    /// planned them again, and audited a road whose corridor had never been cut.
    /// Measured on the fixture: **15.28 % over the ceiling instead of 0 %.**
    pub dry_run: bool,
    /// **Grade the roads onto a levelled corridor plateau** (wave ROAD1).
    ///
    /// On by default, and it is what the island ships: the carve levels a flat
    /// plateau under everything a road draws, and the carriageway is built
    /// planar-with-a-crown on top of it — which is what a built road is, and
    /// what makes the ground under it survive the clipmap's LOD morph unmoved.
    ///
    /// **Off is the pre-ROAD1 island**: no plateau (the levelling eases straight
    /// from the centreline), no crown, and a ribbon that conforms to the ground
    /// at every point of its cross-section. It exists so `road1_gate` can build
    /// the control it compares against rather than asserting that a small number
    /// is small — a switch with one caller, and that caller is the measurement.
    pub graded_roads: bool,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            rederive_layers: false,
            replan_roads: false,
            dry_run: false,
            // **True**, and see the type's own note for why this impl is
            // written out rather than derived.
            graded_roads: true,
        }
    }
}

impl BuildOptions {
    /// The planning half of `inf island route`: plan the ROAD network, write its
    /// layer, and build none of the heavy halves.
    ///
    /// # `rederive_layers` is off, and it was not
    ///
    /// It was `true`, which made `inf island route` — a verb whose whole subject
    /// is the road network — silently overwrite the committed **stream and lake**
    /// layers as well. That is exactly the hazard [`BuildOptions::rederive_layers`]
    /// names two fields up: *"a build that silently rewrote them every run would
    /// make an author's edit last exactly until the next build"*. An author who
    /// moved a reach and then re-routed the roads lost the reach, with nothing
    /// said.
    ///
    /// Nothing is given up by turning it off. The write in
    /// [`build_island`] fires on `rederive_layers || !streams.exists() ||
    /// !lakes.exists()`, so a first run on an island that has no derived water
    /// yet still writes it; a later run reads what is committed — which is the
    /// same water the *second* pass will read, so the two passes now plan and
    /// audit against one ground instead of two.
    /// Re-deriving on purpose is `rederive_layers: true` spelled out.
    pub fn planning_pass() -> Self {
        Self {
            rederive_layers: false,
            replan_roads: true,
            dry_run: true,
            graded_roads: true,
        }
    }
}

/// Everything a build produced.
pub struct IslandBuild {
    pub recipe: IslandRecipe,
    pub anchor: inf_math::geo::GeoAnchor,
    pub grid: IslandGrid,
    pub plan: TilePlan,
    pub terrain: inf_terrain::TerrainData,
    pub asset: Option<inf_terrain::TerrainAsset>,
    pub mesh: Option<crate::roads::RoadMeshes>,
    pub network: StreamNetwork,
    pub routes: Vec<Route>,
    pub biomes: BiomeClassification,
    /// What the ground splat painted (TER2a clause 2) — the per-layer coverage
    /// of the four `TerrainLayer`s, measured over the level-0 samples.
    pub splat: crate::splat::SplatStats,
    pub biome_set: inf_terrain::BiomeSet,
    pub coast: Coastline,
    pub report: IslandReport,
    pub log: Vec<StepLog>,
}

impl IslandBuild {
    /// The world position the report calls the player start.
    ///
    /// # ONE DOOR
    ///
    /// This used to take the elevation from the **built terrain** while the level
    /// took it from the **committed road layer** ([`player_start`]) — two answers
    /// to one question, and `inf island build` printed the one nothing spawns at.
    /// The terrain's answer is not available to the level by construction (the
    /// level is authored from committed design alone, and the terrain is a build
    /// artifact of one machine), so the road door is the one that survives and
    /// this delegates to it. `the_reported_start_is_the_one_the_level_spawns_at`
    /// keeps them the same answer and prints the gap the two used to have.
    pub fn player_start(&self) -> DVec3 {
        player_start(&self.recipe, &self.routes, 0.0)
    }

    /// The elevation the **terrain** carries under the reported start.
    ///
    /// Not the start — see [`IslandBuild::player_start`] — but the number worth
    /// printing beside it, because a large gap between the road layer's planned
    /// ground and the built terrain's is the corridor levelling or the channel
    /// carve having moved the ground since the route was planned.
    pub fn ground_under_start(&self) -> Option<f64> {
        let s = self.player_start();
        self.terrain.height_at(DVec2::new(s.x, s.z))
    }

    /// Which steps ran, in order.
    pub fn steps(&self) -> Vec<BuildStep> {
        self.log.iter().map(|l| l.step).collect()
    }

    /// How many log entries name a step. A **count**, not a substring — see the
    /// module docs.
    pub fn step_count(&self, s: BuildStep) -> usize {
        self.log.iter().filter(|l| l.step == s).count()
    }
}

/// The island's **committed design**, read without building anything.
///
/// # Why this exists, and why the level is authored from it
///
/// A level is committed; a terrain is not. If the level's author needed the
/// built terrain it would be a committed document only one machine could
/// produce, and CI could neither regenerate it nor check that it had not
/// drifted.
///
/// So: everything a level needs — where on Earth the world is, how big it is,
/// where the water is, where the roads go, what the biome palette binds, where
/// the settlements are — is **committed data**, and this is the door onto it. It
/// opens five small files and no elevation tile.
#[derive(Debug)]
pub struct IslandDesign {
    pub recipe: IslandRecipe,
    pub anchor: inf_math::geo::GeoAnchor,
    pub grid: IslandGrid,
    /// The derived water, read back from its committed layers.
    pub network: StreamNetwork,
    /// The designed road network, read back from its committed layer.
    pub routes: Vec<Route>,
    /// The designed shore's rings, world XZ.
    pub coast: Vec<Vec<DVec2>>,
    /// The palette, with its vegetation bound.
    pub biome_set: inf_terrain::BiomeSet,
}

impl IslandDesign {
    /// The world position a player starts at — see [`player_start`].
    pub fn start(&self, lift_m: f64) -> DVec3 {
        player_start(&self.recipe, &self.routes, lift_m)
    }

    /// The shore's length in metres.
    pub fn coastline_m(&self) -> f64 {
        self.coast
            .iter()
            .flat_map(|r| (0..r.len()).map(move |i| (r[(i + 1) % r.len()] - r[i]).length()))
            .sum()
    }

    /// The reaches worth a `WaterBody`, largest catchment first.
    ///
    /// **Not all of them, and the bound is a measurement rather than a taste.**
    /// `WaterSurface::height_at` is `O(frames)` for a river and a `RiverPath`
    /// holds `segments × samples_per_segment` of them; the island's fifty reaches
    /// are about 3 000 segments, so binding every one would put ~48 000 frames
    /// behind every buoyancy query in the world. The rest are still *there* — the
    /// carve cut their channels into the ground — they are dry beds rather than
    /// water bodies, and that is stated rather than hidden.
    pub fn rivers(&self, max: usize) -> Vec<&Stream> {
        let mut v: Vec<&Stream> = self.network.streams.iter().collect();
        v.sort_by(|a, b| {
            b.catchment_m2
                .total_cmp(&a.catchment_m2)
                .then(a.points[0].x.total_cmp(&b.points[0].x))
                .then(a.points[0].z.total_cmp(&b.points[0].z))
        });
        v.truncate(max);
        v
    }
}

/// Read the committed design. No tile is opened and no terrain is built.
pub fn read_design(recipe: &IslandRecipe) -> Result<IslandDesign, IslandError> {
    let anchor = recipe.anchor()?;
    let grid = IslandGrid::of(recipe);
    let hp = HydroParams {
        sea_level_m: recipe.sea.level_m,
        waterfall_grade: recipe.hydro.waterfall_grade,
        ..Default::default()
    };
    let network = committed_network(
        &recipe.resolve(&recipe.streams),
        &recipe.resolve(&recipe.lakes),
        &anchor,
        hp.waterfall_grade,
    )?;
    let road_layer = layers::read_layer(
        &recipe.resolve(&recipe.roads.layer),
        inf_gis::LayerKind::Roads,
        &anchor,
    )?;
    let coast_layer = layers::read_layer(
        &recipe.resolve(&recipe.coast),
        inf_gis::LayerKind::Generic,
        &anchor,
    )?;
    Ok(IslandDesign {
        routes: layers::routes_of(&road_layer),
        coast: layers::rings_of(&coast_layer),
        biome_set: {
            let mut s = biome_set(
                &recipe.name,
                Some(inf_asset::AssetId(cover_pcg_guid(&recipe.name))),
            );
            s.name = format!("{} Biomes", recipe.name);
            s
        },
        recipe: recipe.clone(),
        anchor,
        grid,
        network,
    })
}

/// Run the recipe.
pub fn build_island(
    recipe: &IslandRecipe,
    opts: &BuildOptions,
) -> Result<IslandBuild, IslandError> {
    let mut log: Vec<StepLog> = Vec::new();
    let mut advisories: Vec<Advisory> = Vec::new();
    let mut blocking: Vec<String> = Vec::new();
    let say = |step: BuildStep, note: String, log: &mut Vec<StepLog>| {
        tracing::info!(step = step.label(), "{note}");
        log.push(StepLog { step, note });
    };

    let anchor = recipe.anchor()?;
    let grid = IslandGrid::of(recipe);
    let (min, max) = grid.bounds();

    // ── 1. PLAN ─────────────────────────────────────────────────────────────
    let plan = plan_tiles(recipe)?;
    say(
        BuildStep::Plan,
        format!(
            "{} source tiles at z{}, {:.2} m/px against a {:.2} m grid ({:.2}x)",
            plan.len(),
            plan.zoom,
            plan.ground_m_per_px,
            plan.grid_m_per_sample,
            plan.upsample_ratio()
        ),
        &mut log,
    );
    if plan.upsample_ratio() > crate::source::UPSAMPLE_ADVISORY_RATIO {
        advisories.push(Advisory::new(
            "source.upsampled",
            format!(
                "the world grid is {:.2} m a sample and the finest source available \
                 is {:.2} m a pixel, so a feature shorter than {:.2} m is not in \
                 the survey at all and everything below it is DESIGNED detail: \
                 the carve, the road corridors, the stream channels, and (wave \
                 TER2b) the fBm detail band the `detail` step writes into exactly \
                 that gap. It is not survey, and it is no longer bilinear either.",
                plan.grid_m_per_sample,
                plan.ground_m_per_px,
                2.0 * plan.ground_m_per_px
            ),
        ));
    }

    // ── 2. FETCH (not performed here — see the module docs) ─────────────────
    let cache = recipe.cache_dir();
    let missing = plan.missing_in(&cache);
    if !missing.is_empty() {
        return Err(IslandError::MissingTile {
            z: missing[0].z,
            x: missing[0].x,
            y: missing[0].y,
            cache: cache.display().to_string(),
        });
    }
    say(
        BuildStep::Fetch,
        format!(
            "{} tiles satisfied from the cache at {}; nothing was fetched",
            plan.len(),
            cache.display()
        ),
        &mut log,
    );

    // ── 3. SAMPLE + 4. CARVE ────────────────────────────────────────────────
    let mosaic = TileMosaic::load(&plan, &cache)?;
    if !mosaic.sea_level_tiles().is_empty() {
        advisories.push(Advisory::new(
            "source.sea_level_tiles",
            format!(
                "{} of {} source tiles are uniformly sea level. That is either open \
                 ocean or a fetch that never happened, and the two decode \
                 identically — check the extent before trusting the coastline.",
                mosaic.sea_level_tiles().len(),
                plan.len()
            ),
        ));
    }
    if mosaic.implausible_samples() > 0 {
        advisories.push(Advisory::new(
            "source.implausible",
            format!(
                "{} source samples carry an elevation Earth does not have — the \
                 terrarium codec's own floor is a black pixel at -32 768 m, which \
                 is FINITE and therefore invisible to a finiteness check. They are \
                 treated as nodata and become ocean; if they are inland, the \
                 provider filled that tile rather than surveying it.",
                mosaic.implausible_samples()
            ),
        ));
    }
    let tf = inf_gis::Transform::new("EPSG:4326", &anchor)?;
    let lattice = ProjectionLattice::build(&tf, plan.zoom, min, max)?;
    say(
        BuildStep::Sample,
        format!(
            "{} tiles decoded, {:.1} % of source samples at sea level, source range \
             {:?}; projection lattice {} control points",
            mosaic.tile_count(),
            mosaic.sea_level_fraction() * 100.0,
            mosaic.range().map(|(a, b)| (a.round(), b.round())),
            lattice.control_points()
        ),
        &mut log,
    );

    // The designed coastline.
    let coast_layer = layers::read_layer(
        &recipe.resolve(&recipe.coast),
        inf_gis::LayerKind::Generic,
        &anchor,
    )?;
    let rings = layers::rings_of(&coast_layer);
    if rings.is_empty() {
        return Err(IslandError::Settings(format!(
            "the coastline layer {} carries no polygon — the map would be all sea, \
             because the carve's rule is that everything outside the shore is",
            recipe.coast
        )));
    }
    let coast = Coastline::new(
        rings,
        min,
        max,
        Coastline::field_pitch_m(recipe.sea.beach_width_m),
    );

    // The road corridor, if the design has a road layer yet.
    let road_path = recipe.resolve(&recipe.roads.layer);
    let committed_routes: Vec<Route> = if road_path.exists() && !opts.replan_roads {
        let l = layers::read_layer(&road_path, inf_gis::LayerKind::Roads, &anchor)?;
        layers::routes_of(&l)
    } else {
        Vec::new()
    };
    // **The plateau holds everything the road DRAWS** (wave ROAD1), and the
    // batter eases from its edge out. `built_half_width_m` is the road's own
    // answer — carriageway plus whichever of a sealed shoulder or a
    // kerb-and-pavement its class carries — taken over the routes actually
    // committed rather than assumed, so an island with no roads yet levels
    // nothing and one with a four-lane trunk levels for a four-lane trunk.
    let corridor_flat = if opts.graded_roads {
        committed_routes
            .iter()
            .map(|r| r.built_half_width_m())
            .fold(0.0f64, f64::max)
    } else {
        // The control: no plateau, so the levelling eases from the centreline
        // out exactly as it did before ROAD1.
        0.0
    };
    // The batter keeps the width it always had; the plateau is added OUTSIDE it,
    // so a corridor is now `flat + batter` rather than a bowl of `batter`. On the
    // island that is 9.5 m of plateau (the highway's 7 m half plus its 2.5 m
    // shoulder) and 11.2 m of batter either side.
    let corridor_half = corridor_flat + recipe.roads.shoulder_mult * inf_gis::LANE_WIDTH_M * 2.0;
    let corridor = corridor_index(&committed_routes, corridor_half);

    let pads: Vec<(DVec2, f64, f64)> = recipe
        .sites
        .iter()
        .map(|s| {
            let p = DVec2::new(s.x, s.z);
            // The pad's datum is the ground the SOURCE puts there, sampled once
            // through the same chain the walk uses — so a site on a hillside
            // gets a terrace at its own height rather than one at sea level.
            let (gx, gy) = lattice.pixel_at(p);
            let h = mosaic
                .elevation_at_pixel(gx, gy)
                .unwrap_or(recipe.sea.level_m);
            (p, s.radius_m, h.max(recipe.sea.level_m + 1.0))
        })
        .collect();

    // Cloned, not moved: `CarvePlan` takes the pads by value and the DETAIL step
    // (five steps later) has to exclude the same terraces the carve levelled.
    // Seven sites, so the clone is seven tuples.
    let site_pads = pads.clone();
    let carve = CarvePlan {
        coast: &coast,
        pads,
        corridor: corridor.as_ref(),
        corridor_half_m: corridor_half,
        corridor_flat_m: corridor_flat,
    };
    let (mut data, st) = terrain::sample_terrain(recipe, &mosaic, &lattice, &carve);
    say(
        BuildStep::Carve,
        format!(
            "{} samples, {} on land, {} pad, {} corridor, {} nodata; {:.1}..{:.1} m, \
             land {:.3} km2, shore {:.2} km",
            st.samples,
            st.land,
            st.pad,
            st.corridor,
            st.nodata,
            st.lo_m,
            st.hi_m,
            st.land_area_m2 / 1.0e6,
            coast.perimeter_m() / 1000.0
        ),
        &mut log,
    );

    // ── 5. HYDROLOGY ────────────────────────────────────────────────────────
    let hp = HydroParams {
        sea_level_m: recipe.sea.level_m,
        stream_catchment_m2: recipe.hydro.stream_catchment_m2,
        lake_depth_m: recipe.hydro.lake_depth_m,
        lake_area_m2: recipe.hydro.lake_area_m2,
        waterfall_grade: recipe.hydro.waterfall_grade,
        vertex_stride: recipe.hydro.vertex_stride,
    };
    let coarse = CoarseHeights::of(&data, min, max, DERIVATION_PITCH_M);
    let flow = FlowField::derive(&coarse, &hp);
    let derived = hydro::extract(&flow, &hp);

    let stream_path = recipe.resolve(&recipe.streams);
    let lake_path = recipe.resolve(&recipe.lakes);
    let network = if opts.rederive_layers || !stream_path.exists() || !lake_path.exists() {
        // Written whatever `dry_run` says: the design is the light half and the
        // second pass reads it. See `BuildOptions::dry_run`.
        layers::write_streams(&stream_path, &anchor, &derived.streams)?;
        layers::write_lakes(&lake_path, &anchor, &derived.lakes)?;
        derived.clone()
    } else {
        committed_network(&stream_path, &lake_path, &anchor, hp.waterfall_grade)?
    };
    let stream_drift = LayerDrift {
        committed: network.streams.len(),
        derived: derived.streams.len(),
        committed_measure: network.total_length_m(),
        derived_measure: derived.total_length_m(),
    };
    let lake_drift = LayerDrift {
        committed: network.lakes.len(),
        derived: derived.lakes.len(),
        committed_measure: network.total_lake_area_m2(),
        derived_measure: derived.total_lake_area_m2(),
    };
    if !stream_drift.agrees_within(0.02) || !lake_drift.agrees_within(0.02) {
        advisories.push(Advisory::new(
            "layers.drift",
            format!(
                "the committed water layers and a fresh derivation differ: streams \
                 {} vs {} ({:.2} % of length), lakes {} vs {} ({:.2} % of area). \
                 That is expected across platforms — the sample step goes through \
                 the projection modules the portability law exempts — and it is a \
                 real change if it happens on the machine that authored them.",
                stream_drift.committed,
                stream_drift.derived,
                stream_drift.relative() * 100.0,
                lake_drift.committed,
                lake_drift.derived,
                lake_drift.relative() * 100.0
            ),
        ));
    }

    // A stream needs a bed, or P20's own advisory calls it a basin. Cut one.
    let widest = network
        .streams
        .iter()
        .map(|s| s.width_m())
        .fold(2.0f64, f64::max);
    // **Each reach to its own profile** (wave ROAD1, clause 3). The index still
    // answers out to the widest channel's batter, because one index has to serve
    // every reach; what changed is that the CUT is per-owner, so a 1.5 m creek
    // gets a 1.5 m channel instead of a slot down the middle of the widest
    // stream's trench. The DETAIL step excludes the same beds five steps later
    // but needs an index that answers past them — see the note there, and
    // `detail::fade_reach_m`.
    let profiles = hydro::channel_profiles(&network.streams);
    let reach = widest * 0.5 * (1.0 + hydro::CHANNEL_BANK_MULT);
    let channels = hydro::channel_index(&network.streams, reach);
    let cut = hydro::carve_channels(
        &mut data,
        &channels,
        &profiles,
        // A reach with no profile cannot happen — `channel_profiles` is built
        // from the same slice the index is — so the fallback is the narrowest
        // real channel rather than a number that would look plausible.
        hydro::ChannelProfile {
            half_width_m: 0.75,
            depth_m: 0.35,
        },
    );
    say(
        BuildStep::Hydrology,
        format!(
            "{} reaches / {:.2} km, {} lakes / {:.4} km2, {} waterfalls, max \
             catchment {:.2} km2; {} samples cut for channels up to {:.1} m wide",
            network.streams.len(),
            network.total_length_m() / 1000.0,
            network.lakes.len(),
            network.total_lake_area_m2() / 1.0e6,
            network.waterfalls.len(),
            network.max_catchment_m2 / 1.0e6,
            cut,
            widest
        ),
        &mut log,
    );

    // ── 6. BIOMES ───────────────────────────────────────────────────────────
    let mask_path = recipe.resolve(&recipe.biomes.masks);
    let (masks, skipped) = if mask_path.exists() {
        let l = layers::read_layer(&mask_path, inf_gis::LayerKind::Biomes, &anchor)?;
        layers::masks_of(&l)
    } else {
        (Vec::new(), Vec::new())
    };
    // A mask that names no biome is a typo in committed design data — something
    // an author can fix, and therefore blocking rather than informational.
    for s in &skipped {
        advisories.push(Advisory::new("biomes.mask_skipped", s.clone()));
        blocking.push(format!("a biome mask was skipped: {s}"));
    }
    // Re-read the ground after the channels were cut, so a stream bed is
    // classified as the bed it now is.
    let coarse = CoarseHeights::of(&data, min, max, DERIVATION_PITCH_M);
    let (classifier, classification) = classify_biomes(recipe, &coarse, &masks);
    let painted = terrain::stamp_biomes(&mut data, |p| {
        let i = (((p.x - coarse.min.x) / coarse.pitch).round() as i64)
            .clamp(0, coarse.nx as i64 - 1) as usize;
        let j = (((p.y - coarse.min.y) / coarse.pitch).round() as i64)
            .clamp(0, coarse.nz as i64 - 1) as usize;
        if !coarse.known[j * coarse.nx + i] {
            return inf_terrain::UNASSIGNED_BIOME;
        }
        classifier.at(p, f64::from(coarse.at(i, j)), coarse.slope_deg(i, j))
    });
    // ── the splat, off the same classification ──────────────────────────────
    //
    // TER2a clause 2. The ids above are categorical and NEAREST by ruling; the
    // *weights* are what the four ground materials blend by, and they are
    // derived here rather than in a step of their own because they are the same
    // decision expressed as a mix — and because `BuildStep::ALL` is frozen and
    // the fixture counts one log line per step.
    //
    // The field is read off the classifier cell by cell (the same call the id
    // stamp makes, so the two cannot disagree about what is where) and then
    // interpolated BILINEARLY at 1 m, which is where the feather comes from.
    // See `crate::splat`.
    let splat_field = crate::splat::SplatField::of(&coarse, |i, j| {
        if !coarse.known[j * coarse.nx + i] {
            return inf_terrain::UNASSIGNED_BIOME;
        }
        let p = coarse.position(i, j);
        classifier.at(p, f64::from(coarse.at(i, j)), coarse.slope_deg(i, j))
    });
    // **The banks get the hydrology's own ground** (wave ROAD1, clause 3): the
    // gravel a watercourse lays either side of itself, on the SAME index and the
    // same profiles the carve cut with, so the material's edge is the bank's
    // edge rather than a second guess at where it is.
    let bank_fallback = hydro::ChannelProfile {
        half_width_m: 0.75,
        depth_m: 0.35,
    };
    let banks = crate::splat::ChannelBanks {
        index: &channels,
        profiles: &profiles,
        fallback: bank_fallback,
    };
    let splat = crate::splat::stamp_splat(
        &mut data,
        &splat_field,
        crate::splat::SplatRules::of(recipe),
        Some(&banks),
    );
    if splat.sum_violations > 0 {
        // The splat invariant is a contract, not a tolerance: a weight that does
        // not sum to 255 darkens or brightens that metre of ground by however
        // much it missed by, and the shader's defensive renormalisation hides
        // it. A build that produced one is reporting a defect in the quantizer.
        blocking.push(format!(
            "{} splat weights do not sum to 255 — the ground would shade by an \
             amount no author asked for",
            splat.sum_violations
        ));
    }
    let mut set = biome_set(
        &recipe.name,
        Some(inf_asset::AssetId(cover_pcg_guid(&recipe.name))),
    );
    set.name = format!("{} Biomes", recipe.name);
    say(
        BuildStep::Biomes,
        format!(
            "breaks {:?} over a {:.0}..{:.0} m band; {} samples painted, {} masked, \
             {} reserved; {}",
            classification.breaks,
            classification.band_m.0,
            classification.band_m.1,
            painted,
            classification.masked,
            classification.reserved,
            splat.summary()
        ),
        &mut log,
    );

    // ── 7. ROADS ────────────────────────────────────────────────────────────
    let coarse_for_routing = CoarseHeights::of(&data, min, max, DERIVATION_PITCH_M);
    let routes = if opts.replan_roads || committed_routes.is_empty() {
        let planned = roads::plan_network(recipe, &coarse_for_routing)?;
        // Written whatever `dry_run` says — see `BuildOptions::dry_run`.
        layers::write_roads(&road_path, &anchor, &planned)?;
        planned
    } else {
        committed_routes
    };
    let audit = roads::grade_audit(
        &routes,
        recipe.roads.max_grade,
        recipe.roads.grade_step_m,
        |p| data.height_at(p),
    );
    if !audit.is_clean() {
        advisories.push(Advisory::new(
            "roads.over_grade",
            format!(
                "{} of {} measured stretches ({:.2} %) exceed the {:.3} grade \
                 ceiling; the worst is {:.3} at ({:.1}, {:.1}). This generator \
                 builds no grade separation, so every crossing of two routes at \
                 different elevations leaves one step. Re-plan with `inf island \
                 route`, move the site, or raise the ceiling deliberately.",
                audit.over.len(),
                audit.samples,
                audit.over_fraction() * 100.0,
                audit.ceiling,
                audit.worst,
                audit.worst_at.x,
                audit.worst_at.y
            ),
        ));
        if audit.over_fraction() > crate::report::ROAD_OVER_GRADE_CEILING {
            blocking.push(format!(
                "{:.2} % of the road network exceeds its own grade ceiling, past \
                 the {:.2} % the crossings account for",
                audit.over_fraction() * 100.0,
                crate::report::ROAD_OVER_GRADE_CEILING * 100.0
            ));
        }
    }
    let mut road_report = RoadReport {
        audit: audit.clone(),
        ..Default::default()
    };
    let mut mesh = None;
    if !opts.dry_run && !routes.is_empty() {
        // The mesh goes through IB-4's own door, over a layer written and read
        // back the way a shipped build reads it.
        let tmp = recipe.resolve(&recipe.roads.layer);
        let mut layer = if tmp.exists() {
            layers::read_layer(&tmp, inf_gis::LayerKind::Roads, &anchor)?
        } else {
            inf_gis::GeoLayer::new("roads", inf_gis::LayerKind::Roads, &anchor.crs)
        };
        // **The settlement streets join the SAME layer** (wave ROAD1b), so
        // `RoadGraph::from_layer` snaps the arterial that arrives at a town's
        // centre into the grid it arrives on and `build_surface` fans that
        // junction and paints its crossings. Two graphs would have produced two
        // ribbons at one height, which is a z-fight rather than a junction.
        //
        // Absent is not an error: an island whose Ring-1 generator has not been
        // run yet paves the GIS roads alone, and the road line says how many
        // street spans it found so a reader can tell the two cases apart.
        let streets_path = recipe.resolve(&recipe.roads.streets);
        let street_spans = if streets_path.exists() {
            let s = layers::read_layer(&streets_path, inf_gis::LayerKind::Roads, &anchor)?;
            let n = s.features.len();
            layer.features.extend(s.features);
            n
        } else {
            0
        };
        let mut ground = |x: f64, z: f64| data.height_at(DVec2::new(x, z));
        let (m, mr, rr) = roads::build_mesh(
            &layer,
            recipe.grid.meters_per_sample,
            opts.graded_roads,
            corridor_flat,
            &mut ground,
        )?;
        say(
            BuildStep::Roads,
            format!(
                "{:.2} km over {} segments ({} of them settlement street spans) \
                 and {} junctions; carriageway {} \
                 vertices / {} triangles, quantisation {:.4} m; furniture {}; \
                 {} footway triangles clipped off the carriageway; \
                 worst grade {:.3} against {:.3}, {} over",
                rr.total_km,
                street_spans,
                rr.segments,
                rr.junctions,
                mr.vertices,
                mr.triangles,
                mr.quantisation_m,
                // **What the kerbs, the pavements and the paint cost**
                // (wave ROAD1). One line per material group, because they
                // are one entity and one draw each and the only honest
                // budget for road furniture is what it adds to the frame.
                if m.furniture.is_empty() {
                    "none".to_string()
                } else {
                    m.furniture
                        .iter()
                        .map(|(part, a)| {
                            format!(
                                "{} {} v / {} t",
                                part.label(),
                                a.vertex_count(),
                                a.triangle_count()
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                },
                // **How much of the network runs over itself** (audit ROAD1) —
                // a footway laid beside its own segment and over somebody
                // else's asphalt, dropped. Reported beside the furniture it is
                // subtracted from, because a number that only exists inside the
                // builder is a number nobody reads.
                rr.kerbs_clipped,
                audit.worst,
                audit.ceiling,
                audit.over.len()
            ),
            &mut log,
        );
        road_report = RoadReport {
            audit: audit.clone(),
            ..rr
        };
        mesh = Some(m);
    } else {
        say(
            BuildStep::Roads,
            format!(
                "{} routes, worst grade {:.3} against {:.3}, {} over; no mesh (dry run \
                 or empty network)",
                routes.len(),
                audit.worst,
                audit.ceiling,
                audit.over.len()
            ),
            &mut log,
        );
    }

    // ── 8. DETAIL ───────────────────────────────────────────────────────────
    // The band the survey could not carry, put there on purpose (wave TER2b).
    //
    // **This slot is the design.** It is after ROADS because the grade audit and
    // the road mesh are both built up there against `data.height_at`, so nothing
    // this step writes can move either; after HYDROLOGY because the committed
    // water design and its drift comparison are derived three steps earlier; and
    // before PYRAMID because the coarse levels must be built from the ground the
    // world actually has. Every fixture arm that measures a committed number
    // measures it before this line runs.
    //
    // The corridor index is rebuilt from the FINAL routes rather than reused from
    // the carve's: the carve's is `None` whenever the roads were re-planned, and
    // a re-plan is exactly the case where the corridor moved.
    //
    // **Both of the detail stage's indices reach past the feature, not to it**
    // (TER2b audit). A `SegmentIndex` answers `None` beyond its own reach and the
    // mask reads `None` as "far away, take all the detail", so an index built to
    // the half-width has no fade in it at all — the ground was measured stepping
    // from 0.000 m of relief at 7 m from a corridor centreline to 0.127 m at 8 m,
    // which is the crease along every road the stage exists to avoid. The CARVE
    // keeps its own half-width index: levelling wants the feature's own width.
    let detail_corridor = corridor_index(&routes, crate::detail::fade_reach_m(corridor_half));
    let detail_channels =
        hydro::channel_index(&network.streams, crate::detail::fade_reach_m(widest));
    let band = crate::detail::DetailBand::of(plan.grid_m_per_sample, plan.ground_m_per_px);
    let detail = crate::detail::apply_detail(
        &mut data,
        &crate::detail::DetailPlan {
            seed: recipe.seed,
            sea_level_m: recipe.sea.level_m,
            band,
            coast: &coast,
            corridor: detail_corridor.as_ref(),
            corridor_half_m: corridor_half,
            channels: Some(&detail_channels),
            channel_half_m: widest,
            pads: &site_pads,
        },
    );
    say(
        BuildStep::Detail,
        match detail.band {
            Some(b) => format!(
                "{} of {} samples took designed relief in a {:.2}..{:.2} m band \
                 ({} octaves); mean {:.3} m, worst {:.3} m; excluded {} water, \
                 {} corridor, {} channel, {} pad",
                detail.written,
                detail.samples,
                b.finest_wavelength_m,
                b.base_wavelength_m,
                b.octaves,
                detail.mean_abs_m,
                detail.max_abs_m,
                detail.masked_water,
                detail.masked_road,
                detail.masked_channel,
                detail.masked_pad
            ),
            None => format!(
                "skipped: the {:.2} m grid is not finer than the {:.2} m source, \
                 so there is no band under the survey to fill",
                plan.grid_m_per_sample, plan.ground_m_per_px
            ),
        },
        &mut log,
    );

    // ── 9. PYRAMID + 10. WRITE ──────────────────────────────────────────────
    let origin = DVec3::new(
        anchor.origin_easting_m,
        anchor.origin_height_m,
        anchor.origin_northing_m,
    );
    let mut asset = None;
    let mut tiles_total = 0usize;
    let mut lod_levels = 0u32;
    let mut terrain_bytes = 0usize;
    if !opts.dry_run {
        let (a, pyr) = terrain::build_asset(&data, origin, inf_terrain::PyramidOptions::default())?;
        tiles_total = a.reader().tile_count();
        lod_levels = a.reader().lod_levels();
        terrain_bytes = a.as_bytes().len();
        say(
            BuildStep::Pyramid,
            format!(
                "{} coarse levels, {} tiles in the catalog over {} at level 0",
                pyr.len(),
                tiles_total,
                data.tile_count()
            ),
            &mut log,
        );
        asset = Some(a);
    } else {
        say(
            BuildStep::Pyramid,
            "skipped (dry run)".to_string(),
            &mut log,
        );
    }

    let mut report = IslandReport {
        name: recipe.name.clone(),
        extent_m: recipe.grid.extent_m(),
        map_km2: recipe.grid.extent_m() * recipe.grid.extent_m() / 1.0e6,
        coastline_km: coast.perimeter_m() / 1000.0,
        tiles_level0: recipe.grid.tile_count(),
        tiles_total,
        lod_levels,
        terrain_bytes,
        biome_breaks: classification.breaks.clone(),
        biome_share: classification.land_fractions(),
        roads: road_report,
        steps: log.iter().map(|l| l.step).collect(),
        advisories,
        blocking,
        stream_drift,
        lake_drift,
        ..Default::default()
    }
    .with_samples(&st)
    .with_plan(&plan)
    .with_hydrology(&network);
    report.steps = log.iter().map(|l| l.step).collect();

    say(
        BuildStep::Write,
        format!(
            "terrain {:.1} MB, {} road mesh, {} biome definitions",
            terrain_bytes as f64 / 1.0e6,
            if mesh.is_some() { "one" } else { "no" },
            set.biomes.len()
        ),
        &mut log,
    );
    report.steps = log.iter().map(|l| l.step).collect();

    Ok(IslandBuild {
        recipe: recipe.clone(),
        anchor,
        grid,
        plan,
        terrain: data,
        asset,
        mesh,
        network,
        routes,
        biomes: classification,
        splat,
        biome_set: set,
        coast,
        report,
        log,
    })
}

/// Read the committed stream and lake layers back into a network.
fn committed_network(
    streams: &Path,
    lakes: &Path,
    anchor: &inf_math::geo::GeoAnchor,
    waterfall_grade: f64,
) -> Result<StreamNetwork, IslandError> {
    let sl = layers::read_layer(streams, inf_gis::LayerKind::Streams, anchor)?;
    let ll = layers::read_layer(lakes, inf_gis::LayerKind::Lakes, anchor)?;
    let mut out = StreamNetwork::default();
    for f in &sl.features {
        let inf_gis::GeoGeometry::Polyline { points, .. } = &f.geometry else {
            continue;
        };
        if points.len() < 2 {
            continue;
        }
        let length: f64 = points
            .windows(2)
            .map(|w| (DVec2::new(w[1].x, w[1].z) - DVec2::new(w[0].x, w[0].z)).length())
            .sum();
        out.streams.push(hydro::Stream {
            catchment_m2: f.attr_number(&["catchment_m2"]).unwrap_or(0.0),
            fall_m: f
                .attr_number(&["fall_m"])
                .unwrap_or(points[0].y - points[points.len() - 1].y),
            points: points.clone(),
            length_m: length,
        });
    }
    for f in &ll.features {
        let inf_gis::GeoGeometry::Polygon { exterior, .. } = &f.geometry else {
            continue;
        };
        let outline: Vec<DVec2> = exterior.iter().map(|p| DVec2::new(p.x, p.z)).collect();
        if outline.is_empty() {
            continue;
        }
        let lo = outline
            .iter()
            .fold(DVec2::splat(f64::INFINITY), |a, b| a.min(*b));
        let hi = outline
            .iter()
            .fold(DVec2::splat(f64::NEG_INFINITY), |a, b| a.max(*b));
        out.lakes.push(hydro::Lake {
            level_m: f.attr_number(&["level_m"]).unwrap_or(0.0),
            centre: (lo + hi) * 0.5,
            half_extent: DVec2::new(
                f.attr_number(&["half_x_m"]).unwrap_or((hi.x - lo.x) * 0.5),
                f.attr_number(&["half_z_m"]).unwrap_or((hi.y - lo.y) * 0.5),
            ),
            area_m2: f.attr_number(&["area_m2"]).unwrap_or(f.geometry.area_m2()),
            max_depth_m: f.attr_number(&["max_depth_m"]).unwrap_or(0.0),
            outline,
        });
    }
    // A committed network's waterfalls and its largest catchment are functions of
    // the reaches, so they are re-derived rather than left empty — a network read
    // back with no waterfall in it reads as "this island has none".
    out.waterfalls = hydro::waterfalls_of(&out.streams, waterfall_grade);
    out.max_catchment_m2 = out
        .streams
        .iter()
        .map(|s| s.catchment_m2)
        .fold(0.0f64, f64::max);
    Ok(out)
}

/// Write everything a build produced into a project's `Content`.
///
/// One function so the CLI, a test and a future cook step cannot disagree about
/// which files an island consists of.
pub fn write_content(build: &IslandBuild, content: &Path) -> Result<Vec<String>, IslandError> {
    std::fs::create_dir_all(content)?;
    let mut written = Vec::new();
    if let Some(a) = &build.asset {
        let p = content.join(format!("{}.inf_terrain", slug(&build.recipe.name)));
        let bytes = inf_terrain::write_terrain_asset(&p, a)?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(terrain_guid(&build.recipe.name)),
            inf_asset::AssetKind::Terrain,
            inf_asset::ContentHash::of(bytes),
        )
        .save(&p)
        .map_err(|e| IslandError::Io(e.to_string()))?;
        written.push(p.display().to_string());
    }
    if let Some(m) = &build.mesh {
        // **One writer, four meshes** (wave ROAD1). The carriageway keeps the
        // name and the GUID it has always had — a committed `.inf_lvl` names
        // both — and each furniture group gets its own beside it, because an
        // `inf_ecs::Material` binds ONE `.inf_mat` and a road wears four.
        let write = |stem: String, guid: uuid::Uuid, asset: &inf_mesh::MeshAsset| {
            let p = content.join(stem);
            let bytes = inf_asset::encode(asset).map_err(|e| IslandError::Io(e.to_string()))?;
            std::fs::write(&p, &bytes)?;
            inf_asset::AssetSidecar::new(
                inf_asset::AssetId(guid),
                inf_asset::AssetKind::Mesh,
                inf_asset::ContentHash::of(&bytes),
            )
            .save(&p)
            .map_err(|e| IslandError::Io(e.to_string()))?;
            Ok::<_, IslandError>(p.display().to_string())
        };
        if let Some((asset, _)) = &m.carriageway {
            written.push(write(
                format!("{}Roads.inf_mesh", slug(&build.recipe.name)),
                road_mesh_guid(&build.recipe.name),
                asset,
            )?);
        }
        for (part, asset) in &m.furniture {
            written.push(write(
                format!(
                    "{}{}.inf_mesh",
                    slug(&build.recipe.name),
                    road_part_stem(*part)
                ),
                road_part_mesh_guid(&build.recipe.name, *part),
                asset,
            )?);
        }
    }
    let p = content.join(format!("{}.inf_biomes", slug(&build.recipe.name)));
    let bytes = inf_asset::encode(&build.biome_set).map_err(|e| IslandError::Io(e.to_string()))?;
    std::fs::write(&p, &bytes)?;
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(biome_set_guid(&build.recipe.name)),
        inf_asset::AssetKind::BiomeSet,
        inf_asset::ContentHash::of(&bytes),
    )
    .save(&p)
    .map_err(|e| IslandError::Io(e.to_string()))?;
    written.push(p.display().to_string());

    // The committed halves the recipe names — the level, its assets, the
    // `.inf_pcg` the biome set binds. Copied so that one command produces a
    // project that cooks.
    for rel in &build.recipe.content {
        let src = build.recipe.resolve(rel);
        let name = std::path::Path::new(rel)
            .file_name()
            .ok_or_else(|| IslandError::Settings(format!("[content] {rel:?} names no file")))?;
        let dst = content.join(name);
        std::fs::copy(&src, &dst).map_err(|e| {
            IslandError::Io(format!(
                "copying {} into the project: {e} — `[content]` names files that \
                 live beside the recipe",
                src.display()
            ))
        })?;
        written.push(dst.display().to_string());
    }
    Ok(written)
}

/// The world position a level's player start belongs at, from the **committed
/// road layer**.
///
/// # Why the roads and not the terrain
///
/// The level is committed and the terrain is not, so a start whose elevation came
/// from the terrain would be a committed number that only one machine could
/// produce. The road network *is* committed, it passes through every settlement
/// by construction, and its vertices carry the ground each was planned at — so
/// the nearest road vertex to the first city is a ground height in the design
/// rather than in a build artifact.
///
/// `lift_m` is added on top, because a character spawned exactly on the ground
/// is a character the first ground snap has to resolve out of the floor.
pub fn player_start(recipe: &IslandRecipe, routes: &[Route], lift_m: f64) -> DVec3 {
    let site = recipe
        .sites_of(crate::recipe::SiteKind::City)
        .next()
        .or_else(|| recipe.sites.first());
    let p = site.map(|s| DVec2::new(s.x, s.z)).unwrap_or(DVec2::ZERO);
    // ONE door for "the nearest ground the design commits" (island wave VEH1a):
    // the fleet the level parks at each settlement and the connectivity walk ask
    // the same question, and three copies of this walk is three chances for one
    // of them to break the tie differently.
    let y = crate::roads::nearest_route_vertex(routes, p).map_or(0.0, |(v, _)| v.y);
    DVec3::new(p.x, y + lift_m, p.y)
}

/// **The road corridor as a queryable index** — one door, two callers.
///
/// The CARVE levels the ground inside the corridor so the road has something to
/// sit on; the DETAIL step (wave TER2b) excludes the same ground for the same
/// reason, five steps later, and would otherwise be eight lines of transcription
/// answering the same question. `None` when there are no routes or no reach,
/// which both callers read as "there is no corridor".
///
/// The two callers differ in **both** arguments, on purpose:
///
/// * the carve sees the *committed* routes (it runs before the planner) and the
///   detail step the *final* ones, which differ exactly when `--replan-roads`
///   moved them;
/// * the carve asks for the corridor's own half-width, because that is the ground
///   it levels, and the detail step asks for
///   [`detail::fade_reach_m`](crate::detail::fade_reach_m) of it, because an index
///   that stops answering at the half-width turns the mask's fade into a **cut**
///   — the query returns `None` one metre out and the mask reads `None` as "take
///   all the detail" (the TER2b audit's measurement).
fn corridor_index(routes: &[Route], reach_m: f64) -> Option<SegmentIndex> {
    (!routes.is_empty() && reach_m > 0.0).then(|| {
        let lines: Vec<Vec<Vertex3>> = routes
            .iter()
            .map(|r| {
                r.points
                    .iter()
                    .map(|p| Vertex3 {
                        xz: DVec2::new(p.x, p.z),
                        y: p.y,
                    })
                    .collect()
            })
            .collect();
        SegmentIndex::new(&lines, reach_m)
    })
}

/// A file-name slug from an island's name.
pub fn slug(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .replace(' ', "")
}

/// Derive a stable GUID from a name and a salt.
///
/// The `pcg_structure_guid` pattern: a content-derived id, so a rebuild produces
/// the same asset identity and a level committed against it keeps resolving.
fn derived_guid(name: &str, salt: &str) -> uuid::Uuid {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in salt.as_bytes().iter().chain(b"/").chain(name.as_bytes()) {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    let mut lo: u64 = 0x9e37_79b9_7f4a_7c15 ^ h;
    lo = lo.wrapping_mul(0xff51_afd7_ed55_8ccd);
    lo ^= lo >> 33;
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&h.to_be_bytes());
    bytes[8..].copy_from_slice(&lo.to_be_bytes());
    uuid::Uuid::from_bytes(bytes)
}

/// The `.inf_terrain` asset id for an island.
pub fn terrain_guid(name: &str) -> uuid::Uuid {
    derived_guid(name, "island.terrain")
}

/// The road mesh asset id — the **carriageway's**, unchanged since IB-4 because
/// a committed `.inf_lvl` names it.
pub fn road_mesh_guid(name: &str) -> uuid::Uuid {
    derived_guid(name, "island.roads")
}

/// The file-name suffix one road-furniture mesh is written under (wave ROAD1).
///
/// Frozen with the GUID below: a committed `.inf_lvl` names the GUID and a
/// cooked pack names the file, so a stem that moved would leave the island's
/// kerbs bound to nothing.
pub fn road_part_stem(part: inf_gis::RoadPart) -> &'static str {
    match part {
        inf_gis::RoadPart::Carriageway => "Roads",
        inf_gis::RoadPart::Kerb => "Kerbs",
        inf_gis::RoadPart::MarkingWhite => "RoadMarkings",
        inf_gis::RoadPart::MarkingYellow => "RoadMarkingsYellow",
    }
}

/// One road-furniture mesh's asset id (wave ROAD1).
///
/// Derived from the island's name and a **per-part salt**, on
/// `road_mesh_guid`'s own pattern, so the four ids are a pure function of the
/// recipe and the level generator can name them without the build having run.
/// The carriageway keeps `island.roads` rather than taking a new salt, because
/// the committed level already names that id.
pub fn road_part_mesh_guid(name: &str, part: inf_gis::RoadPart) -> uuid::Uuid {
    match part {
        inf_gis::RoadPart::Carriageway => road_mesh_guid(name),
        inf_gis::RoadPart::Kerb => derived_guid(name, "island.roads.kerb"),
        inf_gis::RoadPart::MarkingWhite => derived_guid(name, "island.roads.marking.white"),
        inf_gis::RoadPart::MarkingYellow => derived_guid(name, "island.roads.marking.yellow"),
    }
}

/// The `.inf_biomes` asset id.
pub fn biome_set_guid(name: &str) -> uuid::Uuid {
    derived_guid(name, "island.biomes")
}

/// The ground-cover `.inf_pcg` asset id.
pub fn cover_pcg_guid(name: &str) -> uuid::Uuid {
    derived_guid(name, "island.cover")
}

/// The level's own asset id.
pub fn level_guid(name: &str) -> uuid::Uuid {
    derived_guid(name, "island.level")
}

/// The biome ids that scatter, for a caller wiring the PCG binding.
pub fn scattering_biomes() -> Vec<IslandBiome> {
    IslandBiome::ALL
        .into_iter()
        .filter(|b| b.scatters())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_guids_are_stable_distinct_and_a_function_of_the_name() {
        let a = "Vancouver Island";
        let b = "Other Island";
        let ids = [
            terrain_guid(a),
            road_mesh_guid(a),
            biome_set_guid(a),
            cover_pcg_guid(a),
            level_guid(a),
        ];
        let set: std::collections::BTreeSet<_> = ids.iter().collect();
        assert_eq!(set.len(), ids.len(), "two assets share an id");
        assert_eq!(terrain_guid(a), terrain_guid(a), "stable across calls");
        assert_ne!(terrain_guid(a), terrain_guid(b), "a function of the name");
        assert_ne!(terrain_guid(a), road_mesh_guid(a), "a function of the salt");
        assert_ne!(terrain_guid(a), uuid::Uuid::nil());
        assert_eq!(slug("Vancouver Island"), "VancouverIsland");
        assert_eq!(slug("A-1 Test!"), "A1Test");
        assert_eq!(scattering_biomes().len(), 6, "everything but urban");
    }

    #[test]
    fn build_options_default_to_leaving_the_committed_design_alone() {
        let o = BuildOptions::default();
        assert!(
            !o.rederive_layers,
            "a build must not overwrite an author's edit"
        );
        assert!(!o.replan_roads);
        assert!(!o.dry_run);
    }
}
