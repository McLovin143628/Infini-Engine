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
#[derive(Clone, Debug, Default, PartialEq)]
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
    pub mesh: Option<inf_mesh::MeshAsset>,
    pub network: StreamNetwork,
    pub routes: Vec<Route>,
    pub biomes: BiomeClassification,
    pub biome_set: inf_terrain::BiomeSet,
    pub coast: Coastline,
    pub report: IslandReport,
    pub log: Vec<StepLog>,
}

impl IslandBuild {
    /// The world position the report calls the player start: the first city
    /// site, lifted onto its own ground.
    pub fn player_start(&self) -> DVec3 {
        let s = self
            .recipe
            .sites_of(crate::recipe::SiteKind::City)
            .next()
            .or_else(|| self.recipe.sites.first());
        let p = s.map(|s| DVec2::new(s.x, s.z)).unwrap_or(DVec2::ZERO);
        let y = self.terrain.height_at(p).unwrap_or(0.0);
        DVec3::new(p.x, y, p.y)
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
    if plan.upsample_ratio() > 1.05 {
        advisories.push(Advisory::new(
            "source.upsampled",
            format!(
                "the world grid is {:.2} m a sample and the finest source available \
                 is {:.2} m a pixel, so detail below {:.2} m is interpolation plus \
                 whatever the design puts there — the carve, the road corridors and \
                 the stream channels. It is not survey.",
                plan.grid_m_per_sample, plan.ground_m_per_px, plan.ground_m_per_px
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
    let corridor_half = recipe.roads.shoulder_mult * inf_gis::LANE_WIDTH_M * 2.0;
    let corridor = (!committed_routes.is_empty() && corridor_half > 0.0).then(|| {
        let lines: Vec<Vec<Vertex3>> = committed_routes
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
        SegmentIndex::new(&lines, corridor_half)
    });

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

    let carve = CarvePlan {
        coast: &coast,
        pads,
        corridor: corridor.as_ref(),
        corridor_half_m: corridor_half,
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
    let channels = hydro::channel_index(&network.streams, widest);
    let cut = hydro::carve_channels(&mut data, &channels, widest, 1.25);
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
    let mut set = biome_set(
        &recipe.name,
        Some(inf_asset::AssetId(cover_pcg_guid(&recipe.name))),
    );
    set.name = format!("{} Biomes", recipe.name);
    say(
        BuildStep::Biomes,
        format!(
            "breaks {:?} over a {:.0}..{:.0} m band; {} samples painted, {} masked, \
             {} reserved",
            classification.breaks,
            classification.band_m.0,
            classification.band_m.1,
            painted,
            classification.masked,
            classification.reserved
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
        let layer = if tmp.exists() {
            layers::read_layer(&tmp, inf_gis::LayerKind::Roads, &anchor)?
        } else {
            inf_gis::GeoLayer::new("roads", inf_gis::LayerKind::Roads, &anchor.crs)
        };
        let mut ground = |x: f64, z: f64| data.height_at(DVec2::new(x, z));
        let (m, mr, rr) = roads::build_mesh(&layer, recipe.grid.meters_per_sample, &mut ground)?;
        say(
            BuildStep::Roads,
            format!(
                "{:.2} km over {} segments and {} junctions; mesh {} vertices / {} \
                 triangles, quantisation {:.4} m; worst grade {:.3} against {:.3}, \
                 {} over",
                rr.total_km,
                rr.segments,
                rr.junctions,
                mr.vertices,
                mr.triangles,
                mr.quantisation_m,
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

    // ── 8. PYRAMID + 9. WRITE ───────────────────────────────────────────────
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
        let p = content.join(format!("{}Roads.inf_mesh", slug(&build.recipe.name)));
        let bytes = inf_asset::encode(m).map_err(|e| IslandError::Io(e.to_string()))?;
        std::fs::write(&p, &bytes)?;
        inf_asset::AssetSidecar::new(
            inf_asset::AssetId(road_mesh_guid(&build.recipe.name)),
            inf_asset::AssetKind::Mesh,
            inf_asset::ContentHash::of(&bytes),
        )
        .save(&p)
        .map_err(|e| IslandError::Io(e.to_string()))?;
        written.push(p.display().to_string());
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
    let mut best: Option<(f64, f64)> = None;
    for r in routes {
        for v in &r.points {
            let d = (DVec2::new(v.x, v.z) - p).length_squared();
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, v.y));
            }
        }
    }
    DVec3::new(p.x, best.map(|(_, y)| y).unwrap_or(0.0) + lift_m, p.y)
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

/// The road mesh asset id.
pub fn road_mesh_guid(name: &str) -> uuid::Uuid {
    derived_guid(name, "island.roads")
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
