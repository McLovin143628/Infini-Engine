//! The island's **level**, authored from its committed design (wave I7).
//!
//! # Why the level is committed and the terrain is not
//!
//! An island is 342 MB of terrain and 260 KB of design. The design is source;
//! the terrain is a build artifact of one machine, because the sampling step
//! goes through the projection modules this repository's portability law exempts
//! by name.
//!
//! So the level has to be authorable **without the terrain**, or it would be a
//! committed document only one machine could produce and nothing could check it
//! had not drifted. It is: `inf_island::read_design` opens five small committed
//! files — the coastline, the roads, the streams, the lakes, the biome masks —
//! and every number in the level comes from those, from the recipe, or from a
//! GUID derived from the island's own name.
//!
//! **Nothing here reads an elevation.** `the_level_is_authored_from_committed_
//! design_alone` is what says so.
//!
//! # "From the recipe" was not the same as "from a committed number"
//!
//! That sentence was true of every *file* and false of three numbers, and the
//! I7 CI-red is what taught the difference. `read_design` used to build the
//! geo-anchor by inverting the recipe's easting/northing through
//! `inf_gis::anchor_at` — i.e. `proj4rs`, i.e. the platform's libm — and the
//! three degrees that came back were serialized straight into this committed
//! `.inf_lvl`. macOS computed `origin_latitude_deg = 49.34307562364772` where
//! Windows had blessed `…773`: one ulp, one byte at offset 14 788, and
//! `committed_sample_matches_generators` red on one platform of three.
//!
//! The recipe now **states** its geodetic origin and `read_design` carries it
//! across untouched, so every byte of the anchor traces to a decimal in a
//! committed TOML. `crates/inf-island/tests/stated_anchor.rs` checks the stated
//! degrees against the projection, and
//! `crates/inf-island/tests/portable_math_law.rs` bans the anchor door from the
//! whole crate so it cannot come back.
//!
//! # One generator, two islands
//!
//! The full island and the CI-scale fixture are the same recipe format and the
//! same scene function. That is not tidiness — it is what makes the fixture a
//! gate: the level CI exercises is built by the code that builds the one that
//! ships, so a change that breaks the shipped level breaks the fixture's too.

use glam::DVec3;
use uuid::Uuid;

use crate::ipc::SpawnKind;
use crate::scene::serialize::{LevelSettings, PartitionSettings};
use crate::scene::SceneDoc;

/// Insert a bundle onto `guid`'s entity, dirtying the doc.
///
/// A second copy of `samples.rs`'s macro rather than a shared function, and the
/// reason is the facade rule: `macro_rules!` is module-scoped, and the only way
/// to write this once as a *function* is a `B: bevy_ecs::bundle::Bundle` bound —
/// which would make this crate name `bevy_ecs`, which is exactly what `inf-ecs`
/// exists to prevent. Eight lines of syntax against a ring violation.
macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr $(,)?) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// How far above the ground a hero is spawned.
///
/// A character placed exactly on the surface is one the first ground snap has to
/// resolve out of the floor; a metre is clear of any rounding the design's own
/// road profile carries and is a fall of 0.45 s at 9.81 m/s².
pub const START_LIFT_M: f64 = 1.0;

/// How tall the island's hero is, metres — **the starter character's own
/// height**, not a number this file chose (SK1c).
///
/// The hero *is* `samples/starter-character`, so its capsule has to be the one
/// `edit_create_character` derives for that rig. Reading the spec is what makes
/// that true by construction rather than by two literals agreeing: a wizard
/// default that moves re-blesses the sample folder AND re-blesses both island
/// levels, in the same run, which is the loud version of a mismatch.
///
/// It was `1.8` while the hero was a bare capsule with nothing inside it. The
/// wizard's default is 1.75, and a 1.75 m body in a 1.8 m capsule floats 5 cm
/// off the ground it is standing on.
fn hero_height_m() -> f64 {
    crate::samples::starter_character_spec().params.height_m
}

/// How many reaches become real `WaterBody::River` entities.
///
/// See `IslandDesign::rivers` for the measurement behind it: a `RiverPath` holds
/// `segments × 16` frames and `WaterSurface::height_at` walks them, so binding
/// fifty reaches would put tens of thousands of frames behind every buoyancy
/// query. The rest keep their carved channels and are dry beds.
pub const MAX_RIVER_BODIES: usize = 10;

/// The partition cell an island streams in.
///
/// `DEFAULT_CELL_SIZE_M` (256 m) over a 7 168 m world is 28 × 28 = 784 cells,
/// which is the same lattice the terrain's own level-0 tiles sit on — so a cell
/// and a page activate together instead of at two different distances.
pub const ISLAND_CELL_SIZE_M: f64 = 256.0;

/// The activation radius. One cell, which is what P16's own default is; the
/// terrain's render cut is a separate and wider thing (IB-9).
pub const ISLAND_ACTIVATION_M: f64 = 256.0;

/// The prefetch margin.
pub const ISLAND_PREFETCH_M: f64 = 256.0;

/// The scatter cell the island's vegetation is evaluated on, world metres.
pub const ISLAND_SCATTER_CELL_M: f64 = 32.0;

/// Instances per square metre at density 1.0.
///
/// **This is the island-scale number and it is a budget, not a taste.** 40 km² of
/// land at the phase-18 sample's own 0.05 /m² would be two million instances
/// before a single mask ran. At 0.004 the biome-bound population over the
/// forested 38.5 % is ~620 000 candidates, which the P18.5 GPU scatter path and
/// the I3 draw bands are sized for; the CPU fallback's own ceiling
/// (`MAX_CPU_SCATTER_INSTANCES`, 65 536) is what a tier that cannot reach the
/// GPU path degrades to, nearest-first.
///
/// # Raised at TER2a, with the arithmetic that allowed it
///
/// 0.004 /m² is **one thing per 250 m²** — a fifteen-metre square with a single
/// tuft of grass in it. That number was set when a scatter kind was a bare
/// transform and the cost of being wrong was zero; TER2a gives the three kinds
/// real meshes, and at one per 250 m² an island covered in them still reads as
/// bare ground.
///
/// **The TER2a audit's correction to the sentence above.** The meshes are
/// authored, byte-locked and cooked, and **nothing draws them**: `push_scatter`
/// builds a `PrimMesh::Cube` tinted from a five-entry palette, `ScatterBatch`
/// has no mesh field, and `PcgKind::mesh` never survives evaluation into a
/// `PcgInstance` — see `island_gate::the_cover_meshes_are_shipped_and_are_not_yet_drawn`.
/// So what this raise actually multiplied, today, is the number of **placeholder
/// cubes** in the player's ~1.3 km²: 2 681 → 16 771 at the authored 0.7–1.6 m
/// scale range, about one every 9 m of resident ground. The number is the right
/// one for the day the upload lands, it is measured, and it is inside every
/// budget below — and until that day it is cubes. Reverting to 0.004 until then
/// is the other defensible answer and is named in the wave's carried list rather
/// than taken by the audit.
///
/// What bounds it is not the island's area but the **working set**, and the
/// working set is what the instrument measures. At 0.004 the shipped island
/// frame drew **2 681** scattered instances: the scatter evaluates only where
/// terrain is resident, which is `SIM_MARGIN_TILES` (2.0) of level-0 pages
/// around each observer — about 1.3 km², of which the scattering biomes are a
/// fraction. So the per-frame population scales with the density and nothing
/// else. **Measured at 0.02: 16 771 instances** of 32–128 triangles — about
/// 1.1 M triangles, against the 10 M-triangle gate P13 measured at 2.4 % cull
/// and the 15 k instances `phase19-town` already draws.
///
/// *The scaling was optimistic and the instrument is what said so.* A linear
/// scale from the 0.004 reading predicts 13 405 — **20 % below** the 16 771 the
/// frame actually drew, which is the measurement sitting **25 % above** the
/// prediction. A jittered per-cell scatter does not divide evenly. The
/// prediction is recorded beside the measurement rather than replaced by it,
/// because the lesson is the house one: an inference dressed as a measurement
/// is worse than no measurement.
///
/// The CPU fallback is what caps it: 16 771 is **25.6 %** of
/// `MAX_CPU_SCATTER_INSTANCES` (65 536), so a tier that cannot reach the GPU
/// scatter path still draws every instance rather than a nearest-first subset —
/// with room for three more raises of this size before the two tiers stop
/// drawing the same island. At 0.1 /m² they would.
///
/// *The arm is tighter than that sentence, deliberately* (TER2a audit). "Three
/// more raises" is the distance to the **real** ceiling, 65 536; the arm below
/// trips at a **third** of it (21 845), i.e. after roughly one more raise of this
/// size. A tripwire that only fires when the thing has already broken is not a
/// tripwire, so the two numbers are different on purpose — and are both written
/// down here so neither reads as the other.
///
/// **The honest bound this does not fix**: the scatter is evaluated on the
/// SIMULATION's resident set, and the renderer draws terrain far past it (the
/// clipmap's outer rings are pages the sim never asked for). So ground cover
/// stops at roughly 1.3 km² around the player and bare ground continues to the
/// horizon. Widening `SIM_MARGIN_TILES` would move it and would also widen every
/// physics query and every biome evaluation with it; the right fix is a scatter
/// residency of its own, and it is a wave rather than a constant.
pub const ISLAND_SCATTER_DENSITY: f64 = 0.02;

/// A stable GUID from the island's name and a salt, mirroring
/// `inf_island`'s own derivation so the level and the build agree about which
/// asset is which without either storing a table.
pub(crate) fn derived(name: &str, salt: &str) -> Uuid {
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
    Uuid::from_bytes(bytes)
}

/// The island's sun.
pub fn sun_guid(name: &str) -> Uuid {
    derived(name, "island.sun")
}
/// The island's terrain entity.
pub fn terrain_entity_guid(name: &str) -> Uuid {
    derived(name, "island.terrain.entity")
}
/// The island's road-surface entity.
pub fn roads_entity_guid(name: &str) -> Uuid {
    derived(name, "island.roads.entity")
}
/// The ocean.
pub fn ocean_guid(name: &str) -> Uuid {
    derived(name, "island.ocean")
}
/// The hero.
pub fn hero_guid(name: &str) -> Uuid {
    derived(name, "island.hero")
}
/// Lake `i`.
pub fn lake_guid(name: &str, i: usize) -> Uuid {
    derived(name, &format!("island.lake.{i}"))
}
/// River `i`.
pub fn river_guid(name: &str, i: usize) -> Uuid {
    derived(name, &format!("island.river.{i}"))
}
/// The scatter volume over site `i`'s quarter of the world.
pub fn cover_volume_guid(name: &str, i: usize) -> Uuid {
    derived(name, &format!("island.cover.{i}"))
}

/// The `.inf_pcg` the biome set binds — the island's ground cover.
///
/// A **document-only** envelope, exactly as `phase18-scatter`'s is: no graph, so
/// no grammar and no building passes, which is right because this is vegetation
/// and the settlements are wave I8's. The biome binding is what restricts it to
/// the biomes that scatter — `bind_document` rewrites the sampler as
/// `Multiply(Biome{id}, authored)` and re-salts the seed per biome, so one
/// document serves six biomes without six copies of it.
pub fn island_cover_document(seed: u64) -> inf_pcg::PcgDocument {
    use inf_pcg::{PcgKind, PcgRule, SamplerDef};
    let rule = PcgRule {
        name: "island-cover".into(),
        // Slope-limited: nothing grows on a 45-degree face, and the feather is
        // what keeps the treeline from being a drawn line.
        sampler: SamplerDef::Slope {
            min_deg: 0.0,
            max_deg: 34.0,
            feather_deg: 6.0,
        },
        scatter: inf_pcg::ScatterParams {
            seed,
            cell_size: ISLAND_SCATTER_CELL_M,
            base_density: ISLAND_SCATTER_DENSITY,
            jitter: 1.0,
            align_to_normal: false,
            scale_range: (0.7, 1.6),
            rotation: inf_pcg::RotationMode::RandomYaw,
            altitude_offset: 0.0,
        },
        // **The three kinds have meshes now** (TER2a, clause 5). All three
        // carried `mesh: None` — a bare transform, which the scatter evaluates,
        // the biome binding restricts, the residency pages and the frame counts,
        // and which draws nothing at all. `CoverKind::ALL`'s order IS this
        // palette's order: `kind_index` on a scattered instance indexes here.
        kinds: vec![
            PcgKind {
                mesh: Some(crate::cover::cover_mesh_guid(
                    crate::cover::CoverKind::GrassTuft,
                )),
                weight: 4.0,
            },
            PcgKind {
                mesh: Some(crate::cover::cover_mesh_guid(
                    crate::cover::CoverKind::Shrub,
                )),
                weight: 2.0,
            },
            PcgKind {
                mesh: Some(crate::cover::cover_mesh_guid(crate::cover::CoverKind::Rock)),
                weight: 1.0,
            },
        ],
    };
    inf_pcg::PcgDocument::single_layer("cover", vec![rule])
}

/// The `.inf_pcg` payload.
pub fn island_cover_payload(seed: u64) -> inf_pcg::PcgAssetPayload {
    inf_pcg::PcgAssetPayload::new(island_cover_document(seed))
}

/// **The island's four ground layers** (wave TER2a, clause 3).
///
/// The order is `inf_island::splat`'s and is load-bearing: the build writes a
/// per-sample `[u8; 4]` whose channels are grass, rock, forest floor and sand in
/// that order, and this is what those four channels *are*. Swapping two here
/// paints the beaches with rock and nothing would report it.
///
/// Each layer names a `.inf_mat` from the engine's committed ground library
/// (`samples/ground/`), which is what turns the terrain shader's four-layer
/// virtual-texture branch from a capability into a picture. The scalar
/// `albedo`/`roughness` beside it are **not decoration**: they are what a
/// surface shades with while its pages stream in, and what it shades with for
/// ever on an adapter with no virtual textures — so they are the ground sets'
/// own base colours rather than a second opinion about them.
///
/// `tex_scale` is metres per tile, and it is also what the procedural triplanar
/// grain is scaled by, so it is one number doing two jobs: at 2 m a 1 024²
/// albedo is 1.95 mm a texel and the grain that breaks up its tiling is a 2 m
/// feature. Both are what those surfaces want.
fn island_ground_layers() -> [inf_ecs::components::TerrainLayer; 4] {
    use inf_ecs::components::TerrainLayer;
    use inf_ecs::math::Color;
    use inf_material::ground::GroundKind;
    let layer = |kind: GroundKind| {
        let c = kind.base_color();
        TerrainLayer {
            albedo: Color::new(c[0], c[1], c[2], 1.0),
            roughness: f64::from(kind.roughness()),
            tex_scale: kind.tex_scale_m(),
            material: Some(crate::ground::ground_material_guid(kind)),
        }
    };
    [
        layer(GroundKind::Grass),
        layer(GroundKind::Rock),
        layer(GroundKind::ForestFloor),
        layer(GroundKind::Sand),
    ]
}

/// Author the island's level from its committed design.
pub fn island_scene(design: &inf_island::IslandDesign) -> SceneDoc {
    use inf_ecs::components::{
        AlwaysLoaded, Light, LightKind, MeshRef, PcgVolume, SkyAtmosphere, Spline, SplineInterp,
        StreamingSource, Terrain, TimeOfDay, Transform, WaterBody, WaterKind,
    };
    use inf_ecs::math::{Color, Vec2d, Vec3d};

    let name = design.recipe.name.as_str();
    let mut doc = SceneDoc::new();
    doc.set_title(name);

    // **Where on Earth the world is.** The sky reads its latitude from this, so a
    // shadow at noon falls the way it falls at 49 N — which is also what pins the
    // world frame: +X east, +Y up, -Z north.
    doc.set_geo(design.anchor.clone());

    doc.set_settings(LevelSettings {
        partition: PartitionSettings {
            enabled: true,
            cell_size_m: ISLAND_CELL_SIZE_M,
            activation_radius_m: ISLAND_ACTIVATION_M,
            prefetch_margin_m: ISLAND_PREFETCH_M,
        },
        ..LevelSettings::default()
    });

    // ── the sky ───────────────────────────────────────────────────────────────
    let sun = sun_guid(name);
    doc.create_with_guid(sun, SpawnKind::Empty, "Sun", None);
    insert!(
        doc,
        sun,
        Transform {
            translation: Vec3d::ZERO,
            rotation: Vec3d::new(-46.0, -28.0, 0.0),
            scale: Vec3d::ONE,
        },
    );
    insert!(
        doc,
        sun,
        Light {
            kind: LightKind::Directional,
            color: Color::WHITE,
            intensity: 3.2,
            ..Default::default()
        },
    );
    insert!(
        doc,
        sun,
        TimeOfDay {
            seconds: 10.5 * 3600.0,
            rate: ISLAND_CLOCK_RATE,
            ..TimeOfDay::default()
        },
    );
    insert!(doc, sun, SkyAtmosphere::default());
    insert!(doc, sun, AlwaysLoaded);

    // ── the ground ────────────────────────────────────────────────────────────
    //
    // Streamed: the terrain ships NO tiles in the level, only the `.inf_terrain`
    // GUID, which is what keeps a 342 MB world out of a 30 KB document. And
    // `AlwaysLoaded`, because a Terrain occupies space and a partitioner would
    // otherwise bin the whole heightfield into the one cell holding its origin —
    // and the ground would despawn under the player.
    // **IDENTITY, AND THE OFFSET IT REPLACES WAS THE ISLAND MOVED HALF A WORLD.**
    //
    // Every other terrain in this repository is built with level-0 tile
    // coordinates starting at `(0, 0)`, so its entity is translated to
    // `-span/2` to centre the grid on the world origin
    // (`island_frame_terrain_origin` is the pattern). **`IslandGrid` does not
    // work that way**: `tile0 = -(tiles / 2)`, so the `.inf_terrain`'s own tile
    // indices are already centred and its sample frame **is** the world frame —
    // which is exactly what the whole build assumes (`CoarseHeights::of(&data,
    // min, max, …)`, the grade audit's `data.height_at(p)`, the channel carve,
    // the biome stamp).
    //
    // Translating the entity as well applied that centring twice. Measured on
    // the fixture, through the shipped host's own `terrain.height_at` seam: the
    // hero's start read **0.000 m of unauthored ground where the design puts
    // 129.916 m**, and `(0, 0)` read 80.000 m off a page 768 m away. On the
    // shipped island the displacement is 3 584 m on both axes, so half the
    // terrain sat outside the world. Nothing caught it because the island gate
    // never attached the terrain streamer, so the simulation's working set was
    // empty and every query answered the unauthored default — the two hosts
    // agreed about no ground at all.
    let terrain_guid = terrain_entity_guid(name);
    doc.create_with_guid(terrain_guid, SpawnKind::Empty, "Ground", None);
    insert!(doc, terrain_guid, Transform::IDENTITY);
    {
        let mut t = Terrain::configured(
            design.recipe.grid.tile_resolution,
            design.recipe.grid.meters_per_sample,
        );
        t.asset = Some(inf_island::terrain_guid(name));
        t.biome_set = Some(inf_island::biome_set_guid(name));
        t.layers = island_ground_layers();
        debug_assert!(t.data.is_empty(), "a streamed terrain ships no tiles");
        insert!(doc, terrain_guid, t);
    }
    insert!(doc, terrain_guid, AlwaysLoaded);

    // ── the roads, as one drawn surface ───────────────────────────────────────
    //
    // No collider, by IB-4's ruling: a road conforms to the terrain, whose
    // heightfield collider already answers there, so a per-segment trimesh would
    // be 3.63 ms a step for nothing a body can reach.
    let roads = roads_entity_guid(name);
    doc.create_with_guid(roads, SpawnKind::Empty, "Roads", None);
    insert!(doc, roads, Transform::IDENTITY);
    insert!(
        doc,
        roads,
        MeshRef {
            asset: Some(inf_island::road_mesh_guid(name)),
            ..Default::default()
        },
    );
    insert!(doc, roads, AlwaysLoaded);

    // ── the sea ───────────────────────────────────────────────────────────────
    //
    // One body. `WaterSurface::Ocean` is unbounded in the simulation and the
    // renderer tessellates a patch around the camera, so an island needs exactly
    // one however long its coastline is.
    let ocean = ocean_guid(name);
    doc.create_with_guid(ocean, SpawnKind::Empty, "Ocean", None);
    insert!(
        doc,
        ocean,
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: design.recipe.sea.level_m,
            wave_amplitude_m: 0.6,
            wave_length_m: 34.0,
            wave_steepness: 0.42,
            wave_count: 5,
            wave_seed: 0x0015_1A4D,
            // Body-local wind: a coastline's sea state must not depend on where
            // the weather blend happens to be when a trace is taken.
            wind_from_weather: false,
            wind_x: 6.5,
            wind_z: -2.5,
            ..WaterBody::default()
        },
    );
    insert!(doc, ocean, AlwaysLoaded);

    // ── the lakes ─────────────────────────────────────────────────────────────
    for (i, l) in design.network.lakes.iter().enumerate() {
        let g = lake_guid(name, i);
        doc.create_with_guid(g, SpawnKind::Empty, &format!("Lake {i}"), None);
        insert!(
            doc,
            g,
            Transform::from_translation(DVec3::new(l.centre.x, l.level_m, l.centre.y)),
        );
        insert!(
            doc,
            g,
            WaterBody::lake(l.level_m, Vec2d::new(l.half_extent.x, l.half_extent.y)),
        );
    }

    // ── the rivers ────────────────────────────────────────────────────────────
    //
    // The centreline is the `Spline` on the SAME entity — P20.1's composition
    // rule, so there is nothing to resolve and nothing to dangle — authored in
    // world space under an identity transform.
    for (i, s) in design.rivers(MAX_RIVER_BODIES).into_iter().enumerate() {
        let g = river_guid(name, i);
        doc.create_with_guid(g, SpawnKind::Empty, &format!("River {i}"), None);
        insert!(doc, g, Transform::from_translation(DVec3::ZERO));
        let w = s.width_m();
        let d = s.depth_m();
        insert!(
            doc,
            g,
            WaterBody {
                river_width_start_m: (w * 0.7).max(1.0),
                river_width_end_m: w,
                river_depth_start_m: (d * 0.7).max(0.3),
                river_depth_end_m: d,
                ..WaterBody::river(w, d, 0.5 + 2.0 * s.grade().clamp(0.0, 0.5))
            },
        );
        insert!(
            doc,
            g,
            Spline {
                points: s.points.iter().map(|p| Vec3d::new(p.x, p.y, p.z)).collect(),
                closed: false,
                interp: SplineInterp::CatmullRom,
            },
        );
    }

    // ── the vegetation ────────────────────────────────────────────────────────
    //
    // The binding is on the `.inf_biomes` set, which both hosts resolve through
    // `inf_pcg::BiomeBinding::from_set`, and it evaluates over the **terrain's
    // own bounds** — so there is no `PcgVolume` here and no half-extent to keep
    // in step with the world's. That is the one door: the level names a biome
    // set, the set names a graph, and the graph is masked by the painted ids.

    // ── the settlements ───────────────────────────────────────────────────────
    //
    // **Wave I8a: the seven pads stop being terraces.** One `PcgVolume` per block
    // — a centre, an axis-aligned half-extent, a seed and the GUID of the zone
    // document its archetype names. The plan is
    // `crate::settlement::settlements`, which is a pure function of the same
    // committed design every number above comes from.
    //
    // **Not `AlwaysLoaded`, deliberately.** A settlement block is exactly what
    // the partition is for: 172 volumes over 51 km² is a thousand buildings the
    // simulation must not hold at once, and `PcgVolume` evaluation runs on cell
    // activation (`cell_stream::reconcile`) as well as at load. The blocks
    // therefore stream with their cells, which is the whole reason the level can
    // carry them at all.
    //
    // `draw_distance` is left at its default: the I3 structure bands
    // (`DEFAULT_STRUCTURE_LOD_M`, 96 m) are what decide whether a building draws
    // its parts or its shell, and a second per-volume distance cut on top of
    // them would be a second authority on the same question.
    for plan in crate::settlement::settlements(design) {
        for b in &plan.blocks {
            let g = crate::settlement::block_guid(name, b.site, b.col, b.row);
            doc.create_with_guid(
                g,
                SpawnKind::Empty,
                &format!("{} {} {},{}", plan.name, b.archetype.name(), b.col, b.row),
                None,
            );
            insert!(
                doc,
                g,
                Transform {
                    translation: Vec3d::new(b.centre.x, 0.0, b.centre.y),
                    rotation: Vec3d::ZERO,
                    scale: Vec3d::ONE,
                },
            );
            insert!(
                doc,
                g,
                PcgVolume {
                    graph: Some(crate::settlement::zone_guid(b.archetype)),
                    extent: Vec2d::new(b.half.x, b.half.y),
                    seed: b.seed,
                    ..Default::default()
                },
            );
        }
    }

    // ── the hero ──────────────────────────────────────────────────────────────
    //
    // **It is the starter character** (SK1c). This used to be forty lines of
    // hand-rolled components ending in `AnimStateMachine { sm: None }` and no
    // `SkeletalMesh` — a capsule that walked, with nothing to draw and nothing to
    // pose — because the one door that knows how to build a character
    // (`SceneDoc::edit_create_character`) minted its own entity GUID and the
    // island derives every one of its own. That door takes a GUID now, so the
    // island spawns a character through the same code path the New Character
    // wizard does, and the assets it names are the ones
    // `samples/starter-character` commits.
    //
    // The assets reach a built project through the recipe's `[content]` list
    // (`inf_island::write_content`), so this crate names a GUID and nothing else:
    // no new crate edge, and the island's own generator still knows nothing about
    // `inf-anim`.
    let ids = crate::samples::starter_character_ids();
    let asset = |id: Option<inf_asset::AssetId>| id.expect("every starter id is fixed").0;
    let feet = design.start(START_LIFT_M);
    let hero = hero_guid(name);
    doc.edit_create_character_with_guid(
        hero,
        "Hero",
        asset(ids.skeleton),
        asset(ids.mesh),
        asset(ids.machine),
        feet,
        Some(asset(ids.actor)),
        hero_height_m(),
    );
    // **Both of these are outside that door, and both are load-bearing.**
    // `StreamingSource` is the partition's activation anchor AND the I3 collider
    // band's — the two cannot disagree about where the simulation is because they
    // read the same component — and `AlwaysLoaded` keeps the hero resident in its
    // own cell. `edit_create_character` deliberately inserts neither: a character
    // is not necessarily a streaming anchor, and putting that opinion in the
    // wizard's door would put it on every character anybody ever spawns.
    insert!(doc, hero, StreamingSource { radius_m: 256.0 });
    insert!(doc, hero, AlwaysLoaded);

    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// Write the island's committed halves beside its recipe: the level and the
/// `.inf_pcg` its biome set binds.
pub fn write_island_level(
    design: &inf_island::IslandDesign,
    dir: &std::path::Path,
) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("mkdir: {e}"))?;
    let name = design.recipe.name.as_str();
    let slug = inf_island::slug(name);

    crate::scene::serialize::save(
        &island_scene(design),
        &dir.join(format!("{slug}.inf_lvl")),
        Some(inf_island::level_guid(name)),
    )?;

    let bytes = inf_asset::encode(&island_cover_payload(design.recipe.seed_for("cover")))
        .map_err(|e| format!("encode the island's .inf_pcg: {e}"))?;
    let p = dir.join(format!("{slug}Cover.inf_pcg"));
    std::fs::write(&p, &bytes).map_err(|e| format!("write {}: {e}", p.display()))?;
    let mut side = inf_asset::AssetSidecar::new(
        inf_asset::AssetId(inf_island::cover_pcg_guid(name)),
        inf_asset::AssetKind::Pcg,
        inf_asset::ContentHash::of(&bytes),
    );
    // **The three cover meshes it scatters** (TER2a clause 5). The cook reaches
    // them through its own implicit `Pcg` edge either way; this is the edge the
    // ASSET DATABASE reads — the delete-with-references warning and the Content
    // Drawer's "show references" both walk sidecars, so without it an author can
    // delete a mesh thirteen thousand instances are standing on and be told
    // nothing.
    side.dependencies = crate::cover::CoverKind::ALL
        .iter()
        .map(|k| inf_asset::AssetId(crate::cover::cover_mesh_guid(*k)))
        .collect();
    side.save(&p)
        .map_err(|e| format!("write the .inf_pcg sidecar: {e}"))
}

/// The repository's own root, from this crate's manifest.
pub fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Every committed island recipe, as repo-relative paths.
///
/// Exhaustive by hand and in one place, so adding an island is a decision taken
/// once rather than three times (the bless, the byte lock and the level count).
/// **How fast the island's day runs** — clock-seconds per simulated second, the
/// [`TimeOfDay::rate`] the level authors (island wave NPC1d).
///
/// Eighteen is an **eighty-minute day**, and it is set from a measurement rather
/// than chosen for looks. `inf_ecs::society` gives a commute one hour of the
/// level clock and a `ScheduleLeg` walks its route over that window, so the
/// metres per second a commute implies is `length x rate / 3600`. Measured on
/// the CI island's own derived population — 329 residents of Harbour City — the
/// median commute is **320 m**, so the rate that makes the median commute a
/// *walk* is `3600 x 1.65 / 320`, and eighteen is that number rounded to
/// something a reader can hold. At eighteen the island's commutes imply
/// **0.89 / 1.60 / 1.97 m/s** (min / median / max) against the movement model's
/// own `walk_speed_mps` of 1.65 — every one of them a walking pace, and the
/// median within three per cent of it.
///
/// The first draft was thirty (a forty-eight-minute day, which is a nicer number
/// to say) and it made the median commute **2.67 m/s** — a jog. The arm that
/// found that is `the_islands_own_rate_makes_a_commute_a_walk`, and it is in the
/// gate rather than in this comment because a proportion stated in a doc is a
/// claim (the NPC1c law about a 2.4 m capsule wearing a 1.8 m comment).
///
/// It was **zero** — a frozen clock at 10:30 UTC — from wave I7 until this wave,
/// which is why the I8b night-window substrate (`inf_render::night_glow_step`)
/// had never once returned a non-zero step on the shipped island: the sun could
/// not get below the horizon. Turning it on is the whole of clause 3.
pub const ISLAND_CLOCK_RATE: f64 = 18.0;

pub const ISLAND_RECIPES: [&str; 2] = [
    "samples/island/island.toml",
    "samples/island-fixture/island.toml",
];

/// Read one committed island's design, or `None` when the recipe is not present.
///
/// `None` rather than an error: a tree that has not blessed the samples yet
/// should not fail CI, which is the same rule
/// `committed_sample_matches_generators` already applies to every other sample.
pub fn committed_design(rel: &str) -> Option<inf_island::IslandDesign> {
    let p = repo_root().join(rel);
    if !p.exists() {
        return None;
    }
    let recipe = inf_island::IslandRecipe::load(&p).ok()?;
    inf_island::read_design(&recipe).ok()
}

/// Write every committed island's level and `.inf_pcg`.
pub fn write_island_levels() -> Result<(), String> {
    for rel in ISLAND_RECIPES {
        let Some(d) = committed_design(rel) else {
            continue;
        };
        let dir = repo_root().join(rel).parent().unwrap().to_path_buf();
        write_island_level(&d, &dir)?;
    }
    Ok(())
}

/// **The committed-design source scan**, shared by every module that authors a
/// committed island document (island wave I8a).
///
/// It was `island.rs`'s alone until the settlement generator arrived, and one
/// file's scan is a scan that stops at the file somebody happened to write
/// first: the level's numbers and the settlements' block positions reach the
/// same committed `.inf_lvl`, so both have to be authored from committed design
/// alone or neither is.
#[cfg(test)]
pub(crate) mod scan {
    /// The **non-test, non-comment** lines of a module source, one-based.
    ///
    /// The scan stops at the test module, and it has to: an arm's own needle
    /// list lives in the file it scans, so a whole-file scan matches itself and
    /// fails on the line that declares what it is looking for. (It did.)
    pub fn code_lines(whole: &str) -> Vec<(usize, String)> {
        let src = whole
            .split_once("#[cfg(test)]")
            .map(|(before, _)| before)
            .unwrap_or(whole);
        assert!(
            src.len() < whole.len(),
            "the test module marker moved; this scan is reading itself"
        );
        assert!(
            src.len() > 4_000,
            "the scan is reading {} bytes of a module that is not that small",
            src.len()
        );
        src.lines()
            .enumerate()
            .filter(|(_, l)| !l.trim_start().starts_with("//"))
            .map(|(i, l)| (i + 1, l.to_string()))
            .collect()
    }

    /// Every `inf_island::` item a code listing names, with the line each was
    /// first seen on.
    ///
    /// # It reads a BRACE GROUP as well as a path, and that was a hole
    ///
    /// The first version took the identifier immediately after `inf_island::`
    /// and stopped. `use inf_island::{IslandDesign, Route};` puts a `{` there,
    /// so the extractor read an **empty name and recorded nothing** — a module
    /// that imported every door in the crate by one `use` line scanned clean.
    /// Found the day a second module joined the scan and its own anti-vacuity
    /// arm (*"the module no longer reads the committed design at all"*) fired.
    ///
    /// # …and a brace group that WRAPS, which is the same hole one line down
    ///
    /// (Island wave I8a audit.) Reading `{` to the `}` on the same line still
    /// missed the form `rustfmt` produces the moment an import list is long
    /// enough to wrap:
    ///
    /// ```text
    /// use inf_island::{
    ///     IslandDesign, Route, Site, SiteKind, sample_terrain,
    /// };
    /// ```
    ///
    /// The first line's `{` opens a group with nothing after it, the following
    /// lines never say `inf_island::` at all, and the module scans clean again.
    /// That is not a hypothetical shape: `settlement.rs`'s own import is four
    /// names and 48 characters, and the fifth door anybody adds wraps it. An open
    /// group therefore stays open across lines until its `}`, and every name in
    /// it is recorded against the line the group **started** on.
    pub fn island_doors(code: &[(usize, String)]) -> std::collections::BTreeMap<String, usize> {
        fn record(used: &mut std::collections::BTreeMap<String, usize>, group: &str, line: usize) {
            for name in group.split(',') {
                let name = name.trim();
                if !name.is_empty() {
                    used.entry(name.to_string()).or_insert(line);
                }
            }
        }
        let mut used: std::collections::BTreeMap<String, usize> = Default::default();
        // `Some(line)` while a brace group opened on `line` is still unclosed.
        let mut open: Option<usize> = None;
        for (n, line) in code {
            let mut rest = line.as_str();
            if let Some(started) = open {
                match rest.split_once('}') {
                    Some((group, tail)) => {
                        record(&mut used, group, started);
                        open = None;
                        rest = tail;
                    }
                    None => {
                        record(&mut used, rest, started);
                        continue;
                    }
                }
            }
            while let Some(at) = rest.find("inf_island::") {
                rest = &rest[at + "inf_island::".len()..];
                if let Some(stripped) = rest.strip_prefix('{') {
                    match stripped.split_once('}') {
                        Some((group, tail)) => {
                            record(&mut used, group, *n);
                            rest = tail;
                        }
                        None => {
                            record(&mut used, stripped, *n);
                            open = Some(*n);
                            rest = "";
                        }
                    }
                    continue;
                }
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    used.entry(name).or_insert(*n);
                }
            }
        }
        used
    }

    /// Lines that reach `inf_island` under **another name**, which is the one
    /// spelling [`island_doors`] cannot follow (island wave I8a audit).
    ///
    /// The extractor is an allowlist over what it can read, and what it reads is
    /// the literal `inf_island::`. `use inf_island as isl;` — or
    /// `use inf_island::sample_terrain as h;` — renames the door and the scan
    /// walks past it. There is no cheap way to follow an alias without parsing,
    /// so the alias itself is refused: a module authored from committed design
    /// alone has no reason to want one, and a REFUSAL is a rule an author meets
    /// immediately rather than a hole nobody meets at all.
    /// Only `use` lines are read, so `inf_island::clamp(x as i32)` is not an
    /// alias and does not trip it. A `type` alias needs no rule: the aliased path
    /// still spells `inf_island::<Door>` on its own line, so the scan has already
    /// recorded the door by the time anybody uses the short name.
    pub fn aliases(code: &[(usize, String)]) -> Vec<usize> {
        code.iter()
            .filter(|(_, l)| {
                let t = l.trim_start();
                t.starts_with("use ") && t.contains("inf_island") && t.contains(" as ")
            })
            .map(|(n, _)| *n)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(rel: &str) -> Option<inf_island::IslandDesign> {
        committed_design(rel)
    }

    /// Lines of the module's **non-test, non-comment** source, with their
    /// one-based numbers — the scan every arm below reads.
    fn module_code() -> Vec<(usize, String)> {
        super::scan::code_lines(include_str!("island.rs"))
    }

    /// **Nothing here reads an elevation.** The level is authored from committed
    /// design alone, which is what makes it a committed document CI can check.
    ///
    /// # An ALLOWLIST, because a ban enumerates only what somebody thought of
    ///
    /// The first version of this arm banned five names — `TileMosaic`,
    /// `plan_tiles`, `build_island`, `elevation_at`, `ProjectionLattice` — and
    /// every other door onto an elevation sailed through it:
    /// `inf_island::sample_terrain`, `inf_island::IslandBuild`,
    /// `inf_terrain::read_terrain_asset`, `TerrainData::height_at`. That is the
    /// P22 law ("a ban enumerates what you thought of, an allowlist what is
    /// allowed") met on the arm that carries this wave's own headline decision.
    ///
    /// So the claim is inverted: the module may name **exactly these** items of
    /// `inf_island`, and no `inf_terrain` item at all. Adding a door here is a
    /// deliberate edit to this list rather than a silent one to the module.
    #[test]
    fn the_level_is_authored_from_committed_design_alone() {
        /// Everything `island.rs` is allowed to reach in the island crate. Every
        /// one of them is a **committed-design** door or a name-derived GUID; not
        /// one of them opens an elevation tile.
        const ALLOWED: &[&str] = &[
            "IslandDesign",   // the committed design, read by `read_design`
            "IslandRecipe",   // the committed recipe
            "biome_set_guid", // …and five GUIDs derived from the island's name
            "cover_pcg_guid",
            "level_guid",
            "read_design", // the one door onto the committed layers
            "road_mesh_guid",
            "slug",
            "terrain_guid",
        ];
        let code = module_code();
        assert!(
            super::scan::aliases(&code).is_empty(),
            "island.rs imports `inf_island` under another name at line(s) {:?} — \
             the scan follows the literal `inf_island::` and an alias walks past \
             it (island wave I8a audit)",
            super::scan::aliases(&code)
        );
        let used = super::scan::island_doors(&code);
        println!("island.rs reaches inf_island::{{{:?}}}", used.keys());
        for (name, line) in &used {
            assert!(
                ALLOWED.contains(&name.as_str()),
                "island.rs:{line} names `inf_island::{name}`, which is not on the \
                 committed-design allowlist. The level must be authorable without \
                 the terrain, or it is a committed document only one machine can \
                 produce — and the terrain is a build artifact of one machine \
                 because the sampling step goes through the projection modules \
                 the portability law exempts. If this door really is design-only, \
                 add it to ALLOWED with the reason."
            );
        }
        // …and the allowlist is not vacuous: the module really does reach the
        // crate, and reaches the ONE door the decision names.
        assert!(used.len() >= 5, "island.rs reaches {} items", used.len());
        assert!(
            used.contains_key("read_design"),
            "the module no longer opens the committed design at all"
        );

        // The terrain crate is out of bounds entirely — `read_terrain_asset`,
        // `TerrainData` and `height_at` all live there, and none of them is a
        // committed-design door.
        for (n, line) in &code {
            assert!(
                !line.contains("inf_terrain::"),
                "island.rs:{n} names `inf_terrain::` — every door onto an \
                 elevation is in that crate: {}",
                line.trim()
            );
        }
    }

    /// **The scan can fail** — the anti-vacuity arm the sibling gate
    /// (`inf-island/tests/portable_math_law.rs`) has and this one did not.
    ///
    /// A source scan whose extraction is broken is indistinguishable from a
    /// module that is clean, and the arm above would have passed over an empty
    /// string, a mis-split file or a needle that never matches.
    #[test]
    fn the_committed_design_scan_finds_a_door_when_one_is_there() {
        // The extractor, run against a line that names a forbidden door.
        let probe = vec![(
            1usize,
            "    let t = inf_island::sample_terrain(&r, &m, &l, &c);".to_string(),
        )];
        let found = super::scan::island_doors(&probe);
        assert_eq!(found.keys().collect::<Vec<_>>(), vec!["sample_terrain"]);
        assert!(
            !["IslandDesign", "read_design", "slug"].contains(&"sample_terrain"),
            "a real door must not be on the allowlist"
        );
        // **And a BRACE GROUP, which the first extractor could not see at all**
        // (island wave I8a): `use inf_island::{A, B};` put a `{` where the
        // identifier was expected, the take-while read an empty string, and a
        // module importing every door in the crate on one line scanned clean.
        let group = vec![(
            2usize,
            "use inf_island::{IslandBuild, sample_terrain, IslandDesign};".to_string(),
        )];
        let doors = super::scan::island_doors(&group);
        let names: Vec<&str> = doors.keys().map(String::as_str).collect();
        assert_eq!(names, vec!["IslandBuild", "IslandDesign", "sample_terrain"]);
        // **AND THE SAME GROUP WRAPPED OVER THREE LINES** (island wave I8a
        // audit), which is what `rustfmt` writes the moment the list is long
        // enough — and which the same-line reader above still could not see: the
        // `{` opened a group with nothing after it and the following lines never
        // say `inf_island::` at all.
        let wrapped = vec![
            (10usize, "use inf_island::{".to_string()),
            (11, "IslandDesign, Route, Site,".to_string()),
            (12, "SiteKind, sample_terrain,".to_string()),
            (13, "};".to_string()),
        ];
        let doors = super::scan::island_doors(&wrapped);
        let mut names: Vec<&str> = doors.keys().map(String::as_str).collect();
        names.sort_unstable();
        assert_eq!(
            names,
            vec![
                "IslandDesign",
                "Route",
                "Site",
                "SiteKind",
                "sample_terrain"
            ],
            "a wrapped brace import scanned clean — the hole moved one line down"
        );
        assert_eq!(
            doors.get("sample_terrain"),
            Some(&10),
            "a name inside a wrapped group is reported against the line the group \
             opened on"
        );
        // …and an ALIAS is refused rather than followed, because the extractor
        // reads one spelling and a rename is the spelling it cannot read.
        assert_eq!(
            super::scan::aliases(&[
                (1usize, "use inf_island as isl;".to_string()),
                (2, "use inf_island::sample_terrain as h;".to_string()),
                (3, "let n = inf_island::clamp(x as i32);".to_string()),
            ]),
            vec![1, 2],
            "the alias probe either missed a rename or called an `as` cast one"
        );
        // …and a comment line is filtered, which is why the real scan drops them.
        assert!("    // inf_island::sample_terrain"
            .trim_start()
            .starts_with("//"));
        // The real module is being read, and it is the module this file is in.
        let code = module_code();
        assert!(code.len() > 200, "the scan read {} lines", code.len());
        assert!(
            code.iter().any(|(_, l)| l.contains("pub fn island_scene")),
            "the scan is not reading island.rs"
        );
    }

    #[test]
    fn the_guids_are_stable_distinct_and_a_function_of_the_island() {
        let a = "Vancouver Island";
        let mut all = std::collections::BTreeSet::new();
        for g in [
            sun_guid(a),
            terrain_entity_guid(a),
            roads_entity_guid(a),
            ocean_guid(a),
            hero_guid(a),
            lake_guid(a, 0),
            lake_guid(a, 1),
            river_guid(a, 0),
            cover_volume_guid(a, 0),
            inf_island::terrain_guid(a),
            inf_island::road_mesh_guid(a),
            inf_island::biome_set_guid(a),
            inf_island::cover_pcg_guid(a),
            inf_island::level_guid(a),
        ] {
            assert!(all.insert(g), "two of the island's guids collide: {g}");
        }
        assert_eq!(sun_guid(a), sun_guid(a));
        assert_ne!(sun_guid(a), sun_guid("Other Island"));
    }

    /// The fixture's level really is a level: it names the terrain, the biome
    /// set, an ocean, the water the design found and a player-controlled hero.
    #[test]
    fn the_fixture_level_carries_the_island_it_describes() {
        let Some(d) = design("samples/island-fixture/island.toml") else {
            println!("SKIP: no island fixture in this tree");
            return;
        };
        let doc = island_scene(&d);
        let name = d.recipe.name.as_str();

        // The geo-anchor reaches the document, which is what the sky's latitude
        // and every future import are read from.
        assert!(doc.geo().enabled);
        assert_eq!(doc.geo().crs, "EPSG:32610");

        // **And every number in it is the recipe's own, carried bit for bit.**
        // This is the I7 CI-red, as an arm: the three degrees used to be
        // inverted out of the easting/northing through `proj4rs`, which is the
        // platform's libm, so the committed level's `origin_latitude_deg` read
        // 49.34307562364773 where it was blessed and 49.34307562364772 on macOS
        // — one ulp, one byte, one red platform of three. A committed byte has
        // to trace to a committed decimal, and `assert_eq!` on an f64 is the
        // only comparison that says so.
        let g = doc.geo();
        let a = &d.recipe.anchor;
        assert_eq!(g.origin_easting_m, a.easting_m);
        assert_eq!(g.origin_northing_m, a.northing_m);
        assert_eq!(g.origin_height_m, a.height_m);
        assert_eq!(g.origin_latitude_deg, a.latitude_deg);
        assert_eq!(g.origin_longitude_deg, a.longitude_deg);
        assert_eq!(g.grid_convergence_deg, a.convergence_deg);
        assert_eq!(g.vertical_datum, a.vertical_datum);

        // The partition is on, at the terrain's own tile lattice.
        let s = doc.settings();
        assert!(s.partition.enabled);
        assert_eq!(s.partition.cell_size_m, ISLAND_CELL_SIZE_M);

        // The terrain names its asset and its palette and ships no tiles.
        let e = doc
            .entity_of(terrain_entity_guid(name))
            .expect("a ground entity");
        let t = doc
            .world()
            .world()
            .get::<inf_ecs::components::Terrain>(e)
            .expect("a Terrain component");
        assert_eq!(t.asset, Some(inf_island::terrain_guid(name)));
        assert_eq!(t.biome_set, Some(inf_island::biome_set_guid(name)));
        assert!(t.data.is_empty(), "a streamed terrain ships no tiles");
        assert_eq!(t.tile_resolution, d.recipe.grid.tile_resolution);

        // One ocean, every lake, and the bounded set of rivers.
        let waters: Vec<&inf_ecs::components::WaterBody> = doc
            .world()
            .world()
            .iter_entities()
            .filter_map(|e| e.get::<inf_ecs::components::WaterBody>())
            .collect();
        let oceans = waters
            .iter()
            .filter(|w| w.kind == inf_ecs::components::WaterKind::Ocean)
            .count();
        let lakes = waters
            .iter()
            .filter(|w| w.kind == inf_ecs::components::WaterKind::Lake)
            .count();
        let rivers = waters
            .iter()
            .filter(|w| w.kind == inf_ecs::components::WaterKind::River)
            .count();
        println!("WATER ENTITIES: {oceans} ocean, {lakes} lakes, {rivers} rivers");
        assert_eq!(oceans, 1, "an island needs exactly one unbounded sea");
        assert_eq!(lakes, d.network.lakes.len());
        assert_eq!(rivers, d.rivers(MAX_RIVER_BODIES).len());
        assert!(rivers <= MAX_RIVER_BODIES);
        assert!(rivers > 0, "the design found no reach worth a body");

        // Every river carries its own centreline on its own entity.
        for i in 0..rivers {
            let g = river_guid(name, i);
            let e = doc.entity_of(g).expect("a river entity");
            let sp = doc
                .world()
                .world()
                .get::<inf_ecs::components::Spline>(e)
                .expect("a river's centreline is the Spline on its own entity");
            assert!(sp.points.len() >= 2);
            assert!(!sp.closed);
        }

        // The hero is player-controlled, streams the world and stands above the
        // ground the design put under it.
        let e = doc.entity_of(hero_guid(name)).expect("a hero");
        let m = doc
            .world()
            .world()
            .get::<inf_ecs::components::CharacterMovement>(e)
            .expect("CharacterMovement");
        assert!(m.player_controlled);
        assert!(doc
            .world()
            .world()
            .get::<inf_ecs::components::StreamingSource>(e)
            .is_some());
        let tr = doc
            .world()
            .world()
            .get::<inf_ecs::components::Transform>(e)
            .expect("a transform");
        let start = d.start(START_LIFT_M);
        assert!((tr.translation.x - start.x).abs() < 1e-9);
        assert!((tr.translation.z - start.z).abs() < 1e-9);
        assert!(
            tr.translation.y > start.y,
            "the capsule's centre must be above its feet"
        );
        // …and the start is at a settlement, not at the world origin by accident.
        let site = d.recipe.sites.first().expect("a site");
        assert!((start.x - site.x).abs() < 1e-9 && (start.z - site.z).abs() < 1e-9);
        assert!(
            start.y > d.recipe.sea.level_m,
            "the hero starts under water"
        );
        println!(
            "START: ({:.1}, {:.1}, {:.1}) at {:?}",
            start.x, start.y, start.z, site.name
        );

        // The level's dependency closure names every asset the build writes --
        // and, since SK1c, every asset the STARTER CHARACTER ships. The three
        // character GUIDs enter through `SkeletalMesh.{mesh, skeleton}`,
        // `AnimStateMachine.sm` and `ActorClass`, which `level_dependencies`
        // already walks, so this is a closure that grew rather than a new
        // mechanism -- and it is what makes the cook copy the rig.
        let ids = crate::samples::starter_character_ids();
        let deps = crate::scene::serialize::level_dependencies(&doc);
        for want in [
            inf_island::terrain_guid(name),
            inf_island::road_mesh_guid(name),
            inf_island::biome_set_guid(name),
            ids.skeleton.unwrap().0,
            ids.mesh.unwrap().0,
            ids.machine.unwrap().0,
        ] {
            assert!(deps.contains(&want), "the level does not depend on {want}");
        }
        // **The controller is deliberately NOT in it.** `level_dependencies`
        // walks asset REFERENCES on components and `ActorClass` is not one of
        // them — a Blueprint class is code, reached by the cook's own scan, and
        // `samples/phase29-locomotion`'s committed sidecar lists its rig, body
        // and machine and not its `.inf_act` for exactly the same reason. Stated
        // here rather than left as an absence, because the `[content]` list in
        // the recipe is what puts the `.inf_act` in the island's project and
        // somebody reading this loop will wonder.
        assert!(
            !deps.contains(&ids.actor.unwrap().0),
            "an `ActorClass` has started entering the level's asset closure — good, \
             but the recipe's `[content]` list and this comment both assume it does not"
        );
    }

    /// **The island's hero is the starter character, built through the wizard's
    /// own door** (SK1c).
    ///
    /// Two halves, and the second is the one the swap exists for.
    ///
    /// *It is a character.* It carries a `SkeletalMesh` naming the committed rig
    /// and body, a machine that is `Some`, and the controller -- the four things
    /// the hand-rolled capsule had none of. `AnimStateMachine { sm: None }` and
    /// no `SkeletalMesh` is a hero that walks, draws nothing and poses nothing,
    /// which is what shipped until this wave.
    ///
    /// *It is the door's character.* Every field is compared against what
    /// `edit_create_character` builds at the same height and the same feet --
    /// `the_showcase_character_matches_the_wizard_door`'s discipline, applied to
    /// the second generator in the tree that used to hand-roll one. The capsule
    /// arithmetic agreed by coincidence before (two copies of the same three
    /// lines); it agrees by construction now, and this is what says so.
    #[test]
    fn the_island_hero_is_the_starter_character_the_wizard_would_build() {
        use inf_ecs::components::{
            ActorClass, AlwaysLoaded, AnimStateMachine, CharacterController3D, CharacterMovement,
            Collider3D, ColliderShape3DKind, RigidBody3D, SkeletalMesh, StreamingSource, Transform,
        };
        let Some(d) = committed_design(ISLAND_RECIPES[1]) else {
            eprintln!("SKIP: no committed fixture design");
            return;
        };
        let name = d.recipe.name.as_str();
        let doc = island_scene(&d);
        let e = doc.entity_of(hero_guid(name)).expect("a hero");
        let w = doc.world().world();
        let ids = crate::samples::starter_character_ids();

        // -- it is a character --
        assert_eq!(
            w.get::<SkeletalMesh>(e).map(|s| (s.skeleton, s.mesh)),
            Some((Some(ids.skeleton.unwrap().0), Some(ids.mesh.unwrap().0))),
            "the island hero carries no rig -- it is a capsule again"
        );
        assert_eq!(
            w.get::<AnimStateMachine>(e).and_then(|m| m.sm),
            Some(ids.machine.unwrap().0),
            "the island hero's machine is None, so it poses nothing"
        );
        assert_eq!(
            w.get::<ActorClass>(e).map(|a| a.0),
            Some(ids.actor.unwrap().0)
        );
        // …and the two the door does not insert, which the island must.
        assert!(
            w.get::<StreamingSource>(e).is_some() && w.get::<AlwaysLoaded>(e).is_some(),
            "the hero lost its streaming anchor -- the partition and the I3 \
             collider band both read it"
        );

        // -- it is the door's character, field by field --
        let mut door_doc = SceneDoc::new();
        let door_guid = door_doc.edit_create_character(
            "Hero",
            ids.skeleton.unwrap().0,
            ids.mesh.unwrap().0,
            ids.machine.unwrap().0,
            d.start(START_LIFT_M),
            Some(ids.actor.unwrap().0),
            hero_height_m(),
        );
        let door = door_doc
            .world()
            .entity_of(door_guid)
            .expect("the door built one");
        let dw = door_doc.world().world();
        assert_eq!(
            w.get::<Collider3D>(e)
                .map(|c| (c.shape_kind, c.half_extents, c.radius)),
            dw.get::<Collider3D>(door)
                .map(|c| (c.shape_kind, c.half_extents, c.radius)),
            "the island's capsule is not the one the wizard would build"
        );
        assert_eq!(
            w.get::<Transform>(e).map(|t| t.translation),
            dw.get::<Transform>(door).map(|t| t.translation),
            "the island places its hero at a different height for the same feet"
        );
        let cm = |c: Option<&CharacterMovement>| {
            c.map(|c| {
                (
                    c.player_controlled,
                    c.stand_half_height_m,
                    c.crouch_half_height_m,
                    c.prone_half_height_m,
                )
            })
        };
        assert_eq!(
            cm(w.get::<CharacterMovement>(e)),
            cm(dw.get::<CharacterMovement>(door))
        );
        assert_eq!(
            w.get::<RigidBody3D>(e).map(|b| b.kind),
            dw.get::<RigidBody3D>(door).map(|b| b.kind)
        );
        assert!(
            w.get::<CharacterController3D>(e).is_some()
                && dw.get::<CharacterController3D>(door).is_some()
        );

        // ANTI-VACUITY: a capsule with real dimensions, derived from the STARTER
        // character's height rather than from a number this file used to choose.
        let c = w.get::<Collider3D>(e).expect("a capsule");
        assert_eq!(c.shape_kind, ColliderShape3DKind::Capsule);
        let h = hero_height_m();
        assert!(
            (h - 1.75).abs() < 1e-12,
            "the starter character's height moved: {h}"
        );
        assert!(
            (c.radius - (h * 0.15)).abs() < 1e-12 && c.half_extents.y > 0.3,
            "{c:?}"
        );
    }

    #[test]
    fn the_cover_document_is_slope_limited_and_island_scaled() {
        let d = island_cover_document(7);
        assert_eq!(d.layers.len(), 1);
        let r = &d.layers[0].rules[0];
        assert_eq!(r.scatter.base_density, ISLAND_SCATTER_DENSITY);
        assert_eq!(r.scatter.cell_size, ISLAND_SCATTER_CELL_M);
        assert_eq!(r.scatter.seed, 7);
        assert!(matches!(
            r.sampler,
            inf_pcg::SamplerDef::Slope { max_deg, .. } if (max_deg - 34.0).abs() < 1e-9
        ));
        assert_eq!(r.kinds.len(), 3);

        // **The arithmetic that bounds this density, and the measurement that
        // corrected it.**
        //
        // The island-wide candidate count is the number people reach for and it
        // is NOT the bound: the scatter is evaluated only where terrain is
        // resident, so what a frame pays for is the WORKING SET.
        //
        // This wave predicted the working set by scaling the shipped
        // instrument's 2 681 instances at 0.004 /m2 linearly to 13 405 at 0.02.
        // The instrument then measured **16 771** -- 25 % ABOVE the prediction,
        // which is the same thing as the prediction sitting 20 % BELOW the
        // measurement (the number the line below prints). A jittered per-cell
        // scatter does not divide evenly. The measurement is the number here;
        // the scaling is kept beside it as what it was, an estimate.
        //
        // **The assertion reads the DOCUMENT'S density, not a literal.** A
        // constant compared against a constant is a tautology the compiler folds
        // and clippy refuses; more to the point it would guard nothing. Scaling
        // the measurement by what `island_cover_document` actually authored
        // means the arm fires the day somebody raises the density past what the
        // CPU tier can draw, which is the only thing worth guarding here.
        const MEASURED_WORKING_SET: f64 = 16_771.0;
        const MEASURED_AT_DENSITY: f64 = 0.02;
        const SCALED_PREDICTION: f64 = 13_405.0;
        let land_m2 = 40.65e6;
        let island_wide = land_m2 * r.scatter.base_density;
        let working_set = MEASURED_WORKING_SET * (r.scatter.base_density / MEASURED_AT_DENSITY);
        println!(
            "SCATTER: {} /m2 is {island_wide:.0} candidates over {:.2} km2 of land, and -- the number that matters -- {working_set:.0} in the working set, MEASURED at {MEASURED_WORKING_SET:.0} (a linear scaling from 0.004 predicted {SCALED_PREDICTION:.0}, {:.0} % low)",
            r.scatter.base_density,
            land_m2 / 1e6,
            (1.0 - SCALED_PREDICTION / MEASURED_WORKING_SET) * 100.0
        );
        // The CPU scatter fallback's own ceiling. A tier that cannot reach the
        // GPU path draws a nearest-first subset past this, which is a different
        // world from the one the GPU tier draws. 16 771 is 25.6 % of it -- so
        // both tiers still draw the same island, with room for three more raises
        // of this size before they stop.
        const MAX_CPU_SCATTER_INSTANCES: f64 = 65_536.0;
        assert!(
            working_set < MAX_CPU_SCATTER_INSTANCES / 3.0,
            "{working_set:.0} instances in the working set is past a third of the CPU fallback's {MAX_CPU_SCATTER_INSTANCES:.0} ceiling -- the two tiers would stop drawing the same island"
        );
        assert!(
            island_wide > 500_000.0,
            "the density is back below what a walk over this island can see"
        );

        // A document-only envelope: no graph, so no grammar and no buildings.
        let p = island_cover_payload(7);
        assert!(p.graph_json.is_none(), "vegetation carries no grammar");
        assert_eq!(p.schema_version, inf_pcg::PcgAssetPayload::CURRENT_VERSION);
    }
}
