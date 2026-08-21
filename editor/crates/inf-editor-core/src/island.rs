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

/// The tallest a designed character on this island is, metres.
///
/// Matches `PHASE29_HEIGHT_M`'s shape — the capsule is derived from it exactly
/// as the New Character wizard derives one, so the hero the island spawns is the
/// hero the movement gates already measure.
const HERO_HEIGHT_M: f64 = 1.8;

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
pub const ISLAND_SCATTER_DENSITY: f64 = 0.004;

/// A stable GUID from the island's name and a salt, mirroring
/// `inf_island`'s own derivation so the level and the build agree about which
/// asset is which without either storing a table.
fn derived(name: &str, salt: &str) -> Uuid {
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
        kinds: vec![
            PcgKind {
                mesh: None,
                weight: 4.0,
            },
            PcgKind {
                mesh: None,
                weight: 2.0,
            },
            PcgKind {
                mesh: None,
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

/// Author the island's level from its committed design.
pub fn island_scene(design: &inf_island::IslandDesign) -> SceneDoc {
    use inf_ecs::components::{
        AlwaysLoaded, AnimStateMachine, CharacterController3D, CharacterMovement, Collider3D,
        ColliderShape3DKind, Light, LightKind, MeshRef, RigidBody3D, SkyAtmosphere, Spline,
        SplineInterp, StreamingSource, Terrain, TimeOfDay, Transform, WaterBody, WaterKind,
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
            rate: 0.0,
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
    let terrain_guid = terrain_entity_guid(name);
    doc.create_with_guid(terrain_guid, SpawnKind::Empty, "Ground", None);
    let (min, _) = design.grid.bounds();
    insert!(
        doc,
        terrain_guid,
        Transform::from_translation(DVec3::new(min.x, 0.0, min.y)),
    );
    {
        let mut t = Terrain::configured(
            design.recipe.grid.tile_resolution,
            design.recipe.grid.meters_per_sample,
        );
        t.asset = Some(inf_island::terrain_guid(name));
        t.biome_set = Some(inf_island::biome_set_guid(name));
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
            wave_seed: 0x1_5_1A_4D,
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

    // ── the hero ──────────────────────────────────────────────────────────────
    //
    // Carries `StreamingSource`, which is both the partition's activation anchor
    // and the I3 collider band's — the two cannot disagree about where the
    // simulation is because they read the same component.
    let radius = (HERO_HEIGHT_M * 0.15).clamp(0.1, 0.5);
    let half_h = (HERO_HEIGHT_M * 0.5 - radius).max(0.05);
    let feet = design.start(START_LIFT_M);
    let hero = hero_guid(name);
    doc.create_with_guid(hero, SpawnKind::Empty, "Hero", None);
    insert!(
        doc,
        hero,
        Transform::from_translation(DVec3::new(feet.x, feet.y + half_h + radius, feet.z)),
    );
    insert!(
        doc,
        hero,
        RigidBody3D {
            kind: inf_ecs::components::BodyKind3D::Kinematic,
            ..Default::default()
        },
    );
    insert!(
        doc,
        hero,
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(radius, half_h, radius),
            radius,
            ..Default::default()
        },
    );
    insert!(doc, hero, CharacterController3D::default());
    insert!(
        doc,
        hero,
        CharacterMovement {
            player_controlled: true,
            stand_half_height_m: half_h,
            crouch_half_height_m: (half_h * 0.5).max(0.05),
            prone_half_height_m: (radius * 0.6).max(0.03),
            ..Default::default()
        },
    );
    insert!(
        doc,
        hero,
        AnimStateMachine {
            sm: None,
            ..Default::default()
        },
    );
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
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(inf_island::cover_pcg_guid(name)),
        inf_asset::AssetKind::Pcg,
        inf_asset::ContentHash::of(&bytes),
    )
    .save(&p)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn design(rel: &str) -> Option<inf_island::IslandDesign> {
        committed_design(rel)
    }

    /// **Nothing here reads an elevation.** The level is authored from committed
    /// design alone, which is what makes it a committed document CI can check.
    #[test]
    fn the_level_is_authored_from_committed_design_alone() {
        // **The scan stops at the test module**, and it has to: this arm's own
        // needle list is in this file, so a whole-file scan matches itself and
        // fails on the line that declares what it is looking for. (It did.)
        let whole = include_str!("island.rs");
        let src = whole
            .split_once("#[cfg(test)]")
            .map(|(before, _)| before)
            .unwrap_or(whole);
        assert!(
            src.len() < whole.len(),
            "the test module marker moved; this scan is reading itself"
        );
        for needle in [
            "TileMosaic",
            "plan_tiles",
            "build_island",
            "elevation_at",
            "ProjectionLattice",
        ] {
            let hits: Vec<usize> = src
                .lines()
                .enumerate()
                .filter(|(_, l)| {
                    let t = l.trim_start();
                    !t.starts_with("//") && l.contains(needle)
                })
                .map(|(i, _)| i + 1)
                .collect();
            assert!(
                hits.is_empty(),
                "island.rs names `{needle}` at {hits:?} — the level must be \
                 authorable without the terrain, or it is a committed document \
                 only one machine can produce"
            );
        }
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

        // The level's dependency closure names every asset the build writes.
        let deps = crate::scene::serialize::level_dependencies(&doc);
        for want in [
            inf_island::terrain_guid(name),
            inf_island::road_mesh_guid(name),
            inf_island::biome_set_guid(name),
        ] {
            assert!(deps.contains(&want), "the level does not depend on {want}");
        }
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

        // **The island-scale arithmetic, printed.** 40 km2 of land at this density
        // is the number the streaming budget has to carry, and the alternative is
        // priced beside it.
        let land_m2 = 40.65e6;
        let here = land_m2 * ISLAND_SCATTER_DENSITY;
        let phase18 = land_m2 * 0.05;
        println!(
            "SCATTER: {ISLAND_SCATTER_DENSITY} /m2 over {:.2} km2 of land is \
             {here:.0} candidates; the phase-18 sample's own 0.05 /m2 would be \
             {phase18:.0}",
            land_m2 / 1e6
        );
        assert!(here < 250_000.0, "{here} candidates before any mask");
        assert!(phase18 > 1.0e6, "the alternative must be materially worse");

        // A document-only envelope: no graph, so no grammar and no buildings.
        let p = island_cover_payload(7);
        assert!(p.graph_json.is_none(), "vegetation carries no grammar");
        assert_eq!(p.schema_version, inf_pcg::PcgAssetPayload::CURRENT_VERSION);
    }
}
