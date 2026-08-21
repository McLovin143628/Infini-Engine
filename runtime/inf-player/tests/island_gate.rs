//! **The island, driven — PIE == shipping over 51 km² of real ground** (wave I7).
//!
//! # What this gate is and what it is not
//!
//! It runs the **CI-scale island**, because the shipped one's terrain is 342.7 MB
//! and is not committed. Everything else is the shipped path: the same recipe
//! format, the same scene generator, the same `.inf_terrain`, the same
//! partition, the same water and the same biome binding — so a change that
//! breaks the island breaks this.
//!
//! # The claim
//!
//! A drive across a streamed island, with the terrain paging under the wheels
//! and the partition activating cells as the source moves, produces **the same
//! bytes** in the shipped player and in the editor's PIE. That is the property
//! every gate in this repository exists to protect, met on the largest world it
//! has: if the terrain streamer, the cell activation or the biome-bound
//! population were a function of anything but sim state, the two would diverge.
//!
//! # Why the drive is scripted through the sim and not through a camera
//!
//! The collider band, the cell activation and the terrain's sim residency all
//! anchor on `StreamingSource` entities — sim state. A camera-driven trace would
//! be measuring the renderer. The hero **is** the streaming source here, and the
//! script moves it.

use std::path::{Path, PathBuf};

use inf_player::runtime_sim::RuntimeSim;
use inf_project::ProjectManifest;

/// The recipe CI builds.
fn fixture_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml")
}

/// The fixed step, matching the level's own `sim_hz`.
const HZ: f64 = 60.0;

/// How many fixed steps the drive runs.
const STEPS: u64 = 900;

/// How far the drive advances per step, metres. 0.4 m at 60 Hz is 24 m/s — a car
/// on a highway, and 360 m over the run, which crosses this fixture's own
/// partition cell and several terrain pages.
const STEP_M: f64 = 0.4;

/// Build the island's project: the recipe's own build, into a temp directory.
///
/// **This is `inf island build`'s own door** (`write_content`), not a
/// reimplementation of it — so a gate that passed while the command produced
/// something different is impossible.
fn build_project(tmp: &Path) -> PathBuf {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    // The design must be non-vacuous BEFORE anything is compared: two empty
    // worlds agree perfectly.
    assert!(
        build.report.land_km2 > 0.5,
        "the fixture has only {:.3} km2 of land",
        build.report.land_km2
    );
    assert!(!build.network.streams.is_empty(), "no water was derived");
    assert!(!build.routes.is_empty(), "no road was designed");

    let proj = tmp.join("island");
    ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    inf_island::write_content(&build, &proj.join("Content")).expect("the island's content writes");
    proj
}

/// Cook it, exactly as `inf cook` does.
fn cook(tmp: &Path) -> PathBuf {
    let proj = build_project(tmp);
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    out
}

/// **The shipping side**: a sim off the cooked pack, **with cell streaming
/// attached** — exactly as `inf_player::run_headless` does.
///
/// Attaching it is not optional and the first draft of this gate found out why:
/// a partitioned level's cooked `.inf_lvl` carries **no entities at all** (they
/// are in the derived `.inf_part`), so a shipping sim without the streamer is a
/// world holding only what `AlwaysLoaded` kept — six records against the
/// editor's fifteen. The two agreed for 411 steps and then did not, which reads
/// like a streaming defect and is a gate that forgot to boot the streamer.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.take_pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    sim
}

/// **The editor side**: the loose `.inf_lvl` the author saved, binned by the same
/// Ring-0 function the cook used.
///
/// This is the pair P16.5's own gate compares, and it is the right one for a
/// partitioned level: a `ScenePayload` carries **no partition** (see
/// `a_scene_payload_carries_no_partition`), so the PIE wire is not the editor's
/// authoritative reading of a streamed world — the document is.
fn loose_sim(content: &Path, slug: &str) -> RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(Vec::new());
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.take_pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    sim
}

/// **The PIE side**: the payload the editor really builds, through
/// `sim_from_payload` — the one PIE boot seam the real `--pie` subprocess takes.
fn pie_sim(proj: &Path) -> RuntimeSim {
    let content = proj.join("Content");
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let doc = inf_editor_core::scene::serialize::load(&content.join(format!("{slug}.inf_lvl")))
        .expect("the island level loads");

    let terrain = std::fs::read(content.join(format!("{slug}.inf_terrain")))
        .expect("the built terrain is on disk");
    let biomes = std::fs::read(content.join(format!("{slug}.inf_biomes")))
        .expect("the built biome set is on disk");
    let pcg = std::fs::read(content.join(format!("{slug}Cover.inf_pcg")))
        .expect("the cover graph is on disk");
    let mesh = std::fs::read(content.join(format!("{slug}Roads.inf_mesh")))
        .expect("the road mesh is on disk");

    let t_guid = inf_island::terrain_guid(&recipe.name);
    let b_guid = inf_island::biome_set_guid(&recipe.name);
    let p_guid = inf_island::cover_pcg_guid(&recipe.name);
    let m_guid = inf_island::road_mesh_guid(&recipe.name);

    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        // resolve (blueprint class), pcg, anim, biome_set, voxel, terrain, mesh,
        // bytes — in that order. Named here because eight closures of the same
        // shape are eight chances to mis-order them, and the first draft did:
        // it put the terrain where the biome set goes and the payload came back
        // with `0 terrain(s)`, which the non-vacuity assertion below caught.
        |_| None,
        |g| (g == p_guid).then(|| pcg.clone()),
        |_| None,
        |g| (g == b_guid).then(|| biomes.clone()),
        |_| None,
        |g| (g == t_guid).then(|| terrain.clone()),
        |g| (g == m_guid).then(|| mesh.clone()),
        |_| None,
        HZ as u32,
        false,
    )
    .expect("the payload builds");

    // **Non-vacuity at the payload.** A payload carrying no terrain would boot a
    // world with no ground, and two hosts with no ground agree perfectly.
    println!(
        "PAYLOAD: {} terrain(s), {} biome set(s), {} pcg(s), {} mesh(es)",
        payload.terrains.len(),
        payload.biome_sets.len(),
        payload.pcgs.len(),
        payload.meshes.len()
    );
    assert_eq!(payload.terrains.len(), 1, "the terrain must ride the wire");
    assert_eq!(
        payload.biome_sets.len(),
        1,
        "the palette must ride the wire"
    );
    assert_eq!(payload.pcgs.len(), 1, "the cover graph must ride the wire");

    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}

/// The drive: a straight run east, sampled every step.
///
/// Deterministic and positional — a *place*, not a time, which is P29's own
/// lesson. Every step the streaming source is moved and the sim advanced, and
/// the sim's own residency sync is what pages the ground.
fn drive(sim: &mut RuntimeSim, from: glam::DVec3) -> Vec<Vec<u8>> {
    let hero = hero_entity(sim).expect("the island has a player-controlled hero");
    let mut trace = Vec::with_capacity(STEPS as usize);
    for step in 0..STEPS {
        let p = glam::DVec3::new(from.x + step as f64 * STEP_M, from.y, from.z);
        set_hero(sim, hero, p);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        trace.push(sim.state_bytes());
    }
    trace
}

fn hero_entity(sim: &RuntimeSim) -> Option<inf_ecs::Entity> {
    let world = sim.world().world();
    let mut found = None;
    for e in world.iter_entities() {
        if e.get::<inf_ecs::components::CharacterMovement>()
            .is_some_and(|m| m.player_controlled)
        {
            found = Some(e.id());
        }
    }
    found
}

fn set_hero(sim: &mut RuntimeSim, e: inf_ecs::Entity, p: glam::DVec3) {
    if let Some(mut t) = sim
        .world_mut()
        .world_mut()
        .get_mut::<inf_ecs::components::Transform>(e)
    {
        t.translation = inf_ecs::math::Vec3d::new(p.x, p.y, p.z);
    }
}

/// Where the drive starts: the design's own player start, lifted clear.
fn start() -> glam::DVec3 {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let s = design.start(inf_editor_core::island::START_LIFT_M);
    glam::DVec3::new(s.x, s.y + 2.0, s.z)
}

/// **THE HEADLINE.** The same drive, byte for byte, on both hosts.
///
/// Un-fix mutations this is armed against: a terrain streamer that read a camera
/// rather than a streaming source; a cell activation keyed on anything but sim
/// state; a biome-bound population evaluated differently by the two boot paths.
#[test]
fn pie_equals_shipping_on_an_island_drive() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let proj = tmp.path().join("island");

    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let from = start();
    let mut ship = pack_sim(&pack);
    let mut pie = loose_sim(&proj.join("Content"), &slug);

    // **Coverage first**, so two identical empty worlds cannot agree their way
    // through: both hosts must have the ground, the water and the population.
    for (who, sim) in [("shipping", &ship), ("pie", &pie)] {
        let world = sim.world().world();
        let terrains = world
            .iter_entities()
            .filter(|e| e.get::<inf_ecs::components::Terrain>().is_some())
            .count();
        let waters = world
            .iter_entities()
            .filter(|e| e.get::<inf_ecs::components::WaterBody>().is_some())
            .count();
        let heroes = world
            .iter_entities()
            .filter(|e| {
                e.get::<inf_ecs::components::CharacterMovement>()
                    .is_some_and(|m| m.player_controlled)
            })
            .count();
        println!("{who}: {terrains} terrain(s), {waters} water bod(ies), {heroes} hero(es)");
        assert_eq!(terrains, 1, "{who} has no ground");
        assert!(waters >= 2, "{who} has {waters} water bodies");
        assert_eq!(heroes, 1, "{who} has no player");
    }

    let a = drive(&mut ship, from);
    let b = drive(&mut pie, from);
    assert_eq!(a.len(), STEPS as usize);
    assert_eq!(b.len(), STEPS as usize);

    // …and the trace is not a constant, or the comparison below is between two
    // recordings of nothing happening.
    let distinct: std::collections::BTreeSet<&Vec<u8>> = a.iter().collect();
    println!(
        "DRIVE: {} steps of {STEP_M} m = {:.0} m, {} distinct states, {} bytes a state",
        STEPS,
        STEPS as f64 * STEP_M,
        distinct.len(),
        a[0].len()
    );
    assert!(
        distinct.len() > STEPS as usize / 2,
        "only {} of {STEPS} states differ — the drive is not moving the world",
        distinct.len()
    );

    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping diverged at step {i} of {STEPS} — the island's \
             streaming or its population is a function of something other than \
             sim state"
        );
    }
    println!("PIE == SHIPPING over {STEPS} steps of an island drive");
}

/// **A `ScenePayload` CARRIES NO PARTITION**, so a PIE preview of the island
/// runs it whole.
///
/// This is a pre-existing engine property, not a defect this wave introduced,
/// and it is measured here rather than described: the wire has `level_bytes`,
/// classes, pcgs, skeletons, clips, machines, biome sets and voxels — and no
/// `.inf_part`, because the partition is **derived at cook** and a payload is
/// what the editor has *before* a cook.
///
/// The consequence for an author is worth stating plainly: previewing a 51 km²
/// island with `--pie` builds every entity in it at once, where the shipped
/// player streams them. For the *fixture* that is fifteen entities against six;
/// for the island it is every lake, river and site at once. It is why
/// `pie_equals_shipping_on_an_island_drive` compares the loose document against
/// the pack — which is the pair P16.5's own gate compares, for this reason.
#[test]
fn a_scene_payload_carries_no_partition() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let payload_sim = pie_sim(&proj);

    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let streamed = loose_sim(&proj.join("Content"), &slug);

    let count = |sim: &RuntimeSim| sim.world().world().iter_entities().count();
    let (whole, part) = (count(&payload_sim), count(&streamed));
    println!(
        "PIE PAYLOAD: {whole} entities built at once; the streamed reading has \
         {part} resident at step 0"
    );
    assert!(
        whole > part,
        "a payload preview ({whole}) should hold MORE than a streamed world's \
         resident set ({part}) — if they are equal the level is not partitioned \
         and this arm is measuring nothing"
    );
    // …and the level really does ask to be partitioned, or the difference above
    // is about something else.
    let doc = inf_editor_core::scene::serialize::load(
        &proj.join("Content").join(format!("{slug}.inf_lvl")),
    )
    .expect("the island level loads");
    assert!(
        doc.settings().partition.enabled,
        "the island level is not partitioned"
    );
}

/// The cooked island really is an island: the pack carries the terrain, the
/// partition, the palette and the vegetation.
#[test]
fn the_cooked_island_carries_every_half_the_recipe_builds() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let reader = inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
        .expect("the pack reader opens");

    let mut kinds: std::collections::BTreeMap<String, usize> = Default::default();
    for e in reader.index() {
        *kinds.entry(e.kind.slug().to_string()).or_default() += 1;
    }
    println!("PACK: {kinds:?}");
    for want in ["terrain", "level", "biome_set", "pcg", "mesh", "partition"] {
        assert!(
            kinds.contains_key(want),
            "the cooked island has no {want}: {kinds:?}"
        );
    }

    // The terrain in the pack is the one the recipe built, with its pyramid.
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let guid = inf_asset::AssetId(inf_island::terrain_guid(&recipe.name));
    let bytes = reader.read(guid).expect("the terrain is in the pack");
    let asset = inf_terrain::TerrainAssetReader::new(&bytes[..]).expect("it decodes");
    println!(
        "TERRAIN: {} tiles, {} LOD levels, {}² samples @ {} m, origin {:?}",
        asset.tile_count(),
        asset.lod_levels(),
        asset.tile_resolution(),
        asset.meters_per_sample(),
        asset.origin()
    );
    assert_eq!(asset.tile_resolution(), recipe.grid.tile_resolution);
    assert_eq!(asset.meters_per_sample(), recipe.grid.meters_per_sample);
    assert!(
        asset.tile_count() as u64 > recipe.grid.tile_count(),
        "the pyramid is missing: {} tiles for {} level-0 pages",
        asset.tile_count(),
        recipe.grid.tile_count()
    );
    // **The georeferenced origin survives the cook**, which is what makes the
    // terrain land where the survey says it does.
    let anchor = recipe.anchor().expect("the anchor builds");
    assert_eq!(asset.origin().x, anchor.origin_easting_m);
    assert_eq!(asset.origin().z, anchor.origin_northing_m);
}

/// The level carries its geo-anchor through the cook, so the sky knows where on
/// Earth it is.
#[test]
fn the_cooked_level_still_knows_where_on_earth_it_is() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    // Read the cooked LEVEL out of the pack and decode it the way the shipped
    // player's own reader does — the geo-anchor is a file-level settings block,
    // not an entity, so it rides the `.inf_lvl` rather than the built world.
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");
    let bytes = source
        .reader()
        .read(source.root_level())
        .expect("the root level is in the pack");
    let level = inf_scene::RuntimeLevel::decode(&bytes).expect("the cooked level decodes");
    let geo = &level.geo;
    println!(
        "GEO: enabled {} crs {:?} at {:.5} N {:.5} E, convergence {:.4} deg",
        geo.enabled,
        geo.crs,
        geo.origin_latitude_deg,
        geo.origin_longitude_deg,
        geo.grid_convergence_deg
    );
    assert!(geo.enabled, "the cooked island lost its geo-anchor");
    assert_eq!(geo.crs, "EPSG:32610");
    assert!((49.0..50.0).contains(&geo.origin_latitude_deg));
    assert!((-124.0..-122.0).contains(&geo.origin_longitude_deg));
    // The solar place is what the sky reads, and it is the anchor's.
    let (lat, lon) = geo.solar_place().expect("an enabled anchor has a place");
    assert_eq!(lat, geo.origin_latitude_deg);
    assert_eq!(lon, geo.origin_longitude_deg);
}

/// **THE VEGETATION IS BOUND AND IT SCATTERS NOTHING ON A STREAMED ISLAND**, and
/// this arm is the number rather than the sentence.
///
/// # What is wired
///
/// The level names a `.inf_biomes` set; the set binds a `.inf_pcg` on every
/// biome that scatters; `inf_pcg::BiomeBinding::from_set` is the one door both
/// hosts resolve it through. All of that is real and the first half of this arm
/// measures it: paged ground, the binding evaluated, **thousands of instances**.
///
/// # What is missing, exactly
///
/// `evaluate_biome_bindings` evaluates over the terrain's **resident**
/// `data.xz_bounds()`, and a streamed terrain ships no tiles — so on the boot
/// path the bounds are `None` and the population is empty. That is the I4 audit's
/// own carried item (*"a streamed cell evaluates its `PcgVolume` and NOT its
/// biome bindings"*), met at island scale with a figure: **{resident} instances
/// with the ground paged, 0 through the shipped boot.**
///
/// The fix is `cell_stream::reconcile`'s missing biome twin — the mirror of
/// `evaluate_pcg_volumes_in` — and it is a change to both hosts' streaming
/// paths, which is why it is measured here and routed rather than smuggled into
/// a content wave.
#[test]
fn the_biome_binding_scatters_when_its_ground_is_resident_and_not_before() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);

    // The palette really binds a graph — the wire the whole thing hangs on.
    let set_bytes = std::fs::read(content.join(format!("{slug}.inf_biomes")))
        .expect("the biome set is written");
    let set = inf_asset::decode::<inf_terrain::BiomeSet>(&set_bytes).expect("it decodes");
    let bound: Vec<&str> = set
        .biomes
        .iter()
        .filter(|b| b.pcg_graph.is_some())
        .map(|b| b.name.as_str())
        .collect();
    println!("BOUND BIOMES: {bound:?}");
    assert_eq!(
        bound.len(),
        6,
        "every biome but urban binds cover: {bound:?}"
    );
    assert!(
        !bound.contains(&"urban"),
        "urban must stay bare for wave I8"
    );

    let pcg_bytes = std::fs::read(content.join(format!("{slug}Cover.inf_pcg")))
        .expect("the cover graph is written");
    let pcg = inf_pcg::PcgAssetPayload::decode(&pcg_bytes).expect("the cover graph decodes");
    let binding = inf_pcg::BiomeBinding::from_set(&set, inf_pcg::DEFAULT_BIOME_FEATHER, |g| {
        (g == inf_island::cover_pcg_guid(&recipe.name)).then(|| pcg.document.clone())
    });
    assert_eq!(
        binding.graphs().len(),
        6,
        "the binding resolved {:?}",
        binding.graphs().len()
    );

    // ── with the ground RESIDENT ──
    let asset = inf_terrain::read_terrain_asset(&content.join(format!("{slug}.inf_terrain")))
        .expect("the built terrain reads");
    let reader = asset.reader();
    let mut data =
        inf_terrain::TerrainData::new(recipe.grid.tile_resolution, recipe.grid.meters_per_sample);
    let (min, max) = inf_island::IslandGrid::of(&recipe).bounds();
    let report = inf_terrain::residency::page_region(
        &mut data,
        &reader,
        glam::DVec2::new(min.x, min.y),
        glam::DVec2::new(max.x, max.y),
    );
    println!(
        "PAGED: {} tiles loaded, {} missing",
        report.loaded.len(),
        report.missing.len()
    );
    assert!(!data.is_empty(), "the whole fixture terrain pages");

    let fields = inf_pcg::OffsetTerrain::new(&data, glam::DVec3::ZERO);
    let height = inf_pcg::FnHeight::new(|x, z| fields.height_at(x, z));
    let bounds = data.xz_bounds().expect("resident ground has bounds");
    let instances = binding.evaluate(
        &height,
        &fields,
        inf_pcg::Region::from_xz(bounds.0.x, bounds.0.y, bounds.1.x, bounds.1.y),
    );
    println!(
        "VEGETATION: {} instances over {:.3} km2 at {} /m2",
        instances.len(),
        (max.x - min.x) * (max.y - min.y) / 1.0e6,
        inf_editor_core::island::ISLAND_SCATTER_DENSITY
    );
    assert!(
        instances.len() > 500,
        "the binding scattered only {} instances with the ground resident",
        instances.len()
    );
    // Every instance is on the ground and inside the world.
    for i in instances.iter().take(200) {
        assert!(i.pos.x >= min.x - 1.0 && i.pos.x <= max.x + 1.0);
        assert!(i.pos.z >= min.y - 1.0 && i.pos.z <= max.y + 1.0);
        assert!(i.pos.y.is_finite());
    }

    // ── through the SHIPPED BOOT ──
    let pack = cook(tmp.path());
    let ship = pack_sim(&pack);
    let world = ship.world().world();
    let population: usize = world
        .iter_entities()
        .filter_map(|e| e.get::<inf_ecs::components::Terrain>())
        .map(|t| t.biome_population.len())
        .sum();
    println!(
        "SHIPPED BOOT: {population} instances — the terrain streams, so \
         `data.xz_bounds()` is None at load and the binding evaluates over nothing"
    );
    assert_eq!(
        population, 0,
        "the streamed boot scattered {population} instances — if this is non-zero \
         the gap this arm records has been closed, and the arm should become an \
         assertion that it stays closed"
    );
}
