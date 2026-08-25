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
//! **And the same forest** (island wave I7b). `Terrain::biome_population` is
//! `#[serde(skip)]`, so it reaches no state fold and two hosts growing different
//! vegetation would have compared equal at every step for ever. The drive folds
//! it separately, out and back, so the ground — and the vegetation on it —
//! pages in **and** out under the comparison.
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
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    // …**and the TERRAIN streamer**, which `run_headless` attaches on the next
    // line and this gate did not. Without it the island's 4.6 MB of pages never
    // move: the `Terrain` component keeps the empty working set a streamed level
    // ships and every `height_at` in the drive answers off nothing. The gate's
    // own headline says "with the terrain paging under the wheels", and
    // `both_hosts_really_streamed` is what now makes that a measurement.
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(source.clone()),
    );
    sim
}

/// **The editor side**: the loose `.inf_lvl` the author saved, binned by the same
/// Ring-0 function the cook used.
///
/// This is the pair P16.5's own gate compares, and it is the right one for a
/// partitioned level: a `ScenePayload` carries **no partition** (see
/// `a_scene_payload_carries_no_partition`), so the PIE wire is not the editor's
/// authoritative reading of a streamed world — the document is.
///
/// **Built the way `build_world`'s own `--level` arm builds it**, and that is not
/// tidiness: the first draft handed the builder `with_defaults(Vec::new())` and
/// nothing else, so the loose host had no biome sets, no PCG payloads and no
/// terrain resolver where the pack host had all three. Two hosts compared for
/// byte equality must be given the same world to disagree about, or the equality
/// is between one real reading and one impoverished one.
fn loose_sim(content: &Path, slug: &str) -> RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

/// What a host's two streamers actually did: `(cell activations, cell
/// deactivations, cells resident, sim-resident level-0 pages, page loads)`.
///
/// # Why the gate needs this and could not do without it
///
/// **Mutation-measured, and it is the reason this function exists.** Deleting
/// `attach_cell_streaming` from *one* host reds the byte compare — that is the
/// wave's own finding D8. Deleting it from **both** left every arm of this file
/// green: the coverage check still found one terrain, two water bodies and one
/// hero (they are all `AlwaysLoaded`), the trace still had 900 distinct states
/// (the drive moves the hero itself), and two hosts that both refuse to stream
/// agree perfectly. A gate whose subject is streaming has to assert that
/// streaming *happened*, not merely that two readings of it match.
fn streaming_counters(sim: &RuntimeSim) -> (u64, u64, usize, usize, u64) {
    let c = sim.cell_streaming().stats();
    let t = sim.terrain_streaming().stats();
    (
        c.activations,
        // …and the DEactivations, since island wave I7b: the drive turns round,
        // so a cell that streamed in streams back out, and "1 resident at the
        // end" stopped being the reading that says the partition worked.
        c.deactivations,
        c.cells_resident,
        t.sim_resident_level0,
        t.loads,
    )
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

/// One host's reading of the drive: the state fold per step, **and** the
/// vegetation the ground grew under it.
///
/// The two are separate because `Terrain::biome_population` is `#[serde(skip)]`
/// — it is what the projector draws, never what the sim reads back — so it does
/// not reach `state_bytes` and no amount of comparing state folds would notice
/// two hosts growing different forests. It is folded here instead, which is
/// where the claim belongs: **the vegetation is a function of the resident
/// ground, and the resident ground is sim state.**
struct Trace {
    states: Vec<Vec<u8>>,
    /// Per step: a digest of every instance's position bits and kind.
    veg: Vec<u128>,
    /// Per step: how many instances stood.
    veg_len: Vec<usize>,
    /// Per step: how many level-0 tiles the simulation held.
    tiles: Vec<usize>,
}

/// Fold a terrain's population into a comparable digest — **positions, not a
/// count** (the I1 law): two forests of the same size in different places must
/// not compare equal.
fn veg_digest(sim: &RuntimeSim) -> (u128, usize) {
    let mut h = xxhash_rust::xxh3::Xxh3::new();
    let mut n = 0usize;
    let world = sim.world().world();
    let mut per_terrain: Vec<(uuid::Uuid, Vec<u8>)> = Vec::new();
    for e in world.iter_entities() {
        let (Some(g), Some(t)) = (
            e.get::<inf_ecs::Guid>(),
            e.get::<inf_ecs::components::Terrain>(),
        ) else {
            continue;
        };
        let mut bytes = Vec::with_capacity(t.biome_population.len() * 28);
        for i in &t.biome_population {
            bytes.extend_from_slice(&i.position.x.to_bits().to_le_bytes());
            bytes.extend_from_slice(&i.position.y.to_bits().to_le_bytes());
            bytes.extend_from_slice(&i.position.z.to_bits().to_le_bytes());
            bytes.extend_from_slice(&i.kind.to_le_bytes());
        }
        n += t.biome_population.len();
        per_terrain.push((g.0, bytes));
    }
    per_terrain.sort_by_key(|(g, _)| *g);
    for (g, bytes) in per_terrain {
        h.update(g.as_bytes());
        h.update(&bytes);
    }
    (h.digest128(), n)
}

/// How many level-0 terrain tiles the **simulation** holds right now.
fn sim_tiles(sim: &RuntimeSim) -> usize {
    sim.world()
        .world()
        .iter_entities()
        .filter_map(|e| e.get::<inf_ecs::components::Terrain>())
        .map(|t| t.data.tile_count())
        .sum()
}

/// The drive: a run east and back again, sampled every step.
///
/// Deterministic and positional — a *place*, not a time, which is P29's own
/// lesson. Every step the streaming source is moved and the sim advanced, and
/// the sim's own residency sync is what pages the ground.
///
/// **It turns round half way** (island wave I7b), and that is not decoration:
/// out and back is what makes the ground page **in and out**, and what makes the
/// second half of the drive re-enter tiles the first half already visited. A
/// population that depended on the order its ground arrived in — P21's
/// first-sight hazard, which the per-tile memo is keyed against — would read
/// differently on the way home.
fn drive(sim: &mut RuntimeSim, from: glam::DVec3) -> Trace {
    let hero = hero_entity(sim).expect("the island has a player-controlled hero");
    let mut t = Trace {
        states: Vec::with_capacity(STEPS as usize),
        veg: Vec::with_capacity(STEPS as usize),
        veg_len: Vec::with_capacity(STEPS as usize),
        tiles: Vec::with_capacity(STEPS as usize),
    };
    for step in 0..STEPS {
        // Out along +x and back — twice the step so the turn is still 360 m out
        // — with a slow drift along +z so **no two steps stand in the same
        // place**. Without the drift the way home would repeat the way out and
        // the "900 distinct states" anti-vacuity arm would be measuring a
        // palindrome rather than a world.
        let out = step.min(STEPS - step);
        let p = glam::DVec3::new(
            from.x + out as f64 * 2.0 * STEP_M,
            from.y,
            from.z + step as f64 * 0.05,
        );
        set_hero(sim, hero, p);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        t.states.push(sim.state_bytes());
        let (d, n) = veg_digest(sim);
        t.veg.push(d);
        t.veg_len.push(n);
        t.tiles.push(sim_tiles(sim));
    }
    t
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

    // What the two had paged before anything moved, so the numbers below are
    // what the DRIVE did rather than what the boot did.
    let (ship0, pie0) = (streaming_counters(&ship), streaming_counters(&pie));

    let a = drive(&mut ship, from);
    let b = drive(&mut pie, from);
    assert_eq!(a.states.len(), STEPS as usize);
    assert_eq!(b.states.len(), STEPS as usize);

    // …and the trace is not a constant, or the comparison below is between two
    // recordings of nothing happening.
    let distinct: std::collections::BTreeSet<&Vec<u8>> = a.states.iter().collect();
    println!(
        "DRIVE: {} steps of {STEP_M} m out and back = {:.0} m, {} distinct \
         states, {} bytes a state",
        STEPS,
        STEPS as f64 * STEP_M,
        distinct.len(),
        a.states[0].len()
    );
    assert!(
        distinct.len() > STEPS as usize / 2,
        "only {} of {STEPS} states differ — the drive is not moving the world",
        distinct.len()
    );

    for (i, (x, y)) in a.states.iter().zip(&b.states).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping diverged at step {i} of {STEPS} — the island's \
             streaming or its population is a function of something other than \
             sim state"
        );
    }
    println!("PIE == SHIPPING over {STEPS} steps of an island drive");

    // **AND THE FOREST AGREES TOO** (island wave I7b). `biome_population` is
    // `#[serde(skip)]`, so it reaches no state fold — two hosts growing
    // different vegetation would have compared equal above, every step, for
    // ever. This is the comparison that says they do not.
    let (vmin, vmax) = (
        *a.veg_len.iter().min().expect("900 steps"),
        *a.veg_len.iter().max().expect("900 steps"),
    );
    let (tmin, tmax) = (
        *a.tiles.iter().min().expect("900 steps"),
        *a.tiles.iter().max().expect("900 steps"),
    );
    let shapes: std::collections::BTreeSet<u128> = a.veg.iter().copied().collect();
    println!(
        "VEGETATION over the drive: {vmin}..{vmax} instances on {tmin}..{tmax} \
         sim tiles, {} distinct forests",
        shapes.len()
    );
    assert!(
        vmin > 0,
        "the drive stood on bare ground at some step — the biome binding is \
         not evaluating over the streamed island"
    );
    assert!(
        tmax > tmin,
        "the simulation held {tmin} terrain tile(s) the whole way, so nothing \
         streamed and this arm cannot see a population stream with it"
    );
    assert!(
        shapes.len() > 1,
        "one forest for the whole drive — the population is not following the \
         ground that pages under it"
    );
    for (i, (x, y)) in a.veg.iter().zip(&b.veg).enumerate() {
        assert_eq!(
            x, y,
            "PIE and shipping grew DIFFERENT vegetation at step {i} of {STEPS} \
             ({} instances against {}) — the biome-bound population is a \
             function of something other than the resident ground",
            a.veg_len[i], b.veg_len[i]
        );
    }
    // …and the ground really paged **out** as well as in, which is the half a
    // one-way drive cannot show: the way home re-enters tiles the way out left.
    let shrank = a.tiles.windows(2).any(|w| w[1] < w[0]);
    let grew = a.tiles.windows(2).any(|w| w[1] > w[0]);
    println!("SIM TILES: grew {grew}, shrank {shrank}");
    assert!(
        grew && shrank,
        "the simulation's tile set only ever {} over the drive — vegetation \
         streaming OUT is not covered by this trace",
        if grew { "grew" } else { "held" }
    );

    // **…AND THE DRIVE REALLY STREAMED**, on both hosts, by the same numbers.
    // See `streaming_counters`: without this the whole file survives having the
    // streamers taken off *both* sides, which is the one mutation the byte
    // compare cannot see.
    let sc = streaming_counters(&ship);
    let pc = streaming_counters(&pie);
    for (who, c) in [("shipping", sc), ("document", pc)] {
        println!(
            "STREAMED {who}: {} cell activation(s), {} deactivation(s), {} \
             cell(s) resident, {} sim L0 page(s), {} page load(s)",
            c.0, c.1, c.2, c.3, c.4
        );
        assert!(
            c.0 > 0 && (c.1 > 0 || c.2 > 0),
            "{who} activated {} cell(s) and deactivated {} over {:.0} m of \
             driving — the partition is not streaming and this gate is \
             comparing two static worlds",
            c.0,
            c.1,
            STEPS as f64 * STEP_M
        );
        assert!(
            c.3 > 0 && c.4 > 0,
            "{who} paged {} terrain tile(s) ({} sim-resident) — the ground is not \
             streaming, so `height_at` answered off an empty working set the \
             whole way",
            c.4,
            c.3
        );
    }
    assert_eq!(
        sc, pc,
        "the two hosts streamed DIFFERENTLY over the same drive — the counters \
         are (cell activations, deactivations, cells resident, sim L0 pages, \
         page loads)"
    );
    // …and the ground paged **under the wheels** rather than only at the boot:
    // the drive itself loaded pages the start position had not asked for.
    println!(
        "PAGED BY THE DRIVE: {} load(s) at the start, {} after {:.0} m out and back",
        ship0.4,
        sc.4,
        STEPS as f64 * STEP_M
    );
    assert!(
        sc.4 > ship0.4 && pc.4 > pie0.4,
        "the drive paged nothing the boot had not already: {} loads at the start, \
         {} at the end. The hero moves {:.0} m across a {}-metre tile span, so a \
         streamer that is working has to fetch something on the way",
        ship0.4,
        sc.4,
        STEPS as f64 * STEP_M,
        recipe.grid.tile_span_m()
    );
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

/// **THE VEGETATION SCATTERS ON THE GROUND THAT IS RESIDENT, AND NOT BEFORE.**
///
/// # What this arm used to say
///
/// Wave I7 measured the gap and asserted it: **4 958 instances with the ground
/// paged by hand and 0 through the shipped boot**, because
/// `evaluate_biome_bindings` ran once at load over `TerrainData::xz_bounds()`
/// and a streamed terrain ships no tiles. Wave I7b closed it — the fixed step
/// refreshes the population from the ground the terrain streamer just paged —
/// so the arm went red as designed and this is its rewrite.
///
/// # What it says now, and why each half is here
///
/// * **not before** — a world built with no terrain streamer attached holds no
///   tiles, so it grows nothing. The population is a function of resident
///   ground and there is none.
/// * **and after** — the shipped boot, with the streamer attached, grows
///   thousands of instances on the pages the hero stands on.
/// * **and it is the SAME forest the author would preview.** Not a count: every
///   instance the streamed world grows is one the fully-paged reading grows, at
///   the same position, and over a tile whose neighbours are all resident the
///   two agree **exactly**. That is the claim "the shipped island grows what
///   the preview shows" reduced to a comparison, and a count could not make it.
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

    // The author's reading: every tile of the island in memory at once, through
    // the same Ring-0 door the shipped step calls tile by tile.
    let instances = binding.evaluate_resident(&data, glam::DVec3::ZERO);
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

    // ── NOT BEFORE: a world with no ground paged ──
    let pack = cook(tmp.path());
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");
    let mut bare = {
        let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
        let partition = built.take_partition();
        let pcg = built.pcg_context();
        let mut sim = inf_player::sim_from_built(built);
        inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
        sim // …and deliberately NO terrain streamer.
    };
    bare.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let (_, bare_pop) = veg_digest(&bare);
    println!(
        "NO GROUND PAGED: {} sim tile(s), {bare_pop} instances",
        sim_tiles(&bare)
    );
    assert_eq!(sim_tiles(&bare), 0, "a streamed level ships no tiles");
    assert_eq!(
        bare_pop, 0,
        "the binding grew {bare_pop} instances over ground that is not there"
    );

    // ── AND AFTER: the shipped boot, streamer attached ──
    let mut ship = pack_sim(&pack);
    ship.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let (_, population) = veg_digest(&ship);
    let tiles = sim_tiles(&ship);
    println!("SHIPPED BOOT: {population} instances on {tiles} sim tile(s)");
    assert!(tiles > 0, "the shipped boot paged no ground");
    assert!(
        population > 500,
        "the streamed boot grew only {population} instances on {tiles} paged \
         tile(s) — the refresh is not reaching the resident ground"
    );

    // ── AND IT IS THE SAME FOREST ──
    let author: std::collections::BTreeSet<(u64, u64, u64)> = instances
        .iter()
        .map(|i| (i.pos.x.to_bits(), i.pos.y.to_bits(), i.pos.z.to_bits()))
        .collect();
    let (shipped, resident): (Vec<_>, Vec<(i32, i32)>) = {
        let w = ship.world().world();
        let t = w
            .iter_entities()
            .find_map(|e| e.get::<inf_ecs::components::Terrain>())
            .expect("the island has ground");
        (
            t.biome_population.clone(),
            t.data.tiles().map(|(&c, _)| c).collect(),
        )
    };
    let stray = shipped
        .iter()
        .filter(|i| {
            !author.contains(&(
                i.position.x.to_bits(),
                i.position.y.to_bits(),
                i.position.z.to_bits(),
            ))
        })
        .count();
    println!(
        "SAME FOREST: {} of {} shipped instances are places the fully-paged \
         reading also grows ({stray} stray)",
        shipped.len() - stray,
        shipped.len()
    );
    assert_eq!(
        stray, 0,
        "the streamed island grew {stray} instance(s) the author's fully-paged \
         reading does not — a streamed forest must be a SUBSET of the whole one, \
         place for place"
    );

    // …and over a tile whose whole neighbourhood is resident, the two are not
    // merely a subset of one another: they are equal. That is the interior of
    // the streamed world reading exactly as the author's does.
    let set: std::collections::BTreeSet<(i32, i32)> = resident.iter().copied().collect();
    let span = recipe.grid.tile_span_m();
    let interior = set
        .iter()
        .copied()
        .find(|c| (-1..=1).all(|dz| (-1..=1).all(|dx| set.contains(&(c.0 + dx, c.1 + dz)))))
        .expect("the sim's resident set has an interior tile");
    let (x0, z0) = (interior.0 as f64 * span, interior.1 as f64 * span);
    let inside = |x: f64, z: f64| (x0..x0 + span).contains(&x) && (z0..z0 + span).contains(&z);
    let mine: std::collections::BTreeSet<(u64, u64, u64)> = shipped
        .iter()
        .filter(|i| inside(i.position.x, i.position.z))
        .map(|i| {
            (
                i.position.x.to_bits(),
                i.position.y.to_bits(),
                i.position.z.to_bits(),
            )
        })
        .collect();
    let theirs: std::collections::BTreeSet<(u64, u64, u64)> = instances
        .iter()
        .filter(|i| inside(i.pos.x, i.pos.z))
        .map(|i| (i.pos.x.to_bits(), i.pos.y.to_bits(), i.pos.z.to_bits()))
        .collect();
    println!(
        "INTERIOR TILE {interior:?}: {} shipped against {} authored",
        mine.len(),
        theirs.len()
    );
    assert!(
        !theirs.is_empty(),
        "the interior tile {interior:?} grows nothing in either reading, so \
         comparing them proves nothing"
    );
    assert_eq!(
        mine, theirs,
        "inside a fully-resident tile the streamed island and the fully-paged \
         reading must place the SAME instances"
    );
}

/// **THE GROUND THE SIMULATION STANDS ON IS THE GROUND THE RECIPE BUILT.**
///
/// # Why this arm exists
///
/// The gate above compares two hosts. Two hosts reading the *same* wrong ground
/// agree perfectly, and for the whole of wave I7 they did: the island's `Terrain`
/// entity carried `Transform::from_translation(grid.bounds().0)` on top of an
/// `.inf_terrain` whose tile indices are **already centred on the world origin**
/// (`IslandGrid::tile0 = -(tiles / 2)`), so the centring was applied twice.
///
/// Measured before the fix, through this same seam: the design's own player start
/// read **0.000 m of unauthored ground where the build puts 129.916 m**, and the
/// world origin read 80.000 m off a page 768 m away. On the shipped island the
/// displacement is 3 584 m on each axis — half the terrain outside the world.
///
/// So the comparison here is host **against the recipe**, not host against host:
/// `RuntimeSim::terrain_height_at` is the exact seam a Blueprint's
/// `terrain.height_at`, the character's ground snap and the physics heightfield
/// all read, and `IslandBuild::terrain` is what `inf island build` wrote.
#[test]
fn the_ground_the_simulation_stands_on_is_the_ground_the_recipe_built() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let mut ship = pack_sim(&pack);

    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let s = start();
    let hero = hero_entity(&ship).expect("a hero");

    // The design's own places: where a player starts, the other settlement, the
    // world origin and a point between them. All four are inside the coastline —
    // a probe on the sea shelf would be a probe on a flat surface, which agrees
    // with itself under any displacement.
    let probes: Vec<(f64, f64)> = build
        .recipe
        .sites
        .iter()
        .map(|q| (q.x, q.z))
        .chain([(0.0, 0.0), (200.0, -200.0)])
        .collect();
    let mut seen: Vec<f64> = Vec::new();
    assert!(probes.len() >= 4, "too few probes to say anything");
    for (x, z) in probes {
        // Stand the streaming source there and let the sim page its own
        // neighbourhood in — residency is derived from sim state, so this is the
        // only honest way to ask.
        set_hero(&mut ship, hero, glam::DVec3::new(x, s.y, z));
        for _ in 0..3 {
            ship.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        let sim_h = ship.terrain_height_at(x, z);
        let built = build
            .terrain
            .height_at(glam::DVec2::new(x, z))
            .unwrap_or_else(|| panic!("({x}, {z}) is off the built terrain"));
        println!("GROUND ({x:>7.1}, {z:>7.1}): sim {sim_h:9.3} m, recipe {built:9.3} m");
        assert!(
            (sim_h - built).abs() < 1.0e-6,
            "the simulation stands at {sim_h} m where the recipe built {built} m at \
             ({x}, {z}) — the terrain entity and the .inf_terrain disagree about \
             where the world is"
        );
        seen.push(built);
    }
    // Anti-vacuity: four probes that all read the same number would agree under
    // any offset at all, and so would four probes on flat sea.
    let lo = seen.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = seen.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        hi - lo > 20.0,
        "the four probes span only {:.3} m of relief, so a displaced terrain \
         could still match them",
        hi - lo
    );
    assert!(
        lo > recipe.sea.level_m,
        "a probe at {lo} m is under the {} m waterline — this arm must stand on land",
        recipe.sea.level_m
    );
}
