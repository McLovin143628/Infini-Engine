//! **The island, driven — PIE == shipping over 51 km² of real ground** (wave I7).
//!
//! # What this gate is and what it is not
//!
//! It runs the **CI-scale island**, because the shipped one's terrain is 549.9 MB
//! and is not committed (342.7 MB was wave I7's figure; wave TER2b's detail band
//! moved it, and the I8a audit re-measured it off `build.report.summary()`).
//! Everything else is the shipped path: the same recipe
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

use uuid::Uuid;

use inf_player::budget::{CITY_STEP_BUDGET_MS, LOAD_BUDGET_MS};
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
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    // **The hero's rig, its machine and its clips** (SK1c). This function's own
    // doc already carried the rule -- *two hosts compared for byte equality must
    // be given the same world to disagree about, or the equality is between one
    // real reading and one impoverished one* -- and the anim index was the third
    // thing it was missing, invisible for as long as the island's hero was a
    // capsule with `AnimStateMachine { sm: None }` and nothing to pose. The pack
    // host gets these from the cook's own index; this one reads the same content
    // root the recipe's `[content]` list filled.
    .with_anim_assets(skeletons, clips, machines)
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

/// **Every asset in a built project's content root, by GUID.**
///
/// The sidecars are the index — a `.toml` beside every payload naming its GUID —
/// which is exactly what `AssetDb`'s own scan reads. A name table here would be
/// a second place the starter character's identity is written down, and the two
/// would disagree the first time a file was renamed.
fn content_assets(content: &Path) -> std::collections::BTreeMap<Uuid, PathBuf> {
    let mut out = std::collections::BTreeMap::new();
    let Ok(dir) = std::fs::read_dir(content) else {
        return out;
    };
    for entry in dir.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(table) = text.parse::<toml::Table>() else {
            continue;
        };
        let Some(guid) = table
            .get("guid")
            .and_then(|v| v.as_str())
            .and_then(|s| Uuid::parse_str(s).ok())
        else {
            continue;
        };
        // `Foo.inf_skel.toml` -> `Foo.inf_skel`, which is the payload it indexes.
        let payload = path.with_extension("");
        if payload.exists() {
            out.insert(guid, payload);
        }
    }
    out
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

    // **The character's assets** (SK1c). The hero is `samples/starter-character`
    // now, so the payload has to carry the rig, the machine and the clips or a
    // `--pie` preview poses nothing while the shipped build poses 161 bones.
    //
    // **Nothing drives this host** (SK1c audit, M1). The first draft of this
    // comment said "the state comparison below said so at step 0"; it did not
    // and could not — that comparison is `pack_sim` against `loose_sim`, and the
    // sim this function returns is only ever *counted*. The counts below are the
    // whole of this seam's cover, which is why the class is counted with the
    // rest.
    //
    // Read off the SIDECARS rather than from a hard-coded name table, which is
    // how the editor's own asset database finds them: the recipe's `[content]`
    // list copies the whole character into `Content/`, and every payload there
    // carries a `.toml` naming its GUID. A table of seven file names here would
    // be a second place the character's identity is written down.
    let assets = content_assets(&content);
    let read_asset = |g: Uuid| assets.get(&g).map(|p| std::fs::read(p).expect("an asset"));

    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        // resolve (blueprint class), pcg, anim, biome_set, voxel, terrain, mesh,
        // bytes — in that order. Named here because eight closures of the same
        // shape are eight chances to mis-order them, and the first draft did:
        // it put the terrain where the biome set goes and the payload came back
        // with `0 terrain(s)`, which the non-vacuity assertion below caught.
        |g| {
            read_asset(g)
                .and_then(|b| serde_json::from_slice::<inf_blueprint::BlueprintClass>(&b).ok())
        },
        |g| (g == p_guid).then(|| pcg.clone()),
        read_asset,
        |g| (g == b_guid).then(|| biomes.clone()),
        |_| None,
        |g| (g == t_guid).then(|| terrain.clone()),
        |g| {
            if g == m_guid {
                Some(mesh.clone())
            } else {
                read_asset(g)
            }
        },
        read_asset,
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
    // **And the hero's rig rides it too** (SK1c). Without this the PIE side
    // publishes no pose at all where the shipping side publishes 6 476 bytes of
    // one. One skeleton, one machine, the machine's three clips reached through
    // the transitive hop, **and the controller class**.
    //
    // **These four assertions are the ONLY cover this payload's character has**
    // (SK1c audit, M1). The first draft of this comment said the step-0 state
    // comparison would notice — it would not, and cannot: the drive gate builds
    // its two hosts from `pack_sim` and `loose_sim`, and `pie_sim` is used here
    // and nowhere else, which the arm below this function says in as many words.
    // A payload seam nothing drives is a payload seam whose only witness is what
    // is counted right here, so the class is counted too: reverting the
    // blueprint resolver alone left all six arms green.
    println!(
        "PAYLOAD CHARACTER: {} skeleton(s), {} machine(s), {} clip(s), {} class(es)",
        payload.skeletons.len(),
        payload.machines.len(),
        payload.clips.len(),
        payload.classes.len()
    );
    assert_eq!(
        payload.skeletons.len(),
        1,
        "the hero's rig must ride the wire"
    );
    assert_eq!(
        payload.machines.len(),
        1,
        "the hero's machine must ride the wire"
    );
    assert_eq!(
        payload.clips.len(),
        3,
        "the machine's clips must ride the wire, or PIE poses every state at rest"
    );
    assert_eq!(
        payload.classes.len(),
        1,
        "the hero's controller class must ride the wire — it is the `.inf_act` \
         the recipe's `[content]` list copies and the one thing in the character \
         that `level_dependencies` does NOT reach, so nothing else would notice"
    );

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
            // Wave TER2b: and the MESH, which is what the instance actually
            // draws. Two populations that agree on every position and differ on
            // which prop stands there are two different worlds, and before this
            // line the fold could not tell them apart.
            bytes.extend_from_slice(&i.mesh.map_or(0u128, |m| m.as_u128()).to_le_bytes());
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

    // **AND THE HERO IS A CHARACTER** (SK1c). The comparison above is a byte
    // equality, and a byte equality is blind to two hosts posing NOTHING
    // identically — which is precisely what this gate did for its whole life
    // before this wave, because the island's hero was a capsule carrying
    // `AnimStateMachine { sm: None }` and no `SkeletalMesh`.
    //
    // So the pose section is measured rather than inferred. **6 476 bytes** is
    // SK1a's arithmetic for a 161-bone rig — a 36-byte header (the entity's GUID,
    // its skeleton's GUID and a joint count) plus 40 bytes a joint — and it is
    // pinned as the number rather than as `> 0` for the reason SK1b's grip gate
    // pins the same one: a rig that silently lost its side tables, or a hero that
    // quietly went back to being a capsule, would still be "greater than zero"
    // on one host and equal on both.
    //
    // It is also the whole of this wave's cost on this trace: the drive went from
    // 403 bytes a state to 6 879.
    const POSED_BYTES: usize = 36 + 161 * 40;
    for (who, sim) in [("shipping", &mut ship), ("pie", &mut pie)] {
        let bytes = inf_ecs::pose::pose_state_bytes(sim.world());
        assert_eq!(
            bytes.len(),
            POSED_BYTES,
            "{who} published {} bytes of pose, not a 161-bone character's {POSED_BYTES} \
             — the island's hero has stopped being the starter character",
            bytes.len()
        );
    }
    println!(
        "POSE: {POSED_BYTES} B a step on both hosts (403 B a state before the hero \
         was a character, {} now)",
        a.states[0].len()
    );

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

    // **AND THE DRIVE IS THROUGH A SETTLEMENT NOW** (island wave I8a), which is
    // what re-prices this trace: the design's start is the first site's own
    // centre (`player_start` reads the committed road layer, and the routes run
    // centre to centre), so the 900 steps leave a settlement, cross its edge and
    // come back. Stated with the numbers rather than left as a change in what
    // the world holds.
    let solids = resident_solids(&ship);
    let doorways = inf_ecs::door::volume_doorways(ship.world());
    let volumes = resident_volumes(&ship);
    println!(
        "SETTLEMENT ON THE DRIVE: {} resident volume(s), {} solids, {} doorways \
         after {:.0} m out and back from {}",
        volumes.len(),
        solids.len(),
        doorways.len(),
        STEPS as f64 * STEP_M,
        recipe
            .sites
            .first()
            .map(|s| s.name.as_str())
            .unwrap_or("the start")
    );
    assert!(
        !volumes.is_empty() && !solids.is_empty() && !doorways.is_empty(),
        "the 900-step drive ended holding no settlement at all — this trace is \
         no longer over a world with a city in it"
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

/// **The settlements stand on the ground urban reserves** (island wave I8a) —
/// the flipped half of the tripwire above.
///
/// Three claims, all of them about the built project rather than about the
/// generator that wrote it:
///
/// * every block a settlement plans sits inside its own site's reservation
///   circle, so it is on ground the carve levelled and the biome map painted
///   urban — which is what makes the vegetation and the buildings disjoint by
///   construction rather than by luck;
/// * every zone document a block names is **in the project**, resolvable by
///   GUID out of the content root the recipe's `[content]` list filled. A
///   `PcgVolume` whose graph does not resolve evaluates to nothing and says
///   nothing, which is the failure this catches;
/// * a settlement that plans no block at all is named, not skipped.
fn the_settlements_stand_where_urban_is_reserved(
    content: &Path,
    recipe: &inf_island::IslandRecipe,
) {
    let design = inf_island::read_design(recipe).expect("the committed design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let assets = content_assets(content);
    let mut blocks = 0usize;
    for p in &plans {
        let site = &recipe.sites[p.site];
        assert!(site.kind.reserves_urban());
        println!(
            "SETTLEMENT {} ({}): {} blocks inside a {:.0} m reservation, {} refused \
             off-pad, {} refused off-land",
            p.name,
            p.kind.label(),
            p.blocks.len(),
            p.radius_m,
            p.refused_off_pad,
            p.refused_off_land
        );
        for b in &p.blocks {
            for c in b.corners() {
                assert!(
                    (c - p.centre).length() <= site.radius_m,
                    "{}'s block {:?} reaches outside the reservation the biome map \
                     paints urban — its buildings would stand in a forest",
                    p.name,
                    (b.col, b.row)
                );
            }
            let g = inf_editor_core::settlement::zone_guid(b.archetype);
            assert!(
                assets.contains_key(&g),
                "{}'s {} block names zone document {g}, which is not in the built \
                 project — the volume would evaluate to nothing, silently",
                p.name,
                b.archetype.name()
            );
        }
        blocks += p.blocks.len();
    }
    println!(
        "SETTLEMENTS: {blocks} blocks over {} settlements, {} distinct zone \
         documents in the project",
        plans.len(),
        inf_pcg::ArchetypeId::ALL
            .iter()
            .filter(|a| assets.contains_key(&inf_editor_core::settlement::zone_guid(**a)))
            .count()
    );
    assert!(
        blocks > 0,
        "the committed design plans no settlement block at all"
    );
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
    // **THE TRIPWIRE FLIPPED, AND THE SENTENCE IT CARRIED IS SPENT** (island
    // wave I8a). This read *"urban must stay bare for wave I8"* — a wave that
    // has now happened. What is still true is that urban binds no VEGETATION
    // graph, and the reason is no longer "nobody has built the settlements yet":
    // it is that a settlement is a `PcgVolume` in the LEVEL, not a biome
    // binding, so the two authorities never meet. What urban reserving the
    // ground buys is exactly what wave I7 said it would — the settlement
    // generator finds bare ground rather than a forest to clear.
    //
    // So the arm asserts the settlements instead, on the world: the level
    // carries one volume per block, every one of them names a committed zone
    // document, and every one of them sits inside a site's own reservation.
    assert!(
        !bound.contains(&"urban"),
        "urban binds a cover graph — the settlements stand on reserved ground \
         and the vegetation must not grow through them"
    );
    the_settlements_stand_where_urban_is_reserved(&content, &recipe);

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
    //
    // **Two controls since wave I8a, and the second one is why.** This used to be
    // one: cell streaming attached, terrain streaming deliberately not, asserting
    // zero tiles and zero instances. With the settlements standing that is no
    // longer true and the reason is a *feature* — IB-1's rule that **PCG pages
    // its own ground**. A settlement block is a `PcgVolume` with
    // `Ground::Terrain`, so activating its cell runs `page_terrains_for_pcg`
    // before evaluating it, and the terrain the level shipped empty now holds the
    // page that volume needed. The vegetation then grows on it, correctly.
    //
    // So the true zero moves to a world with **neither** streamer (no cells, no
    // volumes, no pre-pass, no ground), and the cell-only world becomes what it
    // actually is: a strict, tiny subset of the shipped reading.
    let pack = cook(tmp.path());
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");
    let mut nothing = {
        let built = inf_player::build_world_from_pack(&source).expect("the world builds");
        inf_player::sim_from_built(built)
    };
    nothing.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let (_, nothing_pop) = veg_digest(&nothing);
    println!(
        "NEITHER STREAMER: {} sim tile(s), {nothing_pop} instances",
        sim_tiles(&nothing)
    );
    assert_eq!(sim_tiles(&nothing), 0, "a streamed level ships no tiles");
    assert_eq!(
        nothing_pop, 0,
        "the binding grew {nothing_pop} instances over ground that is not there"
    );

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
    let bare_tiles = sim_tiles(&bare);
    println!(
        "CELLS BUT NO TERRAIN STREAMER: {bare_tiles} sim tile(s) paged by the \
         settlements' own PCG pre-pass, {bare_pop} instances"
    );
    assert!(
        bare_tiles > 0 && bare_pop > 0,
        "the settlement volumes paged no ground of their own — IB-1's pre-pass \
         is not running, and a building with `Ground::Terrain` over an unpaged \
         page fails closed and builds nothing"
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
    // …and the terrain streamer is what did it: the settlements' own pre-pass
    // pages a page or two, the streamer pages the neighbourhood.
    assert!(
        tiles > bare_tiles && population > bare_pop,
        "the terrain streamer added nothing the settlements' PCG pre-pass had \
         not already paged ({tiles} against {bare_tiles} tiles, {population} \
         against {bare_pop} instances)"
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

/// **The island's ground is four real materials, end to end** (wave TER2a).
///
/// Before this wave the island's four `TerrainLayer`s named no material, so the
/// terrain shader's whole per-layer virtual-texture branch was unreachable and
/// the frame reported *zero* virtual textures over 51 km² of ground. Binding
/// them is not one edit: a material GUID on a layer has to survive the level
/// save, the cook's dependency closure, the pack, and the shipped player's
/// binding walk — four separate places that each had a rule for
/// `Material.asset` and none for a terrain layer.
///
/// So this arm walks the whole chain on the real cooked pack and names which
/// link is broken when it breaks, rather than asserting a count at the end and
/// leaving the reader to bisect four crates.
#[test]
fn the_cooked_island_carries_the_ground_its_layers_bind() {
    let tmp = tempfile::tempdir().expect("tmp");
    let pack = cook(tmp.path());
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");

    // 1. THE LEVEL. Its four layers name four DISTINCT materials, in the splat
    //    order `inf_island::splat` writes weights in.
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let bound: Vec<uuid::Uuid> = {
        let world = built.world.world_mut();
        let mut q = world.query::<&inf_ecs::components::Terrain>();
        let ids: Vec<uuid::Uuid> = q
            .iter(world)
            .flat_map(|t| t.layer_materials().collect::<Vec<_>>())
            .collect();
        ids
    };
    assert_eq!(
        bound.len(),
        4,
        "the cooked island's terrain binds {} layer materials, not four: {bound:?}",
        bound.len()
    );
    let mut distinct = bound.clone();
    distinct.sort();
    distinct.dedup();
    assert_eq!(distinct.len(), 4, "two layers share a material: {bound:?}");
    for (k, kind) in [
        inf_material::ground::GroundKind::Grass,
        inf_material::ground::GroundKind::Rock,
        inf_material::ground::GroundKind::ForestFloor,
        inf_material::ground::GroundKind::Sand,
    ]
    .into_iter()
    .enumerate()
    {
        assert_eq!(
            bound[k],
            inf_editor_core::ground::ground_material_guid(kind),
            "layer {k} is not {} — the splat writes its weights into that \
             channel, so a swap here paints the beaches with rock",
            kind.label()
        );
    }

    // 1b. **AND THE LEVEL ITSELF CARRIES NONE OF IT** — the finding this arm
    //     exists to keep found. A cooked partitioned level ships **zero
    //     entities**; they are all in the derived `.inf_part`. So the walk that
    //     collects a level's material bindings found nothing at all on every
    //     partitioned world, for `Material.asset` as much as for a terrain
    //     layer, and every surface in one shipped untextured. Nothing caught it
    //     because until TER2a no content in this repository had a texture to
    //     lose: the island reported "0 virtual textures" over 51 km² of ground,
    //     which read as "the ground names no material" and was also true.
    //
    //     Asserted rather than commented, so the day the cook stops emptying a
    //     partitioned level this arm says so instead of silently measuring a
    //     path that no longer exists.
    {
        let raw = inf_player::level::LevelSource::level_bytes(&source).expect("level bytes");
        let lvl = inf_scene::decode(&raw).expect("level decodes");
        assert!(
            lvl.entities.is_empty(),
            "the cooked island level carries {} entities — if a partitioned \
             level now keeps them, `material_content`'s partition walk is \
             double-counting and this arm's premise has changed",
            lvl.entities.len()
        );
        println!(
            "ISLAND GROUND: the cooked level carries 0 entities (they are in the \
             .inf_part), so every binding below came from the partition walk"
        );
    }

    // 2. THE PACK. The cook's closure followed the level's sidecar edge to each
    //    `.inf_mat`, derived a `.inf_matd` for it, and followed THAT to the
    //    `.inf_tex` containers.
    let content = source.material_content();
    assert_eq!(
        content.materials.len(),
        4,
        "the pack carries {} derived ground records, not four — the cook's \
         closure did not follow `Terrain.layers[*].material`",
        content.materials.len()
    );
    // Fourteen: four albedo + four normal + four ORM + two detail (grass and
    // rock are the only sets that ship one).
    assert_eq!(
        content.textures.len(),
        14,
        "the pack carries {} ground textures, not fourteen",
        content.textures.len()
    );

    // 3. THE RESIDENCY. The registration set a shipped host builds from this
    //    content, and the deterministic floor it admits — the numbers the frame
    //    instrument's "N virtual textures" line reports.
    let mats = content.vt_materials();
    assert_eq!(mats.len(), 4, "the host's material map is not the pack's");
    let order = inf_render::registration_order(&mats);
    assert_eq!(
        order.len(),
        14,
        "the registration order names {} textures, not fourteen — `want_floor` \
         is a pure function of this sequence, so it is the thing two hosts have \
         to agree about",
        order.len()
    );
    println!(
        "ISLAND GROUND: 4 layers -> 4 materials -> {} textures; registration \
         order {:?}",
        content.textures.len(),
        order
            .iter()
            .map(|g| format!("{:x}", *g as u64))
            .collect::<Vec<_>>()
    );

    // 3b. **THE GROUND COVER, AS FAR AS THE PACK** (clause 5). The island's three
    //     scatter kinds bind real meshes now — all three carried `mesh: None` —
    //     and the cook had no `.inf_pcg` -> scatter-mesh edge, so this is the arm
    //     that says both halves landed.
    //
    //     **It ends at the pack, and the TER2a audit made that explicit.** What
    //     this block proves is that the bytes are cooked and reachable — the
    //     fourth link of a five-link chain, and never the fifth. Wave TER2b
    //     closed the fifth: `the_scattered_cover_draws_its_authored_meshes`
    //     below drives the shipped projector over the real population and
    //     asserts the three meshes reach three geometry uploads. This block
    //     stays because a mesh that never reaches the pack cannot be drawn
    //     whatever the projector does.
    {
        let reader = std::sync::Arc::new(
            inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
                .expect("the pack maps"),
        );
        for kind in inf_editor_core::cover::CoverKind::ALL {
            let id = inf_asset::AssetId(inf_editor_core::cover::cover_mesh_guid(kind));
            assert!(
                reader.contains(id),
                "the cooked island scatters {} and the pack does not carry its \
                 mesh -- the `.inf_pcg` -> scatter-kind-mesh edge did not close",
                kind.label()
            );
            let bytes = reader.read(id).expect("the mesh reads");
            let mesh: inf_mesh::MeshAsset =
                inf_asset::decode(&bytes).expect("the cover mesh decodes");
            let tris: usize = mesh.submeshes.iter().map(|s| s.triangle_count()).sum();
            assert!(tris > 0, "{} ships an empty mesh", kind.label());
            println!(
                "ISLAND COVER: {} -> {tris} triangles, {:.3} m tall",
                kind.label(),
                mesh.bounds.max[1] - mesh.bounds.min[1]
            );
        }
    }

    // 4. AND IT IS NOT VACUOUS. Every texture the records name really is in the
    //    pack — a record naming bytes that are absent renders untextured and the
    //    counts above would still be four and fourteen.
    for g in &order {
        assert!(
            content.textures.contains_key(&uuid::Uuid::from_u128(*g)),
            "the pack names texture {g:x} and does not carry it"
        );
    }
}

/// **The scattered cover draws its authored meshes** (wave TER2b).
///
/// # What this arm replaces
///
/// `the_cover_meshes_are_shipped_and_are_not_yet_drawn` — a TER2a-audit tripwire
/// written to assert the WRONG outcome, so that the day a real mesh reached the
/// scatter path it would go red and take the ledger with it. It has now gone red,
/// and it did so in the strongest form it could: its two struct literals stopped
/// **compiling**, because `inf_pcg::PcgInstance` and
/// `inf_ecs::components::ScatteredInstance` each grew the `mesh` field the arm
/// existed to say did not exist.
///
/// What was wrong, and is not any more:
///
/// * `PcgKind::mesh` was read by the packager's dependency closure and by nothing
///   that draws. `rules::evaluate_with_in` now resolves it where the rule that
///   owns the palette is still in hand, and it rides on the instance. It has to
///   be the GUID and not an index: `kind_index` is **rule-local**, populations
///   from every rule of every layer of every biome graph are concatenated with no
///   run boundaries, and `compose_volume` interleaves grammar module indices into
///   the same `u32`.
/// * `push_scatter` built one `ScatterData::build(PrimMesh::Cube, …)` for every
///   instance. It buckets by mesh now, and builds one batch per authored mesh.
/// * `inf_render::ScatterBatch` had nowhere to put geometry. `ScatterData` now
///   carries an `Option<Arc<ScatterGeometry>>`, folded into its content key, and
///   the scatter raster pulls that batch's own vertices and indices out of two
///   storage buffers instead of the shared built-in pack.
///
/// # It drives the SHIPPED door on the REAL island
///
/// The arm it replaces asserted pack membership, which is the fourth link of a
/// five-link chain — the TER2a audit's law: *when a clause's title is about what
/// the world looks like, at least one arm has to be about what the world looks
/// like.* So this one cooks the island, boots the shipping sim, drives it until
/// the ground is resident and the population has scattered, and projects it
/// through `project_scene_full` — the very function the windowed player calls —
/// with the very table `load_scatter_meshes` builds at boot.
///
/// Its anti-vacuity half is the **same projection with an empty table**: that is
/// the pre-TER2b engine exactly, and it must produce one meshless batch where the
/// real one produces three mesh-carrying ones. An arm that could not tell those
/// two apart would be measuring nothing.
#[test]
fn the_scattered_cover_draws_its_authored_meshes() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let pack = cook(tmp.path());
    let reader = inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
        .expect("the pack maps");

    // ── 1. the table the shipped player's projector is handed ──
    //
    // Through the SHIPPED door — `inf_player::scatter_mesh::from_pack` is what
    // `load_scatter_meshes` calls at boot — so this is a real run's table, not
    // one the test assembled.
    let meshes = inf_player::scatter_mesh::from_pack(&reader);
    assert_eq!(
        meshes.len(),
        inf_editor_core::cover::CoverKind::ALL.len(),
        "the cooked island's scatter kinds resolve to {} meshes, not the three the \
         cover library authors",
        meshes.len()
    );
    for kind in inf_editor_core::cover::CoverKind::ALL {
        let id = inf_editor_core::cover::cover_mesh_guid(kind);
        let g = meshes
            .get(&id.as_u128())
            .unwrap_or_else(|| panic!("{} did not resolve to scatter geometry", kind.label()));
        assert!(
            g.triangle_count() > 0 && g.vertex_count() > 0,
            "{} resolved to an empty mesh",
            kind.label()
        );
        assert!(
            g.radius > 0.0 && g.radius < 2.0,
            "{} has a unit bounding radius of {} m, which is not ground cover -- \
             the cull sphere and the impostor card are both sized from it",
            kind.label(),
            g.radius
        );
        // NOT VACUOUS: the resolved geometry is the committed mesh's, through the
        // same one door, and not something the loader synthesized.
        let bytes = reader.read(inf_asset::AssetId(id)).expect("the mesh reads");
        let mesh: inf_mesh::MeshAsset = inf_asset::decode(&bytes).expect("the cover mesh decodes");
        let (p, n, _u, _t, i) = mesh.vgeom_streams();
        assert_eq!(
            g.key(),
            inf_render::ScatterGeometry::from_streams(&p, &n, &i).key(),
            "{}'s resolved geometry is not the committed mesh's",
            kind.label()
        );
        println!(
            "ISLAND COVER: {} -> {} triangles, {} vertices, r = {:.3} m",
            kind.label(),
            g.triangle_count(),
            g.vertex_count(),
            g.radius
        );
    }

    // ── 2. the island's own document names them ──
    let doc = inf_editor_core::island::island_cover_document(7);
    let kinds: Vec<&inf_pcg::PcgKind> = doc
        .layers
        .iter()
        .flat_map(|l| &l.rules)
        .flat_map(|r| &r.kinds)
        .collect();
    assert_eq!(kinds.len(), 3, "the island's cover document is three kinds");
    for (k, cover) in kinds.iter().zip(inf_editor_core::cover::CoverKind::ALL) {
        assert_eq!(
            k.mesh,
            Some(inf_editor_core::cover::cover_mesh_guid(cover)),
            "the island's {} kind names no mesh",
            cover.label()
        );
    }

    // ── 3. the shipping projection, on the real population ──
    let mut sim = pack_sim(&pack);
    let t = drive(&mut sim, start());
    let placed = *t.veg_len.last().expect("the drive traced");
    assert!(
        placed > 0,
        "nothing scattered on the drive, so the projection below would be vacuous"
    );
    // Every instance the sim placed carries the GUID its kind resolved to — the
    // link the audit found broken, asserted on the WORLD rather than on a
    // constructed record.
    {
        let world = sim.world().world();
        let mut named = 0usize;
        let mut total = 0usize;
        for e in world.iter_entities() {
            let Some(terrain) = e.get::<inf_ecs::components::Terrain>() else {
                continue;
            };
            for i in &terrain.biome_population {
                total += 1;
                if i.mesh.is_some_and(|m| meshes.contains_key(&m.as_u128())) {
                    named += 1;
                }
            }
        }
        assert_eq!(
            named, total,
            "{named} of {total} scattered instances name a resolvable mesh"
        );
    }

    let project = |table: &inf_render::ScatterMeshes| {
        let mut scene = inf_render::RenderScene::default();
        inf_player::render::project_scene_full(
            &mut scene,
            &sim,
            1.0,
            &inf_player::vmesh::VmeshRegistry::new(),
            &inf_player::skinned::SkinnedRegistry::new(),
            &inf_voxel::VoxelVolumes::new(),
            &mut inf_render::DebrisCache::default(),
            None,
            table,
        );
        scene
    };

    let scene = project(&meshes);
    let with_geom: Vec<&inf_render::ScatterBatch> = scene
        .scatter
        .iter()
        .filter(|b| b.data.geometry.is_some())
        .collect();
    let drawn: std::collections::BTreeSet<u128> = with_geom
        .iter()
        .filter_map(|b| b.data.geometry.as_ref().map(|g| g.key()))
        .collect();
    assert_eq!(
        drawn.len(),
        3,
        "the island's three cover meshes must reach three DISTINCT geometry \
         uploads; the projection produced {} of {} scatter batches carrying \
         geometry",
        with_geom.len(),
        scene.scatter.len()
    );
    for kind in inf_editor_core::cover::CoverKind::ALL {
        let id = inf_editor_core::cover::cover_mesh_guid(kind);
        let key = meshes.get(&id.as_u128()).expect("resolved").key();
        assert!(drawn.contains(&key), "{} is not drawn", kind.label());
    }
    let instances: usize = with_geom.iter().map(|b| b.data.len()).sum();
    assert_eq!(
        instances, placed,
        "{instances} of {placed} scattered instances reached a mesh-carrying batch"
    );
    // Every batch's cull radius is its OWN geometry's, not the proxy's — the one
    // place the proxy must not be used, because a radius that is too small
    // deletes instances at the frustum edge.
    for b in &with_geom {
        let want = b.data.geometry.as_ref().expect("filtered").radius;
        assert_eq!(b.data.bounding_radius(), want);
    }

    // ── 4. …and the anti-vacuity half: the pre-TER2b engine, exactly ──
    let before = project(&inf_render::ScatterMeshes::new());
    assert!(
        before.scatter.iter().all(|b| b.data.geometry.is_none()),
        "an empty scatter-mesh table must produce the placeholder path -- if it \
         does not, the arm above is not measuring the table"
    );
    // **What "the pre-TER2b engine" is, restated for a world with settlements in
    // it** (island wave I8a). This used to assert `before.scatter.len() == 1`:
    // the island's whole population was the biome-bound vegetation, which is one
    // batch per terrain. Wave I8a stands 172 settlement blocks on the island, and
    // a block's grammar modules go through the same `push_scatter` body, so the
    // batch count is now the vegetation's one plus one per resident block — the
    // fixture measures nine. A count was never the claim; **not one batch carries
    // geometry** is, and it is asserted above.
    assert!(
        !before.scatter.is_empty(),
        "the placeholder projection produced no batch at all, so it is not the \
         same population"
    );
    assert_eq!(
        before.scatter.iter().filter(|b| !b.data.is_empty()).count(),
        before.scatter.len(),
        "an empty batch reached the projection"
    );
    println!(
        "PLACEHOLDER PROJECTION: {} batches, none carrying geometry (the \
         pre-TER2b engine); the settlements are {} of them",
        before.scatter.len(),
        before.scatter.len().saturating_sub(1)
    );

    // The PROXY primitive is still a cube, and that is a CARRIED BOUND rather
    // than an oversight: the impostor card, the CPU fallback and the cascade
    // shadow caster pack all draw it, because those three bind one shared vertex
    // buffer for the whole frame and a per-batch mesh does not fit in it. The
    // impostor is at least sized off the authored radius (`material.w` in
    // `scatter_mesh.wgsl`); the other two are not, and that is named in the
    // wave's carried list.
    assert!(
        scene
            .scatter
            .iter()
            .chain(&before.scatter)
            .all(|b| b.data.mesh == inf_render::PrimMesh::Cube),
        "the scatter proxy primitive moved -- if that is deliberate, the carried \
         bound about impostors, the CPU fallback and shadow casters has to be \
         rewritten with it"
    );

    println!(
        "ISLAND COVER: {placed} instances -> {} batches carrying 3 distinct \
         geometry uploads (was 1 placeholder batch); cull radii off the meshes \
         themselves; proxy primitive still a cube for the impostor / \
         CPU-fallback / shadow-caster paths.",
        with_geom.len()
    );
}

// ── THE SETTLEMENT GATE (island wave I8a) ───────────────────────────────────
//
// Everything below is about the thing wave I8a put on the island's seven pads:
// the blocks, the buildings they stand, the doors those buildings offer, and
// whether a player can walk in through one and go upstairs.
//
// It runs on the CI-scale fixture for the same reason the drive above does — the
// shipped island's terrain is 549.9 MB and is not committed — and it measures a
// SHIPPED-ISLAND city block directly where the block's own size is what is being
// priced (the fixture's reservations take the town's 76 m grid; a Harbour City
// core block is 100 m, and a battery about a city block has to be about one).

/// The band radius the collider band admits solids inside, for the numbers the
/// furnish battery prints. `inf_ecs::band`'s own default, named here so the
/// battery cannot drift from the thing it prices.
const BAND_NEAR_M: f64 = inf_ecs::band::DEFAULT_COLLIDER_NEAR_M;

/// An archetype whose own storey range starts at two or more — a building that
/// is guaranteed to have a stair whatever seed it draws.
///
/// The walk needs one: `Shop` is `(1, 2)` and `House` is `(1, 3)`, so half the
/// buildings in a town have no flight at all and "climb a stair" would be a
/// claim about which seed came up.
fn always_multistorey(a: inf_pcg::ArchetypeId) -> bool {
    inf_pcg::archetype(a).floors.0 >= 2
}

/// Where a settlement walk goes: the settlement holding the most blocks that are
/// **guaranteed** multi-storey, tie-broken by block count and then by name.
///
/// A pure function of the committed design, so both hosts are handed the same
/// number and nothing else.
fn walk_target_settlement(
    design: &inf_island::IslandDesign,
) -> inf_editor_core::settlement::Settlement {
    let tall = |s: &inf_editor_core::settlement::Settlement| {
        s.blocks
            .iter()
            .filter(|b| always_multistorey(b.archetype))
            .count()
    };
    let mut plans = inf_editor_core::settlement::settlements(design);
    plans.sort_by(|a, b| {
        tall(b)
            .cmp(&tall(a))
            .then(b.blocks.len().cmp(&a.blocks.len()))
            .then(a.name.cmp(&b.name))
    });
    let best = plans
        .into_iter()
        .next()
        .expect("the design has a settlement");
    assert!(
        tall(&best) > 0,
        "no settlement on this island has a block that is guaranteed \
         multi-storey, so a walk that climbs a stair has nowhere to go"
    );
    best
}

/// Every `PcgVolume` the simulation currently holds: `(guid, centre, extent,
/// seed)`, in `Guid` order so nothing downstream depends on an archetype walk.
fn resident_volumes(sim: &RuntimeSim) -> Vec<(Uuid, glam::DVec3, glam::DVec2, u32)> {
    let w = sim.world().world();
    let mut out = Vec::new();
    for e in w.iter_entities() {
        let (Some(g), Some(v), Some(t)) = (
            e.get::<inf_ecs::Guid>(),
            e.get::<inf_ecs::components::PcgVolume>(),
            e.get::<inf_ecs::components::GlobalTransform>(),
        ) else {
            continue;
        };
        out.push((
            g.0,
            t.translation(),
            glam::DVec2::new(v.extent.x, v.extent.y),
            v.seed,
        ));
    }
    out.sort_by_key(|(g, _, _, _)| *g);
    out
}

/// Every solid box the simulation currently holds, in `inf_pcg`'s own
/// vocabulary — the one `opening_is_clear` reads.
fn resident_solids(sim: &RuntimeSim) -> Vec<inf_pcg::PcgCollider> {
    let w = sim.world().world();
    let mut out = Vec::new();
    for e in w.iter_entities() {
        let Some(v) = e.get::<inf_ecs::components::PcgVolume>() else {
            continue;
        };
        for s in &v.structures {
            out.push(inf_pcg::PcgCollider {
                center: s.center,
                half_extents: s.half_extents,
                rotation: s.rotation,
            });
        }
    }
    out
}

/// The lowered building passes of one zone document, read out of the built
/// project exactly as the shipped host reads it.
fn zone_passes(content: &Path, a: inf_pcg::ArchetypeId) -> Vec<inf_pcg::BuildingPass> {
    let p = content.join(inf_editor_core::settlement::zone_file_name(a));
    let bytes = std::fs::read(&p).unwrap_or_else(|e| panic!("no {}: {e}", p.display()));
    let payload = inf_pcg::PcgAssetPayload::decode(&bytes).expect("the zone document decodes");
    let graph = payload.graph().expect("the graph is the source of truth");
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    assert!(lowered.ok, "{}: {:?}", a.name(), lowered.issues);
    assert_eq!(
        lowered.buildings.len(),
        1,
        "{} lowers to one building pass",
        a.name()
    );
    lowered.buildings
}

/// **EVERY SETTLEMENT BUILDING IS ENTERABLE, AT SETTLEMENT SCALE** (island wave
/// I8a, clause 3).
///
/// The three phase-19 invariants — `rooms_connected`, `floors_reachable`,
/// `opening_is_clear` — run per **sampled building** over the blocks the
/// simulation is actually holding, against the solids the shipped world
/// actually built. Phase 19 ran them over seven hand-placed lots; this runs them
/// over a settlement.
///
/// It also prints what clause 3 asks for: the doorways per settlement, and the
/// share of them the collider band makes solid.
#[test]
fn every_settlement_building_is_enterable() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let mut sim = pack_sim(&pack);
    let hero = hero_entity(&sim).expect("a hero");
    let mut total_buildings = 0usize;
    let mut total_doorways = 0usize;

    // **Every settlement, one at a time.** They are kilometres apart and the
    // partition holds one neighbourhood at once, so "per settlement" is what the
    // hero walking to each of them means.
    for plan in inf_editor_core::settlement::settlements(&design) {
        set_hero(
            &mut sim,
            hero,
            glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y),
        );
        for _ in 0..8 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }

        let volumes = resident_volumes(&sim);
        let solids = resident_solids(&sim);
        let doorways = inf_ecs::door::volume_doorways(sim.world());
        println!(
            "SETTLED at {} ({:.0}, {:.0}): {} resident volume(s), {} solids, {} doorways",
            plan.name,
            plan.centre.x,
            plan.centre.y,
            volumes.len(),
            solids.len(),
            doorways.len()
        );
        assert!(
            !volumes.is_empty(),
            "no settlement volume is resident — the battery below would be over bare ground"
        );
        assert!(
            !solids.is_empty(),
            "the resident settlement blocks built no solid at all"
        );
        assert!(
            !doorways.is_empty(),
            "the resident settlement blocks planned no doorway — nothing is enterable"
        );

        // **The band, measured rather than described.** A doorway is SOLID when the
        // band admits the building it belongs to; everything past the near radius
        // simulates as a shell, which is the I3 ruling ("doors and walls cannot be
        // solid at different distances").
        let band = inf_ecs::band::SimBand::from_world(
            sim.world(),
            BAND_NEAR_M,
            inf_ecs::band::DEFAULT_COLLIDER_FAR_M,
        );
        let banded = doorways
            .iter()
            .filter(|(_, _, d)| {
                band.tier(
                    d.hinge,
                    glam::DVec3::splat(d.width_m.max(d.height_m) * 0.5),
                    glam::DQuat::IDENTITY,
                ) == inf_math::Tier::Near
            })
            .count();
        println!(
            "DOORWAYS: {} planned, {banded} inside the {BAND_NEAR_M:.0} m collider band ({:.2} %)",
            doorways.len(),
            100.0 * banded as f64 / doorways.len() as f64
        );

        // ── the three invariants, per building ──
        let by_guid: std::collections::BTreeMap<Uuid, inf_editor_core::settlement::Block> = plan
            .blocks
            .iter()
            .map(|b| {
                (
                    inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row),
                    *b,
                )
            })
            .collect();

        let mut buildings = 0usize;
        let mut floors_total = 0u32;
        let mut stairs_total = 0usize;
        let mut doors_total = 0usize;
        let mut by_zone: std::collections::BTreeMap<&'static str, usize> = Default::default();
        for (guid, centre, extent, seed) in &volumes {
            let Some(block) = by_guid.get(guid) else {
                continue;
            };
            let passes = zone_passes(&content, block.archetype);
            let plans = {
                let w = sim.world().world();
                let terrain = w
                    .iter_entities()
                    .find_map(|e| e.get::<inf_ecs::components::Terrain>())
                    .expect("the island has ground");
                let height = inf_pcg::FnHeight::new(|x: f64, z: f64| {
                    terrain.data.height_at(glam::DVec2::new(x, z))
                });
                let cx = inf_pcg::GrammarContext {
                    entity: Some(*guid),
                    center: *centre,
                    extent: *extent,
                    seed_offset: u64::from(*seed),
                };
                inf_pcg::plans_of(&passes, &inf_pcg::NoSplines, &height, &cx)
            };
            assert!(
                !plans.is_empty(),
                "{}'s {} block resolved no building at all — a `Ground::Terrain` lot \
             over unpaged ground fails closed, which is right, but this block's \
             ground IS resident",
                plan.name,
                block.archetype.name()
            );
            *by_zone.entry(block.archetype.name()).or_default() += plans.len();
            buildings += plans.len();
            for p in &plans {
                floors_total += p.floors;
                stairs_total += p.stairs.len();
                // 1. every floor's room graph is connected through doors.
                for f in 0..p.floors {
                    assert!(
                        p.rooms_connected(f),
                        "{} {}: floor {f}'s room graph is not connected",
                        plan.name,
                        block.archetype.name()
                    );
                }
                // 3. and every floor is reachable from OUTSIDE.
                assert!(
                    p.entrance.is_some(),
                    "{} {}: no entrance — the building is sealed",
                    plan.name,
                    block.archetype.name()
                );
                assert!(
                    p.floors_reachable(),
                    "{} {}: a floor cannot be reached from outside",
                    plan.name,
                    block.archetype.name()
                );
                assert_eq!(
                    p.stairs.len(),
                    (p.floors - 1) as usize,
                    "{} {}: wrong flight count for {} floors",
                    plan.name,
                    block.archetype.name(),
                    p.floors
                );
                // 2. no solid the SHIPPED world built intrudes into a door's void.
                let doors: Vec<&inf_pcg::Opening> = p
                    .openings
                    .iter()
                    .filter(|o| o.kind == inf_pcg::OpeningKind::Door)
                    .collect();
                doors_total += doors.len();
                for (i, d) in doors.iter().enumerate() {
                    assert!(
                        p.opening_is_clear(d, &solids, 0.02),
                        "{} {}: door {i} on wall {} is blocked by a collider the \
                     shipped world built",
                        plan.name,
                        block.archetype.name(),
                        d.wall
                    );
                }
                // **THE CONTROL.** Every assertion above is "no solid is here", and
                // such an assertion passes trivially if the predicate cannot say no.
                // A slab through the whole building must read BLOCKED.
                let f = p.footprint;
                let top = p.floor_y(p.floors);
                let slab = [inf_pcg::PcgCollider {
                    center: glam::DVec3::new(f.center().x, (p.base_y + top) * 0.5, f.center().y),
                    half_extents: glam::DVec3::new(
                        f.size_x(),
                        (top - p.base_y) * 0.5 + 2.0,
                        f.size_z(),
                    ),
                    rotation: glam::DQuat::IDENTITY,
                }];
                for d in &doors {
                    assert!(
                        !p.opening_is_clear(d, &slab, 0.02),
                        "a door reads CLEAR through a solid building — the \
                     enterability predicate is vacuous at settlement scale"
                    );
                }
            }
        }
        println!(
            "ENTERABILITY {}: {buildings} buildings over {} resident blocks, \
         {floors_total} storeys, {stairs_total} flights, {doors_total} door \
         openings; by zone {by_zone:?}",
            plan.name,
            volumes.len()
        );
        assert!(
            buildings >= 8,
            "only {buildings} buildings resident at {} — this is not a \
         settlement-scale sample",
            plan.name
        );
        assert!(
            stairs_total > 0,
            "not one building at {} has a stair — 'climb a stair' has nothing to climb",
            plan.name
        );
        total_buildings += buildings;
        total_doorways += doorways.len();
    }
    println!(
        "ENTERABILITY TOTAL: {total_buildings} buildings and {total_doorways} \
         doorways over every settlement of this island, all enterable"
    );
    assert!(total_buildings > 0);
}

/// Does any solid contain `p`?
///
/// The blocks a settlement plans are axis-aligned, so a solid's XZ bounds are
/// exact rather than conservative — `PcgCollider::xz_half_extents` is the same
/// door `solid_bounds` uses and it needs no trigonometry.
fn solid_contains(solids: &[inf_pcg::PcgCollider], p: glam::DVec3) -> usize {
    solids
        .iter()
        .filter(|s| {
            let (ex, ez) = s.xz_half_extents();
            let (lo, hi) = s.y_band();
            p.x >= s.center.x - ex
                && p.x <= s.center.x + ex
                && p.z >= s.center.z - ez
                && p.z <= s.center.z + ez
                && p.y >= lo
                && p.y <= hi
        })
        .count()
}

/// One host's reading of the walk: the state fold per step, and everything the
/// walk **discovered** rather than was told.
///
/// The discovery is deliberately part of the trace. Two hosts handed a
/// hard-coded door guid would agree about it whatever their worlds held; two
/// hosts each asked *"which door is nearest"* agree only if their worlds are the
/// same world.
struct Walk {
    states: Vec<Vec<u8>>,
    door: Uuid,
    prompt: glam::DVec3,
    open_before: bool,
    open_after: bool,
    verdict_moved: bool,
    inside: glam::DVec3,
    upstairs: glam::DVec3,
    inside_blocked: usize,
    upstairs_blocked: usize,
    doorways: usize,
    solids: usize,
    climb_m: f64,
}

/// Steps spent walking from the settlement's centre to the door.
const WALK_STEPS: usize = 40;
/// Steps spent waiting for the leaf to swing after `use_door`.
const SWING_STEPS: usize = 45;
/// Steps spent standing inside, and then upstairs.
const DWELL_STEPS: usize = 15;

/// **THE WALK**: enter the city, find a door, open it, step through, go up.
///
/// Every target is computed from the host's OWN world; nothing is passed in but
/// the settlement's centre, which is a committed number.
fn walk_into_a_building(
    sim: &mut RuntimeSim,
    content: &Path,
    recipe: &inf_island::IslandRecipe,
    plan: &inf_editor_core::settlement::Settlement,
) -> Walk {
    let hero = hero_entity(sim).expect("a hero");
    let centre = glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y);
    let mut states: Vec<Vec<u8>> = Vec::new();

    // ── 1. ENTER THE CITY ── stand at the crossroads and let the cells activate.
    set_hero(sim, hero, centre);
    for _ in 0..8 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }

    // ── 2. FIND A DOOR ── the EXTERIOR doorway on the ground floor nearest the
    //    crossroads, over the doorways the simulation is holding, **on a block
    //    whose archetype is guaranteed multi-storey**. The last clause is not
    //    fussiness: a `Shop` is one or two storeys and a `House` is one to
    //    three, so on any other block "climb a stair" would be a claim about
    //    which seed came up. Ties break on `(volume guid, index)`, which
    //    `volume_doorways` already walks in.
    let tall_blocks: std::collections::BTreeSet<Uuid> = plan
        .blocks
        .iter()
        .filter(|b| always_multistorey(b.archetype))
        .map(|b| inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row))
        .collect();
    let doorways = inf_ecs::door::volume_doorways(sim.world());
    let solids = resident_solids(sim);
    let (vol, idx, slot) = doorways
        .iter()
        .filter(|(v, _, d)| d.exterior && d.floor == 0 && tall_blocks.contains(v))
        .min_by(|a, b| {
            (a.2.hinge - centre)
                .length_squared()
                .total_cmp(&(b.2.hinge - centre).length_squared())
        })
        .copied()
        .expect("a resident multi-storey block offers an exterior door");
    let door = inf_physics::d3::door::pcg_doorway_guid(vol, idx);
    let placement = inf_physics::d3::door::placement_of(sim.world(), door)
        .expect("the doorway the walk found resolves to a placement");
    let inside_dir = {
        let yaw = slot.inside_yaw_deg.to_radians();
        // `+Z` at zero, `+X` at +90 — the compass the doorway carries. The
        // PORTABLE trig, not `std`'s: this reaches a position two hosts compare,
        // and the P14 law does not stop at a file that happens to be a test.
        glam::DVec3::new(
            inf_math::portable::psin64(yaw),
            0.0,
            inf_math::portable::pcos64(yaw),
        )
    };
    // **The walk arrives from the STREET**, which is the side away from the room
    // the wall serves. That is not decoration: `DoorSpec::lock_side` is `Inside`
    // for a grammar door, and `use_door` pressed from the lock side on a shut,
    // unlocked leaf **locks it** rather than opening it — "locked from the
    // inside" has to mean something for the person who locked it. The first
    // draft of this walk stood on `prompt_position`, pressed, and got a verdict
    // that did not move: it had bolted the front door from the hall.
    let prompt = inf_ecs::door::prompt_position(&placement);
    let approach = slot.hinge - inside_dir * 1.2;

    // ── 3. WALK TO IT ── straight from the crossroads, one step at a time, so
    //    the trace is a walk and not a teleport.
    for k in 1..=WALK_STEPS {
        let t = k as f64 / WALK_STEPS as f64;
        set_hero(sim, hero, centre + (approach - centre) * t);
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }

    // ── 4. OPEN IT ── through `use_door`, which is the function the interact
    //    button and the `door.use` node both dispatch to, pressed from the feet
    //    the hero is standing on.
    let feet = glam::DVec3::new(approach.x, approach.y - slot.height_m * 0.5, approach.z);
    let open_before = inf_physics::d3::door::is_open_near(sim.world(), approach);
    let verdict = inf_physics::d3::door::use_door(sim.world_mut(), door, feet);
    for _ in 0..SWING_STEPS {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }
    let open_after = inf_physics::d3::door::is_open_near(sim.world(), approach);

    // ── 5. STEP THROUGH ── a metre and a half along the inside face's own
    //    normal, at knee height, which is where a body would be.
    let inside = slot.hinge + inside_dir * 1.5;
    for _ in 0..DWELL_STEPS {
        set_hero(
            sim,
            hero,
            glam::DVec3::new(inside.x, inside.y - 1.0, inside.z),
        );
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }
    let inside_blocked = solid_contains(&solids, inside);

    // ── 6. CLIMB ── the building this door belongs to, its stair core, at floor
    //    one's own walking height. Found by re-deriving the block's plans and
    //    matching the doorway's hinge BIT FOR BIT against the derivation the
    //    shipped host itself ran — not by a search radius.
    let block = plan
        .blocks
        .iter()
        .find(|b| {
            inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row) == vol
        })
        .copied()
        .expect("the doorway's volume is a settlement block");
    let passes = zone_passes(content, block.archetype);
    let (upstairs, climb_m) = {
        let volumes = resident_volumes(sim);
        let (_, vcentre, vextent, vseed) = volumes
            .iter()
            .find(|(g, _, _, _)| *g == vol)
            .copied()
            .expect("the volume is resident");
        let w = sim.world().world();
        let terrain = w
            .iter_entities()
            .find_map(|e| e.get::<inf_ecs::components::Terrain>())
            .expect("the island has ground");
        let height =
            inf_pcg::FnHeight::new(|x: f64, z: f64| terrain.data.height_at(glam::DVec2::new(x, z)));
        let cx = inf_pcg::GrammarContext {
            entity: Some(vol),
            center: vcentre,
            extent: vextent,
            seed_offset: u64::from(vseed),
        };
        let plans = inf_pcg::plans_of(&passes, &inf_pcg::NoSplines, &height, &cx);
        let mut found = None;
        for p in &plans {
            let mut ds = inf_pcg::building::doorways_of(p);
            inf_pcg::building::place_doorways_in_frame(&mut ds, p.frame);
            if ds.iter().any(|d| {
                d.hinge.x.to_bits() == slot.hinge.x.to_bits()
                    && d.hinge.y.to_bits() == slot.hinge.y.to_bits()
                    && d.hinge.z.to_bits() == slot.hinge.z.to_bits()
            }) {
                found = Some(p.clone());
                break;
            }
        }
        let p = found.expect(
            "the doorway the world offered is not in any plan the same resolution \
             derives — the shipped population and `plans_of` disagree",
        );
        assert!(
            p.floors >= 2,
            "the building this walk entered is single-storey, so there is no stair \
             to climb"
        );
        // **The stair is what got the hero up; the ROOM is where the hero
        // stands.** The core's own footprint is full of treads at floor one's
        // height — that is what a flight from one to two IS — so a "no solid is
        // here" test aimed at the core measures the staircase and reads
        // blocked-by-2. The claim the walk is making is that a floor above the
        // ground is *reachable and standable*, so the point is the first
        // non-stair room on floor one, and the stair's part is asserted
        // separately: its room on floor one must be in the set reachable from
        // outside.
        assert!(p.core.is_some(), "a multi-storey plan has a stair core");
        let reach = p.reachable_rooms();
        let up = p
            .stair_room(1)
            .expect("a multi-storey plan has a stair room on floor one");
        assert!(
            reach.get(up).copied().unwrap_or(false),
            "the stair room on floor one is not reachable from outside — the \
             flight lands nowhere"
        );
        let (ri, room) = p
            .rooms_on(1)
            .find(|(i, r)| r.kind != inf_pcg::RoomType::Stair && reach[*i])
            .expect("floor one has a room that is not the stairwell");
        assert!(reach[ri]);
        let c = p.frame.to_world(room.rect.center());
        // Floor one's walking surface, plus a knee — the point a body standing on
        // the first floor occupies.
        let y = p.floor_y(1) + 0.5;
        (glam::DVec3::new(c.x, y, c.y), p.floor_y(1) - p.floor_y(0))
    };
    for _ in 0..DWELL_STEPS {
        set_hero(
            sim,
            hero,
            glam::DVec3::new(upstairs.x, upstairs.y - 0.5, upstairs.z),
        );
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        states.push(sim.state_bytes());
    }
    let upstairs_blocked = solid_contains(&solids, upstairs);

    Walk {
        states,
        door,
        prompt,
        open_before,
        open_after,
        verdict_moved: verdict.moved(),
        inside,
        upstairs,
        inside_blocked,
        upstairs_blocked,
        doorways: doorways.len(),
        solids: solids.len(),
        climb_m,
    }
}

/// **THE SETTLEMENT GATE** (island wave I8a, clause 4): the shipped player and
/// the editor's document walk into the same building, open the same door and
/// climb the same stair, **byte for byte**.
///
/// # Coverage first, because two empty worlds agree perfectly
///
/// Every claim below is asserted on each host *separately* before the two are
/// compared: the settlement is resident, it built solids, it planned doorways,
/// the door the walk found was shut and is open, the doorway is walkable and the
/// first floor is standable. A gate that compared folds alone would certify two
/// hosts that both found nothing — which is exactly the failure the I7 audit
/// found in the drive gate one file up.
///
/// # Why the walk discovers its own target
///
/// Nothing is passed in but the settlement's centre, which is a committed
/// number. Which door is nearest, which building it belongs to and where that
/// building's stair core is are all read out of the host's own world, so two
/// hosts holding different worlds disagree about the targets as well as about
/// the folds.
#[test]
fn pie_equals_shipping_on_a_walk_into_a_building() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let slug = inf_island::slug(&recipe.name);
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plan = walk_target_settlement(&design);

    let mut ship = pack_sim(&pack);
    let mut editor = loose_sim(&content, &slug);
    let a = walk_into_a_building(&mut ship, &content, &recipe, &plan);
    let b = walk_into_a_building(&mut editor, &content, &recipe, &plan);

    for (label, w) in [("shipping", &a), ("document", &b)] {
        println!(
            "WALK ({label}) at {}: {} doorways / {} solids resident; door {} at \
             ({:.2}, {:.2}, {:.2}); shut {} -> open {}; inside ({:.2}, {:.2}, \
             {:.2}) blocked by {}; upstairs ({:.2}, {:.2}, {:.2}) blocked by {}; \
             climbed {:.2} m",
            plan.name,
            w.doorways,
            w.solids,
            w.door,
            w.prompt.x,
            w.prompt.y,
            w.prompt.z,
            !w.open_before,
            w.open_after,
            w.inside.x,
            w.inside.y,
            w.inside.z,
            w.inside_blocked,
            w.upstairs.x,
            w.upstairs.y,
            w.upstairs.z,
            w.upstairs_blocked,
            w.climb_m
        );
        // ── coverage, on each host, before either is compared to the other ──
        assert!(w.doorways > 0, "{label}: no doorway was resident at all");
        assert!(w.solids > 0, "{label}: the settlement built no solid");
        assert!(
            !w.open_before,
            "{label}: the door was already open, so opening it proves nothing"
        );
        assert!(
            w.verdict_moved,
            "{label}: `use_door` refused — the door the walk found is not usable"
        );
        assert!(
            w.open_after,
            "{label}: the door did not open in {SWING_STEPS} steps"
        );
        assert_eq!(
            w.inside_blocked, 0,
            "{label}: a solid stands where the walk stepped through the doorway — \
             the door opens onto a wall"
        );
        assert_eq!(
            w.upstairs_blocked, 0,
            "{label}: a solid stands in the stair core on the first floor — the \
             stairwell is filled in"
        );
        assert!(
            w.climb_m > 2.0,
            "{label}: the first floor is {:.2} m up, which is not a storey",
            w.climb_m
        );
        assert_eq!(
            w.states.len(),
            8 + WALK_STEPS + SWING_STEPS + 2 * DWELL_STEPS,
            "{label}: the walk did not run its whole script"
        );
    }

    // ── and the two hosts agree, about the targets and about every step ──
    assert_eq!(a.door, b.door, "the two hosts found different doors");
    assert_eq!(
        a.prompt.x.to_bits(),
        b.prompt.x.to_bits(),
        "the two hosts put the same door in different places"
    );
    assert_eq!(a.upstairs.y.to_bits(), b.upstairs.y.to_bits());
    assert_eq!(a.doorways, b.doorways);
    assert_eq!(a.solids, b.solids);
    let mut distinct: std::collections::BTreeSet<&Vec<u8>> = Default::default();
    for (i, (x, y)) in a.states.iter().zip(&b.states).enumerate() {
        assert_eq!(
            x,
            y,
            "the shipping player and the editor's document diverged at step {i} \
             of the walk ({} against {} bytes)",
            x.len(),
            y.len()
        );
        distinct.insert(x);
    }
    println!(
        "SETTLEMENT GATE: {} steps, {} DISTINCT states, byte-identical between \
         the cooked pack and the loose document",
        a.states.len(),
        distinct.len()
    );
    // Anti-vacuity on the fold itself: a walk that never changed the world would
    // produce one state repeated, and comparing it to itself proves nothing.
    assert!(
        distinct.len() > a.states.len() / 2,
        "only {} of {} states are distinct — the walk is not moving the world",
        distinct.len(),
        a.states.len()
    );
}

/// One furnish configuration's price for one block, in counts a machine cannot
/// inflate.
#[derive(Debug, Clone, Copy, Default)]
struct BlockPrice {
    buildings: usize,
    solids: usize,
    instances: usize,
    doorways: usize,
    /// Solids the collider band would admit with an anchor at the block's own
    /// centre — the number a fixed step pays for.
    banded: usize,
}

/// Evaluate one settlement block with `furnish` forced, and price it.
///
/// The ground is held **flat at the site's own datum**, and that is the honest
/// choice for a price rather than a shortcut: a site pad is levelled toward its
/// datum, so a block near a settlement's centre sits on ground that is nearly
/// flat, and holding it exactly flat makes the comparison between three
/// configurations have one variable. What it is NOT is a claim about the
/// island's own relief.
fn price_block(
    passes: &[inf_pcg::BuildingPass],
    furnish: Option<bool>,
    centre: glam::DVec3,
    extent: glam::DVec2,
    seed: u32,
) -> BlockPrice {
    let passes: Vec<inf_pcg::BuildingPass> = passes
        .iter()
        .cloned()
        .map(|mut p| {
            if let Some(f) = furnish {
                p.furnish = f;
            }
            p
        })
        .collect();
    let cx = inf_pcg::GrammarContext {
        entity: None,
        center: centre,
        extent,
        seed_offset: u64::from(seed),
    };
    let height = inf_pcg::FnHeight::new(move |_, _| Some(centre.y));
    let out = inf_pcg::evaluate_buildings(&passes, &inf_pcg::NoSplines, &height, &cx);
    let band = inf_ecs::band::SimBand::from_anchors(
        [centre],
        BAND_NEAR_M,
        inf_ecs::band::DEFAULT_COLLIDER_FAR_M,
    );
    BlockPrice {
        buildings: out.groups.len(),
        solids: out.colliders.len(),
        instances: out.instances.len(),
        doorways: out.doorways.len(),
        banded: out
            .colliders
            .iter()
            .filter(|c| band.tier(c.center, c.half_extents, c.rotation) == inf_math::Tier::Near)
            .count(),
    }
}

/// **THE FURNISH BATTERY** (island wave I8a, clause 3 / ruling 3).
///
/// The ruling was *measure, then decide, default ON*. This is the measurement,
/// and it has three legs because "furnish=true holds" is three different claims:
///
/// 1. **What a city block costs, three ways.** One real **Harbour City** core
///    block — a 100 m block on the shipped island's own grid, not the fixture's
///    76 m one, because a battery about a city block has to be about one —
///    evaluated with furnish off, with furnish as shipped, and with furnish
///    forced on. Counts, which are the same integer on every machine.
/// 2. **What the fixed step pays.** The fixture's own settled world, stepped
///    with the shipped population and again with every resident volume's
///    population replaced by the fully-furnished one, against
///    [`CITY_STEP_BUDGET_MS`]. Same world, same anchors, one variable.
/// 3. **What a load pays**, against [`LOAD_BUDGET_MS`].
///
/// The verdict it produced is `inf_editor_core::settlement::furnishes`, and it
/// is stated in the wave's ledger with these numbers beside it.
#[test]
fn the_furnish_battery_prices_a_city_block_at_island_scale() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let started = std::time::Instant::now();
    let proj = build_project(tmp.path());
    let content = proj.join("Content");
    let pack = cook(tmp.path());
    let build_ms = started.elapsed().as_secs_f64() * 1000.0;

    // ── 1. one REAL Harbour City block, three ways ──
    let shipped_recipe =
        inf_editor_core::island::repo_root().join(inf_editor_core::island::ISLAND_RECIPES[0]);
    if shipped_recipe.exists() {
        let recipe =
            inf_island::IslandRecipe::load(&shipped_recipe).expect("the island recipe loads");
        let design = inf_island::read_design(&recipe).expect("the island design reads");
        let city = inf_editor_core::settlement::settlements(&design)
            .into_iter()
            .find(|s| s.kind == inf_island::SiteKind::City)
            .expect("the island has a city");
        // The core block's own geometry, priced for **every archetype** rather
        // than for the one that happens to be zoned there. That is what makes
        // this a battery: `Shop` is one to two storeys and `Hotel` is four to
        // ten, furniture is per ROOM, and a measurement of the cheap one would
        // have decided the ruling on the wrong building.
        let block = city
            .blocks
            .iter()
            .find(|b| b.ring == 0)
            .copied()
            .expect("a city has a ring-0 block");
        let centre = glam::DVec3::new(block.centre.x, 0.0, block.centre.y);
        let extent = glam::DVec2::new(block.half.x, block.half.y);
        println!(
            "FURNISH BATTERY on a {:.0} x {:.0} m {} block at ({:.0}, {:.0}), \
             every archetype:",
            block.half.x * 2.0,
            block.half.y * 2.0,
            city.name,
            block.centre.x,
            block.centre.y
        );
        let mut worst = 1.0f64;
        let mut worst_name = "";
        for a in inf_pcg::ArchetypeId::ALL {
            let passes = zone_passes(&content, a);
            let off = price_block(&passes, Some(false), centre, extent, block.seed);
            let on = price_block(&passes, Some(true), centre, extent, block.seed);
            let ratio = on.solids as f64 / off.solids.max(1) as f64;
            println!(
                "  {:>10} ({}-{} storeys): {} buildings, {} doorways; bare {} \
                 solids ({} banded, {} drawn instances) -> furnished {} solids \
                 ({} banded, {} drawn), {ratio:.2}x{}",
                a.name(),
                inf_pcg::archetype(a).floors.0,
                inf_pcg::archetype(a).floors.1,
                off.buildings,
                off.doorways,
                off.solids,
                off.banded,
                off.instances,
                on.solids,
                on.banded,
                on.instances,
                if inf_editor_core::settlement::furnishes(a) {
                    "  <- SHIPS FURNISHED"
                } else {
                    ""
                }
            );
            assert_eq!(
                off.buildings,
                on.buildings,
                "{}: furnishing changed how many BUILDINGS a block stands, which \
                 it must not — furniture is what goes inside them",
                a.name()
            );
            assert_eq!(
                off.doorways,
                on.doorways,
                "{}: furnishing changed the doorway count",
                a.name()
            );
            assert!(
                on.solids > off.solids,
                "{}: furnishing added no solid at all — the battery is measuring \
                 nothing",
                a.name()
            );
            if ratio > worst {
                worst = ratio;
                worst_name = a.name();
            }
        }
        println!("  WORST: {worst_name} at {worst:.2}x the solids of a bare block");
    } else {
        println!("SKIP the shipped-island half: no committed island recipe");
    }

    // ── 2. what the fixed step pays, on a settled world ──
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plan = walk_target_settlement(&design);
    let mut sim = pack_sim(&pack);
    let hero = hero_entity(&sim).expect("a hero");
    set_hero(
        &mut sim,
        hero,
        glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y),
    );
    for _ in 0..12 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let shipped_solids = resident_solids(&sim).len();
    println!("FIXED STEP at {} (release asserts nothing here — this REPORTS, on this module's own law that a millisecond is a fact about the machine):", plan.name);
    // **THE SAME CONFIGURATION, TWICE, BEFORE ANYTHING IS CHANGED** (island wave
    // I8a audit). This arm's `A` and `B` are separated by one variable and by a
    // stretch of wall clock, and the wall clock is running **inside a test binary
    // whose other eleven arms are executing on other threads** — `cargo test`
    // runs a file's tests concurrently, and nothing here asks it not to.
    //
    // Measured: this same comparison read **+0.663 ms** in the wave's own run,
    // **+0.821 / +0.915 / +0.805 ms** in three runs of this arm ALONE, and
    // **−0.690 ms** — the opposite sign — in a run of the whole file. The step
    // itself read 6.348, 8.1–8.2 and 10.693 ms in those three regimes over an
    // identical world of 21 453 solids.
    //
    // So the first measurement is repeated with nothing changed between them,
    // and `|A' − A|` is printed beside `B − A'` as this run's own noise floor. A
    // difference inside the floor is not a measurement, and a reader who is
    // handed one number cannot tell.
    let (shipped_ms, shipped_prof) = step_profile_of(&mut sim, 60, 90);
    print_step(
        "as shipped",
        shipped_ms,
        &shipped_prof,
        shipped_solids,
        &sim,
    );
    let (control_ms, control_prof) = step_profile_of(&mut sim, 10, 90);
    print_step(
        "as shipped (again)",
        control_ms,
        &control_prof,
        shipped_solids,
        &sim,
    );
    let floor = (control_ms - shipped_ms).abs();
    println!("  NOISE FLOOR: the same configuration twice differs by {floor:.3} ms");

    // …and again with every resident block's population replaced by the fully
    // furnished one, through the same door the host writes it with.
    let by_guid: std::collections::BTreeMap<Uuid, inf_editor_core::settlement::Block> = plan
        .blocks
        .iter()
        .map(|b| {
            (
                inf_editor_core::settlement::block_guid(&recipe.name, b.site, b.col, b.row),
                *b,
            )
        })
        .collect();
    let mut replaced = 0usize;
    for (guid, centre, extent, seed) in resident_volumes(&sim) {
        let Some(block) = by_guid.get(&guid) else {
            continue;
        };
        let passes: Vec<inf_pcg::BuildingPass> = zone_passes(&content, block.archetype)
            .into_iter()
            .map(|mut p| {
                p.furnish = true;
                p
            })
            .collect();
        let out = {
            let w = sim.world().world();
            let terrain = w
                .iter_entities()
                .find_map(|e| e.get::<inf_ecs::components::Terrain>())
                .expect("the island has ground");
            let height = inf_pcg::FnHeight::new(|x: f64, z: f64| {
                terrain.data.height_at(glam::DVec2::new(x, z))
            });
            let cx = inf_pcg::GrammarContext {
                entity: Some(guid),
                center: centre,
                extent,
                seed_offset: u64::from(seed),
            };
            inf_pcg::compose_volume(
                Vec::new(),
                inf_pcg::evaluate_buildings(&passes, &inf_pcg::NoSplines, &height, &cx),
            )
        };
        let (baked, solid, groups, doorways) = inf_player::level::population_of(out);
        let e = sim
            .world()
            .entity_of(guid)
            .expect("the volume the walk found is in the world");
        if let Some(mut v) = sim
            .world_mut()
            .world_mut()
            .get_mut::<inf_ecs::components::PcgVolume>(e)
        {
            v.set_population(baked, solid, groups, doorways);
            replaced += 1;
        }
    }
    // Two steps for the physics bridge to reconcile the new change stamp before
    // the clock starts — the cost being measured is the STEADY step, not the
    // rebuild the swap itself forces.
    for _ in 0..2 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let furnished_solids = resident_solids(&sim).len();
    let (furnished_ms, furnished_prof) = step_profile_of(&mut sim, 60, 90);
    print_step(
        "fully furnished",
        furnished_ms,
        &furnished_prof,
        furnished_solids,
        &sim,
    );
    let cost = furnished_ms - control_ms;
    println!(
        "  {replaced} volume(s) swapped; furnishing moves the step {cost:+.3} ms \
         against a {floor:.3} ms noise floor — {}. **The COUNT is the half a \
         machine cannot move**: {shipped_solids} -> {furnished_solids} solids \
         ({:.2}x), and that is what the `furnishes` verdict rests on",
        if cost.abs() > floor * 2.0 {
            "a difference"
        } else {
            "INSIDE the floor, i.e. no measurement at all"
        },
        furnished_solids as f64 / shipped_solids.max(1) as f64
    );
    println!("LOAD: build + cook of the whole fixture project took {build_ms:.0} ms against a {LOAD_BUDGET_MS} ms load budget");
    assert!(
        furnished_solids > shipped_solids,
        "the swap changed nothing — the step comparison is between one \
         configuration and itself"
    );
    // **Reported, not asserted, and the reason is this module's own law**: a
    // millisecond is a fact about the machine, `[profile.dev]` is `opt-level = 1`
    // with debug assertions, and every CI runner reports rather than asserts on
    // a wall clock. What IS asserted is the solid count, which is the same
    // integer everywhere.
    assert!(
        shipped_ms.is_finite() && control_ms.is_finite() && furnished_ms.is_finite(),
        "the step clock produced no number"
    );
}

/// The mean fixed-step time over `n` steps, milliseconds, **with the step's own
/// phase breakdown beside it**.
///
/// A step that cannot say where its milliseconds went is the CPU twin of the
/// frame that could not say where its GPU milliseconds went — wave I4b's own
/// finding, and the reason `RuntimeSim` carries a step clock at all. The first
/// draft of this battery printed one number and a reader could not tell a
/// physics regression from a paging one.
///
/// A discarded warm-up pass first: the first steps after a population swap seat
/// the collider band and take every `structure_stamps` miss there is, and
/// measuring them is measuring a step that happens once.
fn step_profile_of(
    sim: &mut RuntimeSim,
    warmup: usize,
    n: usize,
) -> (f64, inf_player::step_profile::StepProfile) {
    for _ in 0..warmup {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    sim.set_step_profiling(true);
    let mut acc = inf_player::step_profile::StepProfile::default();
    let t = std::time::Instant::now();
    for _ in 0..n {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        acc.accumulate(&sim.step_profile());
    }
    let wall = t.elapsed().as_secs_f64() * 1000.0 / n as f64;
    acc.scale(1.0 / n as f64);
    sim.set_step_profiling(false);
    (wall, acc)
}

/// Print one step profile, dearest phase first, **with what the physics world
/// actually admitted beside it**.
///
/// A step whose dearest phase is the solver over a world with one moving thing
/// in it is a step paying for its own STATIC geometry, and the admitted-collider
/// count is the evidence — the fps instrument's own arrangement, met on a
/// settlement.
fn print_step(
    label: &str,
    wall: f64,
    prof: &inf_player::step_profile::StepProfile,
    solids: usize,
    sim: &RuntimeSim,
) {
    let (tracked, touching) = sim.bridge3d().world().contact_pair_counts();
    println!(
        "  {label:<18} {wall:7.3} ms/step over {solids} resident solids (phases \
         sum to {:.3} ms; the ratchet is {CITY_STEP_BUDGET_MS} ms). Physics: {} \
         bodies, {} ADMITTED structure colliders, {tracked} contact pairs \
         tracked ({touching} touching)",
        prof.total_ms(),
        sim.bridge3d().body_count(),
        sim.bridge3d().admitted_structures(),
    );
    for (n, ms) in prof.dearest_first() {
        if ms <= 0.02 {
            continue;
        }
        println!(
            "      {n:<18} {ms:7.3} ms ({:4.1} %)",
            ms / prof.total_ms().max(1.0e-9) * 100.0
        );
    }
}

/// **THE `is_open` WALK, RE-MEASURED AT SETTLEMENT SCALE** (island wave I8a,
/// clause 3).
///
/// The I6 audit found `door.is_open` walking **all 19 790 doorways** of the
/// composed city to answer a question about one, and fixed it by checking the
/// reach *before* building a placement — so the walk is still `O(doorways)` and
/// the allocation is not. Wave I8a is the first content in this repository that
/// puts real doorways on a streamed world, so the cost class is re-measured
/// here rather than assumed.
///
/// **The number that changed is not the constant, it is the N.** The walk is
/// over the doorways the SIMULATION holds, and a streamed island holds one
/// neighbourhood: the whole island plans two orders of magnitude more doorways
/// than any step ever walks.
#[test]
fn the_is_open_walk_costs_what_it_costs_at_settlement_scale() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    build_project(tmp.path());
    let pack = cook(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plan = walk_target_settlement(&design);
    let mut sim = pack_sim(&pack);
    let hero = hero_entity(&sim).expect("a hero");
    set_hero(
        &mut sim,
        hero,
        glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y),
    );
    for _ in 0..12 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let resident = inf_ecs::door::volume_doorways(sim.world()).len();
    let centre = glam::DVec3::new(plan.centre.x, 0.0, plan.centre.y);
    // Two questions, and the difference between them is the whole point: one
    // asked where there IS a door, one asked out at sea.
    let near = {
        let d = inf_ecs::door::volume_doorways(sim.world());
        d.iter()
            .map(|(_, _, s)| s.hinge)
            .min_by(|a, b| {
                (*a - centre)
                    .length_squared()
                    .total_cmp(&(*b - centre).length_squared())
            })
            .expect("a doorway")
    };
    let far = glam::DVec3::new(centre.x + 5_000.0, centre.y, centre.z);
    const CALLS: usize = 200;
    let mut answered = 0usize;
    let t = std::time::Instant::now();
    for _ in 0..CALLS {
        if inf_physics::d3::door::is_open_near(sim.world(), near) {
            answered += 1;
        }
    }
    let near_us = t.elapsed().as_secs_f64() * 1.0e6 / CALLS as f64;
    let t = std::time::Instant::now();
    for _ in 0..CALLS {
        if inf_physics::d3::door::is_open_near(sim.world(), far) {
            answered += 1;
        }
    }
    let far_us = t.elapsed().as_secs_f64() * 1.0e6 / CALLS as f64;
    // **What the walk's N actually is.** The resident set, against the blocks
    // this island holds in all: the ratio is the whole point of the class, and
    // the second number is a block COUNT rather than a doorway count because
    // nothing evaluates the far blocks and inventing a doorway figure for them
    // would be inference dressed as measurement.
    let resident_blocks = resident_volumes(&sim).len();
    let island_blocks: usize = inf_editor_core::settlement::settlements(&design)
        .iter()
        .map(|s| s.blocks.len())
        .sum();
    println!(
        "IS_OPEN WALK: {resident} doorways over {resident_blocks} resident blocks \
         at {}, {near_us:.1} us a call beside a door and {far_us:.1} us a call \
         five kilometres from one ({answered} of {} answered open). This island \
         has {island_blocks} blocks in all and a step walks the resident ones \
         only — the walk is O(RESIDENT doorways), which is what streaming buys \
         and what the I6 measurement (19 790 doorways on an unstreamed city) did \
         not have.",
        plan.name,
        2 * CALLS
    );
    assert!(resident > 100, "only {resident} doorways were resident");
    // The class: the far call does the same walk and allocates nothing, so it
    // must not be dramatically dearer than the near one. A regression that
    // rebuilt every placement would make the far call the expensive one.
    assert!(
        far_us <= near_us * 4.0 + 50.0,
        "a call with NO door in reach costs {far_us:.1} us against {near_us:.1} \
         beside one — the reach check has stopped happening before the \
         allocation (the I6 audit's finding, returned)"
    );
}
