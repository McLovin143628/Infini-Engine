//! **WAVE PAR1b — STREET LIGHTING + STREET FURNITURE** — the gate.
//!
//! The island's settlement streets carry lamp posts (a mesh, an emissive
//! luminaire and a REAL Dusk light), traffic signals the traffic obeys (the
//! obedience arm is `inf_physics`' `par1b_signals_3d`), utility poles with
//! catenary cables, stop signs, benches, bins, hydrants and mailboxes — every
//! piece derived from the streets in both hosts (`inf_pcg::street`,
//! `inf_ecs::furniture`), appended to the population of the block whose
//! frontage it stands in front of, so it is drawn, lit and solid through the
//! doors every block already passes. The arms:
//!
//! * the CI island loaded by BOTH hosts: the furniture census world-side (by
//!   kind, per 100 m of street), NOTHING in a carriageway / junction box /
//!   parking slot / doorway / building, every foot on its pavement, no crowd
//!   agent inside a piece over 600 steps, and PIE == shipping on the census,
//!   the light list at four hours and the signal phase trace;
//! * the lamp's pool on the asphalt: under a lamp vs midway vs no lamp, at a
//!   fixed night exposure, with the light removed as the mutation;
//! * a car at 15 m/s into a lamp post stops at the post and its bodywork takes
//!   the blow.

use std::collections::BTreeMap;
use std::path::PathBuf;

use glam::{DVec2, DVec3, Vec3};
use inf_math::FloatingOrigin;
use inf_pcg::building::fixtures::FixtureRow;
use inf_pcg::street::{self as street, FurniturePiece, PieceKind};
use inf_player::runtime_sim::RuntimeSim;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightBound, LightKind, MeshInstance, RenderLight,
    RenderScene, RenderSettings, RenderView, HEADLESS_FORMAT,
};

// ── the loaded island ────────────────────────────────────────────────────────

fn fixture_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml")
}

/// Build + cook the CI-scale island; the shipping sim (the pack) and the PIE
/// sim (the loose level), both with their streamers — `par1a_rooms_gate`'s
/// two hosts exactly.
fn fixture_hosts(tmp: &std::path::Path) -> (RuntimeSim, RuntimeSim) {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.join("island");
    inf_project::ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    inf_island::write_content(&build, &content).expect("the island's content writes");
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the island cooks");
    let source = inf_player::level::PackLevelSource::open(&out).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut ship = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut ship, &partition, pcg);
    inf_player::attach_terrain_streaming(
        &mut ship,
        &inf_player::TerrainContent::Pack(source.clone()),
    );

    let slug = inf_island::slug(&recipe.name);
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(&content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(&content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(&content),
    )
    .with_bindings(inf_player::level::load_actor_classes_by_guid_from_dir(
        &content,
    ))
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        &content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(
        &content,
    ))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(&content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the loose level builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut pie = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut pie, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut pie, &inf_player::TerrainContent::Dir(terrains));
    (ship, pie)
}

/// Put the hero at `at`, freeze the clock at local `hour`, and step `steps`.
fn stand(sim: &mut RuntimeSim, at: DVec3, hour: f64, steps: usize) {
    let hero = sim
        .world()
        .world()
        .iter_entities()
        .find(|e| {
            e.get::<inf_ecs::components::CharacterMovement>()
                .is_some_and(|m| m.player_controlled)
        })
        .map(|e| e.id());
    if let Some(e) = hero {
        if let Some(mut t) = sim
            .world_mut()
            .world_mut()
            .get_mut::<inf_ecs::components::Transform>(e)
        {
            t.translation = inf_ecs::math::Vec3d::new(at.x, at.y + 2.0, at.z);
        }
    }
    {
        let w = sim.world_mut().world_mut();
        let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
        for mut tod in q.iter_mut(w) {
            tod.seconds = (hour * 3600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
            tod.rate = 0.0;
        }
    }
    sim.world_mut().mark_dirty();
    for _ in 0..steps {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
}

fn project_world(sim: &RuntimeSim) -> RenderScene {
    let mut scene = RenderScene::default();
    let voxels = inf_voxel::VoxelVolumes::default();
    let mut meshes = inf_render::ScatterMeshes::new();
    inf_player::scatter_mesh::add_building_modules(&mut meshes);
    inf_player::render::project_scene_full(
        &mut scene,
        sim,
        1.0,
        &inf_player::vmesh::VmeshRegistry::default(),
        &inf_player::skinned::SkinnedRegistry::new(),
        &voxels,
        &mut inf_render::DebrisCache::default(),
        None,
        &meshes,
        &std::collections::HashMap::new(),
    );
    scene
}

/// The projected light list, its bounds and the drawn lens instances as bytes —
/// what the two hosts must agree on.
fn light_bytes(scene: &RenderScene) -> Vec<u8> {
    let mut b = Vec::new();
    for l in &scene.lights {
        b.push(l.kind as u8);
        for f in l
            .color
            .iter()
            .chain([l.intensity, l.range, l.inner_cos, l.outer_cos].iter())
        {
            b.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        for f in [l.position.x, l.position.y, l.position.z] {
            b.extend_from_slice(&f.to_bits().to_le_bytes());
        }
        b.push(u8::from(l.cast_shadows));
    }
    for lb in &scene.light_bounds {
        b.extend_from_slice(&lb.light.to_le_bytes());
        b.extend_from_slice(&lb.draw_m.to_bits().to_le_bytes());
    }
    for i in &scene.instances {
        for f in i.emissive {
            b.extend_from_slice(&f.to_bits().to_le_bytes());
        }
    }
    b
}

/// One furniture solid, classified world-side by its own box (each kind's
/// collider is a distinct shape — `inf_pcg::street`'s parts).
fn kind_of_solid(h: DVec3) -> Option<PieceKind> {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-6;
    Some(
        if close(h.x, street::LAMP_SHAFT_HALF_M) && close(h.y, street::LAMP_HEIGHT_M * 0.5) {
            PieceKind::LampPost
        } else if close(h.x, street::SIGNAL_MAST_HALF_M) {
            PieceKind::SignalPost
        } else if close(h.x, street::POLE_HALF_M) {
            PieceKind::UtilityPole
        } else if close(h.x, 0.05) && close(h.y, 1.25) {
            PieceKind::Sign
        } else if close(h.y, 0.45) {
            PieceKind::Bench
        } else if close(h.x, 0.28) {
            PieceKind::Bin
        } else if close(h.x, 0.15) {
            PieceKind::Hydrant
        } else if close(h.x, 0.26) {
            PieceKind::Mailbox
        } else {
            return None;
        },
    )
}

/// Every resident block's furniture tail, world-side:
/// `(solids, lights, instances, shells)`.
#[allow(clippy::type_complexity)]
fn furniture_of(
    sim: &RuntimeSim,
) -> (
    Vec<inf_ecs::components::ScatteredSolid>,
    Vec<inf_ecs::components::ScatteredLight>,
    usize,
    Vec<inf_ecs::components::ScatteredSolid>,
    Vec<DVec2>,
) {
    let (mut solids, mut lights, mut inst, mut shells, mut doors) =
        (Vec::new(), Vec::new(), 0usize, Vec::new(), Vec::new());
    for e in sim.world().world().iter_entities() {
        let Some(v) = e.get::<inf_ecs::components::PcgVolume>() else {
            continue;
        };
        let (i, s, l) = inf_ecs::furniture::tail(v);
        solids.extend_from_slice(s);
        lights.extend_from_slice(l);
        inst += i.len();
        shells.extend(v.structure_groups.iter().map(|g| g.shell));
        doors.extend(
            v.doorways
                .iter()
                .filter(|d| d.exterior)
                .map(|d| DVec2::new(d.hinge.x, d.hinge.z)),
        );
    }
    (solids, lights, inst, shells, doors)
}

/// The kerb face of a street at the plan point nearest `p`, and how far `p`
/// stands from the centreline: `(lateral, along-inside-span)`.
fn street_lateral(s: &inf_ecs::traffic::Street, p: DVec2) -> (f64, bool) {
    let (lo, hi, along, perp) = if s.along_x() {
        (s.a.x.min(s.b.x), s.a.x.max(s.b.x), p.x, s.a.y)
    } else {
        (s.a.y.min(s.b.y), s.a.y.max(s.b.y), p.y, s.a.x)
    };
    let lateral = if s.along_x() { p.y - perp } else { p.x - perp };
    (lateral, along >= lo && along <= hi)
}

/// **THE ISLAND IS FURNISHED, AND NOTHING STANDS WHERE IT MUST NOT** (clauses
/// 1-3), with PIE == shipping.
///
/// Reads: the CI island loaded by both hosts, the hero in the first
/// settlement's core at 21:00 for 600 steps. World-side, over every resident
/// block's furniture tail: (1) the census by kind and per 100 m of street;
/// (2) 0 furniture solids reaching a carriageway (a street's kerb back), 0 in a
/// parking slot (`kerb_slots`), 0 within `DOOR_CLEAR_M` of an exterior door,
/// 0 inside a building shell; (3) every foot on its pavement surface
/// (`street::pavement_y` over the sim's own terrain, within 0.05 m) and never
/// under the terrain at the foot; (4) no crowd agent's capsule inside a piece
/// over the 600 steps; (5) PIE == shipping: the census, the projected light
/// list (lamps, signal heads, lenses) at 00:30 / 12:30 / 19:30 / 21:30 and
/// the signal phase trace. Mutation (measured, stated in the report):
/// `FURNITURE_BACK_M` 0.35 -> 2.5 puts the line on the kerb stone and (2)
/// reds with the count.
#[test]
fn the_island_is_furnished_and_nothing_stands_where_it_must_not() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let (mut ship, mut pie) = fixture_hosts(tmp.path());
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let design = inf_island::read_design(&recipe).expect("the design reads");
    let plans = inf_editor_core::settlement::settlements(&design);
    let core = plans[0].centre;
    let at = DVec3::new(core.x, 0.0, core.y);

    // (0) D-18 (clause 5): the hero spawns ON its ground in both hosts — the
    // island's own spawn, before anything moves it: the height it settles by
    // over five seconds standing still is at most 5 cm either way.
    for (name, sim) in [("shipping", &mut ship), ("PIE", &mut pie)] {
        let hero_y = |sim: &RuntimeSim| {
            sim.world()
                .world()
                .iter_entities()
                .find(|e| {
                    e.get::<inf_ecs::components::CharacterMovement>()
                        .is_some_and(|m| m.player_controlled)
                })
                .and_then(|e| e.get::<inf_ecs::components::Transform>())
                .map(|t| t.translation.y)
                .expect("the island has a hero")
        };
        let spawn = hero_y(sim);
        for _ in 0..300 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        let settled = hero_y(sim);
        println!(
            "PAR1b D-18 ({name}): spawned at y {spawn:.4}, at rest at y {settled:.4} — {:+.4} m",
            spawn - settled
        );
        assert!(
            (spawn - settled).abs() <= 0.05,
            "{name}: the hero spawned {:+.4} m off its ground",
            spawn - settled
        );
    }

    let mut worst_agent = f64::INFINITY;
    let mut agent_samples = 0usize;
    let mut overlaps = 0usize;
    for (k, sim) in [&mut ship, &mut pie].into_iter().enumerate() {
        stand(sim, at, 21.0, 0);
        for step in 0..600 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
            // (4) on the shipping host, every 20 steps: crowd capsules vs pieces.
            if k == 0 && step % 20 == 19 {
                let (solids, _, _, _, _) = furniture_of(sim);
                for e in sim.world().world().iter_entities() {
                    let Some(cm) = e.get::<inf_ecs::components::CharacterMovement>() else {
                        continue;
                    };
                    if cm.player_controlled {
                        continue;
                    }
                    let Some(t) = e.get::<inf_ecs::components::GlobalTransform>() else {
                        continue;
                    };
                    let p = t.translation();
                    agent_samples += 1;
                    for s in &solids {
                        let dx = ((p.x - s.center.x).abs() - s.half_extents.x).max(0.0);
                        let dz = ((p.z - s.center.z).abs() - s.half_extents.z).max(0.0);
                        let d = (dx * dx + dz * dz).sqrt();
                        if (p.y - s.center.y).abs() < s.half_extents.y + 1.0 {
                            worst_agent = worst_agent.min(d);
                            if d < 0.3 - 0.02 {
                                overlaps += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    // (1) the census, world-side.
    let streets = inf_ecs::traffic::streets_of(ship.world());
    let (solids, lights, instances, shells, doors) = furniture_of(&ship);
    let mut by_kind: BTreeMap<PieceKind, usize> = BTreeMap::new();
    let mut unknown = 0usize;
    for s in &solids {
        match kind_of_solid(s.half_extents) {
            Some(k) => *by_kind.entry(k).or_default() += 1,
            None => unknown += 1,
        }
    }
    let lamps = lights
        .iter()
        .filter(|l| FixtureRow::from_code(l.row) == Some(FixtureRow::StreetLamp))
        .count();
    let heads = lights
        .iter()
        .filter(|l| FixtureRow::from_code(l.row) == Some(FixtureRow::SignalHead))
        .count();
    let length: f64 = streets.iter().map(|s| (s.b - s.a).length()).sum();
    println!(
        "PAR1b CENSUS (CI island, resident): {} streets {:.0} m; {} furniture instances, {} solids, {} lamps, {} signal heads; by kind {:?}; unclassified {unknown}",
        streets.len(),
        length,
        instances,
        solids.len(),
        lamps,
        heads,
        by_kind
    );
    let per100 = |n: usize| n as f64 / (length / 100.0).max(1e-9);
    for (k, n) in &by_kind {
        println!(
            "  {:<13} {n:>5}  {:>5.2} per 100 m of street",
            k.name(),
            per100(*n)
        );
    }
    // The CI island's resident streets are two (272 m at the wave): fifteen
    // lamps is the lattice on both sides at 30 m less the junction clearance.
    assert!(
        lamps >= 10,
        "only {lamps} street lamps on the resident streets"
    );
    assert_eq!(
        by_kind.get(&PieceKind::LampPost).copied().unwrap_or(0),
        lamps,
        "a lamp light without its post's collider, or a post without its light"
    );
    assert!(
        by_kind.get(&PieceKind::UtilityPole).copied().unwrap_or(0) >= 3,
        "no pole line"
    );
    assert_eq!(unknown, 0, "a furniture solid of no kind");
    // Every street lamp is a Dusk light, unboxed, and every signal head burns
    // at every hour.
    for l in &lights {
        match FixtureRow::from_code(l.row) {
            Some(FixtureRow::StreetLamp) => {
                assert_eq!(l.schedule, inf_ecs::components::LightSchedule::Dusk);
                assert!(l.clip.is_none(), "a street lamp in a box");
            }
            Some(FixtureRow::SignalHead) => {
                assert_eq!(l.schedule, inf_ecs::components::LightSchedule::Always);
                assert_eq!(l.phases, 2);
            }
            other => panic!("a furniture light of row {other:?}"),
        }
    }

    // (2) nothing where it must not stand.
    let slots = inf_ecs::traffic::kerb_slots(&streets);
    let (mut in_carriageway, mut in_slot, mut at_door, mut in_building) = (0, 0, 0, 0);
    for s in &solids {
        let p = DVec2::new(s.center.x, s.center.z);
        let r = s.half_extents.x.max(s.half_extents.z);
        for st in &streets {
            let (lat, inside) = street_lateral(st, p);
            let kerb_back =
                inf_ecs::traffic::street_kerb_offset_m(st.gap_m) + inf_ecs::traffic::KERB_WIDTH_M;
            if inside && lat.abs() - r < kerb_back {
                in_carriageway += 1;
            }
        }
        for (q, _) in &slots {
            if (q.x - p.x).abs() < 1.0 + r && (q.z - p.y).abs() < 1.0 + r {
                in_slot += 1;
            }
        }
        for d in &doors {
            if (*d - p).length() < street::DOOR_CLEAR_M - 0.2 {
                at_door += 1;
            }
        }
        for sh in &shells {
            let l = sh.rotation.inverse() * (s.center - sh.center);
            if l.x.abs() < sh.half_extents.x && l.z.abs() < sh.half_extents.z {
                in_building += 1;
            }
        }
    }
    println!(
        "PAR1b PLACEMENT: of {} solids: {in_carriageway} in a carriageway, {in_slot} in a parking slot ({} slots), {at_door} at a door ({} doors), {in_building} inside a building",
        solids.len(),
        slots.len(),
        doors.len()
    );
    assert_eq!(in_carriageway, 0, "furniture in a carriageway");
    assert_eq!(in_slot, 0, "furniture in a parking slot");
    assert_eq!(at_door, 0, "furniture in a doorway");
    assert_eq!(in_building, 0, "furniture inside a building");

    // (3) every foot on its pavement.
    let mut worst_foot = 0.0f64;
    let mut buried = 0usize;
    for s in &solids {
        let foot = s.center.y - s.half_extents.y;
        let p = DVec2::new(s.center.x, s.center.z);
        let ground_here = ship.terrain_height_at(p.x, p.y);
        if foot < ground_here - 0.05 {
            buried += 1;
        }
        // The pavement surface the derivation stood it on, over the SIM's own
        // terrain query (a different door from the derivation's).
        let Some(st) = streets
            .iter()
            .filter(|st| street_lateral(st, p).1)
            .min_by(|a, b| {
                street_lateral(a, p)
                    .0
                    .abs()
                    .total_cmp(&street_lateral(b, p).0.abs())
            })
        else {
            continue;
        };
        let (lat, _) = street_lateral(st, p);
        if lat.abs() > st.gap_m * 0.5 + 1.0 {
            continue;
        }
        let kerb = inf_ecs::traffic::street_kerb_offset_m(st.gap_m);
        let face = if st.along_x() {
            DVec2::new(p.x, st.a.y + lat.signum() * kerb)
        } else {
            DVec2::new(st.a.x + lat.signum() * kerb, p.y)
        };
        let paved = ship.terrain_height_at(face.x, face.y)
            + street::ROAD_LIFT_M
            + street::KERB_HEIGHT_M
            + (lat.abs() - kerb - street::KERB_WIDTH_M).max(0.0) * street::PAVEMENT_FALL;
        // The visible surface: the slab, or the bank where the ground across
        // the footway climbs over it.
        let want = paved.max(ground_here);
        worst_foot = worst_foot.max((foot - want).abs());
    }
    println!(
        "PAR1b FEET: worst foot vs its pavement surface {worst_foot:.4} m; {buried} under the terrain at the foot"
    );
    assert!(
        worst_foot <= 0.05,
        "a foot {worst_foot:.3} m off its pavement"
    );
    assert_eq!(buried, 0, "a piece under the ground");

    // (4) the crowd.
    println!(
        "PAR1b CROWD: {agent_samples} agent samples over 600 steps, nearest approach to a piece {worst_agent:.2} m, {overlaps} overlaps"
    );
    assert_eq!(overlaps, 0, "a crowd agent inside a piece of furniture");

    // (5) PIE == shipping.
    assert_eq!(
        inf_ecs::furniture::census(ship.world()),
        inf_ecs::furniture::census(pie.world()),
        "the two hosts furnished different streets"
    );
    let bytes = |sim: &RuntimeSim| -> Vec<u8> {
        let (s, l, _, _, _) = furniture_of(sim);
        let mut b = Vec::new();
        for x in &s {
            for f in [x.center.x, x.center.y, x.center.z, x.half_extents.y] {
                b.extend_from_slice(&f.to_bits().to_le_bytes());
            }
        }
        for x in &l {
            for f in [x.at.x, x.at.y, x.at.z] {
                b.extend_from_slice(&f.to_bits().to_le_bytes());
            }
        }
        b
    };
    assert_eq!(
        bytes(&ship),
        bytes(&pie),
        "the furniture differs between the hosts"
    );
    let mut trace_ship = Vec::new();
    let mut trace_pie = Vec::new();
    for (h, n) in [(0.5, 120), (12.5, 120), (19.5, 120), (21.5, 120)] {
        for sim in [&mut ship, &mut pie] {
            stand(sim, at, h, n);
        }
        let (a, b) = (project_world(&ship), project_world(&pie));
        assert_eq!(
            light_bytes(&a),
            light_bytes(&b),
            "{h:.1} h: the two hosts projected different lights / lenses"
        );
        for (sim, trace) in [(&ship, &mut trace_ship), (&pie, &mut trace_pie)] {
            let t = inf_ecs::traffic::signal_clock_of(sim.world());
            if let Some(r) = inf_ecs::traffic::carriageway_of(sim.world()) {
                for j in &r.junctions {
                    trace.push((
                        inf_ecs::traffic::signal_aspect(j, true, t).as_u8(),
                        inf_ecs::traffic::signal_aspect(j, false, t).as_u8(),
                    ));
                }
            }
        }
        let lamps_lit = a
            .lights
            .iter()
            .filter(|l| l.kind == LightKind::Spot && (l.range - street::LAMP_RANGE_M).abs() < 1e-3)
            .count();
        println!(
            "PAR1b HOUR {h:04.1}: {} lights projected, {lamps_lit} street lamps lit, {} lens instances",
            a.lights.len(),
            a.instances.iter().filter(|i| i.mesh == inf_render::PrimMesh::Sphere).count()
        );
        if h < 13.0 && h > 12.0 {
            assert_eq!(lamps_lit, 0, "a street lamp lit at noon");
        }
        if h > 21.0 {
            assert!(lamps_lit > 0, "no street lamp lit at 21:30");
        }
    }
    assert_eq!(trace_ship, trace_pie, "the signal phase traces differ");
    println!(
        "PAR1b PHASE TRACE: {} junction samples, identical",
        trace_ship.len()
    );
}

// ── the lamp's pool ──────────────────────────────────────────────────────────

const W: u32 = 640;
const H: u32 = 360;
const WARM: u32 = 6;
/// The fixed exposure the pool is read at — PAR1a's night arms' x8.
const NIGHT_EXPOSURE: f32 = 8.0;

fn gpu() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP par1b_street_gate: no GPU adapter ({e})");
            None
        }
    }
}

fn look(eye: DVec3, target: DVec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

fn render(gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    let mut s = RenderSettings::default();
    s.gi.enabled = true;
    s.shadows.enabled = false;
    s.vsm.enabled = false;
    s.lights.local_shadow_budget = 0;
    s.exposure = NIGHT_EXPOSURE;
    r.set_settings(s);
    for _ in 0..WARM {
        r.render(gpu, scene, view, &target.view, (W, H));
        let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
    }
    target.read_rgba(gpu).expect("readback")
}

fn luma(img: &[u8], x: u32, y: u32) -> f64 {
    let i = ((y * W + x) * 4) as usize;
    0.2126 * f64::from(img[i]) + 0.7152 * f64::from(img[i + 1]) + 0.0722 * f64::from(img[i + 2])
}

fn patch_at(img: &[u8], view: &RenderView, p: DVec3, px: u32) -> f64 {
    let c = view.view_proj() * (p - view.origin.origin()).as_vec3().extend(1.0);
    let n = c.truncate() / c.w;
    let (x, y) = (
        ((n.x * 0.5 + 0.5) * W as f32).clamp(0.0, W as f32 - 1.0) as u32,
        ((0.5 - n.y * 0.5) * H as f32).clamp(0.0, H as f32 - 1.0) as u32,
    );
    let mut v = Vec::new();
    for yy in y.saturating_sub(px / 2)..(y + px / 2).min(H) {
        for xx in x.saturating_sub(px / 2)..(x + px / 2).min(W) {
            v.push(luma(img, xx, yy));
        }
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// The lamp lights a straight 20 m street over flat ground derives, mapped the
/// way both projectors' light fence maps a `ScatteredLight` at night (level 1).
fn street_scene(with_lights: bool) -> (RenderScene, Vec<FurniturePiece>) {
    let line = [street::StreetLine {
        a: DVec2::new(-150.0, 0.0),
        b: DVec2::new(150.0, 0.0),
        gap_m: 20.0,
    }];
    let f = street::furnish(&line, &[], &[], &inf_pcg::FnHeight::new(|_, _| Some(0.0)));
    let mut scene = RenderScene::default();
    // A moon, in BOTH scenes: an empty light list is the editor's default sun,
    // which would light the "dark" control at noon strength.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [0.6, 0.7, 1.0],
        intensity: 0.002,
        direction: Vec3::new(0.2, 1.0, 0.1).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        inner_cos: 1.0,
        outer_cos: 1.0,
        cast_shadows: false,
    });
    // The asphalt: a wide dark slab whose top is the road.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        glam::Quat::IDENTITY,
        Vec3::new(400.0, 1.0, 400.0),
        [0.16, 0.16, 0.17, 1.0],
        1,
    ));
    let lamps: Vec<FurniturePiece> = f
        .pieces
        .into_iter()
        .filter(|p| p.kind == PieceKind::LampPost)
        .collect();
    if with_lights {
        for p in &lamps {
            for l in &p.lights {
                scene.light_bounds.push(LightBound {
                    light: scene.lights.len() as u32,
                    clip: None,
                    draw_m: l.draw_m,
                });
                scene.lights.push(RenderLight {
                    kind: LightKind::Spot,
                    color: l.sweep.0,
                    intensity: l.intensity,
                    direction: (-l.dir).as_vec3(),
                    position: l.at,
                    range: l.range_m,
                    inner_cos: l.inner_deg.to_radians().cos(),
                    outer_cos: l.outer_deg.to_radians().cos(),
                    cast_shadows: false,
                });
            }
        }
    }
    (scene, lamps)
}

/// **THE LAMP'S POOL IS REAL** (clause 1): on the asphalt under a lamp's head,
/// midway between two lamps, and the same spots with no lamp lit — fixed
/// camera, fixed night exposure x8, GI on. The lamp is unboxed (terrain and
/// water read it: `par0_lights_gate::a_point_light_over_terrain_raises_its_luminance`).
/// Mutation: the lights removed -> under == between == dark.
#[test]
fn a_lamp_pools_on_the_asphalt_under_it_and_between_lamps_against_the_dark() {
    let Some(gpu) = gpu() else { return };
    let (lit, lamps) = street_scene(true);
    let (dark, _) = street_scene(false);
    // Two consecutive lamps on the +side, the camera on the far pavement.
    let mut plus: Vec<&FurniturePiece> = lamps.iter().filter(|p| p.foot.z > 0.0).collect();
    plus.sort_by(|a, b| a.foot.x.total_cmp(&b.foot.x));
    let a = plus
        .iter()
        .find(|p| p.foot.x >= -1.0)
        .expect("a lamp near the origin");
    let b = plus
        .iter()
        .find(|p| p.foot.x > a.foot.x + 1.0)
        .expect("its neighbour");
    let head = a.lights[0].at;
    let under = DVec3::new(head.x, 0.0, head.z);
    let between = DVec3::new((a.foot.x + b.foot.x) * 0.5, 0.0, head.z);
    let view = look(
        DVec3::new(a.foot.x - 6.0, 1.7, -8.5),
        DVec3::new(a.foot.x + 8.0, 0.0, head.z),
    );
    let (li, di) = (render(&gpu, &lit, &view), render(&gpu, &dark, &view));
    let (u, m, ud, md) = (
        patch_at(&li, &view, under, 8),
        patch_at(&li, &view, between, 8),
        patch_at(&di, &view, under, 8),
        patch_at(&di, &view, between, 8),
    );
    println!(
        "PAR1b POOL (x{NIGHT_EXPOSURE}): under the lamp {u:.1}, midway between lamps {m:.1}; no lamp: {ud:.1} / {md:.1}; spacing {:.1} m, intensity {:.1}",
        b.foot.x - a.foot.x,
        a.lights[0].intensity
    );
    assert!(
        u >= ud + 20.0,
        "the asphalt under a lamp {u:.1} vs dark {ud:.1}"
    );
    assert!(
        m > md + 2.0,
        "the asphalt between lamps {m:.1} vs dark {md:.1}"
    );
    assert!(u > m, "under {u:.1} is not brighter than between {m:.1}");
}

// ── a post stops a car ───────────────────────────────────────────────────────

const DT: f64 = 1.0 / 60.0;
const CHASSIS: uuid::Uuid = uuid::Uuid::from_u128(0x5A1B_0001);
const GROUND: uuid::Uuid = uuid::Uuid::from_u128(0x5A1B_0002);
const BLOCK: uuid::Uuid = uuid::Uuid::from_u128(0x5A1B_0003);

/// **A POST STOPS A CAR** (clause 3): a saloon at 15 m/s into a lamp post the
/// furniture door built — its shaft's collider, appended to a block's
/// population through `set_furniture` and described by the physics bridge as
/// an ungrouped static box — stops within the post's radius plus a crush depth,
/// and the bodywork registers the blow. Mutation: the post's solid removed ->
/// the car rolls on past it.
#[test]
fn a_car_at_fifteen_metres_a_second_stops_at_a_lamp_post() {
    use inf_ecs::components::{
        BodyKind3D, Collider3D, ColliderShape3DKind, PcgVolume, RigidBody3D, Transform,
    };
    use inf_ecs::math::{Vec2d, Vec3d};
    let run = |with_post: bool| -> (f64, f64, f64) {
        let mut world = inf_ecs::EcsWorld::default();
        let g = world.spawn_with_guid(GROUND, "Ground", None);
        world.world_mut().entity_mut(g).insert((
            Transform {
                translation: Vec3d::new(0.0, -0.5, 0.0),
                ..Default::default()
            },
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(300.0, 0.5, 300.0),
                friction: 0.9,
                ..Default::default()
            },
        ));
        // The post, through the furniture door: a block whose frontage it is.
        let post_at = DVec3::new(0.0, 0.0, 40.0);
        let b = world.spawn_with_guid(BLOCK, "Block", None);
        world.world_mut().entity_mut(b).insert(Transform {
            translation: Vec3d::new(0.0, 0.0, 60.0),
            ..Default::default()
        });
        let mut vol = PcgVolume {
            extent: Vec2d::new(10.0, 10.0),
            ..Default::default()
        };
        if with_post {
            vol.set_furniture(
                7,
                Vec::new(),
                vec![inf_ecs::components::ScatteredSolid {
                    center: post_at + DVec3::Y * (street::LAMP_HEIGHT_M * 0.5),
                    half_extents: DVec3::new(
                        street::LAMP_SHAFT_HALF_M,
                        street::LAMP_HEIGHT_M * 0.5,
                        street::LAMP_SHAFT_HALF_M,
                    ),
                    rotation: glam::DQuat::IDENTITY,
                }],
                Vec::new(),
            );
        }
        world.world_mut().entity_mut(b).insert(vol);
        let def = *inf_editor_core::vehicle::island_vehicles()
            .get("sedan")
            .expect("the catalogue has a sedan");
        let y = inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.1;
        inf_ecs::vehicle::spawn_rig(
            &mut world,
            CHASSIS,
            &def,
            &inf_ecs::vehicle::RigSpawn {
                name: "Runner".into(),
                at: DVec3::new(0.0, y, -40.0),
                yaw_deg: 0.0,
                paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
                clip: None,
                engine_voice: false,
                livery: None,
            },
        );
        world.mark_dirty();
        world.propagate();
        let mut bridge = inf_physics::d3::PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        let step = |world: &mut inf_ecs::EcsWorld,
                    bridge: &mut inf_physics::d3::PhysicsBridge3D,
                    c: inf_ecs::vehicle::VehicleControls| {
            bridge.sync_from_world(world);
            if let Some(v) = bridge.vehicle_mut(CHASSIS) {
                v.control(c);
            }
            inf_physics::d3::step_vehicles(world, bridge, DT);
            bridge.step(DT);
            bridge.write_back_into(world);
            world.propagate();
        };
        let speed = |bridge: &inf_physics::d3::PhysicsBridge3D| {
            bridge
                .body_of(CHASSIS)
                .and_then(|b| bridge.world().body_linvel(b))
                .map(|v| v.z)
                .unwrap_or(0.0)
        };
        let pos = |bridge: &inf_physics::d3::PhysicsBridge3D| {
            bridge
                .body_of(CHASSIS)
                .and_then(|b| bridge.world().body_translation(b))
                .unwrap_or(DVec3::ZERO)
        };
        for _ in 0..60 {
            step(&mut world, &mut bridge, Default::default());
        }
        let mut hit = 0.0;
        for _ in 0..1200 {
            let throttle = if speed(&bridge) < 15.0 { 1.0 } else { 0.0 };
            if pos(&bridge).z > post_at.z - 12.0 && hit == 0.0 {
                hit = speed(&bridge);
            }
            step(
                &mut world,
                &mut bridge,
                inf_ecs::vehicle::VehicleControls {
                    throttle: if hit > 0.0 { 0.0 } else { throttle },
                    ..Default::default()
                },
            );
        }
        let front = pos(&bridge).z + def.half_extents.z.abs();
        let damage = inf_physics::d3::bodywork::damage_or_default(&world, CHASSIS);
        (
            hit,
            front,
            damage.hull_j + damage.parts.values().map(|p| p.dent_m).sum::<f64>(),
        )
    };
    let (hit, front, lost) = run(true);
    let (_, front_free, _) = run(false);
    println!(
        "PAR1b POST: hit at {hit:.2} m/s, the car's nose stopped at z {front:.2} (post face {:.2}); bodywork lost {lost:.3}; without the post the nose reached {front_free:.2}",
        40.0 - street::LAMP_SHAFT_HALF_M
    );
    assert!(hit > 13.0, "the car met the post at only {hit:.2} m/s");
    assert!(
        front <= 40.0 + street::LAMP_SHAFT_HALF_M + 1.2,
        "the car's nose went {:.2} m past the post",
        front - 40.0
    );
    assert!(front_free > 45.0, "without the post the car stopped anyway");
    assert!(lost > 0.0, "the post took no blow off the bodywork");
}
