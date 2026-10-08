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
    render_at(gpu, scene, view, NIGHT_EXPOSURE)
}

fn render_at(gpu: &GpuContext, scene: &RenderScene, view: &RenderView, exposure: f32) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut r = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    let mut s = RenderSettings::default();
    s.gi.enabled = true;
    s.shadows.enabled = false;
    s.vsm.enabled = false;
    s.lights.local_shadow_budget = 0;
    s.exposure = exposure;
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

/// **THE LAMP'S POOL IS REAL** (clause 1; audit PAR1b re-aimed): on the
/// asphalt where a lamp's beam lands, midway between two lamps on the same
/// line, and the same spots with no lamp lit — fixed camera, fixed night
/// exposure x8, GI on. The lamp is unboxed (terrain and water read it:
/// `par0_lights_gate::a_point_light_over_terrain_raises_its_luminance`).
///
/// A night street is a CHAIN of pools (`steal-car/0035`: the pool reads 2.2x
/// the road beside it in 8-bit), so the arm asks for a pool at least
/// [`POOL_MIN_LIFT`] codes over dark AND for the asphalt between two lamps to
/// stay within [`BETWEEN_MAX_LIFT`] of dark — one wash over the whole street
/// is a defect. Mutations (measured): the lights removed -> under == dark,
/// red; the first cut's 8 000 lm / 75 degrees / 28 m -> between 124.7 vs dark
/// 104.5, red.
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
    // Where the beam lands: the head's light, along its leaned axis, to the
    // road.
    let l = &a.lights[0];
    let land = l.at + l.dir * (l.at.y / -l.dir.y);
    let under = DVec3::new(land.x, 0.0, land.z);
    let between = DVec3::new((a.foot.x + b.foot.x) * 0.5, 0.0, land.z);
    let view = look(
        DVec3::new(a.foot.x - 6.0, 1.7, -8.5),
        DVec3::new(a.foot.x + 8.0, 0.0, land.z),
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
        u >= ud + POOL_MIN_LIFT,
        "the asphalt under a lamp {u:.1} vs dark {ud:.1}"
    );
    assert!(
        m <= md + BETWEEN_MAX_LIFT,
        "the asphalt between lamps {m:.1} vs dark {md:.1}: one wash, not a chain of pools"
    );
    assert!(u > m, "under {u:.1} is not brighter than between {m:.1}");
}

/// The least a lamp's pool lifts the asphalt over dark, codes at x8.
const POOL_MIN_LIFT: f64 = 40.0;
/// The most the asphalt midway between two lamps may sit over dark, codes at
/// x8 — the dark between the pools.
const BETWEEN_MAX_LIFT: f64 = 6.0;

/// The exposure the signal arm reads at — the pool arm's x8 (this synthetic
/// scene's default sky is far brighter than the island's night, so the
/// shipped kerb eyes' x12 - x46 would clip its dark control).
const KERB_EXPOSURE: f32 = 8.0;

/// **A SIGNAL IS SEEN, IT DOES NOT LIGHT THE JUNCTION** (audit PAR1b (d')): the
/// first cut's lens light (350 lm, 30 m) tinted the hero's shirt and the
/// junction asphalt green in the window at 21:00. One signalised crossing of
/// two 20 m streets over dark asphalt, the +X approach's head lit RED (light +
/// lens), then GREEN, then the green lens alone (its GI share), then nothing
/// (the dark control), at x8:
///
/// * the asphalt a metre before the approach's stop line reads the lens's
///   colour FAINTLY — lit, but at most [`SIGNAL_ASPHALT_MAX`] codes over dark;
/// * a pale shirt standing in the junction box in the beam (chest height)
///   shifts its lit channel by at most [`SIGNAL_SHIRT_MAX`] codes over dark.
///
/// * the corner facade beside the approach stays within [`SIGNAL_FACADE_MAX`].
///
/// Mutations (measured): the first cut's 350 lm / 24 degrees -> the shirt
/// +3.6, red; 100 lm / 24 degrees -> the corner facade +0.8, red.
#[test]
fn a_signal_head_tints_its_approach_faintly_and_not_the_hero() {
    let Some(gpu) = gpu() else { return };
    let streets = [
        street::StreetLine {
            a: DVec2::new(-150.0, 0.0),
            b: DVec2::new(150.0, 0.0),
            gap_m: 20.0,
        },
        street::StreetLine {
            a: DVec2::new(0.0, -150.0),
            b: DVec2::new(0.0, 150.0),
            gap_m: 20.0,
        },
    ];
    let site = street::SignalSite {
        centre: DVec2::ZERO,
        gap_x: 20.0,
        gap_z: 20.0,
        offset_s: 0,
    };
    let f = street::furnish(
        &streets,
        &[site],
        &[],
        &inf_pcg::FnHeight::new(|_, _| Some(0.0)),
    );
    // The head serving the approach travelling +X (from -X toward the centre):
    // its light's beam points back down that approach (-X).
    let head = f
        .pieces
        .iter()
        .filter(|p| p.kind == PieceKind::SignalPost)
        .flat_map(|p| p.lights.iter())
        .find(|l| l.dir.x < -0.5)
        .expect("the +X approach's head")
        .clone();
    // `(light colour, lens colour)`: the lit lens is drawn as the projectors'
    // `push_signal_lenses` draws it (an emissive sphere on the housing's face,
    // `SIGNAL_LENS_EMISSIVE` = 1.6 x the aspect), because the GI carries a lit
    // emitter's colour too.
    let scene_of = |colour: Option<[f32; 3]>, lens: Option<[f32; 3]>| {
        let mut scene = RenderScene::default();
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
        scene.instances.push(MeshInstance::lit(
            DVec3::new(0.0, -0.5, 0.0),
            glam::Quat::IDENTITY,
            Vec3::new(400.0, 1.0, 400.0),
            [0.16, 0.16, 0.17, 1.0],
            1,
        ));
        // A pale shirt (a 0.5 m torso) IN the junction box, 10 m back along the
        // beam from the far-side head — where the window's hero stood when
        // the first cut turned his shirt green (`fin5-window-21\200`). At 5 m
        // a chest is 40 degrees under the head, outside its 24-degree cone; at
        // 10 m it is inside it.
        let shirt = DVec3::new(head.at.x - 10.0, 1.35, head.at.z);
        scene.instances.push(MeshInstance::lit(
            shirt,
            glam::Quat::IDENTITY,
            Vec3::new(0.25, 0.6, 0.45),
            [0.7, 0.7, 0.7, 1.0],
            2,
        ));
        // The corner facade beside the approach (the block's face 10.5 m off
        // the centreline, the window's green-washed wall): a pale panel.
        scene.instances.push(MeshInstance::lit(
            DVec3::new(-inf_ecs::traffic::STOP_LINE_M - 8.0, 3.0, -10.6),
            glam::Quat::IDENTITY,
            Vec3::new(16.0, 6.0, 0.2),
            [0.6, 0.6, 0.6, 1.0],
            4,
        ));
        if let Some(c) = lens {
            let mut m = MeshInstance::lit(
                head.at + DVec3::new(-0.08, 0.0, 0.0),
                glam::Quat::IDENTITY,
                Vec3::new(0.05, 0.22, 0.22),
                [c[0] * 0.12, c[1] * 0.12, c[2] * 0.12, 1.0],
                3,
            );
            m.mesh = inf_render::PrimMesh::Sphere;
            m.emissive = [c[0] * 1.6, c[1] * 1.6, c[2] * 1.6];
            scene.instances.push(m);
        }
        if let Some(c) = colour {
            scene.light_bounds.push(LightBound {
                light: scene.lights.len() as u32,
                clip: None,
                draw_m: head.draw_m,
            });
            scene.lights.push(RenderLight {
                kind: LightKind::Spot,
                color: c,
                intensity: head.intensity,
                direction: (-head.dir).as_vec3(),
                position: head.at,
                range: head.range_m,
                inner_cos: head.inner_deg.to_radians().cos(),
                outer_cos: head.outer_deg.to_radians().cos(),
                cast_shadows: false,
            });
        }
        (scene, shirt)
    };
    // The camera on the approach, behind the line, looking at the junction.
    let line_x = -inf_ecs::traffic::STOP_LINE_M;
    let view = look(
        DVec3::new(line_x - 14.0, 1.7, -4.0),
        DVec3::new(line_x, 1.0, -1.75),
    );
    let before_line = DVec3::new(line_x - 1.0, 0.0, -1.75);
    let rgb_at = |img: &[u8], p: DVec3| -> [f64; 3] {
        let c = view.view_proj() * (p - view.origin.origin()).as_vec3().extend(1.0);
        let n = c.truncate() / c.w;
        let (x, y) = (
            ((n.x * 0.5 + 0.5) * W as f32).clamp(0.0, W as f32 - 1.0) as u32,
            ((0.5 - n.y * 0.5) * H as f32).clamp(0.0, H as f32 - 1.0) as u32,
        );
        let mut s = [0.0f64; 3];
        let mut k = 0.0;
        for yy in y.saturating_sub(3)..(y + 3).min(H) {
            for xx in x.saturating_sub(3)..(x + 3).min(W) {
                let i = ((yy * W + xx) * 4) as usize;
                for ch in 0..3 {
                    s[ch] += f64::from(img[i + ch]);
                }
                k += 1.0;
            }
        }
        s.map(|v| v / k)
    };
    let red = inf_pcg::street::aspect_rgb(2);
    let green = inf_pcg::street::aspect_rgb(0);
    let (sr, shirt) = scene_of(Some(red), Some(red));
    let (sg, _) = scene_of(Some(green), Some(green));
    let (sl, _) = scene_of(None, Some(green));
    let (sd, _) = scene_of(None, None);
    let (ir, ig, il, id) = (
        render_at(&gpu, &sr, &view, KERB_EXPOSURE),
        render_at(&gpu, &sg, &view, KERB_EXPOSURE),
        render_at(&gpu, &sl, &view, KERB_EXPOSURE),
        render_at(&gpu, &sd, &view, KERB_EXPOSURE),
    );
    let (ar, ag, al, ad) = (
        rgb_at(&ir, before_line),
        rgb_at(&ig, before_line),
        rgb_at(&il, before_line),
        rgb_at(&id, before_line),
    );
    let (hr, hg, hl, hd) = (
        rgb_at(&ir, shirt),
        rgb_at(&ig, shirt),
        rgb_at(&il, shirt),
        rgb_at(&id, shirt),
    );
    println!(
        "PAR1b SIGNAL (x{KERB_EXPOSURE}, {:.0} lm-row intensity {:.2}, range {:.0} m): asphalt before the line red {ar:.1?} / green {ag:.1?} / green lens alone {al:.1?} / dark {ad:.1?}; shirt in the box red {hr:.1?} / green {hg:.1?} / green lens alone {hl:.1?} / dark {hd:.1?}",
        inf_pcg::building::fixtures::fixture(FixtureRow::SignalHead).lumens,
        head.intensity,
        head.range_m
    );
    // The lens's own channel: red's R, green's G.
    let asphalt_red = ar[0] - ad[0];
    let asphalt_green = ag[1] - ad[1];
    let shirt_red = hr[0] - hd[0];
    let shirt_green = hg[1] - hd[1];
    let wall = DVec3::new(line_x - 3.0, 2.5, -10.45);
    let (wr, wg, wd) = (rgb_at(&ir, wall), rgb_at(&ig, wall), rgb_at(&id, wall));
    let (wall_red, wall_green) = (wr[0] - wd[0], wg[1] - wd[1]);
    println!(
        "PAR1b SIGNAL corner facade: red {wr:.1?} / green {wg:.1?} / dark {wd:.1?} (lift +{wall_red:.1} / +{wall_green:.1})"
    );
    assert!(
        wall_red <= SIGNAL_FACADE_MAX && wall_green <= SIGNAL_FACADE_MAX,
        "the corner facade takes the lens colour: red +{wall_red:.1}, green +{wall_green:.1}"
    );
    assert!(
        asphalt_red >= SIGNAL_ASPHALT_MIN && asphalt_green >= SIGNAL_ASPHALT_MIN,
        "the asphalt before the line does not read the lens at all: red +{asphalt_red:.1}, green +{asphalt_green:.1}"
    );
    assert!(
        asphalt_red <= SIGNAL_ASPHALT_MAX && asphalt_green <= SIGNAL_ASPHALT_MAX,
        "the junction is washed in the lens colour: red +{asphalt_red:.1}, green +{asphalt_green:.1} (max {SIGNAL_ASPHALT_MAX})"
    );
    assert!(
        shirt_red <= SIGNAL_SHIRT_MAX && shirt_green <= SIGNAL_SHIRT_MAX,
        "the shirt in the box takes the lens colour: red +{shirt_red:.1}, green +{shirt_green:.1} (max {SIGNAL_SHIRT_MAX})"
    );
}

/// The most the lens may lift its own channel on the corner facade beside the
/// approach, codes at x8 — the beam is down the lanes only (measured: 0.0 at
/// 40 lm / 14 degrees; 0.8 at the 100 lm / 24-degree cone the window showed
/// washing that facade green).
const SIGNAL_FACADE_MAX: f64 = 0.3;
/// The most the lens may lift its own channel on the asphalt before the line,
/// 8-bit codes at x8 — "faintly" (audit PAR1b).
const SIGNAL_ASPHALT_MAX: f64 = 6.0;
/// The least it must lift it — the lens is SEEN on its approach, codes at x8.
const SIGNAL_ASPHALT_MIN: f64 = 0.5;
/// The most the lens may lift its own channel on a pale shirt in the box,
/// codes at x8.
const SIGNAL_SHIRT_MAX: f64 = 2.0;

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

// ── the whole shipped island's census (off CI, prints) ─────────────────────

/// **THE SHIPPED ISLAND'S FURNITURE, COUNTED** (the census the brief asks for
/// beside the reference frames): every street the island's blocks imply, the
/// derivation over all of them (flat ground — counts do not depend on it), by
/// kind, per 100 m of city (20 m) and town (16 m) street, island totals, and the
/// signalised junctions nearest the hero's start with a place to stand at each
/// approach's stop line (for the frames). Asserts the island-wide placement
/// (audit PAR1b (i'), below); the counts it reports.
#[test]
#[ignore = "prints the shipped island's census; run by hand"]
fn the_shipped_islands_furniture_census() {
    let recipe = inf_island::IslandRecipe::load(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island/island.toml"),
    )
    .expect("the shipped recipe");
    let design = inf_island::read_design(&recipe).expect("the design");
    let streets = inf_editor_core::island::island_streets(&design);
    let junctions = inf_ecs::traffic::signal_junctions(&streets);
    let lines: Vec<street::StreetLine> = streets
        .iter()
        .map(|s| street::StreetLine {
            a: s.a,
            b: s.b,
            gap_m: s.gap_m,
        })
        .collect();
    let sites: Vec<street::SignalSite> = junctions
        .iter()
        .map(|j| street::SignalSite {
            centre: j.centre,
            gap_x: j.gap_x,
            gap_z: j.gap_z,
            offset_s: j.offset_s as u32,
        })
        .collect();
    let f = street::furnish(
        &lines,
        &sites,
        &[],
        &inf_pcg::FnHeight::new(|_, _| Some(0.0)),
    );
    let mut len_by_class: BTreeMap<u32, f64> = BTreeMap::new();
    for s in &streets {
        *len_by_class.entry(s.gap_m.round() as u32).or_default() += (s.b - s.a).length();
    }
    let total: f64 = len_by_class.values().sum();
    let mut by_kind: BTreeMap<PieceKind, usize> = BTreeMap::new();
    let (mut inst, mut solids, mut lights) = (0usize, 0usize, 0usize);
    for p in &f.pieces {
        *by_kind.entry(p.kind).or_default() += 1;
        inst += p.instances.len();
        solids += p.colliders.len();
        lights += p.lights.len();
    }
    println!(
        "PAR1b ISLAND CENSUS: {} streets, {:.0} m ({:?} m by reserve); {} signalised junctions; {} pieces, {inst} instances, {solids} solids, {lights} lights; refused at junctions {}, at doors {}",
        streets.len(),
        total,
        len_by_class.iter().map(|(k, v)| (*k, v.round())).collect::<Vec<_>>(),
        junctions.len(),
        f.pieces.len(),
        f.at_junction,
        f.at_door
    );
    for (k, n) in &by_kind {
        println!(
            "  {:<13} {n:>6}  {:>5.2} per 100 m of street",
            k.name(),
            *n as f64 / (total / 100.0)
        );
    }
    // ── audit PAR1b (i'): THE ISLAND-WIDE PLACEMENT, over every piece of the
    //    SHIPPED island (the gate's own arm reads the CI island's 33). Plan
    //    checks — the derivation's flat ground does not move a foot in plan:
    //    a foot in any street's CARRIAGEWAY (within its kerb face + the kerb
    //    stone; the parking slots lie in it), in any crossing's BOX (the two
    //    reserves' rectangle, crossings + junction fans), inside a BLOCK (the
    //    building plots), or within a metre of the hero's spawn; and every
    //    cable span's catenary sampled against the blocks (a span through a
    //    building) and against every lamp head and signal head (within 0.3 m).
    let rects = inf_editor_core::island::island_block_rects(&design);
    let mut bad: BTreeMap<(&'static str, &'static str), usize> = BTreeMap::new();
    let start0 = design.start(0.0);
    let heads: Vec<DVec3> = f
        .pieces
        .iter()
        .filter(|p| matches!(p.kind, PieceKind::LampPost | PieceKind::SignalPost))
        .flat_map(|p| p.lights.iter().map(|l| l.at))
        .collect();
    let mut checked = 0usize;
    for p in &f.pieces {
        checked += 1;
        let foot = DVec2::new(p.foot.x, p.foot.z);
        if p.kind != PieceKind::Span {
            for s in &streets {
                let (lo, hi, perp, along) = if s.along_x() {
                    (s.a.x.min(s.b.x), s.a.x.max(s.b.x), s.a.y, foot.x)
                } else {
                    (s.a.y.min(s.b.y), s.a.y.max(s.b.y), s.a.x, foot.y)
                };
                let lat = if s.along_x() {
                    foot.y - perp
                } else {
                    foot.x - perp
                };
                let face = street::kerb_offset_m(s.gap_m) + street::KERB_WIDTH_M;
                if along >= lo && along <= hi && lat.abs() < face {
                    *bad.entry((p.kind.name(), "carriageway")).or_default() += 1;
                }
            }
            for sx in streets.iter().filter(|s| s.along_x()) {
                for sz in streets.iter().filter(|s| !s.along_x()) {
                    let c = DVec2::new(sz.a.x, sx.a.y);
                    let (xl, xh) = (sx.a.x.min(sx.b.x), sx.a.x.max(sx.b.x));
                    let (zl, zh) = (sz.a.y.min(sz.b.y), sz.a.y.max(sz.b.y));
                    if !(c.x >= xl - 0.5 && c.x <= xh + 0.5 && c.y >= zl - 0.5 && c.y <= zh + 0.5) {
                        continue;
                    }
                    let (hx, hz) = (
                        street::kerb_offset_m(sz.gap_m) + street::KERB_WIDTH_M,
                        street::kerb_offset_m(sx.gap_m) + street::KERB_WIDTH_M,
                    );
                    if (foot.x - c.x).abs() < hx && (foot.y - c.y).abs() < hz {
                        *bad.entry((p.kind.name(), "crossing box")).or_default() += 1;
                    }
                }
            }
            for r in &rects {
                let d = (foot - r.centre).abs();
                if d.x < r.half.x && d.y < r.half.y {
                    *bad.entry((p.kind.name(), "block")).or_default() += 1;
                }
            }
            if (foot - DVec2::new(start0.x, start0.z)).length() < 1.0 {
                *bad.entry((p.kind.name(), "hero spawn")).or_default() += 1;
            }
        } else {
            for i in &p.instances {
                let c = DVec2::new(i.pos.x, i.pos.z);
                for r in &rects {
                    let d = (c - r.centre).abs();
                    if d.x < r.half.x && d.y < r.half.y {
                        *bad.entry(("cable span", "through a block")).or_default() += 1;
                    }
                }
                if heads.iter().any(|h| (*h - i.pos).length() < 0.3) {
                    *bad.entry(("cable span", "through a head")).or_default() += 1;
                }
            }
        }
    }
    println!(
        "PAR1b ISLAND PLACEMENT: {checked} pieces checked (spans by every segment); defects {:?}",
        bad
    );
    assert!(bad.is_empty(), "furniture where it must not stand: {bad:?}");
    // The nearest piece of each kind to the hero's start (for the frames).
    for k in PieceKind::ALL {
        if let Some(p) = f.pieces.iter().filter(|p| p.kind == k).min_by(|a, b| {
            let s = DVec3::new(start0.x, 0.0, start0.z);
            let (da, db) = (
                DVec3::new(a.foot.x, 0.0, a.foot.z) - s,
                DVec3::new(b.foot.x, 0.0, b.foot.z) - s,
            );
            da.length().total_cmp(&db.length())
        }) {
            println!(
                "PAR1b NEAREST {:<13} foot ({:.1}, {:.1})",
                k.name(),
                p.foot.x,
                p.foot.z
            );
        }
    }
    let start = design.start(0.0);
    let mut near: Vec<&inf_ecs::traffic::SignalJunction> = junctions.iter().collect();
    near.sort_by(|a, b| {
        (a.centre - DVec2::new(start.x, start.z))
            .length()
            .total_cmp(&(b.centre - DVec2::new(start.x, start.z)).length())
    });
    println!("PAR1b START {:.1} {:.1} {:.1}", start.x, start.y, start.z);
    for j in near.iter().take(6) {
        let d = (j.centre - DVec2::new(start.x, start.z)).length();
        // A place 12 m back along the +X approach, on its lane, facing +X.
        let stand = DVec2::new(j.centre.x - 14.0, j.centre.y - 1.75);
        println!(
            "PAR1b JUNCTION at ({:.1}, {:.1}) gaps {}x{} offset {:.0} s, {d:.0} m from the start; stand on the +X approach at ({:.1}, {:.1}) facing yaw 90",
            j.centre.x, j.centre.y, j.gap_x, j.gap_z, j.offset_s, stand.x, stand.y
        );
    }
}

/// **THE SPAWN JUNCTION OBEYS ITS SIGNAL, ON THE SHIPPED ISLAND** (audit
/// PAR1b — the world arm's single wait over 2 400 steps on a synthetic grid
/// was vacuous). The cooked island (`INF_ISLAND_PACK`), the hero standing at
/// the signalised crossing it spawns at (-1750, 2050), the morning rush
/// (08:24), a whole 50 s cycle ([`SPAWN_STEPS`]) after a warm-up that lets the
/// traffic plan its day: every traffic car within 64 m of the crossing is
/// watched, steered (`Full`) and clock-moved (`Near`) alike. Asserts:
///
/// * at least [`SPAWN_MIN_WAITS`] distinct cars stand at a red line there
///   (steered: at rest, nose within 4 m short of / 1 m past the line; clock: held
///   by `clock_signal_hold`), and at least [`SPAWN_MIN_HOLD_STEPS`] clock-tier
///   hold-steps island-wide;
/// * ZERO steps on which two cars' footprints overlap inside the crossing's box;
/// * ZERO stop lines crossed on a red by a car that was not already over it.
///
/// Off CI (CI never cooks the island): skips without `INF_ISLAND_PACK`.
#[test]
#[ignore = "needs a cooked island pack (INF_ISLAND_PACK); run by hand"]
fn the_spawn_junction_holds_its_traffic_at_a_red_and_keeps_its_box_clear() {
    let Some(pack) = std::env::var_os("INF_ISLAND_PACK").map(PathBuf::from) else {
        println!("SKIP: INF_ISLAND_PACK names no pack");
        return;
    };
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Pack(source));
    let centre = DVec2::new(-1750.0, 2050.0);
    let ground = sim.terrain_height_at(centre.x, centre.y);
    stand(
        &mut sim,
        DVec3::new(centre.x + 9.0, ground, centre.y + 9.0),
        8.4,
        SPAWN_WARMUP,
    );
    let Some(j) = inf_ecs::traffic::carriageway_of(sim.world()).and_then(|r| {
        r.junctions
            .iter()
            .find(|j| (j.centre - centre).length() < 1.0)
            .copied()
    }) else {
        panic!("the spawn crossing is not signalised");
    };
    let (hx, hz) = (
        inf_ecs::traffic::street_kerb_offset_m(j.gap_z),
        inf_ecs::traffic::street_kerb_offset_m(j.gap_x),
    );
    let mut waits: std::collections::BTreeSet<uuid::Uuid> = Default::default();
    let mut holds = 0usize;
    let mut box_contacts = 0usize;
    let mut red_crossings = 0usize;
    let mut watched: std::collections::BTreeSet<uuid::Uuid> = Default::default();
    let mut moving: std::collections::BTreeSet<uuid::Uuid> = Default::default();
    let mut noses: BTreeMap<uuid::Uuid, f64> = BTreeMap::new();
    for _ in 0..SPAWN_STEPS {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        holds += sim.traffic_stats().signal_holds;
        let t_s = inf_ecs::traffic::signal_clock_of(sim.world());
        let recs = inf_physics::d3::traffic::records(sim.world());
        let mut in_box: Vec<(DVec3, f64)> = Vec::new();
        let mut next: BTreeMap<uuid::Uuid, f64> = BTreeMap::new();
        for (g, rec) in &recs {
            if rec.taken {
                continue;
            }
            let at = sim
                .world()
                .entity_of(*g)
                .and_then(|e| sim.world().world().get::<inf_ecs::components::Transform>(e))
                .map(|t| t.translation.to_dvec3())
                .unwrap_or(rec.last);
            if DVec2::new(at.x - centre.x, at.z - centre.y).length() > 64.0 {
                continue;
            }
            watched.insert(*g);
            if (rec.last - at).length() > 0.0 || rec.signal_held {
                moving.insert(*g);
            }
            let half = rec.def.half_extents.z.abs();
            if (at.x - centre.x).abs() < hx && (at.z - centre.y).abs() < hz {
                in_box.push((at, (half * half + rec.def.half_extents.x.powi(2)).sqrt()));
            }
            let yaw = rec.yaw_deg.to_radians();
            let fwd = DVec3::new(inf_math::psin64(yaw), 0.0, inf_math::pcos64(yaw));
            let along_x = fwd.x.abs() >= fwd.z.abs();
            let (d_along, d_lat) = if along_x {
                ((centre.x - at.x) * fwd.x.signum(), (centre.y - at.z).abs())
            } else {
                ((centre.y - at.z) * fwd.z.signum(), (centre.x - at.x).abs())
            };
            if d_lat > inf_ecs::traffic::SIGNAL_LANE_REACH_M || !(0.0..=40.0).contains(&d_along) {
                continue;
            }
            let nose = d_along - half - inf_ecs::traffic::STOP_LINE_M;
            let red =
                inf_ecs::traffic::signal_aspect(&j, along_x, t_s) == inf_ecs::traffic::Aspect::Red;
            if let Some(prev) = noses.get(g) {
                if *prev > 0.0 && nose <= 0.0 && red {
                    red_crossings += 1;
                }
            }
            next.insert(*g, nose);
            let still = rec.signal_held
                || (rec.tier == inf_ecs::crowd::CrowdTier::Full
                    && (rec.last - at).length() < 0.01
                    && (-1.0..=4.0).contains(&nose));
            if still && red {
                waits.insert(*g);
            }
        }
        noses = next;
        let mut touched = false;
        for a in 0..in_box.len() {
            for b in a + 1..in_box.len() {
                let d = DVec2::new(in_box[a].0.x - in_box[b].0.x, in_box[a].0.z - in_box[b].0.z)
                    .length();
                touched |= d < 0.75 * (in_box[a].1 + in_box[b].1);
            }
        }
        box_contacts += usize::from(touched);
    }
    println!(
        "PAR1b SPAWN JUNCTION ({:.0}, {:.0}) offset {:.0} s: {} cars watched within 64 m ({} of them moved), {} distinct waited at a red, {} clock hold-steps island-wide, {} box-contact steps, {} red line crossings over {SPAWN_STEPS} steps",
        j.centre.x,
        j.centre.y,
        j.offset_s,
        watched.len(),
        moving.len(),
        waits.len(),
        holds,
        box_contacts,
        red_crossings
    );
    assert!(
        waits.len() >= SPAWN_MIN_WAITS,
        "only {} cars waited at the spawn crossing's reds",
        waits.len()
    );
    assert!(
        holds >= SPAWN_MIN_HOLD_STEPS,
        "only {holds} clock-tier hold-steps island-wide"
    );
    assert_eq!(box_contacts, 0, "two cars met in the spawn crossing's box");
    assert_eq!(red_crossings, 0, "a car crossed a stop line on a red");
}

/// **THE HITCH PROFILE** (audit PAR1b (a')): is re-furnishing on a cell
/// activation what the window's frames > 50 ms are? The cooked island, the hero
/// carried down the strip from the spawn at a walk (1.4 m/s) for 60 s of sim,
/// every step timed; a step whose furniture census changed (a re-furnish that
/// moved pieces) is filed apart, and so is a step that activated a cell. With
/// `INF_NO_STREET_FURNITURE` the same run is the control. REPORTS, never
/// asserts.
#[test]
#[ignore = "needs a cooked island pack (INF_ISLAND_PACK); run by hand"]
fn the_walk_down_the_strip_files_its_slow_steps() {
    let Some(pack) = std::env::var_os("INF_ISLAND_PACK").map(PathBuf::from) else {
        println!("SKIP: INF_ISLAND_PACK names no pack");
        return;
    };
    let source = inf_player::level::PackLevelSource::open(&pack).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Pack(source));
    let start = DVec2::new(-1750.0, 2050.0);
    let g = sim.terrain_height_at(start.x, start.y);
    stand(&mut sim, DVec3::new(start.x, g, start.y), 21.0, 300);
    let mut times: Vec<(f64, bool, bool)> = Vec::new();
    let mut census = inf_ecs::furniture::census(sim.world());
    let mut acts = sim.cell_streaming().stats().activations;
    for k in 0..3600 {
        // Down the street along -Z at a walk, then along +X past the next
        // crossing: the cells a walk crosses.
        let s = k as f64 * 1.4 / 60.0;
        let p = if s < 50.0 {
            DVec2::new(start.x, start.y - s)
        } else {
            DVec2::new(start.x + (s - 50.0), start.y - 50.0)
        };
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
                t.translation.x = p.x;
                t.translation.z = p.y;
            }
        }
        let t0 = std::time::Instant::now();
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let now = inf_ecs::furniture::census(sim.world());
        let a = sim.cell_streaming().stats().activations;
        times.push((ms, now != census, a != acts));
        census = now;
        acts = a;
    }
    let mut all: Vec<f64> = times.iter().map(|t| t.0).collect();
    all.sort_by(f64::total_cmp);
    let p = |v: &[f64], q: f64| v[((v.len() - 1) as f64 * q) as usize];
    let refurnished: Vec<f64> = times.iter().filter(|t| t.1).map(|t| t.0).collect();
    let activated: Vec<f64> = times.iter().filter(|t| t.2).map(|t| t.0).collect();
    let slow = times.iter().filter(|t| t.0 > 25.0).count();
    let slow_ref = times.iter().filter(|t| t.0 > 25.0 && t.1).count();
    let slow_act = times.iter().filter(|t| t.0 > 25.0 && t.2).count();
    println!(
        "PAR1b HITCH PROFILE (furniture {}): {} steps, p50 {:.2} p95 {:.2} p99 {:.2} max {:.2} ms; {} steps re-furnished (max {:.2} ms, sum {:.1}); {} steps activated a cell (max {:.2} ms, sum {:.1}); steps > 25 ms: {slow} ({slow_ref} re-furnished, {slow_act} activated); final census {:?}",
        if std::env::var_os("INF_NO_STREET_FURNITURE").is_some() { "OFF" } else { "on" },
        all.len(),
        p(&all, 0.5),
        p(&all, 0.95),
        p(&all, 0.99),
        all[all.len() - 1],
        refurnished.len(),
        refurnished.iter().cloned().fold(0.0, f64::max),
        refurnished.iter().sum::<f64>(),
        activated.len(),
        activated.iter().cloned().fold(0.0, f64::max),
        activated.iter().sum::<f64>(),
        census
    );
}

/// Steps the shipped-island signal arm warms the traffic up over before it
/// watches (the day plans `TRAFFIC_PLANS_PER_STEP` routes a step).
const SPAWN_WARMUP: usize = 240;
/// Steps it watches — ten seconds, a fifth of a cycle each way.
const SPAWN_STEPS: usize = 3000;
/// The distinct waits it asks for at the spawn crossing — the measured count
/// (audit PAR1b, `cook-h/perf1`): ONE car over a whole 50 s cycle, because a
/// frozen level clock (the arm's, and `INF_PIE_HOUR`'s) freezes every commute
/// where it stands and only the circuits move. The island-wide clock holds
/// (2 459 hold-steps measured) are the arm's real engagement and are asserted
/// beside it.
const SPAWN_MIN_WAITS: usize = 1;
/// The island-wide clock-tier hold-steps the run must show (a tenth of the
/// 2 459 measured).
const SPAWN_MIN_HOLD_STEPS: usize = 245;
