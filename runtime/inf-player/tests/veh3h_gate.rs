//! **WAVE VEH3h -- THE DRIVING CERT GATE**: the gate.
//!
//! A certification wave measures the arc it closes. This file carries the arms
//! that make `docs/memos/driving-parity.md` a set of CITATIONS rather than a set
//! of sentences: every arm the memo cites exists and is not ignored; the lap of
//! the island's circuit logs its telemetry at 60 Hz on the SHIPPED host with
//! the shape the plot reads, and both hosts drive it bit-identically; three
//! classes are re-measured against the published 0-100 figures of their
//! real-world inspirations; and the memo's own numbers are the numbers the arms
//! print.
//!
//! Every arm's header says what it READS and whether a tree with no VEH3 arc
//! would pass it.

use std::path::{Path, PathBuf};

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{CharacterMovement, MovementMode, Transform};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::VehicleDef;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

// ── the CI island, both hosts (`island_gate`'s harness, verbatim) ───────────

/// The recipe CI builds.
fn fixture_recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../samples/island-fixture/island.toml")
}

/// Build the island's project through `inf island build`'s own door.
fn build_project(tmp: &Path) -> PathBuf {
    let recipe =
        inf_island::IslandRecipe::load(&fixture_recipe()).expect("the fixture recipe loads");
    let build = inf_island::build_island(&recipe, &inf_island::BuildOptions::default())
        .expect("the fixture island builds");
    let proj = tmp.join("island");
    inf_project::ProjectManifest::new(&recipe.name, "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    inf_island::write_content(&build, &proj.join("Content")).expect("the island's content writes");
    proj
}

/// Cook it, exactly as `inf cook` does.
fn cook(proj: &Path, out: &Path) -> PathBuf {
    inf_packager::cook(proj, out, &inf_packager::CookOptions::default()).expect("the island cooks");
    out.to_path_buf()
}

/// **The shipping side**: a sim off the cooked pack, with both streamers, as
/// `inf_player::run_headless` boots it.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(source.clone()),
    );
    sim
}

/// **The editor side**: the loose level, built the way `island_gate::loose_sim`
/// builds it (the same world to disagree about).
fn loose_sim(content: &Path, slug: &str) -> RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join(format!("{slug}.inf_lvl")));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_bindings(inf_player::level::load_actor_classes_by_guid_from_dir(
        content,
    ))
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
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

// ── the lap ─────────────────────────────────────────────────────────────────

/// The car the lap is driven in: a guid no level mints.
const LAP_CAR: Uuid = Uuid::from_u128(0x5645_4833_4c41_5000_0000_0000_0000_0001);

/// The row the CI lap drives -- a SHELL muscle coupe with a turbocharger and
/// rear drive (the doc's Dodge Challenger SRT Demon, `shell_coupe`): with the
/// traction aid off, a full-throttle launch spins its rear axle (the burnout
/// heats the tyres), the throttle spools the compressor, and the brakes move
/// load onto the front axle -- all three on one car. (The first choice, the
/// AWD `annis_elegy_rh8`, never let its tyres go: +0.06 C in the burnout.)
const LAP_ROW: &str = "bravado_gauntlet_hellfire";

/// The lap's speed limit on the straights, m/s -- the driver's (`traffic::
/// drive_intent`'s) own bend rule decides every corner.
const LAP_LIMIT_MPS: f64 = 22.0;

/// Steps of the burnout at the line: three seconds.
const BURNOUT_STEPS: usize = 180;

/// The lap's own ceiling, steps (three minutes): a car that never gets round is
/// a red arm, not a hang.
const LAP_MAX_STEPS: usize = 10_800;

/// One 60 Hz sample of the lap.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct LapRow {
    t: f64,
    /// 0 boarding, 1 burnout, 2 lap, 3 cool-down straight.
    phase: u8,
    x: f64,
    z: f64,
    s_m: f64,
    speed: f64,
    long_g: f64,
    lat_g: f64,
    throttle: f64,
    brake: f64,
    rpm: f64,
    gear: i32,
    boost: f64,
    slip: [f64; 4],
    slip_lat: [f64; 4],
    load: [f64; 4],
    temp: [f64; 4],
    /// The surface census of the four contacts (`vehicle::surface_census`).
    surface: &'static str,
}

/// The CSV's header -- the SHAPE the plot reads and `the_lap_logs_at_sixty_
/// hertz_...` pins.
const LAP_COLUMNS: [&str; 30] = [
    "t",
    "phase",
    "x",
    "z",
    "s_m",
    "speed_mps",
    "long_g",
    "lat_g",
    "throttle",
    "brake",
    "rpm",
    "gear",
    "boost",
    "slip_fl",
    "slip_fr",
    "slip_rl",
    "slip_rr",
    "slip_lat_fl",
    "slip_lat_fr",
    "slip_lat_rl",
    "slip_lat_rr",
    "load_fl",
    "load_fr",
    "load_rl",
    "load_rr",
    "temp_fl",
    "temp_fr",
    "temp_rl",
    "temp_rr",
    "surface",
];

impl LapRow {
    fn csv(&self) -> String {
        let mut s = format!(
            "{:.4},{},{:.3},{:.3},{:.2},{:.3},{:.4},{:.4},{:.3},{:.3},{:.1},{},{:.4}",
            self.t,
            self.phase,
            self.x,
            self.z,
            self.s_m,
            self.speed,
            self.long_g,
            self.lat_g,
            self.throttle,
            self.brake,
            self.rpm,
            self.gear,
            self.boost
        );
        for v in self.slip.iter().chain(&self.slip_lat) {
            s.push_str(&format!(",{v:.4}"));
        }
        for v in &self.load {
            s.push_str(&format!(",{v:.1}"));
        }
        for v in &self.temp {
            s.push_str(&format!(",{v:.3}"));
        }
        s.push(',');
        s.push_str(self.surface);
        s
    }
}

/// One host's lap.
struct Lap {
    rows: Vec<LapRow>,
    /// A digest of `RuntimeSim::state_bytes` every step -- what PIE == shipping
    /// compares.
    digests: Vec<u64>,
    /// Where the circuit runs: its corners.
    corners: Vec<DVec3>,
    length_m: f64,
    completed: bool,
    /// The level's own vehicles the lap set aside (see `clear_circuit`).
    set_aside: usize,
}

/// FNV-1a over a state -- a digest, compared between two hosts.
fn digest(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn hero_guid(sim: &RuntimeSim) -> Uuid {
    inf_ecs::movement::camera_subject(sim.world()).expect("the island has a hero")
}

fn hero_mode(sim: &RuntimeSim) -> Option<(MovementMode, inf_ecs::boarding::BoardPhase)> {
    let w = sim.world();
    let e = w.entity_of(hero_guid(sim))?;
    w.world()
        .get::<CharacterMovement>(e)
        .map(|m| (m.mode, m.runtime.boarding.phase))
}

fn put_hero(sim: &mut RuntimeSim, at: DVec3) {
    let hero = hero_guid(sim);
    if let Some(e) = sim.world().entity_of(hero) {
        if let Some(mut t) = sim.world_mut().world_mut().get_mut::<Transform>(e) {
            t.translation = Vec3d::from_dvec3(at);
        }
    }
    if let Some(b) = sim.bridge3d().body_of(hero) {
        sim.bridge3d_mut().world_mut().set_body_translation(b, at);
    }
}

/// **The circuits a settlement offers**: the street centreline round each
/// 2 x 2 group of its planned blocks -- the blocks' own rectangles grown by
/// half a street -- nearest the settlement's centre first, each clockwise
/// seen from above and starting half way down its south side. Answered with
/// the settlement's centre.
fn circuits(design: &inf_island::IslandDesign) -> (DVec2, Vec<Vec<DVec2>>) {
    let mut plans = inf_editor_core::settlement::settlements(design);
    plans.sort_by(|a, b| {
        b.blocks
            .len()
            .cmp(&a.blocks.len())
            .then(a.name.cmp(&b.name))
    });
    let s = plans
        .into_iter()
        .next()
        .expect("the fixture has a settlement");
    let mut found: Vec<(f64, [f64; 4])> = Vec::new();
    for (w, h) in [(2, 2), (2, 1), (1, 2), (1, 1)] {
        for b in &s.blocks {
            let group: Vec<_> = s
                .blocks
                .iter()
                .filter(|o| {
                    (b.col..b.col + w).contains(&o.col) && (b.row..b.row + h).contains(&o.row)
                })
                .collect();
            if group.len() != (w * h) as usize {
                continue;
            }
            let (mut x0, mut x1, mut z0, mut z1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
            for o in &group {
                x0 = x0.min(o.centre.x - o.half.x);
                x1 = x1.max(o.centre.x + o.half.x);
                z0 = z0.min(o.centre.y - o.half.y);
                z1 = z1.max(o.centre.y + o.half.y);
            }
            let c = DVec2::new((x0 + x1) * 0.5, (z0 + z1) * 0.5);
            // Bigger groups first, then nearest the centre.
            found.push((
                f64::from(4 - w * h) * 1.0e6 + (c - s.centre).length(),
                [x0, x1, z0, z1],
            ));
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    let g = s.street_m * 0.5;
    let loops = found
        .into_iter()
        .map(|(_, [x0, x1, z0, z1])| {
            let (x0, x1, z0, z1) = (x0 - g, x1 + g, z0 - g, z1 + g);
            vec![
                DVec2::new((x0 + x1) * 0.5, z0),
                DVec2::new(x1, z0),
                DVec2::new(x1, z1),
                DVec2::new(x0, z1),
                DVec2::new(x0, z0),
                DVec2::new((x0 + x1) * 0.5, z0),
            ]
        })
        .collect();
    (s.centre, loops)
}

/// **The circuit, with the town streamed in and its parked vehicles set
/// aside**: the hero stands at the settlement's centre for ten seconds (both
/// streamers, the cells that carry the town's authored vehicles), then every
/// resident rig is taken out of the world -- the crowd's and the traffic's
/// reason: a scripted driver has no eyes. (The CI island parks a camp car on
/// the town's centre street and a fire appliance on its lane; the appliance
/// was the cert's finding that a parked vehicle beyond the 64 m collider band
/// rolled off -- 36 m in 15 s with the hero 120 m away -- closed by the VEH3h
/// audit's parking hold, see `a_parked_vehicle_beyond_the_collider_band_holds_its_pad`.)
/// Answers the largest circuit and how many vehicles were set aside.
fn clear_circuit(sim: &mut RuntimeSim, design: &inf_island::IslandDesign) -> (Vec<DVec2>, usize) {
    let (centre, loops) = circuits(design);
    let y0 = sim.terrain_height_at(centre.x, centre.y);
    for _ in 0..600 {
        put_hero(sim, DVec3::new(centre.x, y0 + 1.5, centre.y));
        sim.step_once(RuntimeInput::default());
    }
    let rigs: Vec<Uuid> = {
        let w = sim.world();
        w.world()
            .iter_entities()
            .filter_map(|e| e.get::<inf_ecs::Guid>().map(|g| g.0))
            .filter(|g| inf_ecs::vehicle::rig_of(w, *g).is_some())
            .collect()
    };
    for g in &rigs {
        if let Some(e) = sim.world().entity_of(*g) {
            sim.world_mut().despawn(e);
        }
    }
    sim.world_mut().mark_dirty();
    // **The first circuit whose line stays on graded ground** (the VEH3h
    // audit). The cert drove the settlement's largest circuit, and its west
    // street crosses a TERRACE BANK -- the terrain itself rises 1.5 m at
    // ~39 deg between z -262 and -264 on x 354, where two levelled block rows
    // meet -- so the lap car struck an earth face at 13.8 m/s (-36.75 g for
    // one step, the fronts unloaded). It is neither a kerb nor a collider
    // seam (the ray under it answers `Terrain` all the way up); it is a line
    // a road car should not be driven on, so the lap's line moves.
    let mut chosen = None;
    for (k, l) in loops.iter().enumerate() {
        let grade = steepest_grade(sim, l);
        println!(
            "LAP CANDIDATE {k}: the steepest 0.5 m of its line climbs {grade:.3} (rise over run)"
        );
        if chosen.is_none() && grade <= LAP_MAX_GRADE {
            chosen = Some(l.clone());
        }
    }
    (chosen.expect("a circuit on graded ground"), rigs.len())
}

/// The steepest rise over run the lap's line may meet anywhere, sampled
/// every half metre -- 0.20 (11.3 deg). The island's graded streets read
/// under it; the terrace bank the cert's lap struck reads 0.8 (39 deg).
const LAP_MAX_GRADE: f64 = 0.20;

/// The steepest half metre of a closed line's terrain, rise over run.
fn steepest_grade(sim: &mut RuntimeSim, pts: &[DVec2]) -> f64 {
    let mut worst = 0.0f64;
    for w in pts.windows(2) {
        let (a, b) = (w[0], w[1]);
        let n = ((b - a).length() / 0.5).ceil().max(1.0) as usize;
        let mut prev = sim.terrain_height_at(a.x, a.y);
        for i in 1..=n {
            let p = a + (b - a) * (i as f64 / n as f64);
            let h = sim.terrain_height_at(p.x, p.y);
            let run = (b - a).length() / n as f64;
            worst = worst.max((h - prev).abs() / run.max(1e-9));
            prev = h;
        }
    }
    worst
}

/// A step's telemetry off the car the hero is driving -- the bridge's own
/// `WheelState` and `DrivetrainState`, the rapier body's velocity.
fn sample(
    sim: &RuntimeSim,
    prev_v: DVec3,
    t: f64,
    phase: u8,
    s_m: f64,
    input: (f64, f64),
) -> (LapRow, DVec3) {
    let bridge = sim.bridge3d();
    let body = bridge.body_of(LAP_CAR).expect("the lap car has a body");
    let w = bridge.world();
    let at = w.body_translation(body).unwrap_or(DVec3::ZERO);
    let rot = w.body_rotation(body).unwrap_or(glam::DQuat::IDENTITY);
    let v = w.body_linvel(body).unwrap_or(DVec3::ZERO);
    let fwd = rot * DVec3::Z;
    let right = rot * DVec3::X;
    let a = (v - prev_v) * 60.0;
    let veh = bridge
        .vehicle_of(LAP_CAR)
        .expect("the lap car is a vehicle");
    let wheels = veh.wheels();
    let dt = veh.drivetrain().unwrap_or_default();
    let mut row = LapRow {
        surface: inf_ecs::vehicle::surface_census(wheels).0,
        t,
        phase,
        x: at.x,
        z: at.z,
        s_m,
        speed: v.dot(fwd),
        long_g: a.dot(fwd) / 9.81,
        lat_g: a.dot(right) / 9.81,
        throttle: input.0,
        brake: input.1,
        rpm: dt.rpm,
        gear: dt.gear,
        boost: dt.boost,
        ..Default::default()
    };
    // The columns are CORNERS, named off each wheel's own mount on the rig
    // (front: +Z; left: -X, the engine's handedness -- `inf_nav::lane::
    // right_of`), not off the rig's wheel order.
    for (wh, mount) in wheels.iter().zip(&veh.rig().wheels).take(4) {
        let m = mount.mount_local;
        let i = if m.z > 0.0 { 0 } else { 2 } + usize::from(m.x > 0.0);
        row.slip[i] = wh.slip_ratio;
        row.slip_lat[i] = wh.slip_lat;
        row.load[i] = wh.load_n;
        row.temp[i] = wh.temp_c;
    }
    (row, v)
}

/// **Drive the lap on one host**: the car spawned at the circuit's start
/// through the traffic's own rig door, the hero stood beside the driver's door
/// and boarded through the shipped interact press (VEH3d's pipeline), a
/// three-second burnout at the line, one lap driven by the engine's own lane
/// driver (`inf_ecs::traffic::drive_intent`, the stick a traffic car holds) on
/// a loop path round the circuit, then a cool-down straight. Every step's
/// `state_bytes` is digested.
fn drive_lap(sim: &mut RuntimeSim, design: &inf_island::IslandDesign, row: &str) -> Lap {
    use inf_ecs::movement::actions::{HANDBRAKE, INTERACT, MOVE_X, MOVE_Y};
    // A lap is a telemetry instrument: the crowd and the traffic are set aside
    // (a scripted driver has no eyes), everything else -- the ground, the
    // buildings, both streamers -- is the shipped island.
    sim.set_crowd_population(Default::default());
    sim.set_traffic_population(Default::default());
    let (pts, set_aside) = clear_circuit(sim, design);
    let y_at = |sim: &mut RuntimeSim, p: DVec2| sim.terrain_height_at(p.x, p.y);
    let mut corners = Vec::new();
    for p in &pts {
        let y = y_at(sim, *p);
        corners.push(DVec3::new(p.x, y, p.y));
    }
    let path = inf_nav::NavPath::new(corners.clone());
    let def: VehicleDef = *inf_ecs::roster::roster()
        .get(row)
        .unwrap_or_else(|| panic!("the roster has no `{row}`"));
    // The car on the start line, pointing down the first leg (+X).
    let start = corners[0];
    let ground = start.y;
    let at = DVec3::new(
        start.x,
        inf_ecs::vehicle::resting_origin_y(&def, ground),
        start.z,
    );
    let yaw = 90.0; // +Z rotated to +X.
    inf_ecs::vehicle::spawn_rig_at(
        sim.world_mut(),
        LAP_CAR,
        &def,
        &inf_ecs::vehicle::RigSpawn {
            name: "Lap Car".into(),
            at,
            yaw_deg: yaw,
            paint: inf_ecs::math::Color::new(0.8, 0.1, 0.1, 1.0),
            clip: None,
            engine_voice: true,
            livery: None,
        },
        true,
    );
    sim.world_mut().propagate();
    sim.world_mut().mark_dirty();
    let mut digests = Vec::new();
    let mut rows = Vec::new();
    let step = |sim: &mut RuntimeSim, input: RuntimeInput, digests: &mut Vec<u64>| {
        sim.step_once(input);
        digests.push(digest(&sim.state_bytes()));
    };
    for _ in 0..30 {
        step(sim, RuntimeInput::default(), &mut digests);
    }
    // ── board: stand at the driver's flank, land, one press.
    let seat = inf_physics::d3::vehicle::seat_pose(sim.bridge3d(), LAP_CAR)
        .expect("the lap car has a seat")
        .0;
    let rot = glam::DQuat::from_rotation_y(yaw.to_radians());
    let beside = seat + rot * DVec3::new(1.7, 0.6, 0.0);
    for _ in 0..24 {
        put_hero(sim, beside);
        step(sim, RuntimeInput::default(), &mut digests);
    }
    for _ in 0..120 {
        if hero_mode(sim).is_some_and(|(m, _)| m.is_grounded_family()) {
            break;
        }
        step(sim, RuntimeInput::default(), &mut digests);
    }
    step(sim, RuntimeInput::default(), &mut digests);
    step(sim, RuntimeInput::default().press(INTERACT), &mut digests);
    let mut boarded = false;
    for _ in 0..900 {
        step(sim, RuntimeInput::default(), &mut digests);
        if hero_mode(sim).is_some_and(|(m, p)| {
            m == MovementMode::Driving && p == inf_ecs::boarding::BoardPhase::Driving
        }) {
            boarded = true;
            break;
        }
    }
    assert!(
        boarded,
        "the hero never reached the wheel: {:?}",
        hero_mode(sim)
    );
    let mut t = 0.0f64;
    let mut prev_v = DVec3::ZERO;
    // ── the burnout: the handbrake locks the rear axle and the throttle is
    //    wide open -- the front axle is the one that is driven and held by
    //    nothing.
    // The traction aid off for the burnout -- the driver's own button on a
    // real car; here the vehicle's tuning door (`Vehicle::tune`, the one
    // `INF_PIE_TUNE_VEHICLE` reaches), on both hosts alike. Back on after.
    if let Some(v) = sim.bridge3d_mut().vehicle_mut(LAP_CAR) {
        v.tune("traction_control_slip", 0.0);
    }
    for _ in 0..BURNOUT_STEPS {
        let input = RuntimeInput::default().axis_at(MOVE_Y, 1.0);
        step(sim, input, &mut digests);
        t += 1.0 / 60.0;
        let (row, v) = sample(sim, prev_v, t, 1, 0.0, (1.0, 0.0));
        prev_v = v;
        rows.push(row);
    }
    if let Some(v) = sim.bridge3d_mut().vehicle_mut(LAP_CAR) {
        v.tune("traction_control_slip", def.class.traction_control_slip);
    }
    // ── the lap: the lane driver's stick, round the loop once.
    let len = path.length_m();
    let mut travelled = 0.0f64;
    let mut last_s = path.project(at).s_m;
    let mut completed = false;
    for _ in 0..LAP_MAX_STEPS {
        let bridge = sim.bridge3d();
        let body = bridge.body_of(LAP_CAR).expect("a body");
        let w = bridge.world();
        let here = w.body_translation(body).unwrap_or(DVec3::ZERO);
        let rot = w.body_rotation(body).unwrap_or(glam::DQuat::IDENTITY);
        let lin = w.body_linvel(body).unwrap_or(DVec3::ZERO);
        let forward = rot * DVec3::Z;
        let s_m = path.project(here).s_m;
        let view = inf_ecs::traffic::DriveView {
            at: here,
            forward,
            forward_mps: lin.dot(forward),
            path: &path,
            s_m,
            speed_limit_mps: LAP_LIMIT_MPS,
            gap_m: None,
            lateral_bias_m: 0.0,
            loops: true,
        };
        let intent = inf_ecs::traffic::drive_intent(&view);
        let (steer, fwd) = (intent.move_input.x, intent.move_input.y);
        let mut input = RuntimeInput::default()
            .axis_at(MOVE_X, steer as f32)
            .axis_at(MOVE_Y, fwd as f32);
        if intent.handbrake {
            input = input.press(HANDBRAKE);
        }
        step(sim, input, &mut digests);
        t += 1.0 / 60.0;
        let mut ds = s_m - last_s;
        if ds < -len * 0.5 {
            ds += len;
        }
        if ds > 0.0 && ds < 5.0 {
            travelled += ds;
        }
        last_s = s_m;
        let (row, v) = sample(sim, prev_v, t, 2, s_m, (fwd.max(0.0), (-fwd).max(0.0)));
        prev_v = v;
        rows.push(row);
        if travelled >= len {
            completed = true;
            break;
        }
    }
    // ── the cool-down: straight on, a light throttle, for five seconds.
    for _ in 0..300 {
        let input = RuntimeInput::default().axis_at(MOVE_Y, 0.25);
        step(sim, input, &mut digests);
        t += 1.0 / 60.0;
        let (row, v) = sample(sim, prev_v, t, 3, 0.0, (0.25, 0.0));
        prev_v = v;
        rows.push(row);
    }
    Lap {
        rows,
        digests,
        corners,
        length_m: len,
        completed,
        set_aside,
    }
}

// ── the tier ladder, priced per car ─────────────────────────────────────────

/// A flat world with a streaming anchor at the origin (the band's hero), and
/// `n` parked saloons of the island's catalogue on an 8 x 8 grid centred on
/// `centre` -- as TRAFFIC records, so the tier ladder decides what each one is.
fn ladder_sim(n: usize, centre: DVec3) -> RuntimeSim {
    use inf_ecs::components::{
        BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, StreamingSource,
    };
    let mut world = inf_ecs::EcsWorld::new();
    let g = world.spawn_with_guid(
        Uuid::from_u128(0x5645_4833_4c41_4400_0000_0000_0000_0001),
        "Ground",
        None,
    );
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(300.0, -0.5, 0.0);
    world.world_mut().entity_mut(g).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(700.0, 0.5, 200.0),
            friction: 0.9,
            ..Default::default()
        },
    ));
    let a = world.spawn_with_guid(
        Uuid::from_u128(0x5645_4833_4c41_4400_0000_0000_0000_0002),
        "Anchor",
        None,
    );
    world
        .world_mut()
        .entity_mut(a)
        .insert((StreamingSource { radius_m: 512.0 }, Transform::IDENTITY));
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
    let def = *inf_editor_core::vehicle::island_vehicles()
        .get("sedan")
        .expect("the island saloon");
    let y = inf_ecs::vehicle::resting_origin_y(&def, 0.0);
    let mut records = std::collections::BTreeMap::new();
    for k in 0..n {
        let at = DVec3::new(
            centre.x + (k % 8) as f64 * 6.0 - 21.0,
            y,
            centre.z + (k / 8) as f64 * 8.0 - 28.0,
        );
        records.insert(
            Uuid::from_u128(0x5645_4833_4c41_4400_0000_0000_0001_0000 + k as u128),
            inf_ecs::traffic::TrafficRecord::parked(
                def,
                inf_ecs::math::Color::new(0.3, 0.3, 0.3, 1.0),
                at,
                0.0,
            ),
        );
    }
    sim.set_traffic_population(records);
    sim
}

/// The step, min of five rounds of the mean over sixty steps after a settle.
fn ladder_step_ms(sim: &mut RuntimeSim) -> inf_player::step_profile::StepProfile {
    for _ in 0..120 {
        sim.step_once(RuntimeInput::default());
    }
    sim.set_step_profiling(true);
    let mut best: Option<inf_player::step_profile::StepProfile> = None;
    for _ in 0..5 {
        let mut m = inf_player::step_profile::StepProfile::default();
        for _ in 0..60 {
            sim.step_once(RuntimeInput::default());
            m.accumulate(&sim.step_profile());
        }
        m.scale(1.0 / 60.0);
        if best.as_ref().is_none_or(|b| m.total_ms() < b.total_ms()) {
            best = Some(m);
        }
    }
    best.expect("five rounds")
}

/// **THE TIER LADDER, PRICED PER CAR** -- 64 parked island saloons as traffic
/// records, placed so the ladder puts every one of them on ONE rung: inside
/// `TRAFFIC_FULL_M` (a real rig, handbraked, the vehicle phase solving its four
/// wheels), between it and `TRAFFIC_NEAR_M` (a `Body`: chassis and panels, a
/// kinematic transform written from the clock), or beyond it (`Dormant`: a
/// record and nothing built) -- each against a CONTROL, the same world with no
/// records, so the price is the whole step's difference over 64.
///
/// READS: `traffic_stats().per_tier` (the rung every car is on -- asserted, so
/// a placement that straddled a boundary is red, not a mislabelled row), the
/// step's own phase clock (min of five rounds of sixty steps), and
/// `bridge.vehicle_guids()` (how many rigs the Full rung really built).
/// `Far` is unreachable for a car by construction (`TRAFFIC_RADII` sets near ==
/// far) and the row says so. Clocks: printed everywhere, asserted nowhere (the
/// row is a table for the memo, not a ceiling). A tree with no VEH3 arc prices
/// the same ladder with a lighter rig; its Full row would be cheaper.
#[test]
fn the_tier_ladder_prices_a_car_per_rung() {
    let mut control = ladder_sim(0, DVec3::ZERO);
    let c = ladder_step_ms(&mut control);
    println!(
        "TIER LADDER (release: {}), control step {:.4} ms (no records)",
        !cfg!(debug_assertions),
        c.total_ms()
    );
    let rungs = [
        ("Full", DVec3::new(0.0, 0.0, 0.0), 0usize),
        ("Near", DVec3::new(96.0, 0.0, 0.0), 1),
        ("Dormant", DVec3::new(600.0, 0.0, 0.0), 3),
    ];
    let idx = |name: &str| {
        inf_player::step_profile::STEP_PHASE_NAMES
            .iter()
            .position(|n| *n == name)
            .expect("a phase")
    };
    let mut rows = Vec::new();
    for (rung, centre, tier) in rungs {
        let mut sim = ladder_sim(64, centre);
        let m = ladder_step_ms(&mut sim);
        let st = sim.traffic_stats();
        assert_eq!(
            st.per_tier[tier], 64,
            "{rung}: the ladder put the cars on {:?}, not all on one rung",
            st.per_tier
        );
        assert_eq!(
            st.per_tier[2], 0,
            "{rung}: a car reached the Far rung, which TRAFFIC_RADII makes unreachable"
        );
        let rigs = sim.bridge3d().vehicle_guids().len();
        if tier == 0 {
            assert_eq!(rigs, 64, "the Full rung built {rigs} rigs");
        } else {
            assert_eq!(rigs, 0, "{rung}: {rigs} rigs built off the Full rung");
        }
        let per_car_us = (m.total_ms() - c.total_ms()) * 1000.0 / 64.0;
        println!(
            "TIER {rung:<8} 64 cars, per_tier {:?}, {rigs} rigs: step {:.4} ms -> {per_car_us:.2} us a car (traffic {:.4} ms, vehicle {:.4} ms, solver {:.4} ms, physics3d sync {:.4} ms)",
            st.per_tier,
            m.total_ms(),
            m.ms[idx("traffic")],
            m.ms[idx("vehicle")],
            m.ms[idx("solver")],
            m.ms[idx("physics3d sync")],
        );
        rows.push((rung, per_car_us));
    }
    println!(
        "TIER Far      unreachable for a car: TRAFFIC_RADII = {:?} (near == far)",
        inf_ecs::traffic::TRAFFIC_RADII
    );
    assert_eq!(
        inf_ecs::traffic::TRAFFIC_RADII.1,
        inf_ecs::traffic::TRAFFIC_RADII.2,
        "a car's Far rung became reachable -- the ladder table needs a fourth row"
    );
}

// ── the lap, as arms ────────────────────────────────────────────────────────

/// The static front-axle share of the lap car's weight -- the burnout's
/// first row, before the throttle has moved anything -- and the peak share
/// while it brakes at better than 0.8 g.
fn front_share(r: &LapRow) -> f64 {
    let total: f64 = r.load.iter().sum();
    if total > 0.0 {
        (r.load[0] + r.load[1]) / total
    } else {
        0.0
    }
}

/// What the lap says, reduced to the numbers the memo quotes.
#[derive(Debug, Clone, Copy)]
struct LapFacts {
    rows: usize,
    lap_s: f64,
    burnout_rear_rise_c: f64,
    straight_rear_fall_c: f64,
    boost_peak: f64,
    boost_after_lift: f64,
    static_front: f64,
    braking_front: f64,
    peak_brake_g: f64,
    peak_impact_g: f64,
    peak_temp_c: f64,
}

/// Reduce a lap to [`LapFacts`], asserting its SHAPE on the way: 60 Hz rows,
/// every phase present, the columns the plot reads.
fn lap_facts(lap: &Lap) -> LapFacts {
    let r = &lap.rows;
    assert!(lap.completed, "the car never got round the circuit");
    for w in r.windows(2) {
        assert!(
            ((w[1].t - w[0].t) - 1.0 / 60.0).abs() < 1e-9,
            "a gap in the 60 Hz log at t = {:.4}",
            w[0].t
        );
    }
    let burn: Vec<&LapRow> = r.iter().filter(|x| x.phase == 1).collect();
    let lapped: Vec<&LapRow> = r.iter().filter(|x| x.phase == 2).collect();
    assert_eq!(burn.len(), BURNOUT_STEPS, "the burnout's rows");
    assert!(lapped.len() > 600, "a lap of {} rows", lapped.len());
    assert!(r.iter().any(|x| x.phase == 3), "no cool-down rows");
    let rear = |x: &LapRow| x.temp[2].max(x.temp[3]);
    let burnout_rear_rise_c = rear(burn[burn.len() - 1]) - rear(burn[0]);
    // The cool-down straight: a light throttle, no brake, no steer.
    let cool: Vec<&LapRow> = r.iter().filter(|x| x.phase == 3).collect();
    let straight_rear_fall_c = rear(cool[0]) - rear(cool[cool.len() - 1]);
    let boost_peak = r.iter().map(|x| x.boost).fold(0.0, f64::max);
    // Half a second after the first full lift from boost.
    let lift = r
        .windows(2)
        .position(|w| w[0].boost > 0.3 && w[1].throttle == 0.0)
        .expect("the throttle is lifted off boost somewhere on the lap");
    let boost_after_lift = r[(lift + 30).min(r.len() - 1)].boost;
    let static_front = front_share(&r[0]);
    let braking_front = r
        .iter()
        .filter(|x| x.brake > 0.5 && x.long_g < -0.8)
        .map(front_share)
        .fold(0.0, f64::max);
    // The driver's braking, not an impact: rows with the brake pedal down.
    let peak_brake_g = r
        .iter()
        .filter(|x| x.brake > 0.5)
        .map(|x| -x.long_g)
        .fold(0.0, f64::max);
    let peak_impact_g = r.iter().map(|x| -x.long_g).fold(0.0, f64::max);
    let peak_temp_c = r.iter().flat_map(|x| x.temp).fold(f64::MIN, f64::max);
    LapFacts {
        rows: r.len(),
        lap_s: lapped.len() as f64 / 60.0,
        burnout_rear_rise_c,
        straight_rear_fall_c,
        boost_peak,
        boost_after_lift,
        static_front,
        braking_front,
        peak_brake_g,
        peak_impact_g,
        peak_temp_c,
    }
}

/// **THE LAP, AT 60 HZ, ON BOTH HOSTS** -- the island's circuit (the CI
/// island, cooked twice, booted as the shipped player boots it and as the
/// editor's loose level) driven in a shell muscle coupe: boarded through
/// VEH3d's pipeline (one interact press), a three-second burnout at the line
/// with the traction aid off, one lap on the engine's own lane driver, a
/// cool-down straight.
///
/// READS, every fixed step: the car's four `WheelState`s (slip ratio, slip
/// angle, load, temperature -- by corner, named off each wheel's mount), its
/// `DrivetrainState` (rpm, gear, boost), the rapier body's velocity and the g
/// it differences to, the surface census; and a digest of the whole
/// `state_bytes` fold. ASSERTS: two cooks byte-identical; the CSV's SHAPE (the
/// header the plot reads, 60 Hz rows, every phase); the physics the plot
/// shows -- the rear tyres heat in the burnout and cool on the longest
/// straight, the boost spools past 0.5 and blows off within half a second of
/// the lift, braking moves load onto the front axle; and PIE == shipping: the
/// two hosts' 60 Hz rows equal to the bit and every step's state digest equal.
/// `VEH3H_LAP_CSV=<file>` writes the shipped lap for the plot (never
/// committed). A tree with no VEH3 arc has no tyre temperature, no boost and
/// no boarding pipeline: every physics assertion fails.
#[test]
fn the_lap_logs_at_sixty_hertz_and_pie_equals_shipping() {
    let tmp = tempfile::tempdir().expect("tmp");
    let proj = build_project(tmp.path());
    let pack = cook(&proj, &tmp.path().join("out"));
    let pack2 = cook(&proj, &tmp.path().join("out2"));
    // TWO COOKS: every file of the second is the first's.
    let mut files = 0usize;
    for e in walk(&pack) {
        let rel = e.strip_prefix(&pack).expect("under the pack");
        let a = std::fs::read(&e).expect("read");
        let b = std::fs::read(pack2.join(rel)).expect("the second cook has it");
        assert!(a == b, "two cooks differ at {}", rel.display());
        files += 1;
    }
    assert!(files > 0, "the cook wrote nothing");
    let recipe = inf_island::IslandRecipe::load(&fixture_recipe()).expect("recipe");
    let design = inf_island::read_design(&recipe).expect("design");
    let slug = inf_island::slug(&recipe.name);
    let mut ship = pack_sim(&pack);
    let shipped = drive_lap(&mut ship, &design, LAP_ROW);
    let mut pie = loose_sim(&proj.join("Content"), &slug);
    let previewed = drive_lap(&mut pie, &design, LAP_ROW);
    if let Some(p) = std::env::var_os("VEH3H_LAP_CSV") {
        let mut s = LAP_COLUMNS.join(",");
        s.push('\n');
        for r in &shipped.rows {
            s.push_str(&r.csv());
            s.push('\n');
        }
        std::fs::write(p, s).expect("write the lap CSV");
    }
    println!("LAP CIRCUIT: {:?}", shipped.corners);
    let f = lap_facts(&shipped);
    println!(
        "LAP: {} rows at 60 Hz over {:.1} m ({} corners, {} parked vehicle(s) set aside), lap {:.2} s; two cooks identical over {files} files",
        f.rows,
        shipped.length_m,
        shipped.corners.len() - 2,
        shipped.set_aside,
        f.lap_s
    );
    println!(
        "LAP HEAT: the burnout took the rear tyres +{:.2} C, the longest straight cooled them {:.2} C, peak {:.2} C",
        f.burnout_rear_rise_c, f.straight_rear_fall_c, f.peak_temp_c
    );
    println!(
        "LAP BOOST: peak {:.3}, {:.3} half a second after the lift",
        f.boost_peak, f.boost_after_lift
    );
    println!(
        "LAP LOAD: the front axle carries {:.1} % standing and {:.1} % braking (peak braking {:.2} g; peak one-step deceleration {:.2} g)",
        f.static_front * 100.0,
        f.braking_front * 100.0,
        f.peak_brake_g,
        f.peak_impact_g
    );
    // The header is the plot's contract.
    assert_eq!(LAP_COLUMNS.len(), shipped.rows[0].csv().split(',').count());
    assert!(
        f.burnout_rear_rise_c > 3.0,
        "the burnout heated nothing: {f:?}"
    );
    assert!(
        f.straight_rear_fall_c > 0.3,
        "nothing cooled on the straight: {f:?}"
    );
    assert!(f.boost_peak > 0.5, "the compressor never spooled: {f:?}");
    assert!(
        f.boost_after_lift < 0.5 * f.boost_peak,
        "no blow-off on the lift: {f:?}"
    );
    assert!(
        f.braking_front > f.static_front + 0.05,
        "braking moved no load forward: {f:?}"
    );
    // **NO IMPACT ON THE LINE** (the VEH3h audit): the cert's circuit struck a
    // terrace bank at -36.75 g for one step. The steepest one-step
    // deceleration of the lap is its braking, not a wall.
    assert!(
        f.peak_impact_g < f.peak_brake_g + 0.5,
        "the lap met something: a one-step deceleration of {:.2} g against {:.2} g of braking",
        f.peak_impact_g,
        f.peak_brake_g
    );
    // THE MEMO QUOTES THIS LAP: its `LAP-FACTS` line carries the numbers this
    // arm just measured, digit for digit, or the memo is red.
    let memo = memo_text();
    let line = memo
        .lines()
        .find(|l| l.starts_with("LAP-FACTS:"))
        .expect("the memo prints no LAP-FACTS line");
    for want in [
        format!("{} rows", f.rows),
        format!("+{:.2} C", f.burnout_rear_rise_c),
        format!("{:.2} C", f.straight_rear_fall_c),
        format!("boost {:.3}", f.boost_peak),
        format!("{:.1} %", f.braking_front * 100.0),
    ] {
        assert!(
            line.contains(&want),
            "the memo's LAP-FACTS line does not say {want:?}: {line}"
        );
    }
    // PIE == SHIPPING, the whole lap.
    assert_eq!(
        shipped.rows.len(),
        previewed.rows.len(),
        "the laps differ in length"
    );
    for (i, (a, b)) in shipped.rows.iter().zip(&previewed.rows).enumerate() {
        assert!(
            a == b,
            "PIE and shipping diverge at lap row {i}:\n{}\n{}",
            a.csv(),
            b.csv()
        );
    }
    assert_eq!(shipped.digests.len(), previewed.digests.len());
    let first = shipped
        .digests
        .iter()
        .zip(&previewed.digests)
        .position(|(a, b)| a != b);
    assert_eq!(first, None, "the state digests diverge at step {first:?}");
    println!(
        "PIE == SHIPPING over the whole lap: {} rows and {} state digests equal",
        shipped.rows.len(),
        shipped.digests.len()
    );
}

/// Every file under `dir`, recursively, in a stable order.
fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut todo = vec![dir.to_path_buf()];
    while let Some(d) = todo.pop() {
        for e in std::fs::read_dir(&d).expect("read a dir").flatten() {
            let p = e.path();
            if p.is_dir() {
                todo.push(p);
            } else {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

// ── the memo's citations ────────────────────────────────────────────────────

/// `docs/memos/driving-parity.md`, CR-stripped (a Windows checkout's CRLF
/// must not hide a line from a grep -- the P22 law).
fn memo_text() -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/memos/driving-parity.md");
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("the driving parity memo is not on disk ({e})"))
        .replace('\r', "")
}

/// The test roots a citation may live under, `(label, dir)`.
fn citation_roots() -> Vec<PathBuf> {
    let m = Path::new(env!("CARGO_MANIFEST_DIR"));
    vec![
        m.join("tests"),
        m.join("../../crates/inf-physics/tests"),
        m.join("../../crates/inf-ecs/src"),
        m.join("../../crates/inf-audio/src"),
        m.join("../../crates/inf-audio/tests"),
        m.join("../../editor/crates/inf-editor-core/tests"),
        m.join("../../editor/crates/inf-editor-core/src"),
        m.join("src"),
    ]
}

/// **EVERY ARM THE MEMO CITES EXISTS AND IS NOT IGNORED** -- the cert's own
/// law: a row whose citation does not exist, or is `#[ignore]`d, is a false
/// cert.
///
/// READS the memo (CR-stripped) and every `` `file::name` `` citation in it
/// whose file is a test file (`*_gate`, `*_3d`, `vehicle_grade`, ...) and
/// whose name is a sentence (four words or more); finds `fn name(` in that
/// file under the test roots (CR-stripped), and refuses one whose attribute
/// block carries `#[ignore`. Also asserts the memo's honest half names the
/// waves the carried rows route to. A tree with no VEH3 arc has no memo and
/// fails at the read. Mutation: renaming a cited arm, or `#[ignore]` on one.
#[test]
fn every_arm_the_driving_memo_cites_exists_and_is_not_ignored() {
    let memo = memo_text();
    let mut cites: Vec<(String, String)> = Vec::new();
    for line in memo.lines() {
        for tok in line.split('`') {
            let Some((file, name)) = tok.split_once("::") else {
                continue;
            };
            if file.is_empty()
                || !file
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                continue;
            }
            if name.matches('_').count() < 3
                || !name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                continue;
            }
            cites.push((file.to_string(), name.to_string()));
        }
    }
    cites.sort();
    cites.dedup();
    let mut missing = Vec::new();
    let mut ignored = Vec::new();
    for (file, name) in &cites {
        let mut found = None;
        for root in citation_roots() {
            let p = root.join(format!("{file}.rs"));
            if let Ok(text) = std::fs::read_to_string(&p) {
                let text = text.replace('\r', "");
                if let Some(at) = text.find(&format!("fn {name}(")) {
                    found = Some((text, at));
                    break;
                }
            }
        }
        let Some((text, at)) = found else {
            missing.push(format!("{file}::{name}"));
            continue;
        };
        // The attribute block: the lines straight above the `fn`, up to the
        // first that is not an attribute or a doc line.
        let head = &text[..at];
        let attrs: Vec<&str> = head
            .lines()
            .rev()
            .take_while(|l| {
                let t = l.trim_start();
                t.starts_with("#[") || t.starts_with("///") || t.is_empty()
            })
            .collect();
        if attrs.iter().any(|l| l.trim_start().starts_with("#[ignore")) {
            ignored.push(format!("{file}::{name}"));
        }
    }
    println!(
        "THE DRIVING MEMO cites {} arms: {} missing, {} ignored",
        cites.len(),
        missing.len(),
        ignored.len()
    );
    assert!(
        cites.len() >= 60,
        "the memo cites only {} arms",
        cites.len()
    );
    assert!(
        missing.is_empty(),
        "the memo cites arms that do not exist: {missing:?}"
    );
    assert!(
        ignored.is_empty(),
        "the memo cites IGNORED arms: {ignored:?}"
    );
    for must in ["PAR1", "PAR2", "PERF1", "N = 1", "NOT MET", "CARRIED"] {
        assert!(memo.contains(must), "the memo never says {must:?}");
    }
}

// ── the feel against Forza ──────────────────────────────────────────────────

/// A car the feel arm drives.
const FEEL_CAR: Uuid = Uuid::from_u128(0x5645_4833_4645_4500_0000_0000_0000_0001);

/// **The three rows re-measured against Forza**, with the real-world car each
/// is inspired by and that car's STOCK figures in Forza Horizon 5 (the game's
/// own acceleration test, 0-60 mph and 0-100 mph, as published by the
/// `forza.labsgg.com` FH5 car sheets' "simulation results", RE-READ by the
/// VEH3h audit on 2026-09-26 -- the cert's figures (2.293 / 5.615 / 2.500 s)
/// were each 1-2.5 % off what the sheets print): `(row, body kind,
/// inspiration, FH5 0-60 mph s, FH5 0-100 mph s)`. One SHELL row, one IMPORTED
/// row (its art is local-only; the physics is the committed row either way),
/// and one hypercar shell row at the other end of the class.
const FORZA_ROWS: [(&str, &str, &str, f64, f64); 3] = [
    (
        "bravado_gauntlet_hellfire",
        "shell",
        "2018 Dodge Challenger SRT Demon",
        2.259,
        5.847,
    ),
    (
        "karin_asterope_gz",
        "imported",
        "2023 Toyota Camry TRD",
        5.555,
        13.785,
    ),
    (
        "pegassi_zentorno",
        "shell",
        "2011 Lamborghini Sesto Elemento",
        2.442,
        5.088,
    ),
];

/// **The band a row is held to against its inspiration's Forza time, as a
/// fraction** -- 25 %. Justified, not chosen to fit: a roster row authors the
/// inspiration's published MASS and its engine's torque curve on a rigid
/// driveline model with one gearbox shape for every car in its class, no
/// launch control, no drag-radial compound and a mu-0.9 slab (Forza's test
/// surface is dry asphalt, mu ~1.0-1.1 on a road tyre); a 0-60 is dominated by
/// the first-gear traction limit, which scales with mu, so the surface alone
/// is worth ~10 %, and the class gearbox another ~10 %. A row outside 25 % is
/// a row whose engine or mass is not the car it claims, and the memo says so.
const FORZA_BAND: f64 = 0.25;

/// 60 mph, m/s.
const MPH60: f64 = 26.8224;
/// 100 mph, m/s.
const MPH100: f64 = 44.704;

/// One row's sprint on the SHIPPED host: a 6 km mu-0.9 slab, the car settled
/// two seconds, then full throttle; the seconds to 60 and to 100 mph off the
/// rapier body's own ground speed (`None` for a speed the row never reaches in
/// sixty seconds).
fn sprint(row: &str) -> (Option<f64>, Option<f64>) {
    use inf_ecs::components::{BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D};
    let def: VehicleDef = *inf_ecs::roster::roster()
        .get(row)
        .unwrap_or_else(|| panic!("the roster has no `{row}`"));
    let mut world = inf_ecs::EcsWorld::new();
    let g = world.spawn_with_guid(
        Uuid::from_u128(0x5645_4833_4645_4500_0000_0000_0000_0002),
        "Slab",
        None,
    );
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, -0.5, 0.0);
    world.world_mut().entity_mut(g).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(3_000.0, 0.5, 3_000.0),
            friction: 0.9,
            ..Default::default()
        },
    ));
    let at = DVec3::new(0.0, inf_ecs::vehicle::resting_origin_y(&def, 0.0), -2_940.0);
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        FEEL_CAR,
        &def,
        &inf_ecs::vehicle::RigSpawn {
            name: "Car".into(),
            at,
            yaw_deg: 0.0,
            paint: inf_ecs::math::Color::new(0.3, 0.3, 0.7, 1.0),
            clip: None,
            engine_voice: false,
            livery: None,
        },
    );
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
    let drive = |sim: &mut RuntimeSim, throttle: f64| {
        if let Some(v) = sim.bridge3d_mut().vehicle_mut(FEEL_CAR) {
            v.control(inf_ecs::vehicle::VehicleControls {
                throttle,
                occupied: true,
                ..Default::default()
            });
        }
        sim.step_once(RuntimeInput::default());
    };
    for _ in 0..120 {
        drive(&mut sim, 0.0);
    }
    let (mut t60, mut t100) = (None, None);
    for i in 0..3_600usize {
        drive(&mut sim, 1.0);
        let v = sim
            .bridge3d()
            .body_of(FEEL_CAR)
            .and_then(|b| sim.bridge3d().world().body_linvel(b))
            .map(|v| DVec3::new(v.x, 0.0, v.z).length())
            .unwrap_or(0.0);
        let t = (i + 1) as f64 / 60.0;
        if t60.is_none() && v >= MPH60 {
            t60 = Some(t);
        }
        if v >= MPH100 {
            t100 = Some(t);
            break;
        }
    }
    (t60, t100)
}

/// **THREE CLASSES, RE-MEASURED AGAINST FORZA** -- a shell muscle coupe, an
/// imported saloon and a shell hypercar, each sprinted on the shipped host
/// (`sprint`) and set beside its inspiration's stock Forza Horizon 5 0-60 and
/// 0-100 mph ([`FORZA_ROWS`]), with the verdict against [`FORZA_BAND`].
///
/// READS: the rapier body's ground speed every step of a full-throttle run
/// (the world, not a table). ASSERTS: every row reaches 60 mph; the verdict
/// each row gets is the verdict `docs/memos/driving-parity.md` prints for it
/// (the memo's `FEEL-VS-FORZA` lines) -- so the day a retune moves a row across
/// the band, the memo is red until it is rewritten. A tree with no VEH3 arc has
/// none of these rows (VEH3f's roster) and fails at the first lookup.
#[test]
fn three_classes_against_their_forza_inspirations() {
    let mut measured = Vec::new();
    for (row, kind, inspiration, f60, f100) in FORZA_ROWS {
        let (t60, t100) = sprint(row);
        let t60 = t60.unwrap_or_else(|| panic!("{row} never reached 60 mph"));
        let off = (t60 - f60) / f60;
        let verdict = if off.abs() <= FORZA_BAND {
            "WITHIN"
        } else {
            "OUTSIDE"
        };
        let t100s = t100.map_or("never".to_string(), |t| format!("{t:.2} s"));
        let line = format!(
            "FEEL-VS-FORZA {row} ({kind}) vs {inspiration}: 0-60 mph {t60:.2} s against FH5 {f60:.3} s ({:+.0} %), 0-100 mph {t100s} against FH5 {f100:.3} s -- {verdict} the {:.0} % band",
            off * 100.0,
            FORZA_BAND * 100.0
        );
        println!("{line}");
        measured.push((row, t60, verdict, line));
    }
    let memo = memo_text();
    for (row, t60, verdict, line) in measured {
        let pinned = memo
            .lines()
            .find(|l| l.starts_with("FEEL-VS-FORZA") && l.contains(row))
            .unwrap_or_else(|| panic!("the memo prints no FEEL-VS-FORZA line for {row}"));
        assert!(
            pinned.contains(&format!("{t60:.2} s")) && pinned.contains(verdict),
            "the memo's line for {row} is not what the arm measured today:\n  memo: {pinned}\n  arm:  {line}"
        );
    }
}

/// Where the CI island's camp fire appliance is after `seconds` with the hero
/// standing at `hero` (x, z): `(start, end)` of its rapier body, and whether
/// the parking hold held it beyond the band at the end
/// (`PhysicsBridge3D::parked_beyond_band`).
fn appliance_after(hero: (f64, f64), seconds: usize) -> (DVec3, DVec3, bool) {
    let tmp = tempfile::tempdir().expect("tmp");
    let proj = build_project(tmp.path());
    let pack = cook(&proj, &tmp.path().join("out"));
    let mut sim = pack_sim(&pack);
    sim.set_crowd_population(Default::default());
    sim.set_traffic_population(Default::default());
    let at = DVec3::new(hero.0, 0.0, hero.1);
    let y = sim.terrain_height_at(at.x, at.z);
    let find = |sim: &RuntimeSim| {
        let w = sim.world();
        w.world().iter_entities().find_map(|e| {
            (w.name_of(e.id()) == Some("Fixture Camp engine"))
                .then(|| e.get::<inf_ecs::Guid>().map(|g| g.0))
                .flatten()
        })
    };
    let pos = |sim: &RuntimeSim, g: Uuid| {
        let b = sim.bridge3d();
        b.body_of(g)
            .and_then(|body| b.world().body_translation(body))
            .expect("the appliance has a body")
    };
    let mut start = None;
    for _ in 0..(seconds * 60) {
        put_hero(&mut sim, DVec3::new(at.x, y + 1.5, at.z));
        sim.step_once(RuntimeInput::default());
        if start.is_none() {
            if let Some(g) = find(&sim) {
                if sim.bridge3d().body_of(g).is_some() {
                    start = Some((g, pos(&sim, g)));
                }
            }
        }
    }
    let (g, p0) = start.expect("the camp appliance streamed in");
    let held = sim.bridge3d().parked_beyond_band().contains(&g);
    (p0, pos(&sim, g), held)
}

/// **A PARKED VEHICLE BEYOND THE COLLIDER BAND HOLDS ITS PAD** (the VEH3h
/// audit; the cert shipped this arm asserting the DEFECT -- the appliance
/// rolled **36.63 m** in fifteen seconds with the hero 120 m away -- and routed
/// it to PERF1. It is a parking-hold defect, and it is closed here.)
///
/// READS the CI island's camp fire appliance's rapier body over fifteen
/// seconds, the crowd and traffic set aside, with the hero standing 120 m
/// away (outside `DEFAULT_COLLIDER_NEAR_M`, 64 m -- the fine colliders under
/// the appliance are banded out, the vehicle is not) and, as the CONTROL,
/// 20 m away; and `PhysicsBridge3D::parked_beyond_band` (the hold's
/// engagement: the appliance is in it at the end of the far run and not of
/// the near one). ASSERTS: beyond the band it moves **0.00 m** and falls
/// 0.00 m, held; inside it, simulated, under 3 m. THE CONTROL IS ITSELF A
/// FINDING, printed and carried: inside the band the appliance creeps
/// **1.51 m** in fifteen seconds at ~0.11 m/s -- the island generator parks
/// it 6 m off the lane's centreline on a 7.8 deg grade between two
/// buildings, it rolls 18.6 deg within two seconds with one wheel hanging, and
/// its chassis slides on what it leans on (`driving-parity.md` §13). A tree
/// with no VEH3 arc rolls tens of metres beyond the band. Mutation: the hold
/// disabled (`hold_beyond_band` answering `false`) -> RED, the appliance
/// rolls off again.
#[test]
fn a_parked_vehicle_beyond_the_collider_band_holds_its_pad() {
    let (f0, f1, far_held) = appliance_after((430.0, -376.0), 15);
    let (n0, n1, near_held) = appliance_after((392.0, -280.0), 15);
    let far = DVec3::new(f1.x - f0.x, 0.0, f1.z - f0.z).length();
    let near = DVec3::new(n1.x - n0.x, 0.0, n1.z - n0.z).length();
    println!(
        "PARKED APPLIANCE: {far:.2} m in 15 s with the hero 120 m away ({:.2} m of fall, held {far_held}), {near:.2} m with the hero 20 m away (held {near_held})",
        f0.y - f1.y
    );
    assert!(
        far < 0.01 && (f0.y - f1.y).abs() < 0.01,
        "the parked appliance moved {far:.3} m ({:.3} m of fall) beyond the collider band",
        f0.y - f1.y
    );
    assert!(
        far_held,
        "the appliance held still beyond the band without the hold -- the arm measured nothing"
    );
    assert!(!near_held, "the hold froze a vehicle INSIDE the band");
    assert!(
        near < 3.0,
        "the control moved {near:.2} m with the hero beside it"
    );
}

// ── the headlamps ───────────────────────────────────────────────────────────

/// **A HEADLAMP IS A LENS, NOT A LIGHT** -- the cert's PAR1 routing, measured
/// rather than assumed.
///
/// READS the world: every row of the island catalogue and the roster spawned
/// through the one rig door (`spawn_rig`) into one world, then every entity
/// under every chassis -- the drawn `lamp_front` parts and their `Material`
/// (the emissive lens VEH3f.2b gave them), and any `Light` component at all.
/// ASSERTS: lamp parts exist and glow (emissive > 0) and NOT ONE rig entity
/// carries a `Light` -- there is no cone for a night frame to show, which is
/// PAR1's (street lighting and vehicle lamps on PAR0's many-lights substrate:
/// `MAX_LIGHTS` is 16 for the whole frame). The day a car grows a light this
/// arm reds and the memo row is rewritten. A tree with no VEH3 arc has no lamp
/// parts and fails the first assertion.
#[test]
fn a_headlamp_is_an_emissive_lens_and_no_car_carries_a_light() {
    let mut world = inf_ecs::EcsWorld::new();
    let mut rows: Vec<VehicleDef> = inf_editor_core::vehicle::island_vehicles()
        .0
        .values()
        .copied()
        .collect();
    rows.extend(inf_ecs::roster::roster().0.values().copied());
    let mut chassis = Vec::new();
    for (k, def) in rows.iter().enumerate() {
        let g = Uuid::from_u128(0x5645_4833_4c41_4d00_0000_0000_0001_0000 + k as u128);
        inf_ecs::vehicle::spawn_rig(
            &mut world,
            g,
            def,
            &inf_ecs::vehicle::RigSpawn {
                name: "Car".into(),
                at: DVec3::new(k as f64 * 20.0, 5.0, 0.0),
                yaw_deg: 0.0,
                paint: inf_ecs::math::Color::new(0.5, 0.5, 0.5, 1.0),
                clip: None,
                engine_voice: false,
                livery: None,
            },
        );
        chassis.push(g);
    }
    world.propagate();
    let (mut lamps, mut glowing, mut lights) = (0usize, 0usize, 0usize);
    for g in &chassis {
        let Some(root) = world.entity_of(*g) else {
            continue;
        };
        for e in world.subtree(root) {
            if world.world().get::<inf_ecs::components::Light>(e).is_some() {
                lights += 1;
            }
            if world
                .name_of(e)
                .is_some_and(|n| n.starts_with("lamp_front"))
            {
                lamps += 1;
                if world
                    .world()
                    .get::<inf_ecs::components::Material>(e)
                    .is_some_and(|m| m.emissive_intensity > 0.0)
                {
                    glowing += 1;
                }
            }
        }
    }
    println!(
        "HEADLAMPS: {} rows spawned, {lamps} front lamp part(s), {glowing} of them emissive, {lights} Light component(s) on any rig",
        chassis.len()
    );
    assert!(
        lamps > 0 && glowing == lamps,
        "no glowing front lamp to speak of"
    );
    assert_eq!(
        lights, 0,
        "a car carries a Light now -- rewrite the memo's headlight row"
    );
}

/// The IMPORTED row the second lap drives: the doc's Pegassi Toros (the
/// Lamborghini Urus) wears `dd_suv`, one of the Fab packs' cars the island
/// commits as a level dependency, and carries a turbocharger. CI draws its
/// committed fallback; the physics is the row either way.
const IMPORTED_LAP_ROW: &str = "obey_rocoto";

/// **THE SAME LAP IN AN IMPORTED CAR, ON BOTH HOSTS** -- the circuit, the
/// boarding, the burnout, the lane driver and the cool-down of
/// `the_lap_logs_at_sixty_hertz_and_pie_equals_shipping`, driven in
/// [`IMPORTED_LAP_ROW`].
///
/// READS what that arm reads. ASSERTS: the lap completes, the compressor
/// spools past 0.3, braking moves load onto the front axle, and the two
/// hosts' 60 Hz rows and state digests are equal. (The burnout's heat is
/// printed, not asserted: an all-wheel-drive SUV with its aid off spreads the
/// launch over four tyres.) `VEH3H_LAP_CSV_IMPORTED=<file>` writes its CSV. A
/// tree with no VEH3 arc has no such row (VEH3f) and no art table (VEH3f.2a).
#[test]
fn the_imported_row_drives_the_same_lap_on_both_hosts() {
    let tmp = tempfile::tempdir().expect("tmp");
    let proj = build_project(tmp.path());
    let pack = cook(&proj, &tmp.path().join("out"));
    let recipe = inf_island::IslandRecipe::load(&fixture_recipe()).expect("recipe");
    let design = inf_island::read_design(&recipe).expect("design");
    let slug = inf_island::slug(&recipe.name);
    let def = inf_ecs::roster::roster()
        .get(IMPORTED_LAP_ROW)
        .copied()
        .expect("the imported row");
    assert!(
        def.art.is_some_and(|k| !k.shell()),
        "{IMPORTED_LAP_ROW} no longer wears imported art"
    );
    let mut ship = pack_sim(&pack);
    let shipped = drive_lap(&mut ship, &design, IMPORTED_LAP_ROW);
    let mut pie = loose_sim(&proj.join("Content"), &slug);
    let previewed = drive_lap(&mut pie, &design, IMPORTED_LAP_ROW);
    if let Some(p) = std::env::var_os("VEH3H_LAP_CSV_IMPORTED") {
        let mut s = LAP_COLUMNS.join(",");
        s.push('\n');
        for r in &shipped.rows {
            s.push_str(&r.csv());
            s.push('\n');
        }
        std::fs::write(p, s).expect("write the imported lap CSV");
    }
    assert!(shipped.completed, "the imported car never got round");
    let r = &shipped.rows;
    let rear = |x: &LapRow| x.temp[2].max(x.temp[3]);
    let burn: Vec<&LapRow> = r.iter().filter(|x| x.phase == 1).collect();
    let boost = r.iter().map(|x| x.boost).fold(0.0, f64::max);
    let braking = r
        .iter()
        .filter(|x| x.brake > 0.5 && x.long_g < -0.8)
        .map(front_share)
        .fold(0.0, f64::max);
    let lap_s = r.iter().filter(|x| x.phase == 2).count() as f64 / 60.0;
    println!(
        "IMPORTED LAP ({IMPORTED_LAP_ROW}, art {}): {} rows, lap {lap_s:.2} s; burnout rear +{:.2} C; boost peak {boost:.3}; front axle {:.1} % standing, {:.1} % braking",
        def.art.map_or("-", |k| k.name()),
        r.len(),
        rear(burn[burn.len() - 1]) - rear(burn[0]),
        front_share(&r[0]) * 100.0,
        braking * 100.0
    );
    // PRINTED, NOT ASSERTED (the VEH3h audit's close): the shell lap asserts
    // no one-step deceleration above its braking; this lap does not, and the
    // heavier all-wheel-drive SUV runs wide out of a corner and meets
    // something. The line says where, so the memo's sentence is the arm's.
    let hit = r
        .iter()
        .min_by(|a, b| a.long_g.total_cmp(&b.long_g))
        .expect("rows");
    println!(
        "IMPORTED LAP IMPACT: peak one-step deceleration {:.2} g at t {:.2} s, ({:.1}, {:.1}), s {:.1} m (printed, not asserted)",
        -hit.long_g,
        hit.t,
        hit.x,
        hit.z,
        hit.s_m
    );
    assert!(boost > 0.3, "the imported car's compressor never spooled");
    assert!(
        braking > front_share(&r[0]) + 0.05,
        "braking moved no load forward on the imported car"
    );
    assert_eq!(shipped.rows.len(), previewed.rows.len());
    for (i, (a, b)) in shipped.rows.iter().zip(&previewed.rows).enumerate() {
        assert!(a == b, "PIE and shipping diverge at imported lap row {i}");
    }
    assert!(
        shipped.digests == previewed.digests,
        "the imported lap's state digests diverge"
    );
}

// ── the step beside a car (the VEH3h audit) ─────────────────────────────────

/// One row per body kind the island can carry: every committed SHELL, two
/// IMPORTED rows (art-referencing; the CI island draws their fallback) and one
/// PRIMITIVE row.
const BESIDE_ROWS: [(&str, &str); 8] = [
    ("shell_coupe", "bravado_gauntlet_hellfire"),
    ("shell_sedan", "albany_washington"),
    ("shell_suv", "benefactor_xls"),
    ("shell_pickup", "bravado_duneloader"),
    ("shell_cruiser", "bravado_buffalo_cruiser"),
    ("imported dd_suv", "obey_rocoto"),
    ("imported dd_sedan", "karin_asterope_gz"),
    ("primitive", "vulcar_ingot"),
];

/// The guid the beside-a-car arm spawns its car under.
const BESIDE_CAR: Uuid = Uuid::from_u128(0x5645_4833_4253_4400_0000_0000_0000_0001);

/// The whole fixed step (ms) and the scene queries it asked, min of five rounds
/// of twenty steps with the hero held at `at`, after twenty to settle.
fn step_with_hero_at(sim: &mut RuntimeSim, at: DVec3) -> (f64, f64) {
    sim.set_step_profiling(true);
    for _ in 0..20 {
        put_hero(sim, at);
        sim.step_once(RuntimeInput::default());
    }
    let mut best = f64::INFINITY;
    let mut asked = f64::INFINITY;
    for _ in 0..5 {
        let mut ms = 0.0;
        let q0 = sim.bridge3d().world().queries();
        for _ in 0..20 {
            put_hero(sim, at);
            sim.step_once(RuntimeInput::default());
            ms += sim.step_profile().total_ms();
        }
        best = best.min(ms / 20.0);
        asked = asked.min((sim.bridge3d().world().queries() - q0) as f64 / 20.0);
    }
    (best, asked)
}

/// **A FIXED STEP BESIDE A CAR IS A FIXED STEP** (the VEH3h audit) -- the arm
/// the "Harbour City cruiser stall" asked for, on every body kind.
///
/// The cert's hands-on read `character move` at 356-372 ms with the hero 2.5 m
/// off the Harbour City cruiser and carried it undiagnosed, with the obvious
/// suspect the shell's 7 000 triangles. The audit's profile refuted the car
/// (`shell_body` carries no collider; the chassis collider is a box) and named
/// the cause -- a crowd PILE in a building 37 m away
/// (`character_move_cost::a_pile_of_characters_is_set_aside_and_one_in_the_way_still_blocks`
/// is its arm). This one closes the suspicion the cert wrote down: the car
/// itself costs the step nothing.
///
/// READS the CI island's SHIPPED host (the pack), the crowd and traffic set
/// aside and every level rig cleared (the lap's instrument): each
/// `BESIDE_ROWS` car spawned beside the circuit's start through the one rig
/// door, the whole fixed step and `PhysicsWorld3D::queries` per step with
/// the hero in open ground 20 m off it (the CONTROL: the car's own wheel rays
/// are paid wherever the hero stands) and stood 2.5 m off its centre on each
/// of its four sides, and `pawns_set_aside` over all of it. ASSERTS the
/// counted half everywhere: the queries a step beside every car are at most
/// twice the control's, and nothing is set aside (there is no pile beside a
/// car). The clock half -- every step beside every car at most twice the
/// control's, min of five -- in release off CI (the house conditioning).
/// A tree with no VEH3 arc has no shells or imported rows and fails at the
/// roster. VACUOUS BY MUTATION, said: this arm certifies an ABSENCE (no
/// body kind stalls the step beside it) and no production mutation reaches
/// it -- the stall was never the car. `stacked_pawns` answering nothing
/// leaves it green (there is no pile here) and reds the pile arm, which is
/// why the two are named together.
#[test]
fn a_fixed_step_beside_every_body_kind_is_a_fixed_step() {
    let tmp = tempfile::tempdir().expect("tmp");
    let proj = build_project(tmp.path());
    let pack = cook(&proj, &tmp.path().join("out"));
    let recipe = inf_island::IslandRecipe::load(&fixture_recipe()).expect("recipe");
    let design = inf_island::read_design(&recipe).expect("design");
    let mut sim = pack_sim(&pack);
    sim.set_crowd_population(Default::default());
    sim.set_traffic_population(Default::default());
    let (pts, _) = clear_circuit(&mut sim, &design);
    let p = pts[0];
    let ground = sim.terrain_height_at(p.x, p.y);
    let spot = DVec3::new(p.x, ground + 1.0, p.y);
    // Five more seconds at the spot: the first seconds after a clear are the
    // streamers finishing (measured: 2.3 ms falling to 0.3), and a control
    // taken while the world settles would flatter every row after it.
    for _ in 0..300 {
        put_hero(&mut sim, spot);
        sim.step_once(RuntimeInput::default());
    }
    let set_aside0 = sim.bridge3d().world().pawns_set_aside();
    let (bare_ms, bare_q) = step_with_hero_at(&mut sim, spot);
    println!("BESIDE: open ground, no car {bare_ms:.3} ms a step, {bare_q:.1} queries a step");
    let release = !cfg!(debug_assertions) && std::env::var_os("CI").is_none();
    let mut worst: (f64, String) = (0.0, String::new());
    for (kind, row) in BESIDE_ROWS {
        let def: VehicleDef = *inf_ecs::roster::roster()
            .get(row)
            .unwrap_or_else(|| panic!("the roster has no `{row}`"));
        let centre = DVec3::new(
            p.x + 6.0,
            inf_ecs::vehicle::resting_origin_y(&def, ground),
            p.y,
        );
        inf_ecs::vehicle::spawn_rig_at(
            sim.world_mut(),
            BESIDE_CAR,
            &def,
            &inf_ecs::vehicle::RigSpawn {
                name: "Beside Car".into(),
                at: centre,
                yaw_deg: 0.0,
                paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
                clip: None,
                engine_voice: false,
                livery: None,
            },
            true,
        );
        sim.world_mut().propagate();
        sim.world_mut().mark_dirty();
        // The CONTROL is the same world with the same car in it, the hero
        // twenty metres off: the car's own wheel rays are a vehicle-phase
        // cost the step pays wherever the hero stands.
        let (open_ms, open_q) = step_with_hero_at(
            &mut sim,
            DVec3::new(centre.x - 20.0, ground + 1.0, centre.z),
        );
        println!("BESIDE: {kind:<18} {row:<26} 20 m  {open_ms:.3} ms, {open_q:.1} queries a step (the control)");
        for (side, off) in [
            ("front", DVec3::new(0.0, 0.0, 2.5)),
            ("back", DVec3::new(0.0, 0.0, -2.5)),
            ("left", DVec3::new(2.5, 0.0, 0.0)),
            ("right", DVec3::new(-2.5, 0.0, 0.0)),
        ] {
            let at = DVec3::new(centre.x + off.x, ground + 1.0, centre.z + off.z);
            let (ms, q) = step_with_hero_at(&mut sim, at);
            println!(
                "BESIDE: {kind:<18} {row:<26} {side:<5} {ms:.3} ms ({:.2}x), {q:.1} queries a step",
                ms / open_ms
            );
            if ms / open_ms > worst.0 {
                worst = (ms / open_ms, format!("{row} {side}"));
            }
            assert!(
                q <= 2.0 * open_q.max(1.0),
                "beside {row} ({kind}, {side}) the step asked {q:.1} queries against {open_q:.1} in open ground"
            );
            if release {
                assert!(
                    ms <= 2.0 * open_ms,
                    "beside {row} ({kind}, {side}) the fixed step is {ms:.3} ms against {open_ms:.3} in open ground"
                );
            }
        }
        let rigs: Vec<Uuid> = {
            let w = sim.world();
            w.world()
                .iter_entities()
                .filter_map(|e| e.get::<inf_ecs::Guid>().map(|g| g.0))
                .filter(|g| inf_ecs::vehicle::rig_of(w, *g).is_some())
                .collect()
        };
        assert!(!rigs.is_empty(), "the {row} rig never spawned");
        for g in &rigs {
            if let Some(e) = sim.world().entity_of(*g) {
                sim.world_mut().despawn(e);
            }
        }
        sim.world_mut().mark_dirty();
    }
    println!(
        "BESIDE: the dearest spot {:.2}x open ground ({})",
        worst.0, worst.1
    );
    assert_eq!(
        sim.bridge3d().world().pawns_set_aside(),
        set_aside0,
        "a body was set aside beside a car -- there is no pile here"
    );
}

/// Whole-step p50 over the last thirty of `steps` steps with the hero held at
/// `at` (a DWELL: the Harbour City pile took ~1.5 s after it materialized to
/// reach its full price), and the steered characters at the end.
fn dwell_step_p50(sim: &mut RuntimeSim, at: DVec3, steps: usize) -> (f64, usize) {
    sim.set_step_profiling(true);
    let mut totals = Vec::new();
    for k in 0..steps {
        put_hero(sim, at);
        sim.step_once(RuntimeInput::default());
        if k + 30 >= steps {
            totals.push(sim.step_profile().total_ms());
        }
    }
    totals.sort_by(f64::total_cmp);
    let n = sim
        .world()
        .world()
        .iter_entities()
        .filter(|e| e.get::<CharacterMovement>().is_some())
        .count();
    (totals[totals.len() / 2], n)
}

/// **THE HARBOUR CITY "CRUISER STALL", ON THE REAL ISLAND** (the VEH3h audit).
///
/// `#[ignore]`d because it needs the user's cooked island (`INF_ISLAND_PACK`,
/// e.g. `cook-h/art`) -- the fixture island has no Harbour City pile. Run:
///
/// ```text
/// INF_ISLAND_PACK=<cook> cargo test --release -p inf-player --test veh3h_gate
///     -- --ignored the_harbour_city_pile --nocapture
/// ```
///
/// READS the shipped host off the pack with its own crowd and traffic, the
/// hero held at the Harbour City cruiser's spawn for ten seconds, then at the
/// four spots 2.5 m off the first cruiser's centre and one 20 m off its side,
/// 160 steps each: the whole fixed step's p50 over each dwell's last thirty,
/// the steered characters, and `pawns_set_aside`. ASSERTS: the pile is
/// reached (more than 150 steered characters at one spot -- 229 measured)
/// and set aside (the count moves), and no spot's step is more than twice the
/// 20 m spot's. Measured by the audit: before the fix the dwell beside the
/// cruiser read **266.0 ms** a step (`character move` 232.8; an earlier dwell
/// 358.7) against **34.6** at 20 m; after it **50.0** against 34.0 (1.47x).
#[test]
#[ignore = "needs the user's cooked island (INF_ISLAND_PACK)"]
fn the_harbour_city_pile_is_set_aside_on_the_real_island() {
    let Some(pack) = std::env::var_os("INF_ISLAND_PACK").map(PathBuf::from) else {
        println!("SKIP the_harbour_city_pile: INF_ISLAND_PACK names no pack");
        return;
    };
    let mut sim = pack_sim(&pack);
    let spawn = DVec3::new(-1620.9, 0.0, 2178.0);
    let y = sim.terrain_height_at(spawn.x, spawn.z);
    for _ in 0..600 {
        put_hero(&mut sim, DVec3::new(spawn.x, y + 1.0, spawn.z));
        sim.step_once(RuntimeInput::default());
    }
    let cruiser = {
        let w = sim.world();
        let mut found: Vec<(f64, DVec3, f64)> = w
            .world()
            .iter_entities()
            .filter(|e| w.name_of(e.id()) == Some("Harbour City cruiser"))
            .filter_map(|e| {
                e.get::<Transform>()
                    .map(|t| (t.translation.to_dvec3(), t.rotation.y))
            })
            .map(|(p, yaw)| ((p - spawn).length(), p, yaw))
            .collect();
        found.sort_by(|a, b| a.0.total_cmp(&b.0));
        found
            .first()
            .map(|f| (f.1, f.2))
            .expect("the Harbour City cruiser is resident")
    };
    let (p, yaw) = (cruiser.0, cruiser.1.to_radians());
    let fwd = DVec3::new(yaw.sin(), 0.0, yaw.cos());
    let side = DVec3::new(yaw.cos(), 0.0, -yaw.sin());
    let at = |sim: &mut RuntimeSim, q: DVec3| {
        let g = sim.terrain_height_at(q.x, q.z).max(p.y - 0.3);
        DVec3::new(q.x, g + 1.0, q.z)
    };
    let set0 = sim.bridge3d().world().pawns_set_aside();
    let ctl = at(&mut sim, p + side * 20.0);
    let (open, _) = dwell_step_p50(&mut sim, ctl, 160);
    let mut most = 0usize;
    for (label, q) in [
        ("long +2.5", p + fwd * 2.5),
        ("long -2.5", p - fwd * 2.5),
        ("side +2.5", p + side * 2.5),
        ("side -2.5", p - side * 2.5),
    ] {
        let spot = at(&mut sim, q);
        let (ms, n) = dwell_step_p50(&mut sim, spot, 160);
        most = most.max(n);
        println!(
            "HARBOUR CITY: {label}: {ms:.3} ms a step ({:.2}x the 20 m spot's {open:.3}), {n} characters",
            ms / open
        );
        assert!(
            ms <= 2.0 * open,
            "{label}: the step beside the cruiser is {ms:.3} ms against {open:.3} at 20 m"
        );
    }
    let set = sim.bridge3d().world().pawns_set_aside() - set0;
    println!("HARBOUR CITY: at most {most} characters; {set} set aside");
    assert!(
        most > 150,
        "the pile was never reached ({most} characters) -- the arm measured nothing"
    );
    assert!(set > 0, "the pile was reached and nothing was set aside");
}
