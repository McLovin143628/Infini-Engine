//! **The Harbour Heist gate** (wave SCRIPT3) — a whole mission, authored as one
//! `.infini` file, run twice and required identical.
//!
//! `script_gameplay_gate` proved `PIE == shipping` over a script that defines a
//! catalogue and hands out a crate on a timer. This is the arc's dogfood: a
//! mission with objectives, a state machine, a bolted door, loot, stakes and two
//! outcomes — and **no Rust behind any of it**. The level is a quay, a
//! grammar-built vault and a hero; everything the mission needs it makes for
//! itself on `BeginPlay`.
//!
//! # What is compared
//!
//! Per step: the whole world's state hash, and the mission's own member
//! variables as **bit patterns**. Blueprint `vars` are deliberately not in
//! `step_state_hash` (the phase22 gate's `var_bits`, met again for the reason it
//! was written), so a mission's state has to be traced explicitly or two hosts
//! could agree about a world while disagreeing about the mission in it.
//!
//! # Two routes, because a state machine with one path is a sequence
//!
//! * **The clean run** — in, loot the shelf, out on `bars >= 6` with time to
//!   spare and condition to spare: **clear**.
//! * **The interrupted run** — in, take a couple of bars, step back out of the
//!   vault where the shelf stops paying and the bank's staff can see you, let the
//!   clock run out, and reach the quay too hurt to make the boat: **caught**.
//!
//! Two exits from the vault (`bars >= 6` and `clock <= 0`) and two endings
//! (clear and caught), out of one script over one level. A gate that only drove
//! the first would certify a mission whose timeout and whose failure state both
//! did nothing. Both routes are compared PIE-against-shipping in full.
//!
//! # The hero is moved by the GATE, not by a controller
//!
//! The level's hero carries no `CharacterMovement` (see `inf_editor_core::heist`)
//! — this is a mission gate, not a locomotion gate, and a player-controlled mover
//! would fold gravity, ground snap and a camera into every step of the trace.
//! The gate moves the hero the way a player's hands would and the mission reacts,
//! which is the division of labour the mission itself has. Both sims are moved
//! identically, which is what makes the comparison about the mission.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use inf_ecs::components::Transform;
use inf_ecs::math::Vec3d;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;
use uuid::Uuid;

use inf_editor_core::heist::{
    heist_dir, heist_script, HEIST_ALARM_MESH_GUID, HEIST_BOAT_AT, HEIST_HERO_GUID,
    HEIST_HERO_HALF_H, HEIST_HERO_RADIUS, HEIST_HERO_START, HEIST_HOUSING_PCG_GUID, HEIST_PCG_GUID,
    HEIST_SCRIPT_GUID, HEIST_VAULT_AT, HEIST_VAULT_DOOR_AT,
};

const HZ: u32 = 60;

/// The mission's own vocabulary, as the gate reads it back.
const PHASE_APPROACH: i64 = 0;
const PHASE_IN_THE_VAULT: i64 = 1;
const PHASE_RUNNING: i64 = 2;
const PHASE_CLEAR: i64 = 3;
const PHASE_CAUGHT: i64 = 4;

// ── the fixture ─────────────────────────────────────────────────────────────

fn sample_files() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(heist_dir())
        .expect("the committed mission is there")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .collect();
    v.sort();
    v
}

/// **The fixture really is what is committed**, file for file — a cooked mission
/// missing a file the sample has is a gate measuring a smaller world than the one
/// an author opens.
#[test]
fn the_fixture_copies_every_committed_file() {
    let files = sample_files();
    println!("the committed mission is {files:?}");
    for want in [
        "HarbourHeist.infini",
        "HarbourHeist.infini.toml",
        "HarbourHeist.inf_lvl",
        "HarbourHeist.inf_lvl.toml",
        "HarbourVault.inf_pcg",
        "Alarm.inf_mesh",
        "README.md",
    ] {
        assert!(files.iter().any(|f| f == want), "{want} is missing");
    }
}

fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Harbour Heist", "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("a content root");
    for f in sample_files() {
        std::fs::copy(heist_dir().join(&f), content.join(&f)).expect("copy");
    }
    proj
}

fn read_asset(dir: &Path, guid: Uuid) -> Option<Vec<u8>> {
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if path.extension().is_some_and(|e| e == "toml") {
            continue;
        }
        if let Ok(side) = inf_asset::AssetSidecar::load(&path) {
            if side.guid.0 == guid {
                return std::fs::read(&path).ok();
            }
        }
    }
    None
}

/// **The shipping side**: a sim off the cooked pack, the way `--pack` boots.
fn pack_sim(out: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(out).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the world builds");
    inf_player::sim_from_built(built)
}

/// **The PIE side**: the payload the editor really builds, resolving the mission
/// through the same class closure a `.inf_act` goes through.
fn pie_sim() -> RuntimeSim {
    let dir = heist_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("HarbourHeist.inf_lvl"))
        .expect("the level loads");
    let src = heist_script().expect("the mission is committed");
    let (class, warnings) = inf_script::compile_bytes(
        &src,
        "HarbourHeist.infini",
        format!("script:{}", inf_asset::AssetId(HEIST_SCRIPT_GUID)),
    )
    .unwrap_or_else(|d| panic!("{}", inf_script::render(&d)));
    assert!(warnings.is_empty(), "{}", inf_script::render(&warnings));
    let vault = read_asset(&dir, HEIST_PCG_GUID).expect("the vault graph is on disk");
    let housing = read_asset(&dir, HEIST_HOUSING_PCG_GUID).expect("the housing graph is on disk");
    let mesh = read_asset(&dir, HEIST_ALARM_MESH_GUID).expect("the alarm mesh is on disk");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == HEIST_SCRIPT_GUID).then(|| class.clone()),
        |guid| match guid {
            g if g == HEIST_PCG_GUID => Some(vault.clone()),
            g if g == HEIST_HOUSING_PCG_GUID => Some(housing.clone()),
            _ => None,
        },
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |guid| (guid == HEIST_ALARM_MESH_GUID).then(|| mesh.clone()),
        |_| None,
        HZ,
        false,
    )
    .expect("the payload builds");
    // Non-vacuity at the payload, before anything is compared: a payload with no
    // class boots a world where nothing was authored, and two hosts agree
    // perfectly about that.
    assert_eq!(
        payload.classes.len(),
        1,
        "the mission's class must ride the wire"
    );
    // BOTH graphs, and the count is exact: the first cut resolved the vault
    // alone, the housing block grew nothing in PIE, the level had no homes and
    // therefore **no crowd** — and the trace diverged from the shipped one at
    // step 1 with a `witnesses` of zero on one side. An exact expected count
    // taken from the fixture is the P21.4 rule, and this is what it is for.
    assert_eq!(
        payload.pcgs.len(),
        2,
        "the vault AND the housing block must ride the wire"
    );
    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}

// ── the routes ──────────────────────────────────────────────────────────────

/// Where the hero stands at step `i`. Both hosts are driven by the same
/// function, so any difference in the trace is the mission's.
type Route = fn(usize) -> (f64, f64, f64);

/// Metres up from a foot position to the capsule's centre.
fn stand(at: (f64, f64, f64)) -> (f64, f64, f64) {
    (at.0, at.1 + HEIST_HERO_HALF_H + HEIST_HERO_RADIUS, at.2)
}

/// **The clean run**: approach for a third of a second, loot the vault until the
/// shelf is empty, then run for the boat.
fn clean_run(i: usize) -> (f64, f64, f64) {
    if i < 20 {
        stand(HEIST_HERO_START)
    } else if i < 200 {
        stand(HEIST_VAULT_AT)
    } else {
        stand(HEIST_BOAT_AT)
    }
}

/// **The interrupted run**: in, a couple of bars, back out onto the plaza —
/// where the shelf stops paying, the staff can see you and the clock runs at
/// double speed — and on to the quay when it expires.
fn interrupted_run(i: usize) -> (f64, f64, f64) {
    if i < 20 {
        stand(HEIST_HERO_START)
    } else if i < 50 {
        stand(HEIST_VAULT_AT)
    } else if i < 210 {
        stand(HEIST_HERO_START)
    } else {
        // …and on to the quay anyway, which is what makes the ending TERMINAL
        // rather than merely reached: the mission is caught at the top of phase
        // 2 and standing on the boat afterwards changes nothing.
        stand(HEIST_BOAT_AT)
    }
}

fn place(sim: &mut RuntimeSim, at: (f64, f64, f64)) {
    let world = sim.world_mut();
    let e = world.entity_of(HEIST_HERO_GUID).expect("the hero");
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::new(at.0, at.1, at.2);
    }
    world.mark_dirty();
}

// ── the trace ───────────────────────────────────────────────────────────────

/// One step: the whole world, plus the mission's own state as bit patterns.
#[derive(Clone, PartialEq, Debug)]
struct Frame {
    state: u64,
    vars: [u64; 7],
}

fn var_bits(sim: &RuntimeSim, name: &str) -> u64 {
    match sim.actor_var(HEIST_HERO_GUID, name) {
        Some(inf_blueprint::Value::Float(f)) => f.to_bits(),
        Some(inf_blueprint::Value::Int(i)) => *i as u64,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn var_i(sim: &RuntimeSim, name: &str) -> i64 {
    match sim.actor_var(HEIST_HERO_GUID, name) {
        Some(inf_blueprint::Value::Int(i)) => *i,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn var_f(sim: &RuntimeSim, name: &str) -> f64 {
    match sim.actor_var(HEIST_HERO_GUID, name) {
        Some(inf_blueprint::Value::Float(f)) => *f,
        other => panic!("`{name}` is {other:?}"),
    }
}

fn run_trace(sim: &mut RuntimeSim, route: Route, steps: usize) -> Vec<Frame> {
    (0..steps)
        .map(|i| {
            place(sim, route(i));
            sim.step_once(RuntimeInput::default());
            Frame {
                state: inf_player::step_state_hash(sim),
                vars: [
                    var_bits(sim, "phase"),
                    var_bits(sim, "clock"),
                    var_bits(sim, "grab"),
                    var_bits(sim, "bars"),
                    var_bits(sim, "alarm"),
                    var_bits(sim, "witnesses"),
                    var_bits(sim, "condition"),
                ],
            }
        })
        .collect()
}

/// The steps at which the mission's phase changed, and what it changed to.
fn phase_changes(trace: &[Frame]) -> Vec<(usize, i64)> {
    let mut out = Vec::new();
    let mut last = PHASE_APPROACH;
    for (i, f) in trace.iter().enumerate() {
        let p = f.vars[0] as i64;
        if p != last {
            out.push((i, p));
            last = p;
        }
    }
    out
}

fn compare(ship: &[Frame], pie: &[Frame], label: &str) {
    for (i, (s, p)) in ship.iter().zip(pie.iter()).enumerate() {
        assert_eq!(s.state, p.state, "{label}: world diverged at step {i}");
        assert_eq!(
            s.vars, p.vars,
            "{label}: the mission's own state diverged at step {i}"
        );
    }
    assert_eq!(ship.len(), pie.len());
}

/// **Anti-vacuity**: two worlds where nothing happened are identical, and so are
/// two missions that never started.
fn assert_the_mission_ran(trace: &[Frame], label: &str) {
    let states: BTreeSet<u64> = trace.iter().map(|f| f.state).collect();
    let vars: BTreeSet<[u64; 7]> = trace.iter().map(|f| f.vars).collect();
    println!(
        "{label}: {} distinct world states, {} distinct mission states over {} steps; \
         phase changes {:?}",
        states.len(),
        vars.len(),
        trace.len(),
        phase_changes(trace)
    );
    assert!(
        states.len() >= 4,
        "{label}: only {} distinct world states — the mission ran and changed \
         NOTHING in the world, which is the vacuous shape this gate refuses",
        states.len()
    );
    assert!(
        vars.len() >= 8,
        "{label}: only {} distinct mission states",
        vars.len()
    );
}

// ── the arms ────────────────────────────────────────────────────────────────

/// # THE ARM: the mission runs identically in PIE and in a shipped build.
#[test]
fn the_mission_is_the_same_program_in_pie_and_shipping() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = scaffold(tmp.path());
    let out = tmp.path().join("out");
    // The COOK's own cost, printed beside the mission's iteration cost: this is
    // the other half of the arc's compile-time table, and it is what an author
    // pays to ship rather than to try. Printed, never asserted (no wall-clock
    // assertions) — and it is a whole project: two grammar graphs, a level, a
    // mesh and the mission.
    let cooked = std::time::Instant::now();
    let report = inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the mission cooks");
    println!(
        "the mission's project cooks in {:.0} ms",
        cooked.elapsed().as_secs_f64() * 1000.0
    );
    println!("{}", report.render());
    assert!(!report.has_blocking(), "{:?}", report.blocking);
    assert_eq!(
        report.kinds.get("script"),
        Some(&1),
        "the mission must be packed as a script: {:?}",
        report.kinds
    );
    // **THE COOK'S ASSET WALK, BITING ON COMMITTED CONTENT.** `engine.spawn`'s
    // prefab is the node kit's only `StrRole::Asset` port, and until SCRIPT3 it
    // served a verb neither host implemented. The mission names `Alarm`, the
    // walk resolves that stem against the project, and the mesh is in the pack
    // because of that edge and nothing else -- the level does not reference it
    // and no component names it. A stem that resolved to nothing would be a
    // BLOCKING advisory, which is what `has_blocking` above is really testing;
    // this is the positive half of the same claim.
    assert_eq!(
        report.kinds.get("mesh"),
        Some(&1),
        "the asset the MISSION names must be in the pack: {:?}",
        report.kinds
    );

    for (label, route, steps, ending) in [
        ("clean", clean_run as Route, 340, PHASE_CLEAR),
        ("interrupted", interrupted_run as Route, 300, PHASE_CAUGHT),
    ] {
        let mut ship = pack_sim(&out);
        let mut pie = pie_sim();
        let ship_trace = run_trace(&mut ship, route, steps);
        let pie_trace = run_trace(&mut pie, route, steps);
        assert_the_mission_ran(&ship_trace, label);
        assert_the_mission_ran(&pie_trace, label);
        compare(&ship_trace, &pie_trace, label);

        // …and the mission ENDED, on both hosts. A trace that is identical and
        // never leaves phase 0 is two hosts agreeing about a mission that never
        // started.
        for (host, sim) in [("shipping", &ship), ("pie", &pie)] {
            assert_eq!(
                var_i(sim, "phase"),
                ending,
                "{label}/{host}: the mission ended in phase {}, not {ending}",
                var_i(sim, "phase")
            );
        }
        println!(
            "{label}: ended in phase {} with {} bars, condition {:.3}, {} witnesses",
            var_i(&ship, "phase"),
            var_i(&ship, "bars"),
            var_f(&ship, "condition"),
            var_i(&ship, "witnesses")
        );
    }
}

/// **The two routes take the two exits**, which is what makes the mission a state
/// machine rather than a sequence.
///
/// The clean run leaves on `bars >= 6` with the clock still running; the
/// interrupted run steps out of the vault, stops being paid, and leaves on
/// `clock <= 0` with what it had. Same script, same level, two outcomes.
#[test]
fn the_two_routes_take_the_two_exits() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = scaffold(tmp.path());
    let out = tmp.path().join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default()).expect("cooks");

    let mut clean = pack_sim(&out);
    let clean_trace = run_trace(&mut clean, clean_run as Route, 340);
    let mut interrupted = pack_sim(&out);
    let interrupted_trace = run_trace(&mut interrupted, interrupted_run as Route, 300);
    // The two ENDINGS, which is the half the exits alone do not say.
    assert_eq!(var_i(&clean, "phase"), PHASE_CLEAR);
    assert_eq!(var_i(&interrupted, "phase"), PHASE_CAUGHT);
    assert!(
        var_f(&interrupted, "condition") <= 0.5,
        "the interrupted run is caught because it is HURT: {}",
        var_f(&interrupted, "condition")
    );
    assert!(
        var_f(&clean, "condition") > 0.5,
        "the clean run was watched too, and got out before it mattered: {}",
        var_f(&clean, "condition")
    );

    // The clean run: all six bars, and it left with time on the clock.
    assert_eq!(var_i(&clean, "bars"), 6, "the shelf holds six bars");
    let clean_out = phase_changes(&clean_trace)
        .into_iter()
        .find(|(_, p)| *p == PHASE_RUNNING)
        .expect("the clean run leaves the vault");
    let clock_at_exit = f64::from_bits(clean_trace[clean_out.0].vars[1]);
    assert!(
        clock_at_exit > 0.0,
        "the clean run left on the LOOT, so its clock must still be running: \
         {clock_at_exit}"
    );

    // The interrupted run: fewer bars, and the clock is what put it out.
    let taken = var_i(&interrupted, "bars");
    assert!(
        (1..6).contains(&taken),
        "the interrupted run should leave with some of the bullion and not all \
         of it, got {taken}"
    );
    let int_out = phase_changes(&interrupted_trace)
        .into_iter()
        .find(|(_, p)| *p == PHASE_RUNNING)
        .expect("the interrupted run leaves the vault");
    assert!(
        f64::from_bits(interrupted_trace[int_out.0].vars[1]) <= 0.0,
        "the interrupted run left on the CLOCK, so its clock must have expired"
    );
    println!(
        "clean: 6 bars, out at step {} with {clock_at_exit:.2}s left; \
         interrupted: {taken} bars, out at step {} on the clock",
        clean_out.0, int_out.0
    );
}

/// **The mission reaches the world**, in every direction it claims to.
///
/// The trace comparison above proves the two hosts agree; it cannot say the
/// mission *did* anything, because two hosts agreeing about nothing is the
/// vacuous shape. So each verb family is measured where it lands: the bolt on
/// the door, the alarm in the world, the bullion in the bag, the hero turned
/// around.
#[test]
fn every_verb_family_the_mission_uses_reaches_the_world() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = scaffold(tmp.path());
    let out = tmp.path().join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default()).expect("cooks");
    let mut sim = pack_sim(&out);

    let door_at = glam::DVec3::new(
        HEIST_VAULT_DOOR_AT.0,
        HEIST_VAULT_DOOR_AT.1,
        HEIST_VAULT_DOOR_AT.2,
    );
    // `BeginPlay` has already run: the catalogue is defined, the door is hung
    // and the bullion is on the floor.
    assert!(
        !inf_physics::d3::door::is_locked_near(sim.world_mut(), door_at),
        "`door.spawn` must hang the vault door unlocked"
    );

    // Step into the vault and watch the mission take hold of the world.
    let trace = run_trace(&mut sim, clean_run as Route, 340);
    let broke_in = phase_changes(&trace)
        .into_iter()
        .find(|(_, p)| *p == PHASE_IN_THE_VAULT)
        .expect("the mission breaks in");

    // engine.spawn — the alarm is a real entity, under the identity the CONTENT
    // names, and `engine.destroy` takes it away again on the way out.
    let alarm_guid = inf_ecs::prefab::authored_spawn_guid(
        "Alarm",
        Vec3d::new(
            HEIST_VAULT_AT.0,
            HEIST_VAULT_AT.1 + HEIST_HERO_HALF_H + HEIST_HERO_RADIUS,
            HEIST_VAULT_AT.2,
        ),
    );
    let alarm_handle = trace[broke_in.0].vars[4] as i64;
    assert_eq!(
        alarm_handle,
        inf_ecs::prefab::spawn_entity_id(alarm_guid),
        "the handle the mission holds is the one the identity folds to"
    );
    assert_eq!(
        var_i(&sim, "alarm"),
        0,
        "`engine.destroy` must clear the alarm on the way out"
    );
    assert!(
        sim.world_mut().entity_of(alarm_guid).is_none(),
        "the destroyed alarm must be gone from the world, not just from the \
         mission's variable"
    );

    // engine.set_rotation — the hero turned to face the door on the way in.
    let hero = sim
        .world_mut()
        .entity_of(HEIST_HERO_GUID)
        .expect("the hero");
    assert_eq!(
        sim.world_mut()
            .world()
            .get::<Transform>(hero)
            .expect("a transform")
            .rotation
            .y,
        180.0,
        "`engine.set_rotation` must have turned the hero around"
    );

    // item.* — six bars off the shelf and into the bag.
    assert_eq!(var_i(&sim, "bars"), 6);
    assert_eq!(
        inf_ecs::item::inventory_of(sim.world_mut(), HEIST_HERO_GUID)
            .map(|inv| inv.count_of("bullion"))
            .unwrap_or(0),
        6,
        "the bag is the world's opinion; the variable is the mission's"
    );

    // health.* — the hero has a body, and it is the one the mission set.
    let left = inf_ecs::weapon::health_of(sim.world_mut(), HEIST_HERO_GUID)
        .map(|h| h.joules)
        .unwrap_or(0.0);
    assert!(
        left > 0.0,
        "the mission set the hero's health and it stands"
    );
    println!("the hero ends with {left} J and six bars");
}
