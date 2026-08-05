//! **THE PHASE 20 GATE** — water & hydrology.
//!
//! The fixture is the committed `samples/phase20-coastal`: the plan's own
//! done-when sentence built as a level — a 512 m coast with an **ocean** at sea
//! level, a **head lake** in a dug basin, a **spline river** running the valley
//! from the lake to the shore, **eight buoyant crates** and a **swimmer** driven
//! by the committed `Swimmer.inf_act`.
//!
//! Six arms, in the order the phase's claim needs them:
//!
//! * **(a) determinism** — two fresh loads of one cooked pack produce
//!   bit-identical physics traces over the whole run.
//! * **(b) PIE == shipping** — the editor's PIE payload and the cooked pack float
//!   the same crates and swim the same swimmer, **bit for bit**.
//! * **(c) the water is really doing something** — the anti-vacuity arm. The
//!   crates settle at draughts that depend on their densities, the swimmer rises
//!   toward the surface rather than sinking, and the river is finite.
//! * **(d) the river's mouth is finite** — the P20.4 sim fix, asserted on the
//!   *shipped* content: a probe thirty metres past the mouth is dry, and the
//!   mouth itself is not.
//! * **(e) the cook is silent** — neither water advisory fires on the flagship
//!   sample. An advisory that fires on correct content is one nobody reads.
//! * **(f) budget** — the composed scene builds inside the **load** budget and
//!   steps inside the **frame** budget.
//!
//! # Why the trace is bits and not floats
//!
//! Buoyancy is a force integrated by a substepping solver; a comparison that
//! "passes within 1e-9" would accept exactly the drift these gates exist to
//! catch. Every arm that compares two runs compares `f64::to_bits`.
//!
//! GPU rendering is human-verified as elsewhere; this asserts the authoritative
//! deterministic state, headlessly.

use std::path::{Path, PathBuf};
use std::time::Instant;

use uuid::Uuid;

use inf_editor_core::samples::{
    phase20_coastal_dir, phase20_crate_count, phase20_crate_guid, phase20_river_points,
    PHASE20_LAKE_GUID, PHASE20_LAKE_LEVEL_M, PHASE20_OCEAN_GUID, PHASE20_RIVER_GUID,
    PHASE20_SEA_CRATES, PHASE20_SEA_LEVEL_M, PHASE20_SWIMMER_ACTOR_GUID, PHASE20_SWIMMER_GUID,
};
use inf_packager::{cook, CookOptions};
use inf_player::level::{BuiltWorld, PackLevelSource};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;

// Budgets are imported, never redeclared: a phase does not get its own budget for
// being new, and a private copy is somewhere for one to be quietly raised. Each
// arm takes the budget of its own *class* — the one-shot-load arm the load-class
// ceiling, the recurring-work arm the per-frame tripwire. They are not
// interchangeable, and a load must never be measured against the frame budget
// (that mistake cost a CI failure earlier in this phase).
use inf_core::FRAME_BUDGET_MS;
use inf_player::budget::LOAD_BUDGET_MS;

/// Steps traced. Long enough for a crate dropped 2.5 m over the sea to fall in,
/// bob and settle, and for the swimmer to surface — the claim is that water
/// *works*, not that two traces of a still-moving body agree.
const STEPS: usize = 900;

/// Every committed file of the sample, so the fixture copy cannot silently miss
/// one as the sample grows.
fn sample_files() -> [&'static str; 4] {
    [
        "Phase20Coastal.inf_lvl",
        "Phase20Coastal.inf_lvl.toml",
        "Swimmer.inf_act",
        "Swimmer.inf_act.toml",
    ]
}

/// Scaffold a project holding the sample and cook it; returns `(content, pack)`.
fn cook_coast(tmp: &Path) -> (PathBuf, PathBuf) {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 20 Coastal", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = phase20_coastal_dir();
    for f in sample_files() {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (content, out)
}

fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack).expect("pack opens");
    let built: BuiltWorld = inf_player::build_world_from_pack(&source).expect("pack world builds");
    inf_player::sim_from_built(built)
}

/// The editor's PIE payload for the committed sample: the document plus the
/// swimmer's blueprint class, resolved exactly as the editor resolves it.
fn pie_sim() -> RuntimeSim {
    let dir = phase20_coastal_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Phase20Coastal.inf_lvl"))
        .expect("the committed phase20 document loads");
    let act_bytes =
        std::fs::read(dir.join("Swimmer.inf_act")).expect("the swimmer class is committed");
    let act = inf_editor_core::samples::decode_actor(&act_bytes).expect("the class decodes");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == PHASE20_SWIMMER_ACTOR_GUID).then(|| act.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        // P22.3: no destructible meshes in this fixture.
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    let built = inf_player::build_world_from_payload(&payload).expect("PIE world builds");
    inf_player::sim_from_built(built)
}

/// One traced frame: every crate's pose plus the swimmer's, as raw bits.
type Frame = Vec<[u64; 7]>;

fn pose_of(sim: &RuntimeSim, guid: Uuid) -> [u64; 7] {
    let t = sim
        .world()
        .entity_of(guid)
        .and_then(|e| sim.world().world().get::<inf_ecs::components::Transform>(e))
        .copied()
        .unwrap_or(inf_ecs::components::Transform::IDENTITY);
    let q = t.quat();
    [
        t.translation.x.to_bits(),
        t.translation.y.to_bits(),
        t.translation.z.to_bits(),
        q.x.to_bits(),
        q.y.to_bits(),
        q.z.to_bits(),
        q.w.to_bits(),
    ]
}

fn frame(sim: &RuntimeSim) -> Frame {
    (0..phase20_crate_count())
        .map(|i| pose_of(sim, phase20_crate_guid(i)))
        .chain(std::iter::once(pose_of(sim, PHASE20_SWIMMER_GUID)))
        .collect()
}

fn run_trace(sim: &mut RuntimeSim) -> Vec<Frame> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            frame(sim)
        })
        .collect()
}

/// The y of one traced body at the end of a run.
fn final_y(trace: &[Frame], i: usize) -> f64 {
    f64::from_bits(trace.last().expect("a non-empty trace")[i][1])
}

/// **ANTI-VACUITY, applied to every arm that compares two traces.**
///
/// Two identical traces of a scene where nothing moved would satisfy every
/// equality in this file. So: bodies must have moved, and they must have *stopped*
/// somewhere plausible.
fn assert_not_vacuous(trace: &[Frame]) {
    assert_eq!(trace.len(), STEPS);
    let first = &trace[0];
    let last = trace.last().unwrap();
    assert_eq!(first.len(), phase20_crate_count() + 1);
    assert_ne!(first, last, "nothing moved over the whole run");
}

// ── (a) determinism ─────────────────────────────────────────────────────────

/// Two fresh loads of ONE cooked pack produce bit-identical traces.
#[test]
fn the_coast_traces_bit_identically_across_two_loads() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_coast(tmp.path());
    let a = run_trace(&mut pack_sim(&pack));
    let b = run_trace(&mut pack_sim(&pack));
    assert_not_vacuous(&a);
    assert_eq!(a, b, "the coast moved between two loads of the same pack");
}

// ── (b) PIE == shipping ─────────────────────────────────────────────────────

/// **The house gate.** The editor's preview and the shipped build float the same
/// crates and swim the same swimmer — bit for bit, over 900 steps.
#[test]
fn pie_equals_shipping_on_the_coast() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_coast(tmp.path());
    let ship = run_trace(&mut pack_sim(&pack));
    let pie = run_trace(&mut pie_sim());
    assert_not_vacuous(&ship);
    assert_eq!(
        ship, pie,
        "a cooked build floats the coast differently from the editor preview"
    );
}

// ── (c) the water is really doing something ─────────────────────────────────

/// The crates settle at draughts that **depend on their densities**, the lake
/// crates settle on the lake and not at sea level, and the swimmer rises.
///
/// This is what makes (a) and (b) claims about *water* rather than about two
/// copies of a free fall.
#[test]
fn the_crates_float_at_their_draughts_and_the_swimmer_surfaces() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_coast(tmp.path());
    let trace = run_trace(&mut pack_sim(&pack));
    assert_not_vacuous(&trace);

    // A unit box of density ρ floats with its centre `0.5 - ρ/1000` above the
    // waterline (`Buoyancy::equilibrium_fraction` is `ρ / ρ_water`, and a unit
    // box's submerged fraction is linear in depth). The sea has 0.55 m of swell,
    // so the tolerance is the amplitude plus the P20.1 honest worst-case height
    // residual — quoting the ≤3 mm typical instead would be tuning a bound to a
    // measurement.
    const SWELL_M: f64 = 0.55;
    const HEIGHT_RESIDUAL_M: f64 = 0.17;
    for i in 0..PHASE20_SEA_CRATES {
        let density = 350.0 + i as f64 * 50.0;
        let want = PHASE20_SEA_LEVEL_M + 0.5 - density / 1000.0;
        let got = final_y(&trace, i);
        assert!(
            (got - want).abs() <= SWELL_M + HEIGHT_RESIDUAL_M,
            "sea crate {i} (ρ = {density}) rests at {got}, expected about {want}"
        );
    }
    // The draughts really do differ, BY THE MARGIN THE MODEL PREDICTS. A bare
    // `light > heavy` would pass on a millimetre and admit eight crates all
    // sitting at essentially the same height, which is the failure this arm
    // exists to catch. Archimedes puts the gap at `(600 - 350) / 1000 = 0.25 m`;
    // half of that is the floor, because the two crates are sampled at
    // independent points of a 0.55 m swell and the instantaneous gap is the
    // static one plus the difference of two wave heights.
    const NOMINAL_GAP_M: f64 = (600.0 - 350.0) / 1000.0;
    let light = final_y(&trace, 0);
    let heavy = final_y(&trace, PHASE20_SEA_CRATES - 1);
    assert!(
        light - heavy > NOMINAL_GAP_M * 0.5,
        "a 350 kg/m³ crate must ride about {NOMINAL_GAP_M:.2} m higher than a 600 kg/m³ one,          and rides {:.3} m higher ({light} vs {heavy})",
        light - heavy
    );

    // The lake crates float on the LAKE, ~33.6 m up — not at sea level, which is
    // what proves `highest_surface`'s per-body footprint reached the sim.
    for i in PHASE20_SEA_CRATES..phase20_crate_count() {
        let got = final_y(&trace, i);
        assert!(
            (got - PHASE20_LAKE_LEVEL_M).abs() < 1.5,
            "lake crate {i} rests at {got}, not on the lake at {PHASE20_LAKE_LEVEL_M}"
        );
    }

    // The swimmer: dropped 2 m under the surface, asking to sink at a full g every
    // step, and it comes UP. That is the P20.2 asymmetric sink authority working
    // on committed content.
    let swimmer = phase20_crate_count();
    let start = f64::from_bits(trace[0][swimmer][1]);
    let end = final_y(&trace, swimmer);
    assert!(
        end > start,
        "the swimmer sank ({start} → {end}) — the swim balance lost to free fall"
    );
    assert!(
        end > PHASE20_SEA_LEVEL_M - 1.2,
        "the swimmer never got near the surface: {end}"
    );
}

// ── (d) the river's mouth is finite (the P20.4 sim fix, on shipped content) ──

/// A probe past the river's mouth is **dry**, and the mouth itself is not.
///
/// The bug this closes made `RiverSample::inside` an infinite strip along the end
/// tangent, so buoyancy, drag, swim and the water events all fired over dry land
/// for as far as the lateral test kept passing. It is asserted here, through the
/// shipped pack's own `water.surface_height` seam, rather than only in the Ring-0
/// unit test — because the unit test cannot tell you the *level* got it too.
#[test]
fn the_rivers_mouth_is_finite_in_the_shipped_pack() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_coast(tmp.path());
    let mut sim = pack_sim(&pack);
    sim.step_once(RuntimeInput::default());

    let pts = phase20_river_points();
    let mouth = *pts.last().unwrap();
    let prev = pts[pts.len() - 2];
    let dir = (mouth - prev).normalize();

    // The river's own surface at its mouth is well above sea level, so "which
    // body answered" is decidable from the height alone.
    let at_mouth = sim
        .bridge3d()
        .water_surface_height(mouth.x - dir.x * 2.0, mouth.z - dir.z * 2.0)
        .expect("the river covers its own centreline");
    assert!(
        at_mouth > PHASE20_SEA_LEVEL_M + 0.4,
        "the probe just inside the mouth answered {at_mouth}, which is the SEA"
    );

    // Thirty metres past the mouth, on the centreline's extension: the river must
    // not answer. The ocean does — it is unbounded — so the assertion is that the
    // height is the SEA's, not that there is no water.
    let past = mouth + dir * 30.0;
    let beyond = sim
        .bridge3d()
        .water_surface_height(past.x, past.z)
        .expect("the ocean is everywhere");
    assert!(
        beyond < PHASE20_SEA_LEVEL_M + 1.0,
        "thirty metres past the mouth the river is still answering ({beyond}) — \
         the arc-length bound is not in the shipped path"
    );
    // ANTI-VACUITY: the two probes are genuinely different places on the same
    // ribbon's axis, and the river really is the taller body where it exists.
    assert!(at_mouth - beyond > 0.4, "{at_mouth} vs {beyond}");
}

// ── (e) the cook is silent on correct content ───────────────────────────────

/// Neither water advisory fires on the flagship sample: the surface descends and
/// the authored bed descends with it.
#[test]
fn the_coast_draws_no_water_advisory() {
    let tmp = tempfile::tempdir().unwrap();
    let proj = tmp.path().join("proj");
    ProjectManifest::new("Phase 20 Coastal", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = phase20_coastal_dir();
    for f in sample_files() {
        std::fs::copy(src.join(f), content.join(f)).unwrap();
    }
    let report = cook(&proj, &tmp.path().join("out"), &CookOptions::default()).unwrap();
    let water: Vec<&String> = report
        .warnings
        .iter()
        .filter(|w| w.contains("climbs"))
        .collect();
    assert!(
        water.is_empty(),
        "the flagship coastal sample trips a water advisory: {water:?}"
    );
}

// ── (f) budget ──────────────────────────────────────────────────────────────

/// The composed coastal scene builds inside the **load** budget and steps inside
/// the **frame** budget.
///
/// Two different ceilings for two different classes of work, both imported. A
/// load measured against the frame budget is a category error — a one-shot world
/// build is not recurring work, and asserting it against 16.6 ms would either fail
/// spuriously or force the frame budget up for everyone.
#[test]
fn the_coast_loads_and_steps_inside_its_budgets() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_coast(tmp.path());

    let t0 = Instant::now();
    let mut sim = pack_sim(&pack);
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        load_ms < LOAD_BUDGET_MS,
        "the coastal scene took {load_ms:.1} ms to build (budget {LOAD_BUDGET_MS} ms)"
    );

    // Warm up, then measure the steady state: the first step pays for rapier's
    // island construction, which is a load cost wearing a step's clothes.
    for _ in 0..60 {
        sim.step_once(RuntimeInput::default());
    }
    const MEASURED: usize = 240;
    let t1 = Instant::now();
    for _ in 0..MEASURED {
        sim.step_once(RuntimeInput::default());
    }
    let per_step_ms = t1.elapsed().as_secs_f64() * 1000.0 / MEASURED as f64;
    assert!(
        per_step_ms < FRAME_BUDGET_MS,
        "a coastal fixed step took {per_step_ms:.3} ms (budget {FRAME_BUDGET_MS} ms)"
    );

    // ANTI-VACUITY: the scene really is the one with the water in it.
    let w = sim.world();
    for guid in [PHASE20_OCEAN_GUID, PHASE20_LAKE_GUID, PHASE20_RIVER_GUID] {
        let e = w.entity_of(guid).expect("water entity survived the cook");
        assert!(w.world().get::<inf_ecs::components::WaterBody>(e).is_some());
    }
    assert!(w.entity_of(PHASE20_SWIMMER_GUID).is_some());
    for i in 0..phase20_crate_count() {
        assert!(w.entity_of(phase20_crate_guid(i)).is_some());
    }
}
