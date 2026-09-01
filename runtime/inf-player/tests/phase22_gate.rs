//! **THE PHASE 22 GATE** — the dynamic world: deformation & destruction.
//!
//! The fixture is the committed `samples/phase22-playground`: the plan's own
//! done-when sentence built as a level — a **streamed terrain** painted in snow
//! and sand bands, a **roller** that drives itself across both and leaves a
//! trail, a **grass strip** straddling that trail, a **multi-storey block** and a
//! **car on revolute joints** that a Blueprint blows apart on scripted steps, and
//! a **control twin of the block** whose `runtime_destruct` is off.
//!
//! The arms, in the order the phase's claim needs them:
//!
//! * **(a) determinism** — two fresh loads of one cooked pack produce
//!   bit-identical traces, deformation field included.
//! * **(b) cooked == uncooked** — the same level off loose files and off a pack,
//!   up to the charge (a dev directory has no `.inf_fracture` by design), and
//!   demonstrably **different** past it.
//! * **(c) PIE == shipping** on the FULL trace, compared as raw bits — and the
//!   payload's `fractures` is asserted **non-empty before** anything is compared,
//!   which is what makes `pie.rs`'s own prose about this gate true.
//! * **(d) the REAL `--pie` subprocess** — the actual player binary, against the
//!   in-process reference. The guard-law arm.
//! * **(e) the WORLD** — the charge opened the block and left a structure behind,
//!   the car fractured and the structural solve dropped what the charge did not
//!   pay for, the trail is where the roller ran and nowhere else, the debris
//!   budget held, and the control twin survived the identical charge.
//! * **(f) budgets** — the load against `LOAD_BUDGET_MS`, the damage step
//!   measured against `FRAME_BUDGET_MS`.
//! * **(g) pool-size determinism** — a subprocess matrix at 1/2/4/8 workers.
//! * **(h) persistence** — a save after the damage is byte-identical to a save
//!   before it, which is this phase's ruling stated where a gate can fail on it.
//! * **(i) the cook is silent** — no advisory fires on the flagship sample,
//!   including the three this phase added.
//! * **(j) the keying claim, on real physics** — the rubble's content key holds
//!   while chunks demonstrably tumble, and moves when the live set does. The only
//!   arm that touches `inf_render::debris`.
//!
//! The letters are load-bearing: `samples/phase22-playground/README.md` and
//! ROADMAP §12 cite them, so an arm that is added or re-ordered has to be
//! re-cited with it. (The P22.4 audit found the README pointing at "arm (e)" for
//! the cook-silence arm, which is (i).)
//!
//! # Why the trace is bits and not floats
//!
//! Every quantity here is either an integer count or an `f64` the two sides are
//! claimed to agree about *exactly*; a comparison that "passes within 1e-9" would
//! accept precisely the drift this gate exists to catch. Every arm that compares
//! two runs compares `f64::to_bits` (and, for the deformation field, the raw
//! `state_bytes`).
//!
//! GPU rendering is human-verified as elsewhere; this asserts the authoritative
//! deterministic state, headlessly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use uuid::Uuid;

use inf_blueprint::Value;
use inf_editor_core::samples::{
    phase22_layer_at, phase22_playground_dir, PHASE22_BLOCK_MESH_GUID, PHASE22_CHASSIS_GUID,
    PHASE22_CHASSIS_MESH_GUID, PHASE22_CLIP_GUID, PHASE22_CONTROL_GUID, PHASE22_DEMO_ACTOR_GUID,
    PHASE22_GRASS_GUID, PHASE22_ROLLER_ACTOR_GUID, PHASE22_ROLLER_GUID, PHASE22_ROLLER_RADIUS_M,
    PHASE22_ROLLER_SPEED_M_S, PHASE22_ROLLER_START_Z, PHASE22_ROLLER_X, PHASE22_SAND_LAYER,
    PHASE22_SAND_Z, PHASE22_SNOW_LAYER, PHASE22_SNOW_Z, PHASE22_STEPS, PHASE22_TERRAIN_GUID,
    PHASE22_TOWER_GUID, PHASE22_TRIGGER_END, PHASE22_TRIGGER_STEP, PHASE22_WHEEL_GUIDS,
    PHASE22_WRECK_ACTOR_GUID,
};
use inf_packager::{cook, CookOptions};
use inf_player::level::{
    self, BuiltWorld, DevDirLevelSource, InfSceneWorldBuilder, PackLevelSource,
};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_project::ProjectManifest;

// Budgets are imported, never redeclared: a phase does not get its own budget for
// being new. Each arm takes the budget of its own *class* — the one-shot-load arm
// the load-class ceiling, the recurring-work arm the per-frame tripwire.
use inf_core::FRAME_BUDGET_MS;
use inf_player::budget::LOAD_BUDGET_MS;

/// Steps traced.
///
/// **THE WINDOW LAW** (`phase21_gate.rs`'s lesson, restated because this phase's
/// window is harder). The temptation with an expensive per-frame comparison is to
/// trace a prefix and stop. A prefix is exactly wrong here: nothing in this level
/// destroys anything until step [`PHASE22_TRIGGER_STEP`] (60), so **the first
/// quarter of the run is a settling car and a rolling ball** and two hosts that
/// disagree about destruction entirely trace identically through all of it. The
/// window has to contain the moment the world diverges — the block's demolition
/// string (steps 60–65) and the car bomb's single instant (step 60) — the
/// collapse they drive, and the **settle** after it, because a chunk that is
/// still in the air when the trace ends has not yet had a chance to land
/// somewhere different.
///
/// So the trace is the WHOLE run, and the run is sized from the other end: 240
/// steps is 4 s, of which 1 s is the pre-trigger settle, 0.1 s the charge and
/// **2.9 s the collapse and its debris settling**.
///
/// "Settled" is a measurement rather than a feeling, and the honest number is
/// **0.365 m**: that is how far the block's rubble still moves over the last
/// second of the run. It is not zero — a hull resting on a heightfield keeps
/// micro-settling — and it is two orders below the metres the collapse itself
/// covers, which is what the window has to contain. Defended: a run truncated to
/// 120 steps fails 7 of the 15 arms, and a perturbation injected at step 200 is
/// caught at step 200.
///
/// The number lives in the sample, not here, so the content and the test cannot
/// disagree about how long the run is.
const STEPS: usize = PHASE22_STEPS;

/// Every committed file of the sample, so the fixture copy cannot silently miss
/// one as the sample grows.
fn sample_files() -> [&'static str; 17] {
    [
        "Phase22Playground.inf_lvl",
        "Phase22Playground.inf_lvl.toml",
        "Phase22Playground.inf_terrain",
        "Phase22Playground.inf_terrain.toml",
        "Block.inf_mesh",
        "Block.inf_mesh.toml",
        "Chassis.inf_mesh",
        "Chassis.inf_mesh.toml",
        "Collapse.inf_audio",
        "Collapse.inf_audio.toml",
        "Demolition.inf_act",
        "Demolition.inf_act.toml",
        "Wreck.inf_act",
        "Wreck.inf_act.toml",
        "Roller.inf_act",
        "Roller.inf_act.toml",
        "README.md",
    ]
}

/// …and the list really is **every** committed file. This function exists so a
/// sample that grows a file cannot be cooked without it, which only works if
/// something checks the list against the directory (P21.4's finding: it was eight
/// of nine from the day it was written).
#[test]
fn the_fixture_copies_every_committed_sample_file() {
    let listed: std::collections::BTreeSet<String> =
        sample_files().iter().map(|s| s.to_string()).collect();
    let on_disk: std::collections::BTreeSet<String> = std::fs::read_dir(phase22_playground_dir())
        .expect("the sample directory exists")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        listed, on_disk,
        "`sample_files()` and samples/phase22-playground disagree — a cooked \
         fixture would be missing content the committed sample has"
    );
}

/// Scaffold a project holding the sample; returns its `Content` dir.
fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 22 Playground", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = phase22_playground_dir();
    for f in sample_files() {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    content
}

/// Scaffold + cook; returns `(content, pack)`.
fn cook_playground(tmp: &Path) -> (PathBuf, PathBuf) {
    let content = scaffold(tmp);
    let proj = tmp.join("proj");
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (content, out)
}

/// The **shipping** side: a cooked pack, with terrain streaming and the fracture
/// registry attached exactly as `run_headless` attaches them.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = PackLevelSource::open(pack).expect("pack opens");
    let built: BuiltWorld = inf_player::build_world_from_pack(&source).expect("pack world builds");
    let mut sim = inf_player::sim_from_built(built);
    let pack_file = pack.join(inf_player::level::PACK_FILE);
    let reader =
        std::sync::Arc::new(inf_asset::PackReader::open(&pack_file).expect("pack reader opens"));
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Pack(PackLevelSource::open(pack).expect("pack opens")),
    );
    inf_player::fracture::attach_fractures(
        &mut sim,
        &inf_player::fracture::FractureRegistry::from_pack(reader),
    );
    sim
}

/// The three classes the level binds, decoded off disk once.
fn bindings(content: &Path) -> HashMap<Uuid, inf_blueprint::BlueprintClass> {
    [
        ("Demolition.inf_act", PHASE22_DEMO_ACTOR_GUID),
        ("Wreck.inf_act", PHASE22_WRECK_ACTOR_GUID),
        ("Roller.inf_act", PHASE22_ROLLER_ACTOR_GUID),
    ]
    .into_iter()
    .map(|(file, guid)| {
        let bytes = std::fs::read(content.join(file)).expect("the class is committed");
        (
            guid,
            inf_editor_core::samples::decode_actor(&bytes).expect("the class decodes"),
        )
    })
    .collect()
}

/// The **loose-files** side, for the cooked-==-uncooked arm.
///
/// Note what it deliberately does NOT do: attach a fracture registry. There is no
/// `FractureRegistry::from_dir`, on purpose — a `.inf_fracture` is *derived at
/// cook*, so a dev `--level` run previews as indestructible and re-deriving on
/// load would make a `--level` run and a `--pack` run differ in *when* the
/// chunking happened. This arm therefore compares the two sides on everything a
/// dev run can have: the deformation, the physics, the ground and the refusals.
fn dir_sim(content: &Path) -> RuntimeSim {
    let source = DevDirLevelSource::new(content.join("Phase22Playground.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new()).with_bindings(bindings(content));
    let built = level::load(&source, &builder).expect("dev-dir world builds");
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_terrain_streaming(
        &mut sim,
        &inf_player::TerrainContent::Dir(level::terrain_paths_by_guid_from_dir(content)),
    );
    sim
}

/// The **editor preview** side: the payload the editor really builds, through the
/// one PIE boot seam the `--pie` subprocess takes.
fn pie_sim() -> RuntimeSim {
    inf_player::sim_from_payload(&playground_payload())
        .expect("PIE world builds")
        .sim
}

/// The payload the editor really builds for the committed sample — the input to
/// both the in-process PIE arm and the real-subprocess one, so the two cannot
/// disagree about what was sent.
fn playground_payload() -> inf_runtime::pie::ScenePayload {
    let dir = phase22_playground_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Phase22Playground.inf_lvl"))
        .expect("the committed phase22 document loads");
    let classes = bindings(&dir);
    let terrain_bytes =
        std::fs::read(dir.join("Phase22Playground.inf_terrain")).expect("the terrain is committed");
    let block = std::fs::read(dir.join("Block.inf_mesh")).expect("the block mesh is committed");
    let chassis =
        std::fs::read(dir.join("Chassis.inf_mesh")).expect("the chassis mesh is committed");
    let audio = std::fs::read(dir.join("Collapse.inf_audio")).expect("the clip is committed");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| classes.get(&guid).cloned(),
        |_| None,
        |guid| (guid == PHASE22_CLIP_GUID).then(|| audio.clone()),
        |_| None,
        |_| None,
        // The BYTES route (`ScenePayload` v12 kept both) — see `phase21_gate`.
        |_| {
            Some(inf_editor_core::pie::TerrainRef::Bytes(
                terrain_bytes.clone(),
            ))
        },
        |guid| match guid {
            g if g == PHASE22_BLOCK_MESH_GUID => Some(block.clone()),
            g if g == PHASE22_CHASSIS_MESH_GUID => Some(chassis.clone()),
            _ => None,
        },
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    // **THE PROMISE `pie.rs` MADE, KEPT.** Its own doc block says "the phase-22
    // gate asserts the fractures vector is non-empty before comparing"; until this
    // line that sentence described nothing, and a payload that carried no chunk
    // sets would have made every comparison below a comparison of two intact
    // worlds — the P21.4 lesson, which this repository has now paid for twice.
    assert_eq!(
        payload.fractures.len(),
        2,
        "the PIE payload carries {} derived fracture(s); the level has two \
         destructible meshes, so a count other than two means the derivation is \
         not running and PIE == shipping is about to be asserted between two \
         unbreakable buildings",
        payload.fractures.len()
    );
    assert_eq!(
        payload.terrains.len(),
        1,
        "the .inf_terrain must ride the wire"
    );
    payload
}

// ── the trace ───────────────────────────────────────────────────────────────

/// One traced step.
///
/// Four independent claims, because each can hold while another breaks:
///
/// * `deform` — the P22.1 field's `state_bytes`, verbatim. The ground itself.
/// * `fractures` — every destructible's whole `FractureState::bits()` in `Guid`
///   order: the generation, the detach ordering, and every chunk's pose as raw
///   bits. This is simultaneously the *chunk poses* and the *detach events*.
/// * `audit` — the fixed step's own counters: what the structural solve dropped,
///   what the budget reclaimed, how much debris is live.
/// * `probes` — the roller's pose, the two scripts' spent joules, and a ground
///   query, so a trace that agreed about rubble while disagreeing about the world
///   under it fails here.
#[derive(Clone, PartialEq, Debug)]
struct Frame {
    deform: Vec<u8>,
    fractures: Vec<u64>,
    audit: [u64; 5],
    probes: [u64; 6],
    /// The sim's **whole** trace surface — every `Guid`'d entity's pose plus the
    /// deformation field, i.e. exactly what the replay harness and the PIE
    /// `Frame::state_hash` consume. Carried per step (P22.4 audit, M5) because
    /// four hand-picked scalars are not "the terrain, the physics, the joints,
    /// the dispatch and the ground query", however carefully the doc block says
    /// they are.
    state: Vec<u8>,
}

fn var_bits(sim: &RuntimeSim, actor: Uuid, name: &str) -> u64 {
    match sim.actor_var(actor, name) {
        Some(Value::Float(f)) => f.to_bits(),
        Some(Value::Bool(b)) => (*b as u8 as f64).to_bits(),
        Some(Value::Int(i)) => (*i as f64).to_bits(),
        other => panic!("{name} on {actor} is {other:?}"),
    }
}

fn translation(sim: &RuntimeSim, guid: Uuid) -> glam::DVec3 {
    sim.world()
        .entity_of(guid)
        .and_then(|e| {
            sim.world()
                .world()
                .get::<inf_ecs::components::Transform>(e)
                .map(|t| t.translation.to_dvec3())
        })
        .unwrap_or(glam::DVec3::NAN)
}

/// Every destructible's state as raw bits, in `Guid` order.
fn fracture_bits(sim: &RuntimeSim) -> Vec<u64> {
    let mut out = Vec::new();
    for (guid, state) in sim.fractures() {
        out.push(guid.as_u128() as u64);
        out.push((guid.as_u128() >> 64) as u64);
        out.extend_from_slice(&state.bits());
    }
    out
}

fn frame(sim: &mut RuntimeSim) -> Frame {
    let audit = sim.fracture_audit();
    let roller = translation(sim, PHASE22_ROLLER_GUID);
    Frame {
        deform: inf_ecs::deform::deform_state_bytes(sim.world()),
        fractures: fracture_bits(sim),
        audit: [
            audit.tracked as u64,
            audit.collapsed as u64,
            audit.expired as u64,
            audit.over_budget as u64,
            audit.live_debris as u64,
        ],
        probes: [
            roller.x.to_bits(),
            roller.y.to_bits(),
            roller.z.to_bits(),
            var_bits(sim, PHASE22_TOWER_GUID, "total"),
            var_bits(sim, PHASE22_CHASSIS_GUID, "total"),
            // The combined ground query under the roller — the seam a character
            // controller reads.
            sim.terrain_height_at(roller.x, roller.z).to_bits(),
        ],
        state: sim.state_bytes(),
    }
}

fn run_trace(sim: &mut RuntimeSim) -> Vec<Frame> {
    (0..STEPS)
        .map(|_| {
            sim.step_once(RuntimeInput::default());
            frame(sim)
        })
        .collect()
}

/// The trace, plus the sim it left behind — so a world arm can interrogate the
/// world rather than a report about it.
fn run_and_keep(pack: &Path) -> (RuntimeSim, Vec<Frame>) {
    let mut sim = pack_sim(pack);
    let trace = run_trace(&mut sim);
    (sim, trace)
}

/// **ANTI-VACUITY, applied to every arm that compares two traces.**
///
/// Two identical traces of a world where nothing broke and nothing moved would
/// satisfy every equality in this file — which is exactly how the first cut of the
/// P21.4 gate certified a no-op.
fn assert_not_vacuous(trace: &[Frame]) {
    assert_eq!(trace.len(), STEPS);

    // 1. The GROUND was pressed, and it went on being pressed as the roller moved.
    assert!(
        !trace[0].deform.is_empty(),
        "the first step pressed nothing — the roller never touched the ground, so \
         every deformation claim below is about an empty field"
    );
    assert_ne!(
        trace.first().map(|f| &f.deform),
        trace.last().map(|f| &f.deform),
        "the deformation field never changed over the whole run"
    );

    // 2. The CHARGE was spent. Before the trigger step nothing has been, after it
    //    something has — so the window really contains the divergence.
    let before = f64::from_bits(trace[PHASE22_TRIGGER_STEP as usize - 2].probes[3]);
    let after = f64::from_bits(trace.last().unwrap().probes[3]);
    assert_eq!(before, 0.0, "the block lost energy before the trigger step");
    assert!(
        after > 0.0,
        "the block absorbed {after} J over the whole run — the charge never went \
         off, so this is a trace of a level standing still"
    );

    // 3. Something BROKE, and it kept breaking: the fracture bits differ between
    //    the pre-trigger world and the last step.
    assert_ne!(
        trace[PHASE22_TRIGGER_STEP as usize - 2].fractures,
        trace.last().unwrap().fractures,
        "no destructible's state moved across the charge"
    );

    // 4. There was LIVE DEBRIS at some point. A collapse that detached nothing
    //    would satisfy (3) through the generation alone.
    assert!(
        trace.iter().any(|f| f.audit[4] > 0),
        "the level never had a single piece of live debris"
    );

    // 5. The roller really travelled its lane, so "the trail" is a trail.
    let z0 = f64::from_bits(trace[0].probes[2]);
    let z1 = f64::from_bits(trace.last().unwrap().probes[2]);
    assert!(
        z1 - z0 > 20.0,
        "the roller moved {:.2} m along +Z over {STEPS} steps — it is stuck, so \
         the trail is a dot",
        z1 - z0
    );
}

// ── (a) determinism ─────────────────────────────────────────────────────────

/// Two fresh loads of ONE cooked pack produce bit-identical traces — the level,
/// the derived fracture assets and the terrain all included.
#[test]
fn the_playground_traces_bit_identically_across_two_loads() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());

    let mut a = pack_sim(&pack);
    let mut b = pack_sim(&pack);
    let ta = run_trace(&mut a);
    let tb = run_trace(&mut b);
    assert_not_vacuous(&ta);
    assert_eq!(ta, tb, "the playground moved between two loads of one pack");

    // The two worlds ended in the same *state*, not merely on the same trace:
    // `FractureState` derives `PartialEq` over the asset, the chunks, the bonds
    // and the placement.
    assert_eq!(
        a.fractures(),
        b.fractures(),
        "two loads broke different buildings"
    );
    assert!(
        !a.fractures().is_empty(),
        "no fracture was seeded at all — the pack carries no .inf_fracture"
    );
}

// ── (b) cooked == uncooked ──────────────────────────────────────────────────

/// The same level, off loose files and off a cooked pack, is **the same world up
/// to the moment the cook's own output starts mattering** — and demonstrably a
/// different one afterwards.
///
/// # Why the comparison has a horizon, and why that is the honest arm
///
/// A dev directory has no `.inf_fracture`: the cook derives one, there is
/// deliberately no `FractureRegistry::from_dir`, and a `--level` run therefore
/// previews as indestructible. So the two sides *cannot* agree after the charge,
/// and a version of this test that compared the whole run would either fail or —
/// worse — be quietly weakened until it passed.
///
/// What can be compared, and is, is everything up to `PHASE22_TRIGGER_STEP`: the
/// terrain and its splat bands, the physics, the joints, the Blueprint dispatch,
/// the ground query and the whole deformation field. None of that is derived at
/// cook, so a divergence there is a cook bug.
///
/// Past the trigger the two must **differ**, and that is asserted too — because
/// "identical up to step 60" is satisfied by two worlds that are identical for
/// ever, which is exactly what a cook that shipped no fractures would produce.
#[test]
fn cooked_equals_uncooked_up_to_the_charge_and_differs_after_it() {
    let tmp = tempfile::tempdir().unwrap();
    let (content, pack) = cook_playground(tmp.path());
    let mut cooked_sim = pack_sim(&pack);
    let mut loose_sim = dir_sim(&content);
    let cooked = run_trace(&mut cooked_sim);
    let loose = run_trace(&mut loose_sim);
    assert_not_vacuous(&cooked);

    let horizon = PHASE22_TRIGGER_STEP as usize;
    for (i, (c, l)) in cooked.iter().zip(&loose).enumerate().take(horizon) {
        assert_eq!(
            c.deform, l.deform,
            "step {i}: the cook changed what the roller presses into the ground"
        );
        assert_eq!(
            c.probes[..3],
            l.probes[..3],
            "step {i}: the cook moved the roller"
        );
        assert_eq!(
            c.probes[5], l.probes[5],
            "step {i}: the cook changed the ground the roller runs on"
        );
        // **AND THE WHOLE WORLD** (P22.4 audit, M5). The four scalars above were
        // this arm's entire evidence for a doc block promising "the terrain and its
        // splat bands, the physics, the joints, the Blueprint dispatch, the ground
        // query and the whole deformation field". `state_bytes` is the trace
        // surface the replay harness and the PIE `Frame::state_hash` already
        // consume: every `Guid`'d entity's pose plus the deformation field. It is
        // provably equal over this prefix — nothing has broken yet, and everything
        // that HAS happened (the car settling on its joints, the roller crossing
        // two splat bands) is in it.
        assert_eq!(
            c.state, l.state,
            "step {i}: the cooked world and the uncooked one diverged BEFORE \
             anything was destroyed — the cook changed the level itself"
        );
    }
    // Not a vacuous prefix: the roller has already crossed into the snow and
    // pressed real depth into it by the horizon.
    assert!(
        !cooked[horizon - 1].deform.is_empty(),
        "nothing had been pressed by step {horizon}, so the compared prefix is empty"
    );

    // THE DIVERGENCE, on the far side of the horizon.
    assert_ne!(
        cooked.last().unwrap().fractures,
        loose.last().unwrap().fractures,
        "the cooked run and the uncooked run ended with the same fracture state — \
         the cook shipped no chunk sets, so this arm has been comparing two \
         indestructible worlds"
    );

    // …and the loose side reports its indestructibility as a VALUE rather than
    // failing, which is the kit's contract.
    assert!(
        loose_sim.fractures().is_empty(),
        "a dev-directory run resolved fractures — the cook is supposed to be the \
         only thing that derives them"
    );
    assert!(
        loose_sim
            .logs()
            .iter()
            .any(|l| l.contains("no fracture data resident")),
        "an uncooked run broke nothing and said nothing: {:?}",
        loose_sim.logs()
    );
    assert!(
        !cooked_sim.fractures().is_empty(),
        "the cooked side has no fractures either"
    );
}

// ── (c) PIE == shipping ─────────────────────────────────────────────────────

/// **THE HOUSE GATE**, on destruction. The editor's preview and the shipped build
/// press the same ground, break the same buildings into the same chunks, drop them
/// in the same order and leave them in the same poses — bit for bit, over every
/// step of the run.
#[test]
fn pie_equals_shipping_on_the_damage_trace() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());
    let mut ship_sim = pack_sim(&pack);
    let mut pie_side = pie_sim();
    let ship = run_trace(&mut ship_sim);
    let pie = run_trace(&mut pie_side);
    assert_not_vacuous(&ship);

    // Reported per step rather than as one `assert_eq!` on two 240-element vectors:
    // the failure this exists to catch is a *divergence point*, and "which step"
    // is most of the diagnosis.
    for (i, (s, p)) in ship.iter().zip(&pie).enumerate() {
        assert_eq!(
            s.deform, p.deform,
            "step {i}: a cooked build presses different ground from the editor preview"
        );
        assert_eq!(
            s.fractures, p.fractures,
            "step {i}: a cooked build broke the level differently from the editor preview"
        );
        assert_eq!(s.audit, p.audit, "step {i}: the fracture audits differ");
        assert_eq!(s.probes, p.probes, "step {i}: the probes differ");
        assert_eq!(
            s.state, p.state,
            "step {i}: the two worlds' whole trace surfaces differ — some entity \
             the four probes above do not name is in a different place"
        );
    }
    assert_eq!(ship.len(), pie.len());
    assert_eq!(
        ship_sim.fractures(),
        pie_side.fractures(),
        "preview and shipping traced the same numbers over different buildings"
    );
}

// ── (d) the REAL `--pie` subprocess boundary ────────────────────────────────

/// **THE GUARD THE LAW DEMANDS.** Spawn the actual `inf-player` binary in `--pie`
/// mode over the playground payload and compare its per-step state hashes against
/// the in-process reference.
///
/// Every other PIE arm in this file builds its "preview" side **in process**, by
/// calling `sim_from_payload` directly — a comparison of one function against
/// itself. Revert `main.rs`'s `LoadScene` handler to a bare `RuntimeSim::new` and
/// every one of them stays green, because the thing that changed is the boot path
/// neither side runs. That is P21.4's own law:
///
/// > a boot path that forgets an attachment does not crash — it agrees with
/// > itself.
///
/// **The window law applies here too, and harder**, because every frame is a pipe
/// round trip and the temptation to stop at forty is strongest. Forty is inside
/// the pre-trigger settle: a subprocess that attached **no fractures at all**
/// traces identically for the first sixty steps, because nothing has been asked to
/// break yet. `N` is therefore the whole run.
#[test]
fn the_real_pie_subprocess_matches_the_in_process_reference() {
    use inf_editor_core::pie::PieSession;
    use inf_runtime::pie::PlayerToEditor;
    use std::time::Duration;

    let payload = playground_payload();
    const N: u32 = STEPS as u32;

    let mut session =
        PieSession::spawn_scene(&PathBuf::from(env!("CARGO_BIN_EXE_inf-player")), &payload)
            .expect("the player spawns in --pie mode");
    session.step(N).expect("step N");

    let mut got = Vec::with_capacity(N as usize);
    for _ in 0..N {
        let ev = session
            .wait_for(Duration::from_secs(30), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }
    session
        .stop(Duration::from_secs(10))
        .expect("graceful stop");

    let want = inf_player::scene_trace(&payload, N as u64).expect("in-process reference");
    assert_eq!(
        got, want,
        "the REAL --pie subprocess ran a different world from the in-process \
         reference — one of the two boot paths is missing an attachment"
    );
    // ANTI-VACUITY, and the audit's M4: the two guards below are **satisfied by
    // the rolling ball**. Proven, not supposed — a subprocess with no fractures at
    // all still moves its state hash every step and still moves it after the
    // trigger, because the roller never stops. So they stay (a frozen world is
    // still worth catching) but they are not the anti-vacuity this arm needs.
    assert!(
        got.windows(2).any(|w| w[0] != w[1]),
        "the state hash never changed across {N} steps"
    );
    let t = PHASE22_TRIGGER_STEP as usize;
    assert!(
        got[t..].windows(2).any(|w| w[0] != w[1]),
        "the state hash stopped changing at the trigger step"
    );
    // THE ONE THAT BITES: the reference world really destroyed something inside
    // the compared window. If it did not, the hashes above are a comparison of two
    // levels in which the whole subject of this phase never happened.
    let mut reference = pie_sim();
    for _ in 0..=(PHASE22_TRIGGER_END as usize + 1) {
        reference.step_once(RuntimeInput::default());
    }
    assert!(
        !reference.fractures()[&PHASE22_TOWER_GUID].is_intact(),
        "the world these hashes were taken over still had an intact block at step \
         {} — the subprocess and the reference agreed about a level in which \
         nothing broke",
        PHASE22_TRIGGER_END as usize + 1
    );
    assert!(
        !reference.fractures()[&PHASE22_CHASSIS_GUID].is_intact(),
        "…and an intact car"
    );
}

// ── (e) the WORLD ───────────────────────────────────────────────────────────

/// **THE HEADLINE.** The multi-storey block came down, its **control twin under
/// the identical charge did not**, and the structural solve did real work getting
/// there.
#[test]
fn the_block_collapses_and_its_control_twin_survives() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());
    let (sim, trace) = run_and_keep(&pack);
    assert_not_vacuous(&trace);

    let block = &sim.fractures()[&PHASE22_TOWER_GUID];
    let control = &sim.fractures()[&PHASE22_CONTROL_GUID];

    // The two really are twins: same chunk count, same bond graph, and neither's
    // bonds were guessed. `estimated_bonds` is an INVARIANT, not a tolerance — the
    // cook prunes every faceless adjacency edge, so a non-empty set here means a
    // phantom load path reached the support solve.
    assert_eq!(block.chunk_count(), control.chunk_count());
    assert!(block.chunk_count() >= 18, "the block barely fractured");
    for (name, s) in [("block", block), ("control", control)] {
        assert!(
            s.estimated_bonds().is_empty(),
            "the {name}'s fracture has {} bond(s) priced from the fallback estimate \
             — the cook shipped a faceless adjacency edge: {:?}",
            s.estimated_bonds().len(),
            s.estimated_bonds()
        );
    }

    // THE WORLD, not the report: the block's chunk POSES moved.
    //
    // **And how many, exactly, and why not all of them** (P22.4 audit). Ten chunks
    // detach; three leave their rest pose by more than 25 cm and seven do not, and
    // that is correct rather than a weak threshold: damage is cheapest-to-liberate,
    // the cheapest chunks in a block standing on flat ground are the ones with
    // fewest neighbours, and several of those are **base** chunks (chunk 0, three
    // neighbours, is the cheapest in the whole structure). A base chunk that
    // detaches is already resting on the terrain — it has nowhere to fall. So the
    // bound is stated as the pair it is: some rubble fell, and the rest is sitting
    // where it was because it was on the floor to begin with.
    assert!(!block.is_intact(), "the block never broke");
    let detached = block.chunks().iter().filter(|c| c.detached).count();
    let moved = block
        .chunks()
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            c.detached && (c.translation - block.chunk_rest_center(*i)).length() > 0.25
        })
        .count();
    assert!(
        moved >= 3,
        "only {moved} of the block's {detached} detached chunks left their rest \
         pose — a detach that leaves everything where it was is a bookkeeping \
         change, not a collapse"
    );
    // …and the ones that did move came from ABOVE the ground floor, which is what
    // makes "it fell" true rather than "it was nudged". A chunk whose rest centre
    // is below 3 m had nothing to fall.
    let fell_from_height = block
        .chunks()
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            c.detached
                && block.asset().chunks[*i].center_of_mass[1] > 3.0
                && c.translation.y < block.chunk_rest_center(*i).y - 0.5
        })
        .count();
    assert!(
        fell_from_height >= 1,
        "no chunk from above the ground floor ended lower than it started — the \
         charge peeled the base and nothing actually came down"
    );

    // …and the charge OPENED it rather than erasing it: chunks are left standing
    // for the structure to still be a structure afterwards.
    let standing = block.chunks().iter().filter(|c| !c.detached).count();
    assert!(
        standing >= 8,
        "the charge left only {standing} of {} chunks attached — that is not a \
         building with a hole in it, and nothing is left for the support graph to \
         have an opinion about",
        block.chunk_count()
    );

    // **MULTI-STOREY, asserted on the shipped fracture's own graph.** The claim
    // "upper floors are supported THROUGH lower ones" is a property of the chunk
    // adjacency, and this is where it is checkable: every chunk in the top third
    // of the block must reach the bottom third through the neighbour graph, and
    // must not be adjacent to it directly. A block whose chunking produced
    // full-height columns would fail here — and would behave like a row of
    // independent pillars however convincing the screenshot looked.
    let asset = block.asset();
    let y_of = |i: usize| asset.chunks[i].center_of_mass[1];
    let n = asset.chunks.len();
    let top: Vec<usize> = (0..n).filter(|&i| y_of(i) > 6.0).collect();
    let base: Vec<usize> = (0..n).filter(|&i| y_of(i) < 3.0).collect();
    assert!(
        !top.is_empty() && !base.is_empty(),
        "the block's chunking has no top storey ({}) or no base ({}) — it is not \
         a multi-storey structure at all",
        top.len(),
        base.len()
    );
    for &t in &top {
        // Breadth-first over the adjacency: the hop count from this top chunk to
        // the nearest base chunk.
        let mut dist = vec![usize::MAX; n];
        dist[t] = 0;
        let mut q = std::collections::VecDeque::from([t]);
        let mut reached = usize::MAX;
        while let Some(i) = q.pop_front() {
            if base.contains(&i) {
                reached = dist[i];
                break;
            }
            for &nb in &asset.chunks[i].neighbors {
                let nb = nb as usize;
                if nb < n && dist[nb] == usize::MAX {
                    dist[nb] = dist[i] + 1;
                    q.push_back(nb);
                }
            }
        }
        assert!(
            reached != usize::MAX,
            "chunk {t} of the block reaches no base chunk at all — it is a piece \
             of a separate structure that happens to share a mesh"
        );
        assert!(
            reached >= 2,
            "chunk {t} sits above y = 6 m and is {reached} hop(s) from the ground \
             floor — the block's chunking made a full-height column, so its upper \
             storey is not supported THROUGH anything"
        );
    }

    // **THE BLOCK DOES NOT TOPPLE, AND THAT IS THE RULING RATHER THAN A DEFECT.**
    //
    // Damage in this engine is cheapest-to-liberate and has no impact point (the
    // P22.3 ruling, taken so that scene schema v20 could be frozen). On a
    // monolithic block standing on solid ground the cheapest pieces are the
    // least-connected ones — the corners and the top — and every piece left after
    // the charge still reaches the terrain through its neighbours. So the charge
    // *opens* the building and the structural solve has nothing to say about it.
    //
    // Asserted rather than assumed, with the number that says so: this level's
    // out-of-damage collapses are exactly zero. When a hit position arrives (the
    // ledger's "two-line change once there is something to sort by") this
    // assertion is what will fail, and whoever changes it will have to update the
    // ledger with it. The solve's *own* demonstration lives on the car, which is
    // not standing on anything — see
    // `the_car_fractures_and_its_blast_reaches_everything_in_its_radius`.
    let out_of_damage: u64 = trace.iter().map(|f| f.audit[1]).sum();
    assert_eq!(
        out_of_damage, 0,
        "the block collapsed structurally, which this engine's cheapest-to-liberate \
         damage on solid ground is not supposed to produce — either an impact \
         point has arrived (update ROADMAP §12's P22 ledger) or the support solve \
         has started reporting something else"
    );

    // THE CONTROL, under the identical charge, from the identical class.
    assert!(
        control.is_intact(),
        "the control twin broke — its `runtime_destruct` is false and it runs the \
         SAME Blueprint class as the block, so the two differ in that flag alone"
    );
    let spent = match sim.actor_var(PHASE22_CONTROL_GUID, "total") {
        Some(Value::Float(f)) => *f,
        other => panic!("the control's total is {other:?}"),
    };
    assert_eq!(spent, 0.0, "a refused charge absorbed {spent} J");
    // The refusal is REPORTED — the kit's own contract, and the difference between
    // "withheld" and "silently dropped".
    assert!(
        sim.logs()
            .iter()
            .any(|l| l.contains("apply_damage") && l.contains("runtime_destruct")),
        "the control's refusal was swallowed instead of logged"
    );
    // …and it fired once per charge tick rather than once per session, so a level
    // designer sees the whole story.
    let ticks = (PHASE22_TRIGGER_END - PHASE22_TRIGGER_STEP) as usize + 1;
    let refusals = sim
        .logs()
        .iter()
        .filter(|l| l.contains("runtime_destruct"))
        .count();
    assert!(
        refusals >= ticks,
        "{refusals} refusal(s) logged for {ticks} charge ticks"
    );
}

/// **The car fractured, and its own blast reached everything inside its radius.**
///
/// Named for what it checks (P22.4 audit): on the step the blast fires, the
/// bodies in range are the four wheels and the still-unswapped chassis, and
/// **no debris at all** — a chunk body reaches the solver one step later. The
/// arm used to be called "…reaches the debris it made", which was a claim it did
/// not make and could not have made.
///
/// The blast is the one node in the kit deliberately *not* gated by
/// `runtime_destruct` — it pushes bodies, and a body is a body — so "it hit
/// exactly what was there" and "the chassis broke" are two claims and both are
/// made here.
#[test]
fn the_car_fractures_and_its_blast_reaches_everything_in_its_radius() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());
    let (sim, _trace) = run_and_keep(&pack);

    let car = &sim.fractures()[&PHASE22_CHASSIS_GUID];
    assert!(!car.is_intact(), "the car never came apart");
    assert!(
        car.chunks().iter().filter(|c| c.detached).count() >= 2,
        "one chunk off a car is a dent, not a wreck"
    );

    // **WHAT `hit` ACTUALLY COUNTS, stated because this arm's name over-claimed
    // it** (P22.4 audit). The blast fires on the SAME step the charge lands, and a
    // chunk body reaches the solver on the step *after* the swap (the one-beat
    // latency `inf_physics::d3::fracture` documents). So the bodies it pushes are
    // the four wheels and the still-unswapped chassis — **zero debris**. That the
    // blast reaches debris is real and is asserted in `runtime_destruct`'s
    // `a_blueprint_breaks_its_own_wall`, where a second tick can be isolated; the
    // honest claim here is narrower and exact.
    let hit = match sim.actor_var(PHASE22_CHASSIS_GUID, "hit") {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(i)) => *i as f64,
        other => panic!("the car's `hit` is {other:?}"),
    };
    assert_eq!(
        hit,
        (PHASE22_WHEEL_GUIDS.len() + 1) as f64,
        "the blast pushed {hit} bodies. On the step it fires, the four wheels and \
         the not-yet-swapped chassis are what is inside its radius: more means it \
         reached the block 25 m away (and 'the block's chunks moved' stops being a \
         statement about the collapse), fewer means it never reached the solver."
    );

    // **THE STRUCTURAL SOLVE, ON THE ACTOR IT IS HONEST ABOUT — and measured in
    // joules rather than counted.**
    //
    // A car body is held up by four DYNAMIC wheels, and debris resting on debris
    // is not support. So the instant the charge takes its first chunk and the
    // atomic swap removes the chassis' own collider, everything still attached is
    // hanging in mid-air and the solve drops it.
    //
    // The evidence is an *energy budget*: the charge absorbed one chunk's worth of
    // bonds and TWELVE chunks came off. Eleven of them were therefore not paid
    // for — which is what "supported through something else" means, stated as a
    // number nothing else in the level can produce.
    //
    // (`FractureAudit::collapsed` is deliberately NOT the instrument here, and the
    // reason is worth recording: it counts the collapses `step_fractures` finds,
    // and `runtime_destruct` runs the same solve inside the damage call — so a
    // collapse the charge triggered is reported in `DamageReport::detached` and
    // never reaches the audit. Reading the audit here would have shown zero and
    // looked like the solve was dead.)
    let detached = car.chunks().iter().filter(|c| c.detached).count();
    assert_eq!(
        detached,
        car.chunk_count() as usize,
        "{} of the car's chunks are still attached to a body that is no longer \
         standing on anything",
        car.chunk_count() as usize - detached
    );
    let cheapest = (0..car.chunk_count() as usize)
        .map(|i| {
            car.asset().chunks[i]
                .neighbors
                .iter()
                .map(|&n| car.bond_energy_j(i as u32, n))
                .sum::<f64>()
        })
        .fold(f64::INFINITY, f64::min);
    let spent = match sim.actor_var(PHASE22_CHASSIS_GUID, "total") {
        Some(Value::Float(f)) => *f,
        other => panic!("the car's total is {other:?}"),
    };
    assert!(cheapest.is_finite() && cheapest > 0.0);
    // `spent <= the charge` by construction, so a bound of `2 × cheapest`
    // (50 272 J against a 30 000 J charge) could never fail — it read like a
    // measurement and was an identity (P22.4 audit). Two assertions that CAN fail
    // replace it.
    //
    // (1) EXACTLY ONE CHUNK WAS PAID FOR. `apply_damage` spends on the cheapest
    //     bond set and stops when what is left cannot buy another; if the charge
    //     had cascaded — breaking a chunk makes its neighbours cheaper — `spent`
    //     would be several bond sets rather than one. This is the number that
    //     tells "the solve dropped eleven" from "the charge bought twelve", and
    //     raising `PHASE22_CAR_CHARGE_J` breaks it.
    assert!(
        (spent - cheapest).abs() < 1.0,
        "the charge spent {spent:.0} J where the cheapest single chunk costs \
         {cheapest:.0} J — it cascaded, so the damage paid for the wreck and the \
         structural solve did not do the work this arm credits it with"
    );
    // (2) …AND PAYING FOR THE WHOLE CAR WOULD HAVE COST FAR MORE. A lower bound on
    //     that: the sum of every chunk's own bond set, halved because each bond is
    //     counted from both ends. The charge is a small fraction of it.
    let total: f64 = (0..car.chunk_count() as usize)
        .map(|i| {
            car.asset().chunks[i]
                .neighbors
                .iter()
                .map(|&n| car.bond_energy_j(i as u32, n))
                .sum::<f64>()
        })
        .sum::<f64>()
        * 0.5;
    assert!(
        spent * 4.0 < total,
        "the charge ({spent:.0} J) is not small against what liberating every \
         chunk would cost ({total:.0} J) — the fixture no longer demonstrates a \
         collapse nobody paid for"
    );

    // The wheels are still IN the level and still near the wreck: the joints
    // dangling when the chassis body went away must not have deleted them or sent
    // them to the origin — and the blast must be sized against the LIGHTEST body
    // in its radius rather than against the chunk masses. (It was not, once: at
    // 60 kN·s on a wheel this gate found one 13 km up. `PHASE22_BLAST_NS` carries
    // the arithmetic.)
    for guid in PHASE22_WHEEL_GUIDS {
        let at = translation(&sim, guid);
        assert!(at.is_finite(), "a wheel left the world entirely");
        let (cx, cz) = inf_editor_core::samples::PHASE22_CAR_XZ;
        assert!(
            (at.x - cx).abs() < 60.0 && (at.z - cz).abs() < 60.0 && at.y < 60.0,
            "a wheel ended at {at} — the blast is a car bomb, not an ejection seat"
        );
    }
}

/// **The trail is where the roller ran, and nowhere else.**
///
/// Both halves matter. Depth in the snow band and in the sand band is what makes
/// "footprints and tyre tracks in snow and sand" true; zero depth off the lane and
/// on the rock-hard layer is what makes it a statement about the roller rather
/// than about a field that pressed everything.
#[test]
fn the_roller_leaves_a_trail_in_the_snow_and_the_sand_and_nowhere_else() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());
    let (sim, _trace) = run_and_keep(&pack);
    let field = sim
        .deform_field()
        .expect("the roller pressed something into the ground");
    assert!(!field.is_empty());

    // The MIDPOINT of each band — a point the roller passed through, on the lane
    // by construction, and far enough from either edge that a bilinear splat
    // sample cannot be answering for the neighbouring layer.
    let snow_z = 0.5 * (PHASE22_SNOW_Z.0 + PHASE22_SNOW_Z.1);
    let sand_z = 0.5 * (PHASE22_SAND_Z.0 + PHASE22_SAND_Z.1);
    assert_eq!(phase22_layer_at(snow_z), PHASE22_SNOW_LAYER);
    assert_eq!(phase22_layer_at(sand_z), PHASE22_SAND_LAYER);

    let at = |z: f64| glam::DVec2::new(PHASE22_ROLLER_X, z);
    let snow = field.depth_at(at(snow_z));
    let sand = field.depth_at(at(sand_z));
    assert!(
        snow > 0.02,
        "the snow band is pressed {snow:.4} m deep — a track in 40 cm-deep snow \
         should be centimetres, so this is a field that was never stamped"
    );
    assert!(sand > 0.005, "the sand band is pressed only {sand:.4} m");
    // Snow is the deeper archetype (LAYER_RESPONSE: 0.40 m against 0.14 m), so the
    // two bands must not merely both be non-zero — they must differ in the
    // direction the table says. A field that ignored the layer would fail here and
    // pass everything above it.
    assert!(
        snow > sand,
        "the snow track ({snow:.4} m) is no deeper than the sand one ({sand:.4} m) \
         — the per-layer response is not being read"
    );
    assert_eq!(field.layer_at(at(snow_z)), Some(PHASE22_SNOW_LAYER));
    assert_eq!(field.layer_at(at(sand_z)), Some(PHASE22_SAND_LAYER));

    // OFF the lane: eight metres to the side of a 0.5 m roller.
    assert_eq!(
        field.depth_at(glam::DVec2::new(PHASE22_ROLLER_X + 8.0, snow_z)),
        0.0,
        "ground the roller never crossed is deformed — the trail is not a trail"
    );
    // …and BEFORE the run began, on the same lane, the field held nothing at all.
    let cold = pack_sim(&pack);
    assert!(
        cold.deform_field().is_none(),
        "a freshly-loaded level already has a deformation field"
    );

    // The grass really is standing in and beside the trail, which is what the bend
    // shader reads the same window for. Structural, not pixels: the strip exists,
    // it is where the sample says, and the layer under it bends.
    let grass = sim
        .world()
        .entity_of(PHASE22_GRASS_GUID)
        .and_then(|e| {
            sim.world()
                .world()
                .get::<inf_ecs::components::Foliage>(e)
                .map(|f| f.instances.len())
        })
        .expect("the grass strip survived the cook");
    assert!(grass > 100, "the grass strip has {grass} instances");
    // **DRIVEN, not read off a constant** (P22.4 audit). The previous version
    // asserted `LAYER_RESPONSE[SAND].bend > 0.0`, which is a compile-time table
    // lookup: it would hold on a level with no grass, no roller and no field.
    // What the bend shader actually reads is `bend × depth` at the instance's own
    // XZ, so drive it: the strip straddles the lane, so instances stand in ground
    // the roller pressed and in ground it did not, and the two must differ.
    let strip_z = 56.0_f64; // the grass entity's own translation
    let lane_xz = glam::DVec2::new(PHASE22_ROLLER_X, strip_z);
    let lane_layer = field.layer_at(lane_xz).expect("the trail names a layer");
    let on_lane =
        field.depth_at(lane_xz) * inf_terrain::deform::LAYER_RESPONSE[lane_layer as usize].bend;
    let off_lane = field.depth_at(glam::DVec2::new(PHASE22_ROLLER_X + 5.0, strip_z));
    assert!(
        on_lane > 0.0,
        "the grass strip's own lane has no bend drive at all — the blades inside \
         the trail would stand exactly as straight as the ones beside it"
    );
    assert_eq!(
        off_lane, 0.0,
        "ground five metres off the lane is pressed, so 'the blades in the trail \
         bend' would be 'every blade bends'"
    );
}

/// **(j) THE KEYING CLAIM, on real physics** (P22.4 audit, M1).
///
/// `inf_render::debris`'s own unit test can only assert that a pure function is
/// pure: a `DebrisSite` carries no live pose, so "the key holds while the chunks
/// tumble" is unstateable there. It is stateable here, where there are chunks and
/// they are actually tumbling.
///
/// Three things, and the first is what makes the other two mean anything:
///
/// 1. the chunk **poses demonstrably move** between the two samples;
/// 2. the fracture **generation does not**, because nothing detached or was
///    reclaimed in between;
/// 3. so the batch's **content key is identical** and the memo does not re-pack.
///
/// Then the control: a step that *does* bump the generation re-keys.
///
/// The sites are built here the way `project_debris` builds them, which is
/// duplication — held honest by `projector_mirror`'s `DebrisSite` field
/// allowlist, which pins every one of those five expressions on both hosts. This
/// is also the only arm in the gate that touches `inf_render::debris` at all; the
/// render path had zero integration coverage before it.
#[test]
fn the_debris_batch_holds_its_key_while_the_chunks_tumble() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());
    let mut sim = pack_sim(&pack);

    // The same five expressions `project_debris` uses.
    let sites = |sim: &RuntimeSim, guid: Uuid| -> Vec<inf_render::DebrisSite> {
        let state = &sim.fractures()[&guid];
        state
            .chunks()
            .iter()
            .enumerate()
            .filter(|(_, c)| c.detached && !c.gone)
            .map(|(i, c)| inf_render::DebrisSite {
                entity: 7,
                chunk: i as u32,
                order: c.detach_order,
                center: state.chunk_rest_center(i),
                radius_m: state.chunk_radius_m(i),
            })
            .collect()
    };
    let poses = |sim: &RuntimeSim, guid: Uuid| -> Vec<[u64; 3]> {
        sim.fractures()[&guid]
            .chunks()
            .iter()
            .map(|c| {
                [
                    c.translation.x.to_bits(),
                    c.translation.y.to_bits(),
                    c.translation.z.to_bits(),
                ]
            })
            .collect()
    };

    // Run until the car has come apart and its rubble is in the air. The CAR,
    // not the block: it collapses in one step, so a window after it exists in
    // which everything is moving and nothing further detaches.
    let mut cache = inf_render::DebrisCache::default();
    let mut before = None;
    for step in 0..STEPS {
        sim.step_once(RuntimeInput::default());
        if sim.fractures()[&PHASE22_CHASSIS_GUID].is_intact() {
            continue;
        }
        // Two steps clear of the break, so the generation has settled.
        if before.is_none() && step >= PHASE22_TRIGGER_STEP as usize + 2 {
            let s = sites(&sim, PHASE22_CHASSIS_GUID);
            assert!(!s.is_empty(), "the car shed no sites at all");
            let gen = sim.fractures()[&PHASE22_CHASSIS_GUID].generation();
            let batch = cache
                .batch(
                    7,
                    gen,
                    &s,
                    inf_render::DEBRIS_RUBBLE_PER_CHUNK,
                    [1.0; 4],
                    0.8,
                )
                .expect("a broken car sheds a batch");
            before = Some((gen, batch.data.key(), poses(&sim, PHASE22_CHASSIS_GUID)));
            continue;
        }
        if let Some((gen0, key0, poses0)) = &before {
            let gen1 = sim.fractures()[&PHASE22_CHASSIS_GUID].generation();
            if gen1 != *gen0 {
                // Something detached or was reclaimed — not the window this arm is
                // about. (It is the control below.)
                continue;
            }
            let poses1 = poses(&sim, PHASE22_CHASSIS_GUID);
            if &poses1 == poses0 {
                continue; // still settling into the same pose; keep looking
            }
            // (1) the poses MOVED, (2) the generation did not, so (3) the key must
            // not have — and the memo must not have re-packed.
            let packs0 = cache.packs();
            let batch = cache
                .batch(
                    7,
                    gen1,
                    &sites(&sim, PHASE22_CHASSIS_GUID),
                    inf_render::DEBRIS_RUBBLE_PER_CHUNK,
                    [1.0; 4],
                    0.8,
                )
                .expect("still broken");
            assert_eq!(
                batch.data.key(),
                *key0,
                "the rubble's content key moved while the chunks were only tumbling \
                 — the batch re-uploads its whole instance buffer every step, which \
                 is the documented cost the rest-pose rule exists to avoid"
            );
            assert_eq!(
                cache.packs(),
                packs0,
                "the memo re-packed a payload whose generation had not moved"
            );

            // THE CONTROL: a *different* live set is a different batch, so the
            // equality above is a claim about tumbling rather than about a key
            // that never changes.
            let mut fewer = sites(&sim, PHASE22_CHASSIS_GUID);
            fewer.pop();
            let other = cache
                .batch(
                    7,
                    gen1 + 1,
                    &fewer,
                    inf_render::DEBRIS_RUBBLE_PER_CHUNK,
                    [1.0; 4],
                    0.8,
                )
                .expect("a smaller set is still a batch");
            assert_ne!(
                other.data.key(),
                *key0,
                "a reclaim did not re-key the batch"
            );
            assert_eq!(cache.packs(), packs0 + 1, "the control did not re-pack");
            return;
        }
    }
    panic!("the car never broke, or its chunks never moved while its generation held");
}

/// **The debris budget held**, measured with engagement counters rather than
/// pixels — and the counter is proved to be capable of failing.
#[test]
fn the_debris_budget_holds_across_the_collapse() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());
    let (sim, trace) = run_and_keep(&pack);
    assert_not_vacuous(&trace);

    let budget = inf_physics::d3::DebrisBudget::default();
    let peak = trace.iter().map(|f| f.audit[4]).max().unwrap_or(0);
    for (i, f) in trace.iter().enumerate() {
        assert!(
            f.audit[4] <= budget.max_live as u64,
            "step {i}: {} live debris chunks against a cap of {}",
            f.audit[4],
            budget.max_live
        );
    }
    // ANTI-VACUITY: there WAS debris, so "never exceeded" is a bound that was
    // approached rather than a statement about zero.
    assert!(
        peak > 0,
        "the level never had a live debris chunk, so the cap was never tested"
    );
    // …and none of it was reclaimed. **Asserted on the CONDITIONS, not only on
    // the count** (P22.4 audit): `reclaimed == 0` cannot fail while the run is
    // shorter than the lifetime and the peak is under the cap, so on its own it
    // was an identity dressed as a measurement. Both premises are checked, so a
    // fixture that grew past either fails here with the reason rather than
    // silently making the arm vacuous.
    let run_s = STEPS as f64 / 60.0;
    assert!(
        run_s < budget.lifetime_s,
        "the {run_s:.1} s run is longer than the {} s debris lifetime, so chunks \
         expire during it and 'nothing was reclaimed' is no longer the claim this \
         arm makes",
        budget.lifetime_s
    );
    assert!(
        peak < budget.max_live as u64,
        "peak debris {peak} reaches the cap of {} — the level now exercises the \
         over-budget path, which is what the second world below is for",
        budget.max_live
    );
    let reclaimed: u64 = trace.iter().map(|f| f.audit[2] + f.audit[3]).sum();
    assert_eq!(
        reclaimed, 0,
        "the default budget reclaimed {reclaimed} chunk(s) on a level that is \
         under the cap for a run shorter than the lifetime — that is not a \
         budget, it is a bug"
    );

    // **THE FALSIFICATION.** The same level under a cap of two: the counter moves,
    // the reclaims happen, and the cap holds at the new number. Without this the
    // arm above is satisfied by a counter that is always zero.
    let mut capped = pack_sim(&pack);
    capped.set_debris_budget(inf_physics::d3::DebrisBudget {
        max_live: 2,
        lifetime_s: f64::INFINITY,
    });
    let mut over = 0u64;
    let mut worst = 0u64;
    for _ in 0..STEPS {
        capped.step_once(RuntimeInput::default());
        let a = capped.fracture_audit();
        over += a.over_budget as u64;
        worst = worst.max(a.live_debris as u64);
    }
    assert!(
        over > 0,
        "a cap of 2 reclaimed nothing on a level that peaks at {peak}"
    );
    assert!(worst <= 2, "a cap of 2 allowed {worst} live chunks");

    // The reclaimed chunks left the physics world too — despawn-by-omission is one
    // event, not two. `body_count` under the cap must be lower than without it.
    let loose_bodies = sim.bridge3d().body_count();
    let capped_bodies = capped.bridge3d().body_count();
    assert!(
        capped_bodies < loose_bodies,
        "the capped run holds {capped_bodies} bodies and the uncapped one \
         {loose_bodies} — a reclaimed chunk stopped being drawn but kept its body"
    );
}

// ── (i) the cook is silent on correct content ───────────────────────────────

/// **No advisory fires on the flagship sample** — including the three this phase
/// added (the fracture skip, the chunk-yield shortfall and the uncollidable
/// chunk). An advisory that fires on the engine's own flagship content is one
/// nobody reads.
#[test]
fn the_playground_draws_no_cook_advisory() {
    let tmp = tempfile::tempdir().unwrap();
    scaffold(tmp.path());
    let report = cook(
        &tmp.path().join("proj"),
        &tmp.path().join("out"),
        &CookOptions::default(),
    )
    .unwrap();
    assert!(
        report.warnings.is_empty(),
        "the flagship playground sample trips cook advisories: {:?}",
        report.warnings
    );
}

// ── (f) budgets ─────────────────────────────────────────────────────────────

/// The composed playground builds inside the **load** budget, and a step that is
/// *breaking a building* is measured against the **frame** budget. Two different
/// ceilings for two different classes of work, both imported. A load measured
/// against the frame budget is a category error.
#[test]
fn the_playground_loads_and_breaks_inside_its_budgets() {
    let tmp = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_playground(tmp.path());

    let t0 = Instant::now();
    let mut sim = pack_sim(&pack);
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert!(
        load_ms < LOAD_BUDGET_MS,
        "the playground took {load_ms:.1} ms to build (budget {LOAD_BUDGET_MS} ms)"
    );

    // Warm up to just before the charge — the first steps pay for rapier's island
    // construction and for the terrain heightfield colliders, which are load costs
    // wearing a step's clothes.
    for _ in 0..(PHASE22_TRIGGER_STEP as usize - 1) {
        sim.step_once(RuntimeInput::default());
    }
    // THE DAMAGE STEPS: the charge, the swap, the structural solve and the budget
    // sweep, all in one fixed step.
    let t1 = Instant::now();
    let measured = (PHASE22_TRIGGER_END - PHASE22_TRIGGER_STEP) as usize + 1;
    for _ in 0..measured {
        sim.step_once(RuntimeInput::default());
    }
    let damage_ms = t1.elapsed().as_secs_f64() * 1000.0 / measured as f64;

    // …and the settle after it, which is the steady state a player spends most of
    // a collapse in.
    let t2 = Instant::now();
    const SETTLE: usize = 60;
    for _ in 0..SETTLE {
        sim.step_once(RuntimeInput::default());
    }
    let settle_ms = t2.elapsed().as_secs_f64() * 1000.0 / SETTLE as f64;

    // **Printed, and asserted against the EXISTING budget only.** No new ratchet:
    // a phase does not get its own number for being new, and a destruction step's
    // cost is a per-frame cost like any other.
    println!(
        "phase22: damage step {damage_ms:.3} ms, settle step {settle_ms:.3} ms \
         (frame budget {FRAME_BUDGET_MS} ms)"
    );
    assert!(
        damage_ms < FRAME_BUDGET_MS,
        "a damage step took {damage_ms:.3} ms (budget {FRAME_BUDGET_MS} ms) — the \
         charge, the intact→chunks swap, the structural solve and the budget sweep \
         are all per-step work"
    );
    assert!(
        settle_ms < FRAME_BUDGET_MS,
        "a settling step took {settle_ms:.3} ms (budget {FRAME_BUDGET_MS} ms)"
    );

    // ANTI-VACUITY: the steps measured really were the ones that broke it.
    assert!(!sim.fractures()[&PHASE22_TOWER_GUID].is_intact());
    assert!(sim.world().entity_of(PHASE22_TERRAIN_GUID).is_some());
}

// ── (g) pool-size invariance ────────────────────────────────────────────────

/// Launch the `destruct_probe` binary at a fixed pool size and return the
/// `key=value` lines it printed.
fn probe(threads: usize, steps: usize) -> HashMap<String, String> {
    let exe = env!("CARGO_BIN_EXE_destruct_probe");
    let out = Command::new(exe)
        .arg(threads.to_string())
        .arg(steps.to_string())
        .output()
        .expect("spawn destruct_probe");
    assert!(
        out.status.success(),
        "destruct_probe (threads={threads}) failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("probe stdout utf8")
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect()
}

/// **A collapse is invariant to the ECS worker-pool size** — in what the
/// Blueprints saw *and* in what the world became.
///
/// # What this arm is, honestly (P22.4 audit, M3)
///
/// It was written believing that "the `destruct.*` calls run from a Blueprint
/// tick, which runs inside the ECS schedule". **They do not.** `RuntimeSim` runs
/// no `SimSchedule`: `run_all_with_args` is a serial `for` loop over a `Vec` of
/// `BTreeMap` keys, and rapier's `parallel` feature is off by rule. Four pool
/// sizes execute the same serial program, so this cannot *discover* an ordering
/// race — and claiming it does would be the "inference dressed as measurement"
/// mistake this phase's own ledger records.
///
/// It is kept, with the claim corrected, because it is worth exactly two things
/// and they are both real: a **regression tripwire** for the day the tick pass
/// moves onto `inf-ecs`'s parallel schedule (a documented follow-up in
/// `runtime_sim`'s own module docs), already wired and already holding a
/// reference hash; and a **pin** that nothing on the fixed-step path has reached
/// for the process-global `ComputeTaskPool`, which is the half that can change
/// silently. `destruct_probe.rs`'s module docs carry the full statement.
///
/// Four towers, so that the day there IS interleaving there is something to
/// interleave.
#[test]
fn the_collapse_is_identical_across_pool_sizes() {
    const PROBE_STEPS: usize = 180;
    let sizes = [1usize, 2, 4, 8];
    let runs: Vec<HashMap<String, String>> = sizes.iter().map(|n| probe(*n, PROBE_STEPS)).collect();

    // The harness itself works: each subprocess really got the pool it asked for,
    // so the hashes below were produced under four genuinely different pools.
    for (n, run) in sizes.iter().zip(&runs) {
        assert_eq!(
            run.get("threads").map(String::as_str),
            Some(n.to_string().as_str()),
            "the probe did not get the pool size it was asked for: {run:?}"
        );
    }

    for key in ["trace", "field"] {
        let reference = runs[0]
            .get(key)
            .unwrap_or_else(|| panic!("probe printed {key}="));
        for (n, run) in sizes.iter().zip(&runs) {
            assert_eq!(
                run.get(key)
                    .unwrap_or_else(|| panic!("probe printed {key}=")),
                reference,
                "the collapse's {key} depends on the ECS pool size (threads={n}) — \
                 it is not replay-safe"
            );
        }
    }

    // Not vacuous: the towers really came apart over the run, so the equal hashes
    // are equal collapses rather than equal copies of nothing.
    assert_ne!(
        runs[0].get("first"),
        runs[0].get("last"),
        "nothing broke across {PROBE_STEPS} steps — the comparison is vacuous"
    );
}

// ── (h) persistence: the ruling, in the gate ────────────────────────────────

/// **Destruction is runtime-only.** A save after the charge is byte-identical to a
/// save before it, on the committed sample, through the editor's own Simulate
/// session — which is the host that owns the document a Ctrl+S would write.
///
/// `inf-editor-core`'s `simulate_destruction_not_persisted` makes this claim on a
/// synthetic fixture; this makes it on the shipped one, where the fractures are
/// derived from real cook-equivalent bytes and the level carries a streamed
/// terrain (whose working set the save strips — so "byte-identical" is a claim
/// about a real serialization path rather than about a two-entity toy).
#[test]
fn a_save_after_the_charge_is_byte_identical_to_a_save_before_it() {
    use glam::DVec2;
    use inf_editor_core::scene::serialize;
    use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};

    let tmp = tempfile::tempdir().unwrap();
    let dir = phase22_playground_dir();
    let mut doc = serialize::load(&dir.join("Phase22Playground.inf_lvl")).expect("the doc loads");
    let level = Uuid::from_u128(0x2204_0000_0000_4000_8000_0000_0000_0001);

    let before = tmp.path().join("Before.inf_lvl");
    let after = tmp.path().join("After.inf_lvl");
    serialize::save(&doc, &before, Some(level)).expect("the pre-charge save");
    let pre = std::fs::read(&before).expect("read pre");

    let classes = bindings(&dir);
    let actors: Vec<(Uuid, inf_blueprint::BlueprintClass)> = doc
        .order()
        .iter()
        .filter_map(|guid| {
            let e = doc.world().entity_of(*guid)?;
            let ac = doc
                .world()
                .world()
                .get::<inf_ecs::components::ActorClass>(e)?;
            Some((*guid, classes.get(&ac.0)?.clone()))
        })
        .collect();
    assert_eq!(actors.len(), 4, "the level binds four scripted actors");

    let mut session = SimSession::enter(&mut doc, actors, DVec2::ZERO, SIM_HZ);
    // Seeded through the SAME pair of editor helpers Ring 2's `sim_start` uses, so
    // this is the Simulate path rather than a test-local imitation.
    let block = std::fs::read(dir.join("Block.inf_mesh")).unwrap();
    let chassis = std::fs::read(dir.join("Chassis.inf_mesh")).unwrap();
    let params = inf_editor_core::pie::destructible_mesh_params(&doc);
    let states = inf_physics::d3::resolve_fracture_states(doc.world(), |mesh_id| {
        let p = *params.get(&mesh_id)?;
        let raw = if mesh_id == PHASE22_BLOCK_MESH_GUID {
            &block
        } else if mesh_id == PHASE22_CHASSIS_MESH_GUID {
            &chassis
        } else {
            return None;
        };
        let mesh = inf_asset::decode::<inf_mesh::MeshAsset>(raw).ok()?;
        inf_editor_core::pie::derive_fracture(&mesh, mesh_id, p).map(std::sync::Arc::new)
    });
    assert_eq!(states.len(), 3, "the level has three destructible actors");
    session.set_fractures(states);

    for _ in 0..STEPS {
        session.tick(&mut doc, 1.0 / SIM_HZ, SimInput::default());
    }
    // ANTI-VACUITY, before the claim: the session really destroyed something.
    assert!(
        !session.fractures()[&PHASE22_TOWER_GUID].is_intact(),
        "the Simulate session never broke the block, so 'nothing was persisted' is \
         a statement about a session that did nothing"
    );
    session.exit(&mut doc);

    serialize::save(&doc, &after, Some(level)).expect("the post-charge save");
    assert_eq!(
        std::fs::read(&after).expect("read post"),
        pre,
        "a Simulate session's destruction reached the author's .inf_lvl — this \
         phase's ruling is that destruction is RUNTIME-ONLY, and a save that \
         carried rubble would mean an author who pressed Play could not press \
         Ctrl+S again"
    );
    assert_eq!(
        std::fs::read(after.with_extension("inf_lvl.toml")).unwrap(),
        std::fs::read(before.with_extension("inf_lvl.toml")).unwrap(),
        "the level's TOML sidecar changed across a Simulate destruction"
    );
    // …and Simulate left no footprints on the author's document either (P22.1's
    // own rule, re-checked on the shipped sample because this is the level where a
    // regression would be felt).
    assert!(
        inf_ecs::deform::deform_field(doc.world()).is_none(),
        "the deformation field outlived the Simulate session"
    );
}

// ── the sample really is the done-when sentence ─────────────────────────────

/// A sanity check on the fixture itself, asserted on the **generators**, so the
/// arms above cannot pass because the level was quietly retuned into something
/// that no longer demonstrates the phase.
#[test]
fn the_playground_is_actually_the_done_when_sentence() {
    // The roller's lane crosses BOTH deformable bands and starts and ends on the
    // one that barely dents.
    let end_z = PHASE22_ROLLER_START_Z + PHASE22_ROLLER_SPEED_M_S * STEPS as f64 / 60.0;
    assert_eq!(phase22_layer_at(PHASE22_ROLLER_START_Z), 0);
    assert!(
        end_z > PHASE22_SAND_Z.1,
        "the roller stops at z = {end_z}, inside the sand band — the trail has no \
         far end on undeformable ground to compare against"
    );
    assert_eq!(phase22_layer_at(PHASE22_SNOW_Z.0 + 1.0), PHASE22_SNOW_LAYER);
    assert_eq!(phase22_layer_at(PHASE22_SAND_Z.0 + 1.0), PHASE22_SAND_LAYER);
    // …and the roller is narrow enough for its trail to be a LINE inside a band
    // rather than a smear across two of them, which is what makes the off-lane
    // "depth is exactly zero" assertion mean anything.
    let narrowest_band =
        (PHASE22_SNOW_Z.1 - PHASE22_SNOW_Z.0).min(PHASE22_SAND_Z.1 - PHASE22_SAND_Z.0);
    assert!(
        2.0 * PHASE22_ROLLER_RADIUS_M < 0.25 * narrowest_band,
        "the roller is {:.2} m across against a {narrowest_band:.0} m band",
        2.0 * PHASE22_ROLLER_RADIUS_M
    );

    // The charge window sits inside the run with settle on both sides.
    let (trigger, end, run) = (PHASE22_TRIGGER_STEP, PHASE22_TRIGGER_END, STEPS as f64);
    assert!(
        trigger > 30.0,
        "the charge fires at step {trigger}, before the car has settled — the \
         pre-trigger prefix would be a falling world rather than a still one"
    );
    assert!(
        run - end > 120.0,
        "only {} step(s) of collapse and debris settle after the charge; the \
         window law wants seconds, not frames",
        run - end
    );

    // The block's mesh clears the cook's vgeom threshold, so the sample cooks
    // silently (the arm above would otherwise fail with a sub-threshold advisory
    // and the reason would be one file away).
    assert!(
        inf_editor_core::samples::phase22_tower_mesh().triangle_count() >= 2048,
        "the block mesh is below the vgeom derivation threshold"
    );
    assert!(inf_editor_core::samples::phase22_chassis_mesh().triangle_count() >= 2048);
}
