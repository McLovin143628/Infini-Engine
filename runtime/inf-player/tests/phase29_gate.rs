//! **THE PHASE 29 GATE** (P29.6) — the anim & movement wave's acceptance test.
//!
//! §13's "done when": *`samples/phase29-locomotion` — a wizard-produced
//! character driven by a v2 machine over a real movement component — runs in PIE
//! with **PIE == shipping on the (pose, movement-mode) trace**, replays
//! bit-exactly across two independent cooks, agrees between its Blueprint driver
//! and its transpiled Rust, holds foot-slide under a bound measured in metres,
//! and shows a one-line authoring change as a one-line diff.* The movement
//! catalogue amendment adds the sentence this file's anti-vacuity is written
//! against: *"P29.6's course must force every catalogue mode in its one
//! deterministic replay, so the (pose, mode) trace certifies the catalogue and
//! not a subset."*
//!
//! The arms, lettered:
//!
//! * **(a)** PIE == shipping, byte for byte, on the (pose, mode) trace over the
//!   whole course — with an anti-vacuity half that names **every** mode.
//! * **(b)** Bit-exact replay across two **independent cooks**.
//! * **(c)** Blueprint-versus-transpiled parity on a course segment driven
//!   through the `anim.*` kit.
//! * **(d)** **The one-line-diff demonstration** — pillar S1's acceptance test.
//! * **(e)** A deterministic **camera** trace that is *not* part of the sim
//!   trace (the ViewMode ruling's proof).
//! * **(f)** The committed content really is **derived**, and the machine really
//!   is the **proposed** one.
//! * **(g)** `Driving` and `Flying` are typed **refusals**, by name, until P29.7.
//!
//! # Why the course driver reads the world instead of the clock
//!
//! Every other scripted replay in this repository is `fn script(step)` — a pure
//! function of the step index. This one is a pure function of **the character's
//! own position and mode**, and that is deliberate: a station is at a place, not
//! at a time, and a time-scripted run that reached the pool a tenth of a second
//! late would walk into the water standing up and quietly certify a smaller
//! catalogue. Determinism is unaffected — both hosts step the same world and
//! read the same numbers out of it — and PIE == shipping is a *stronger* claim
//! this way, because the two hosts now have to agree about where the character
//! is before they can agree about what it does next.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use glam::DVec2;
use uuid::Uuid;

use inf_ecs::components::{
    CharacterMovement, Collider3D, MovementMode, MovementRefusal, Transform,
};
use inf_ecs::EcsWorld;
use inf_editor_core::samples;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

/// The fixed step the whole gate runs at.
const HZ: f64 = 60.0;
/// How many fixed steps the course takes. Long enough for the character to reach
/// the far side of the pool; the coverage arm asserts it did.
const STEPS: u32 = 5400;
/// The prefix the one-line-diff demonstration runs — long enough for the first
/// transition to fire and for the difference to persist, short enough that the
/// arm is about the diff rather than about the whole course.
const DIFF_STEPS: u32 = 240;
/// The prefix the camera arms run. The course's first thousand steps carry the
/// gaits, the roof and the low stances, which is where a third-person camera has
/// the most to do.
const CAMERA_STEPS: u32 = 1800;
/// The course segment the Blueprint-versus-transpiled parity arm replays. Long
/// enough to contain a run and several footfalls, which its own anti-vacuity
/// half asserts.
const PARITY_STEPS: u32 = 900;

// ── the fixture ─────────────────────────────────────────────────────────────

fn hero() -> Uuid {
    samples::phase29_hero()
}

/// The committed sample's directory.
fn sample_dir() -> PathBuf {
    samples::phase29_locomotion_dir()
}

/// Every file the sample ships, so a fixture that copies the folder is checked
/// against the list rather than against `read_dir`'s mood.
fn sample_files() -> [&'static str; 19] {
    [
        "Hero Body.inf_mesh",
        "Hero Body.inf_mesh.toml",
        "Hero Controller.inf_act",
        "Hero Controller.inf_act.toml",
        "Hero Idle.inf_anim",
        "Hero Idle.inf_anim.toml",
        "Hero Locomotion.inf_sm",
        "Hero Locomotion.inf_sm.toml",
        "Hero Locomotion.inf_sm.txt",
        "Hero Run.inf_anim",
        "Hero Run.inf_anim.toml",
        "Hero Walk.inf_anim",
        "Hero Walk.inf_anim.toml",
        "Hero.inf_skel",
        "Hero.inf_skel.toml",
        "Phase29Locomotion.inf_lvl",
        "Phase29Locomotion.inf_lvl.toml",
        "camera.toml",
        "input.toml",
    ]
}

/// The anim assets the sample commits, decoded, keyed by GUID — what both hosts
/// are seeded with.
struct Assets {
    skeletons: BTreeMap<Uuid, inf_anim::SkeletonAsset>,
    clips: BTreeMap<Uuid, inf_anim::AnimClip>,
    machines: BTreeMap<Uuid, inf_anim::StateMachine>,
}

fn load_assets(machine_override: Option<inf_anim::StateMachine>) -> Assets {
    let dir = sample_dir();
    let ids = samples::phase29_asset_guids();
    let read = |name: &str| std::fs::read(dir.join(name)).expect("the sample file is committed");

    let skel: inf_anim::SkeletonAsset =
        inf_asset::decode(&read("Hero.inf_skel")).expect("the rig decodes");
    let mut clips = BTreeMap::new();
    for (name, guid) in [
        ("Hero Idle.inf_anim", ids.idle),
        ("Hero Walk.inf_anim", ids.walk),
        ("Hero Run.inf_anim", ids.run),
    ] {
        let asset: inf_anim::AnimClipAsset =
            inf_asset::decode(&read(name)).expect("the clip decodes");
        clips.insert(guid, asset.clip);
    }
    let machine = match machine_override {
        Some(m) => m,
        None => {
            let asset: inf_anim::StateMachineAsset =
                inf_asset::decode(&read("Hero Locomotion.inf_sm")).expect("the machine decodes");
            asset.machine
        }
    };
    Assets {
        skeletons: [(ids.skeleton, skel)].into_iter().collect(),
        clips,
        machines: [(ids.machine, machine)].into_iter().collect(),
    }
}

// ── the course driver ───────────────────────────────────────────────────────

/// One step's input, in the vocabulary both hosts resolve
/// (`inf_ecs::movement::actions`).
type Drive = (Vec<&'static str>, BTreeMap<String, f32>);

fn axes(pairs: &[(&str, f32)]) -> BTreeMap<String, f32> {
    pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect()
}

/// Forward at full deflection, with the given held actions.
fn forward(held: &[&'static str]) -> Drive {
    (held.to_vec(), axes(&[("move_y", 1.0)]))
}

/// **The course driver** — a stage machine whose every transition is a fact
/// about the world.
///
/// The first draft was a pure `fn(z, mode)` with no memory at all, and it does
/// not work for a reason worth writing down: a **roll** moves the character
/// sixteen centimetres, so a station expressed as a z-window fires its roll,
/// lands back in the window, and rolls again for ever. Stations are *events*,
/// not places, and an event needs somewhere to record that it happened.
///
/// The memory is still deterministic, and PIE == shipping is unaffected: both
/// hosts step the same world, see the same `(z, mode, grounded)` and therefore
/// advance the same stage on the same step. That is a **stronger** claim than a
/// clock-scripted replay, because the two hosts now have to agree about where
/// the character is before they can agree about what it does next.
#[derive(Debug, Default, Clone, Copy)]
struct Driver {
    stage: usize,
    /// Whether the current stage's target mode has been observed. A station is
    /// "press until it takes, then wait until it is over", and this is the
    /// "it took" half.
    took: bool,
    /// Steps spent in the current stage, for the stations that WAIT rather than
    /// press (the ragdoll settles before it is ended).
    held: u32,
}

impl Driver {
    /// Advance to the next station.
    fn next(&mut self) {
        self.stage += 1;
        self.took = false;
        self.held = 0;
    }

    /// One step's input.
    fn step(&mut self, z: f64, mode: MovementMode, grounded: bool) -> Drive {
        use MovementMode as M;
        // Press `action` until `mode` becomes `want`, then move on once the mode
        // has left it again — the shape every one-shot station has.
        macro_rules! oneshot {
            ($action:expr, $want:expr) => {{
                if mode == $want {
                    self.took = true;
                    forward(&[])
                } else if self.took {
                    self.next();
                    forward(&[])
                } else if matches!(mode, M::Grounded) {
                    (vec![$action], axes(&[("move_y", 1.0)]))
                } else {
                    // Somewhere in between (a crouch left over from the last
                    // station): stand up first.
                    (vec!["crouch"], axes(&[("move_y", 1.0)]))
                }
            }};
        }
        match self.stage {
            // ── open floor: the three gaits ──
            0 => {
                if z >= 4.0 {
                    self.next();
                }
                forward(&["walk"])
            }
            1 => {
                if z >= 8.0 {
                    self.next();
                }
                forward(&[])
            }
            2 => {
                if z >= 10.5 {
                    self.next();
                }
                forward(&["sprint"])
            }
            // ── the 1.4 m roof: a standing capsule does not fit under it ──
            3 => {
                if z >= 17.5 && mode == M::Crouch {
                    self.next();
                }
                if mode == M::Crouch {
                    forward(&[])
                } else {
                    (vec!["crouch"], axes(&[("move_y", 1.0)]))
                }
            }
            // ── prone, and the crawl ──
            4 => {
                if mode == M::Prone {
                    self.took = true;
                }
                if self.took && z >= 20.0 {
                    self.next();
                }
                if mode == M::Prone {
                    forward(&[])
                } else {
                    (vec!["prone"], axes(&[("move_y", 1.0)]))
                }
            }
            // ── back to a stand, one press at a time ──
            5 => match mode {
                M::Prone => (vec!["prone"], axes(&[("move_y", 1.0)])),
                M::Crouch => (vec!["crouch"], axes(&[("move_y", 1.0)])),
                _ => {
                    self.next();
                    forward(&[])
                }
            },
            // ── eight metres of sprint: the runway a 4 m/s slide entry needs ──
            6 => {
                if z >= 30.0 {
                    self.next();
                }
                forward(&["sprint"])
            }
            // ── the slide ──
            7 => {
                if mode == M::Slide {
                    self.took = true;
                    return forward(&["sprint"]);
                }
                if self.took {
                    self.next();
                    return forward(&[]);
                }
                if mode == M::Grounded {
                    (vec!["sprint", "crouch"], axes(&[("move_y", 1.0)]))
                } else {
                    forward(&["sprint"])
                }
            }
            // ── stand out of whatever the slide left behind ──
            8 => match mode {
                M::Slide | M::Crouch => (vec!["crouch"], axes(&[("move_y", 1.0)])),
                _ => {
                    self.next();
                    forward(&[])
                }
            },
            9 => oneshot!("roll", M::Roll),
            10 => oneshot!("dive", M::Dive),
            // ── a jump on open ground: `FallFree`, which only a jump makes ──
            11 => {
                if mode == M::FallFree {
                    self.took = true;
                    return forward(&[]);
                }
                if self.took && grounded {
                    self.next();
                    return forward(&[]);
                }
                match mode {
                    M::Grounded if grounded => (vec!["jump"], axes(&[("move_y", 1.0)])),
                    M::Crouch | M::Prone => (vec!["crouch"], axes(&[("move_y", 1.0)])),
                    _ => forward(&[]),
                }
            }
            // ── the stairs, the landing, and the drop off its far edge ──
            12 => {
                if z >= 63.0 {
                    self.next();
                }
                forward(&[])
            }
            // ── the three ledges. A jump with the stick forward WITHIN REACH of
            //    a face is a mantle; the same jump two metres early is just a
            //    jump, which is how the first draft of this course cleared a 1 m
            //    ledge and reported no mantle at all. ──
            13 | 15 | 17 => {
                if mode == M::Mantle {
                    self.took = true;
                    return forward(&[]);
                }
                if self.took && grounded {
                    self.next();
                    return forward(&[]);
                }
                if grounded {
                    (vec!["jump"], axes(&[("move_y", 1.0)]))
                } else {
                    forward(&[])
                }
            }
            14 => {
                if z >= 69.0 {
                    self.next();
                }
                forward(&[])
            }
            16 => {
                if z >= 75.0 {
                    self.next();
                }
                forward(&[])
            }
            18 => {
                if z >= 81.0 {
                    self.next();
                }
                forward(&[])
            }
            // ── off the 5 m ledge, JUMPING. Five metres is 9.9 m/s and the
            //    ragdoll threshold is 10.0, so the apex a 4.5 m/s jump adds is
            //    what puts the landing over it. ──
            19 => {
                if mode == M::Ragdoll {
                    self.next();
                    return (Vec::new(), BTreeMap::new());
                }
                if grounded && mode == M::Grounded && z >= 81.0 {
                    (vec!["jump"], axes(&[("move_y", 1.0)]))
                } else {
                    forward(&[])
                }
            }
            // ── the ragdoll: let it SETTLE, then end it with a jump. The
            //    get-up follows by itself, pose-matched off the pelvis. ──
            20 => {
                if mode != M::Ragdoll {
                    self.next();
                    return forward(&[]);
                }
                self.held += 1;
                if self.held > 90 {
                    (vec!["jump"], BTreeMap::new())
                } else {
                    (Vec::new(), BTreeMap::new())
                }
            }
            // ── run to the pool ──
            21 => {
                if z >= 99.0 {
                    self.next();
                }
                forward(&[])
            }
            // ── swim across, submerged for the middle stretch ──
            _ => {
                let mut a = axes(&[("move_y", 1.0)]);
                if (104.0..114.0).contains(&z) {
                    a.insert("move_up".to_string(), -1.0);
                }
                (Vec::new(), a)
            }
        }
    }
}

// ── the trace ───────────────────────────────────────────────────────────────

/// `f64`s in the movement half of a record, written before the first
/// discriminant. Pinned against a real record by [`assert_not_vacuous`], which
/// is the P29.4 audit's A10 lesson: a derived offset nothing checks against the
/// writer is an offset that drifts.
const FLOATS: usize = 14;
/// Single-byte discriminants and flags after them.
const FLAGS: usize = 6;

fn record_len(pose: usize) -> usize {
    pose + FLOATS * 8 + FLAGS
}

/// Where the mode byte sits inside a record whose pose half is `pose` bytes.
fn mode_at(pose: usize) -> usize {
    pose + FLOATS * 8
}

/// **The (pose, mode) trace record.** The evaluated pose exactly as
/// `state_bytes` folds it, then the movement state that chose it.
fn record(world: &EcsWorld) -> Vec<u8> {
    let mut out = inf_ecs::pose::pose_state_bytes(world);
    let Some(e) = world.entity_of(hero()) else {
        return out;
    };
    let w = world.world();
    let (Some(t), Some(cm), Some(c)) = (
        w.get::<Transform>(e),
        w.get::<CharacterMovement>(e),
        w.get::<Collider3D>(e),
    ) else {
        return out;
    };
    let rt = &cm.runtime;
    for v in [
        t.translation.x,
        t.translation.y,
        t.translation.z,
        t.rotation.y,
        c.half_extents.y,
        rt.velocity.x,
        rt.velocity.y,
        rt.velocity.z,
        rt.aim_yaw_deg,
        rt.body_yaw_deg,
        rt.mapped_speed,
        rt.gait_scalar,
        rt.land_impact_mps,
        rt.time_in_mode_s,
    ] {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out.push(cm.mode as u8);
    out.push(cm.gait as u8);
    out.push(rt.actual_gait as u8);
    out.push(rt.direction as u8);
    out.push(rt.landing as u8);
    out.push(u8::from(rt.grounded));
    out
}

/// Where the character is, and what it is doing — the driver's input.
fn state_of(world: &EcsWorld) -> (f64, MovementMode, bool) {
    let Some(e) = world.entity_of(hero()) else {
        return (0.0, MovementMode::Grounded, false);
    };
    let w = world.world();
    let z = w
        .get::<Transform>(e)
        .map(|t| t.translation.z)
        .unwrap_or(0.0);
    let (mode, grounded) = w
        .get::<CharacterMovement>(e)
        .map(|cm| (cm.mode, cm.runtime.grounded))
        .unwrap_or((MovementMode::Grounded, false));
    (z, mode, grounded)
}

// ── the two hosts ───────────────────────────────────────────────────────────

/// Copy the committed sample into a scaffolded project and answer its `Content`.
fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    inf_project::ProjectManifest::new("Phase 29 Locomotion", "blank-3d")
        .save(&proj)
        .expect("the manifest saves");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("mkdir Content");
    let src = sample_dir();
    for f in sample_files() {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    content
}

/// Scaffold + cook; answers the pack directory.
fn cook_course(tmp: &Path) -> PathBuf {
    scaffold(tmp);
    let out = tmp.join("out");
    inf_packager::cook(
        &tmp.join("proj"),
        &out,
        &inf_packager::CookOptions::default(),
    )
    .expect("the course cooks");
    out
}

/// **The shipping side**: a `RuntimeSim` off a cooked pack, exactly as
/// `run_headless` builds one.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("pack world builds");
    inf_player::sim_from_built(built)
}

/// The committed document, loaded off disk rather than regenerated — the same
/// bytes a project would open.
fn course_doc() -> SceneDoc {
    inf_editor_core::scene::serialize::load(&sample_dir().join("Phase29Locomotion.inf_lvl"))
        .expect("the committed course loads")
}

/// **The PIE payload** the editor really builds for the committed sample, with
/// an optional machine substituted for the one-line-diff arm.
fn course_payload(machine: Option<&inf_anim::StateMachine>) -> inf_runtime::pie::ScenePayload {
    let dir = sample_dir();
    let ids = samples::phase29_asset_guids();
    let doc = course_doc();
    let read = |name: &str| std::fs::read(dir.join(name)).expect("the sample file is committed");
    let controller =
        samples::decode_actor(&read("Hero Controller.inf_act")).expect("the controller decodes");
    let skel = read("Hero.inf_skel");
    let mesh = read("Hero Body.inf_mesh");
    let idle = read("Hero Idle.inf_anim");
    let walk = read("Hero Walk.inf_anim");
    let run = read("Hero Run.inf_anim");
    let sm = match machine {
        Some(m) => inf_asset::encode(&inf_anim::StateMachineAsset::new(
            m.clone(),
            Some(*ids.skeleton.as_bytes()),
        ))
        .expect("the substituted machine encodes"),
        None => read("Hero Locomotion.inf_sm"),
    };
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == ids.actor).then(|| controller.clone()),
        |_| None,
        |guid| match guid {
            g if g == ids.skeleton => Some(skel.clone()),
            g if g == ids.idle => Some(idle.clone()),
            g if g == ids.walk => Some(walk.clone()),
            g if g == ids.run => Some(run.clone()),
            g if g == ids.machine => Some(sm.clone()),
            _ => None,
        },
        |_| None,
        |_| None,
        |_| None,
        |guid| (guid == ids.mesh).then(|| mesh.clone()),
        |_| None,
        HZ as u32,
        false,
    )
    .expect("the payload builds");
    // **Non-vacuity at the payload**, before anything is compared: a payload that
    // carried no machine, no rig or no clips would make every trace below a
    // comparison of two motionless characters (the P21.4 lesson, met twice since).
    assert_eq!(payload.machines.len(), 1, "the machine must ride the wire");
    assert_eq!(payload.skeletons.len(), 1, "the rig must ride the wire");
    assert_eq!(
        payload.clips.len(),
        3,
        "all three cycles must ride the wire"
    );
    assert_eq!(payload.meshes.len(), 1, "the body must ride the wire");
    payload
}

/// **The PIE side**: through `sim_from_payload`, the one boot seam the `--pie`
/// subprocess takes.
fn pie_sim(machine: Option<&inf_anim::StateMachine>) -> RuntimeSim {
    inf_player::sim_from_payload(&course_payload(machine))
        .expect("the PIE world builds")
        .sim
}

/// The editor's own Simulate over the committed document, seeded by hand — the
/// third host, and the one the author actually watches.
fn editor_session(machine: Option<inf_anim::StateMachine>) -> (SceneDoc, SimSession) {
    let assets = load_assets(machine);
    let mut doc = course_doc();
    let mut session = SimSession::enter(&mut doc, samples::phase29_actors(), DVec2::ZERO, HZ);
    session.set_skeletons(assets.skeletons);
    session.set_pose_clips(assets.clips);
    session.set_state_machines(assets.machines);
    (doc, session)
}

fn run_trace(mut sim: RuntimeSim, steps: u32) -> Vec<Vec<u8>> {
    let mut driver = Driver::default();
    let mut out = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let (z, mode, grounded) = state_of(sim.world());
        let (held, ax) = driver.step(z, mode, grounded);
        sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
        out.push(record(sim.world()));
    }
    out
}

fn editor_trace(machine: Option<inf_anim::StateMachine>, steps: u32) -> Vec<Vec<u8>> {
    let (mut doc, mut session) = editor_session(machine);
    let mut driver = Driver::default();
    let mut out = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let (z, mode, grounded) = state_of(doc.world());
        let (held, ax) = driver.step(z, mode, grounded);
        session.step_once(&mut doc, SimInput::with_down(held).with_axes(ax));
        out.push(record(doc.world()));
    }
    session.exit(&mut doc);
    out
}

/// Every mode the trace visited, by name — the anti-vacuity list the amendment
/// asks for.
fn modes_in(trace: &[Vec<u8>], pose: usize) -> BTreeSet<u8> {
    trace.iter().map(|r| r[mode_at(pose)]).collect()
}

/// …and how many steps each one occupies — the ledger's table, as data the gate
/// can assert on rather than a printout (P29.6 audit, A3).
fn mode_counts(trace: &[Vec<u8>], pose: usize) -> std::collections::BTreeMap<u8, usize> {
    let mut out = std::collections::BTreeMap::new();
    for r in trace {
        *out.entry(r[mode_at(pose)]).or_insert(0) += 1;
    }
    out
}

/// What the course owes a mode: force it, or account for why not.
///
/// The list below is a `match` on purpose (P29.6 audit, A3). `required_modes`
/// used to be a hand-written `vec!` of twelve, and `assert_not_vacuous` tests
/// that nothing on that list is **missing** — so *deleting a row deleted the
/// obligation*, silently, and the whole gate stayed green. Measured: removing
/// `("Prone", M::Prone)` left all ten arms passing.
///
/// A `match` with no wildcard is the pin. Every variant of `MovementMode` is
/// classified here, the compiler refuses the day one is added, and
/// `the_catalogue_is_accounted_for_variant_by_variant` re-derives the twelve
/// from it rather than restating them.
enum ModeDuty {
    /// The course must force it, under this name.
    Forced(&'static str),
    /// P29.7 owns the mechanics; arm (g) asserts the typed refusal instead.
    RefusedUntilP297,
    /// A wire slot with no meaning yet.
    Reserved,
}

fn duty_of(mode: MovementMode) -> ModeDuty {
    use ModeDuty::*;
    use MovementMode as M;
    match mode {
        M::Grounded => Forced("Grounded"),
        M::Crouch => Forced("Crouch"),
        M::Prone => Forced("Prone"),
        M::Slide => Forced("Slide"),
        M::Roll => Forced("Roll"),
        M::Dive => Forced("Dive"),
        M::FallFree => Forced("FallFree"),
        M::FallControlled => Forced("FallControlled"),
        M::SwimSurface => Forced("SwimSurface"),
        M::SwimUnder => Forced("SwimUnder"),
        M::Mantle => Forced("Mantle"),
        M::Ragdoll => Forced("Ragdoll"),
        M::Driving | M::Flying => RefusedUntilP297,
        M::Reserved14 | M::Reserved15 | M::Reserved16 | M::Reserved17 => Reserved,
    }
}

/// Every `MovementMode` the wire declares, in declaration order — the domain
/// [`duty_of`] is total over, and the one thing this file cannot derive from the
/// type system (Rust has no variant enumeration). It is checked against the
/// discriminants rather than trusted: see
/// `the_catalogue_is_accounted_for_variant_by_variant`.
const ALL_MODES: [MovementMode; 18] = {
    use MovementMode as M;
    [
        M::Grounded,
        M::Crouch,
        M::Prone,
        M::Slide,
        M::Roll,
        M::Dive,
        M::FallFree,
        M::FallControlled,
        M::SwimSurface,
        M::SwimUnder,
        M::Mantle,
        M::Ragdoll,
        M::Driving,
        M::Flying,
        M::Reserved14,
        M::Reserved15,
        M::Reserved16,
        M::Reserved17,
    ]
};

/// The modes the course must force — **derived** from [`duty_of`], not restated.
/// `Driving` and `Flying` are absent because that function says so, and arm (g)
/// asserts the refusal they are absent for.
fn required_modes() -> Vec<(&'static str, MovementMode)> {
    ALL_MODES
        .iter()
        .filter_map(|m| match duty_of(*m) {
            ModeDuty::Forced(name) => Some((name, *m)),
            _ => None,
        })
        .collect()
}

/// The modes P29.7 owns — derived from the same `match`, so arm (g) and the
/// anti-vacuity list cannot drift apart.
fn refused_modes() -> Vec<MovementMode> {
    ALL_MODES
        .iter()
        .copied()
        .filter(|m| matches!(duty_of(*m), ModeDuty::RefusedUntilP297))
        .collect()
}

/// **The catalogue is accounted for variant by variant** (P29.6 audit, A3).
///
/// Three claims, and together they are what stops the anti-vacuity list from
/// being shortened by an edit nothing notices:
///
/// 1. [`ALL_MODES`] really is every variant — checked against the
///    discriminants, which are wire-frozen (§13), so a variant added anywhere
///    but the end changes a number here;
/// 2. every one of them is classified, and the classification is a `match` with
///    no wildcard, so adding a variant is a **compile error** rather than a
///    silently unclassified mode;
/// 3. the counts are the ledger's: **twelve** forced, two refused, four
///    reserved.
#[test]
fn the_catalogue_is_accounted_for_variant_by_variant() {
    for (i, m) in ALL_MODES.iter().enumerate() {
        assert_eq!(
            *m as u8, i as u8,
            "`ALL_MODES` is not the wire order at index {i}: {m:?} has \
             discriminant {}",
            *m as u8
        );
    }
    let forced = required_modes();
    let refused = refused_modes();
    let reserved = ALL_MODES
        .iter()
        .filter(|m| matches!(duty_of(**m), ModeDuty::Reserved))
        .count();
    assert_eq!(
        forced.len(),
        12,
        "the course's obligation is twelve modes and this list has {} — a row \
         was deleted, and `assert_not_vacuous` only checks that nothing on the \
         list is MISSING, so it would not have said so",
        forced.len()
    );
    assert_eq!(refused.len(), 2, "P29.7 owns exactly Driving and Flying");
    assert_eq!(reserved, 4, "the wire has four reserved slots");
    assert_eq!(forced.len() + refused.len() + reserved, ALL_MODES.len());
    // Every reserved slot answers `reserved_slot`, and no forced one does —
    // the classification agrees with the engine's own answer rather than with
    // this file's opinion.
    for m in ALL_MODES {
        assert_eq!(
            m.reserved_slot().is_some(),
            matches!(duty_of(m), ModeDuty::Reserved),
            "{m:?} disagrees with `MovementMode::reserved_slot`"
        );
    }
    // The names are distinct and each is its variant's own spelling, so a
    // failure message names the mode a reader can find in the enum.
    let names: BTreeSet<&str> = forced.iter().map(|(n, _)| *n).collect();
    assert_eq!(names.len(), forced.len(), "two modes share a name");
    for (name, m) in &forced {
        assert_eq!(*name, format!("{m:?}"), "the name is not the variant's");
    }
}

#[test]
#[ignore = "the coverage probe: run it by name while tuning the course"]
fn probe_the_course() {
    let mut sim = pie_sim(None);
    let mut driver = Driver::default();
    let mut seen: BTreeMap<u8, u32> = BTreeMap::new();
    let mut log: Vec<(u32, f64, f64, String)> = Vec::new();
    let mut last = MovementMode::Grounded;
    for i in 0..STEPS {
        let (z, mode, grounded) = state_of(sim.world());
        let (held, ax) = driver.step(z, mode, grounded);
        sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
        let (z, mode, _) = state_of(sim.world());
        let y = sim
            .world()
            .entity_of(hero())
            .and_then(|e| sim.world().world().get::<Transform>(e))
            .map(|t| t.translation.y)
            .unwrap_or(0.0);
        *seen.entry(mode as u8).or_default() += 1;
        if mode != last {
            log.push((i, z, y, format!("{mode:?} (stage {})", driver.stage)));
            last = mode;
        }
    }
    for (i, z, y, m) in &log {
        eprintln!("step {i:5}  z={z:8.3}  y={y:7.3}  {m}");
    }
    let (z, _, _) = state_of(sim.world());
    eprintln!("END z = {z:.3}");
    for (name, mode) in required_modes() {
        eprintln!(
            "{name:16} {:>6} steps{}",
            seen.get(&(mode as u8)).copied().unwrap_or(0),
            if seen.contains_key(&(mode as u8)) {
                ""
            } else {
                "   <<<< MISSING"
            }
        );
    }
}

// ── the anti-vacuity half ───────────────────────────────────────────────────

/// The pose half's width, derived from a real record and checked against the
/// writer — the P29.4 audit's A10 lesson, which is that a derived offset nothing
/// pins against the writer is an offset that drifts silently.
fn pose_width(trace: &[Vec<u8>]) -> usize {
    let n = trace[0].len();
    assert!(
        n > FLOATS * 8 + FLAGS,
        "a record of {n} bytes cannot hold the movement half at all — the \
         character published no pose, so every comparison below is between two \
         empty worlds"
    );
    let pose = n - (FLOATS * 8 + FLAGS);
    assert!(pose > 0, "the pose half is empty");
    pose
}

/// **`FLOATS` and `FLAGS` are what `record` actually writes** — pinned against
/// the writer on a world with **no** pose, where the whole record IS the
/// movement half (P29.6 audit, A10-again).
///
/// The arm this replaces was `assert_eq!(record_len(pose), n)` inside
/// `pose_width`, one line under `let pose = n - (FLOATS * 8 + FLAGS)`. Substitute
/// and it reads `n == n`: algebraically unfalsifiable, and advertised in its own
/// comment as the pin that stops a derived offset drifting. It stopped nothing.
/// A fifteenth `f64` added to `record` would have shifted `mode_at` eight bytes
/// early onto the `gait` byte, and the twelve-mode check would have started
/// certifying the wrong field — quietly, since a gait byte takes small values
/// too.
///
/// An empty `EcsWorld` publishes no pose, so `pose_state_bytes` is empty and the
/// record is exactly the movement half. That is a real measurement of the
/// writer, and it fails the moment the writer and these two constants disagree.
#[test]
fn the_record_layout_is_pinned_against_its_writer() {
    let empty = EcsWorld::new();
    let bare = record(&empty);
    assert!(
        bare.is_empty(),
        "an empty world published {} bytes of pose, so the calibration below is \
         not measuring the movement half alone",
        bare.len()
    );

    // A world with the hero but no skeleton: the early return is gone, so the
    // record is exactly `FLOATS` floats and `FLAGS` bytes and nothing else.
    let mut w = EcsWorld::new();
    let e = w.spawn_with_guid(hero(), "Hero", None);
    w.world_mut().entity_mut(e).insert((
        Transform::IDENTITY,
        CharacterMovement::default(),
        Collider3D {
            shape_kind: inf_ecs::components::ColliderShape3DKind::Capsule,
            ..Default::default()
        },
    ));
    let movement_only = record(&w);
    assert_eq!(
        movement_only.len(),
        FLOATS * 8 + FLAGS,
        "`record` writes {} bytes of movement state and this file believes it \
         writes {} — `mode_at` is therefore reading the wrong byte, and the \
         twelve-mode check is certifying the wrong field",
        movement_only.len(),
        FLOATS * 8 + FLAGS
    );
    assert_eq!(record_len(0), movement_only.len());
    // …and `mode_at` really lands on the mode. `Prone` rather than the default,
    // so a record of zeros cannot satisfy it.
    let mut w2 = EcsWorld::new();
    let e2 = w2.spawn_with_guid(hero(), "Hero", None);
    w2.world_mut().entity_mut(e2).insert((
        Transform::IDENTITY,
        CharacterMovement {
            mode: MovementMode::Prone,
            ..Default::default()
        },
        Collider3D {
            shape_kind: inf_ecs::components::ColliderShape3DKind::Capsule,
            ..Default::default()
        },
    ));
    let posed = record(&w2);
    assert_eq!(
        posed[mode_at(0)],
        MovementMode::Prone as u8,
        "`mode_at` does not point at the mode byte"
    );
}

/// **Every catalogue mode, by name.**
///
/// The amendment's sentence is this function's specification: *"P29.6's course
/// must force every catalogue mode in its one deterministic replay, so the
/// (pose, mode) trace certifies the catalogue and not a subset."* So the check
/// is not "the trace visited more than three modes" — it is a list, and a mode
/// that stops appearing fails **by name**.
fn assert_not_vacuous(trace: &[Vec<u8>]) {
    assert_eq!(trace.len() as u32, STEPS);
    let pose = pose_width(trace);
    assert!(
        trace.iter().all(|r| r.len() == trace[0].len()),
        "the record width changed mid-run"
    );
    let seen = modes_in(trace, pose);
    let missing: Vec<&str> = required_modes()
        .into_iter()
        .filter(|(_, m)| !seen.contains(&(*m as u8)))
        .map(|(n, _)| n)
        .collect();
    assert!(
        missing.is_empty(),
        "the course did not force {missing:?} — the (pose, mode) trace would \
         certify a SUBSET of the catalogue, which is exactly what §13's \
         movement-catalogue amendment forbids"
    );
    // **Present for more than one frame** (P29.6 audit, A3). Set membership is
    // satisfied by a single step, and a mode the character flickers through on
    // one frame of a transition is not a mode the trace certifies — it is a
    // glitch that happens to have the right discriminant. The shortest station
    // the shipped course actually visits is the slide, at thirteen steps, so a
    // floor of five is well under every real one and well over an accident.
    const MODE_FLOOR: usize = 5;
    let counts = mode_counts(trace, pose);
    let brief: Vec<(&str, usize)> = required_modes()
        .into_iter()
        .map(|(n, m)| (n, counts.get(&(m as u8)).copied().unwrap_or(0)))
        .filter(|(_, c)| *c < MODE_FLOOR)
        .collect();
    assert!(
        brief.is_empty(),
        "these modes appear for fewer than {MODE_FLOOR} steps: {brief:?} — a \
         one-frame flicker is not a station the course forces"
    );
    // …and the pose really moved, or the mode half is riding a still character.
    assert!(
        trace.iter().any(|r| r[..pose] != trace[0][..pose]),
        "the pose half never changed across {STEPS} steps"
    );
    assert_ne!(
        trace[0],
        trace[trace.len() - 1],
        "the character ended the course in exactly the state it started in"
    );
}

// ── (a) PIE == shipping on the (pose, mode) trace ───────────────────────────

/// **THE HEADLINE.** §13's "done when", first clause.
///
/// The shipped side is a **cooked pack**; the PIE side is the payload the editor
/// really builds, through `sim_from_payload` — the one boot seam the `--pie`
/// subprocess takes. Compared per step rather than as two vectors, because the
/// failure this exists to catch is a *divergence point* and "which step" is most
/// of the diagnosis.
#[test]
fn pie_equals_shipping_on_the_pose_and_mode_trace() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pack = cook_course(tmp.path());
    let ship = run_trace(pack_sim(&pack), STEPS);
    let pie = run_trace(pie_sim(None), STEPS);
    assert_not_vacuous(&ship);
    // Both lengths, before the zip: `zip` truncates to the shorter side, so a
    // PIE run that returned early would compare zero pairs and pass (P29.6
    // audit). `assert_not_vacuous` pins the shipped side only.
    assert_eq!(
        ship.len(),
        pie.len(),
        "the two hosts produced different numbers of steps"
    );
    let pose = pose_width(&ship);
    for (i, (s, p)) in ship.iter().zip(pie.iter()).enumerate() {
        assert_eq!(
            s,
            p,
            "step {i}: the cooked pack and the PIE payload disagree — shipping \
             mode {}, PIE mode {}",
            s[mode_at(pose)],
            p[mode_at(pose)]
        );
    }
}

/// The **third** host: the editor's own Simulate, over the same committed
/// document and the same assets.
///
/// A separate arm and not a third comparison inside the one above, because it is
/// a different claim: `sim_from_payload` and `SimSession` are two different
/// programs over one Ring-0 fixed step, and what this asserts is that the
/// hand-maintained pair still slot the same calls in the same order.
///
/// # THE BOUND THIS ARM FOUND, stated rather than hidden
///
/// The two hosts are byte-identical for the whole course **up to the ragdoll** —
/// two and a half thousand steps, covering every gait, the crouch, the prone
/// crawl, the slide, the roll, the dive, both falls and all three mantles — and
/// then diverge. The divergence begins on the step the ragdoll's articulated
/// bodies are spawned, which is the first moment either world contains a
/// **dynamic** body at all: it starts in the last bits of the character's
/// position and, because a jointed body assembly is chaotic, it grows — nine
/// millimetres four steps later, and by step 2 545 the two hosts have the
/// ragdoll ending one step apart.
///
/// The cause is rapier's determinism contract rather than this engine's. It
/// guarantees the same answer for the same **sequence of operations**, and the
/// two hosts do not have the same one: a `SimSession` is entered over a document
/// that has been built and edited, so its body handles carry a different
/// generation history, and the island solver's iteration order follows the
/// handles. Nothing in this repository had ever compared the two hosts across a
/// dynamic solve — `movement_parity` and `pose_parity` both run worlds whose
/// only dynamic body is a settled crate, and P29.4's own ragdoll fixtures run
/// one host.
///
/// **PIE == shipping is unaffected and is not this arm.** That claim compares a
/// cooked pack with the PIE payload — two `RuntimeSim`s built the same way — and
/// it is byte-exact over the whole course, ragdoll included. What is bounded
/// here is the *editor preview* against a build, and what the bound says is:
/// identical until something dynamic exists, and the same **behaviour** after.
///
/// Carried to P29.7 with a name: either both hosts build their physics world
/// through one construction sequence, or a dynamic solve is host-local and the
/// editor's preview of one is an approximation. Choosing is a design decision,
/// not a fix.
#[test]
fn the_editors_simulate_matches_the_shipped_player() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pack = cook_course(tmp.path());
    let ship = run_trace(pack_sim(&pack), STEPS);
    let editor = editor_trace(None, STEPS);
    assert_not_vacuous(&editor);
    let pose = pose_width(&ship);
    let ragdoll = MovementMode::Ragdoll as u8;
    let first_ragdoll = ship
        .iter()
        .position(|r| r[mode_at(pose)] == ragdoll)
        .expect("the course ragdolls");

    // ── exact, up to the first dynamic body ──
    for (i, (s, e)) in ship
        .iter()
        .zip(editor.iter())
        .enumerate()
        .take(first_ragdoll)
    {
        assert_eq!(
            s,
            e,
            "step {i}: the shipped player and the editor's Simulate disagree \
             BEFORE any dynamic body exists — shipping mode {}, editor mode {}",
            s[mode_at(pose)],
            e[mode_at(pose)]
        );
    }
    assert!(
        first_ragdoll > 2000,
        "the exact half covered only {first_ragdoll} steps, which is not the \
         course"
    );

    // ── behavioural, after it ──
    //
    // The same catalogue, and the same finish. Not the same bytes: see the bound
    // above.
    assert_eq!(
        modes_in(&ship, pose),
        modes_in(&editor, pose),
        "the two hosts certified different catalogues"
    );
    let float_at = |r: &[u8], k: usize| {
        let o = pose + k * 8;
        f64::from_bits(u64::from_le_bytes(r[o..o + 8].try_into().unwrap()))
    };
    let end_gap = (0..3)
        .map(|k| {
            let (a, b) = (
                float_at(&ship[ship.len() - 1], k),
                float_at(&editor[editor.len() - 1], k),
            );
            (a - b) * (a - b)
        })
        .sum::<f64>()
        .sqrt();
    assert!(
        end_gap < 2.0,
        "the two hosts finished the course {end_gap:.3} m apart — that is not a \
         rounding difference amplified by a ragdoll, it is a different run"
    );
    // …and they both finished it, rather than both stopping early in the same
    // place.
    let far = float_at(&ship[ship.len() - 1], 2);
    assert!(
        far > 110.0,
        "the shipped run ended at z = {far:.1}, short of the pool — the arm above \
         is comparing two characters that never got there"
    );
}

// ── (b) bit-exact replay across two independent cooks ───────────────────────

/// **Two cooks, two packs, one trace** (the P9 discipline).
///
/// Not two runs off one pack — that is a determinism check on the *sim*, and
/// this is a determinism check on the **pipeline**: a cook that made a different
/// choice on the second pass (an id, an order, a derived byte) would produce a
/// pack that plays differently, and every gate that compared one machine with
/// itself would go on passing.
#[test]
fn the_course_replays_bit_identically_across_two_independent_cooks() {
    let a = tempfile::tempdir().expect("tempdir a");
    let b = tempfile::tempdir().expect("tempdir b");
    let pack_a = cook_course(a.path());
    let pack_b = cook_course(b.path());
    let trace_a = run_trace(pack_sim(&pack_a), STEPS);
    let trace_b = run_trace(pack_sim(&pack_b), STEPS);
    assert_not_vacuous(&trace_a);
    assert_eq!(
        trace_a.len(),
        trace_b.len(),
        "the two cooks produced different numbers of steps"
    );
    for (i, (x, y)) in trace_a.iter().zip(trace_b.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: two independent cooks diverged");
    }
}

// ── (c) Blueprint versus transpiled, over a course segment ──────────────────

/// What the `anim.*` kit does, as observable effects — the stand-in bridge.
#[derive(Clone, Debug, Default, PartialEq)]
struct Kit {
    /// `(name, value)` in call order.
    params: Vec<(String, f64)>,
    /// Notifies waiting to be consumed this step.
    pending: BTreeSet<String>,
    /// Notifies actually taken, in order.
    taken: Vec<String>,
    /// The machine's current state name.
    state: String,
    /// The actor's member variables.
    vars: BTreeMap<String, f64>,
}

impl Kit {
    fn set_param(&mut self, name: &str, value: f64) -> bool {
        self.params.push((name.to_string(), value));
        true
    }
    fn query_state(&self, name: &str) -> bool {
        self.state == name
    }
    fn consume_notify(&mut self, name: &str) -> bool {
        if self.pending.remove(name) {
            self.taken.push(name.to_string());
            true
        } else {
            false
        }
    }
    fn get(&self, name: &str) -> f64 {
        self.vars.get(name).copied().unwrap_or(0.0)
    }
    fn set(&mut self, name: &str, value: f64) {
        self.vars.insert(name.to_string(), value);
    }
}

/// **The compiled half**: a hand-written Rust mirror of what `generate_fn`
/// emits for the committed controller's `Tick`.
///
/// Hand-written on purpose (the P6.6 / P29.4 shape): compiling the generated
/// source at test time would need a `cargo build` inside a test, and the string
/// pin at the end of the arm is what keeps this mirror honest instead.
mod compiled {
    use super::Kit;

    /// `footsteps` is the rig's own event-marker names, in the order
    /// `DerivedNames` sorts them — `footstep_upper_leg_l`, `footstep_upper_leg_r`
    /// on the template biped. The controller is a function of the rig, so its
    /// compiled mirror has to be too; hard-coding `footstep_l` here is exactly
    /// the defect this arm found in the generator.
    pub fn tick(k: &mut Kit, dt: f64, footsteps: &[String]) {
        let _n1 = k.get("entity");
        for marker in footsteps {
            if k.consume_notify(marker) {
                let v = k.get("steps") + 1.0;
                k.set("steps", v);
            }
        }
        if k.query_state("run") {
            let v = k.get("run_time") + dt;
            k.set("run_time", v);
        }
        let steps = k.get("steps");
        let _ = k.set_param("steps", steps);
    }
}

/// One step of the course, as the `anim.*` kit sees it.
#[derive(Clone, Debug, PartialEq)]
struct KitStep {
    state: String,
    fired: BTreeSet<String>,
}

/// **Record a real course segment** through the `anim.*` doors: which state the
/// machine is in, and which notifies fired, on every step.
///
/// This is what makes the parity arm about the *course* rather than about a
/// synthetic script — the sequence below is the character's own walk, run and
/// footfalls, read off the sim.
fn kit_segment(steps: u32) -> Vec<KitStep> {
    let mut sim = pie_sim(None);
    let mut driver = Driver::default();
    let mut out = Vec::with_capacity(steps as usize);
    for _ in 0..steps {
        let (z, mode, grounded) = state_of(sim.world());
        let (held, ax) = driver.step(z, mode, grounded);
        sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
        let world = sim.world();
        // The state's NAME, straight off the bridge — the same door
        // `anim.query_state` answers from, so the segment is what the kit sees
        // and not a second reading of it.
        let state = inf_ecs::anim_bridge::anim_state(world, hero())
            .map(|s| s.name.clone())
            .unwrap_or_default();
        let fired: BTreeSet<String> = inf_ecs::pose::anim_events(world, hero())
            .iter()
            .cloned()
            .collect();
        out.push(KitStep { state, fired });
    }
    out
}

/// **PILLAR S6, on committed content.** *"The same `anim.*` semantics run
/// interpreted in the editor and compiled in the ship build, because both sides
/// are the same IR."*
///
/// The program is the **sample's own `.inf_act`**, read off disk — not a fixture
/// written to be easy — and the input is a real segment of the course: the
/// states its machine actually entered and the footstep notifies its derived
/// clips actually fired. Interpreted and compiled must drive the kit to the same
/// effects, step for step.
#[test]
fn the_committed_controller_is_the_same_program_interpreted_and_compiled() {
    use inf_blueprint::interp::{eval_fn, FnHost, Value};
    use std::collections::HashMap;

    let class = samples::decode_actor(
        &std::fs::read(sample_dir().join("Hero Controller.inf_act")).unwrap(),
    )
    .expect("the committed controller decodes");
    let tick = class
        .events
        .iter()
        .find(|e| e.event == inf_blueprint::EventKind::Tick)
        .map(|e| e.body.clone())
        .expect("the controller has a Tick");

    // The rig's own footfall names — see `compiled::tick`.
    let footsteps: Vec<String> = inf_anim::DerivedNames::of_skeleton(
        &load_assets(None)
            .skeletons
            .into_values()
            .next()
            .expect("the rig is seeded")
            .skeleton,
    )
    .event_markers
    .into_iter()
    .collect();
    assert_eq!(footsteps.len(), 2, "a biped has two feet: {footsteps:?}");

    let segment = kit_segment(PARITY_STEPS);
    // **NOT VACUOUS**: the segment has to contain both a run and some footfalls,
    // or the two halves agree about a program that never branched.
    assert!(
        segment.iter().any(|s| s.state == "run"),
        "the course segment never entered `run`, so `anim.query_state`'s branch \
         is untaken on both sides"
    );
    let footfalls: usize = segment
        .iter()
        .map(|s| {
            s.fired
                .iter()
                .filter(|n| n.starts_with("footstep_"))
                .count()
        })
        .sum();
    assert!(
        footfalls >= 4,
        "the segment carries {footfalls} footstep notifies — the derivation's \
         own markers are not reaching the kit, so `consume_notify`'s branch is \
         untaken on both sides"
    );

    let dt = 1.0 / HZ;
    let run = |interpret: bool| -> Vec<Kit> {
        let mut kit = Kit::default();
        let mut out = Vec::with_capacity(segment.len());
        for step in &segment {
            kit.state = step.state.clone();
            kit.pending = step.fired.clone();
            kit.taken.clear();
            kit.params.clear();
            if interpret {
                let mut host = FnHost(|path: &[String], args: &[Value]| {
                    let name = |k: usize| args[k].as_str().unwrap_or_default().to_string();
                    Ok(match path {
                        p if p == ["vars", "get"] => Value::Float(kit.get(&name(0))),
                        p if p == ["vars", "set"] => {
                            kit.set(&name(0), args[1].as_float().unwrap_or(0.0));
                            Value::Unit
                        }
                        p if p == ["anim", "set_param"] => {
                            Value::Bool(kit.set_param(&name(1), args[2].as_float().unwrap_or(0.0)))
                        }
                        p if p == ["anim", "query_state"] => Value::Bool(kit.query_state(&name(1))),
                        p if p == ["anim", "consume_notify"] => {
                            Value::Bool(kit.consume_notify(&name(1)))
                        }
                        _ => Value::Unit,
                    })
                });
                let args: HashMap<String, Value> = [("dt".to_string(), Value::Float(dt))].into();
                eval_fn(&tick, &args, &mut host).expect("the interpreted Tick runs");
            } else {
                compiled::tick(&mut kit, dt, &footsteps);
            }
            out.push(kit.clone());
        }
        out
    };

    let interpreted = run(true);
    let built = run(false);
    for (i, (a, b)) in interpreted.iter().zip(built.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step {i}: the interpreted controller and the compiled one drove the \
             `anim.*` kit differently"
        );
    }
    // The counters really counted, or the equality is between two no-ops.
    let last = interpreted.last().expect("a segment");
    assert!(
        last.get("steps") >= 4.0,
        "the controller counted {} footfalls",
        last.get("steps")
    );
    assert!(last.get("run_time") > 0.0, "it never timed a run");

    // **THE STRING PIN** that keeps the compiled mirror honest — the same device
    // `anim_roundtrip` and P6.6's `parity` use. If `generate_fn` stops emitting
    // one of these calls, the mirror above is a fiction and this fails.
    let src = inf_transpile::generate_fn(&tick).expect("the controller transpiles");
    for fragment in [
        "anim::consume_notify(",
        &format!("\"{}\"", footsteps[0]),
        &format!("\"{}\"", footsteps[1]),
        "anim::query_state(",
        "\"run\"",
        "anim::set_param(",
        "\"steps\"",
    ] {
        assert!(
            src.contains(fragment),
            "the generated Rust is missing `{fragment}`:\n{src}"
        );
    }
    assert!(
        !src.contains("set_trigger"),
        "the generated Rust arms a trigger the controller does not"
    );
}

// ── (d) the one-line-diff demonstration ─────────────────────────────────────

/// **PILLAR S1'S ACCEPTANCE TEST.**
///
/// §13's opening finding is that a `.uasset` AnimBlueprint *"cannot be diffed,
/// reviewed, merged or edited outside the editor"*, and S1's answer is that a
/// machine is text. This is that claim, measured, in four parts:
///
/// 1. The committed `.inf_sm.txt` **is** the committed `.inf_sm` — read the text,
///    and it decodes to the machine beside it, byte for byte through the asset
///    encoder.
/// 2. Changing **one transition's blend duration** in that text changes **exactly
///    one line** of the file.
/// 3. The edited text still **validates** — the same door `sm_save` puts a
///    machine through.
/// 4. The (pose, mode) trace is **identical up to the step the affected
///    transition first fires**, and different after it. That is the half that
///    makes this a demonstration rather than a diff: it shows the one line
///    reaching the simulation, and reaching nothing before it.
#[test]
fn one_line_of_text_is_one_line_of_diff_and_one_change_in_the_trace() {
    let dir = sample_dir();
    let text = std::fs::read_to_string(dir.join("Hero Locomotion.inf_sm.txt"))
        .expect("the machine's text face is committed");
    let committed: inf_anim::StateMachineAsset =
        inf_asset::decode(&std::fs::read(dir.join("Hero Locomotion.inf_sm")).unwrap())
            .expect("the payload decodes");

    // 1. The two faces are one machine.
    let from_text = inf_anim::from_toml(&text).expect("the text reads back");
    assert_eq!(
        from_text, committed.machine,
        "the committed text and the committed payload are different machines — \
         the reviewable face is a lie"
    );

    // 2. One line. The edit is the FIRST transition's blend duration, found in
    //    the text rather than assumed, so this arm cannot drift from the file.
    let before_line = text
        .lines()
        .find(|l| l.starts_with("duration = "))
        .expect("a transition owns a `duration` line")
        .to_string();
    let after_line = "duration = 0.42";
    let edited = text.replacen(&before_line, after_line, 1);
    let diff: Vec<(usize, &str, &str)> = text
        .lines()
        .zip(edited.lines())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| (i, a, b))
        .collect();
    assert_eq!(
        text.lines().count(),
        edited.lines().count(),
        "the edit changed the file's shape"
    );
    assert_eq!(diff.len(), 1, "not a one-line diff: {diff:?}");
    assert_eq!(diff[0].1, before_line);
    assert_eq!(diff[0].2, after_line);

    // 3. It goes back in **through the door** (P29.6 audit, A5).
    //
    //    `sm_text::save_from_text` is the Ring-1 door pillar S1 promises: parse,
    //    validate through the same call `sm_save` makes, re-encode the payload,
    //    and rewrite the text from what was stored. The first cut of this arm
    //    parsed and validated by hand and never touched it, while the ledger
    //    said "the gate's demonstration goes through the door" — so the one
    //    thing that makes the text an *authoring* surface rather than a
    //    read-only projection was the one thing not exercised.
    //
    //    On a COPY, because the committed sample is not this test's to rewrite.
    let dir_tmp = tempfile::tempdir().expect("tempdir");
    let copy = dir_tmp.path().join("Hero Locomotion.inf_sm");
    std::fs::copy(dir.join("Hero Locomotion.inf_sm"), &copy).expect("copy the payload");
    std::fs::write(inf_editor_core::sm_text::text_path(&copy), &edited).expect("copy the text");
    let tuned = inf_editor_core::sm_text::save_from_text(&copy, &edited, committed.skeleton)
        .expect("the edited machine passes the same door `sm_save` uses");
    // The payload beside it IS the edited machine, read back through the shipped
    // reader — which is the half a text face nobody can save does not have.
    let stored: inf_anim::StateMachineAsset =
        inf_asset::decode(&std::fs::read(&copy).unwrap()).expect("the saved payload decodes");
    assert_eq!(stored.machine, tuned, "the door stored a different machine");
    assert_eq!(stored.skeleton, committed.skeleton, "the binding survived");
    assert_eq!(
        inf_anim::from_toml(
            &std::fs::read_to_string(inf_editor_core::sm_text::text_path(&copy)).unwrap()
        )
        .unwrap(),
        tuned,
        "the two faces drifted across the save"
    );
    assert_ne!(tuned, committed.machine, "the edit changed nothing");
    // …and it changed exactly the field the line names.
    let changed: Vec<usize> = committed
        .machine
        .transitions
        .iter()
        .zip(tuned.transitions.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        changed,
        vec![0],
        "more than one transition moved: {changed:?}"
    );
    assert_eq!(tuned.transitions[0].duration, 0.42);

    // 4. **The trace changes only after the affected transition fires**, and the
    //    control is what makes that a claim rather than an observation.
    //
    //    The edited edge is `idle -> walk`, so the pair is: a character that
    //    WALKS traces differently, and a character that stands still traces
    //    **identically** — for the whole run, byte for byte, because the one line
    //    that changed belongs to a transition that never fires.
    let still = |m: Option<&inf_anim::StateMachine>| -> Vec<Vec<u8>> {
        let mut sim = pie_sim(m);
        (0..DIFF_STEPS)
            .map(|_| {
                sim.step_once(RuntimeInput::default());
                record(sim.world())
            })
            .collect()
    };
    let idle_base = still(None);
    let idle_edit = still(Some(&tuned));
    assert_eq!(
        idle_base, idle_edit,
        "a character that never leaves `idle` traced differently for a blend \
         duration on the `idle -> walk` edge — the one line reached something \
         it does not name"
    );
    // …and the control is not vacuous: the world really is simulating (the
    // character settles onto the floor over those steps).
    assert!(
        idle_base.iter().any(|r| *r != idle_base[0]),
        "the still control never changed at all, so `identical` is a statement \
         about a frozen world"
    );

    let base = run_trace(pie_sim(None), DIFF_STEPS);
    let edit = run_trace(pie_sim(Some(&tuned)), DIFF_STEPS);
    assert_eq!(base.len(), edit.len(), "the two runs are different lengths");
    assert_eq!(base.len() as u32, DIFF_STEPS);
    let first = base
        .iter()
        .zip(edit.iter())
        .position(|(a, b)| a != b)
        .expect("the one-line edit changed nothing in the simulation at all");
    // **A real prefix** (P29.6 audit, A5). The first cut asserted
    // `base[..first] == edit[..first]`, which is what `position` means — a
    // tautology, and it would have been satisfied by `first == 0`, i.e. by an
    // edit that changed the very first frame. The claim is "identical UP TO the
    // step the affected transition fires", so the load-bearing number is that
    // `first` is where the edge actually fires, not merely somewhere.
    assert!(
        first > 0,
        "the trace diverged on step 0 — the edit reached the character before \
         the `idle -> walk` edge could possibly have fired, so \"identical up to \
         the transition\" is a claim about an empty prefix"
    );
    let walked = base
        .iter()
        .position(|r| {
            r[mode_at(pose_width(&base))] == MovementMode::Grounded as u8 && r != &base[0]
        })
        .unwrap_or(0);
    assert!(
        first >= walked,
        "the divergence at step {first} came BEFORE the character started \
         moving at step {walked}, so it is not the `idle -> walk` edge"
    );
    // …and what changed is the POSE, not the movement: a blend duration is an
    // animation decision and must not move the character.
    let pose = pose_width(&base);
    assert_ne!(
        base[first][..pose],
        edit[first][..pose],
        "the pose is identical at the divergence step, so something else moved"
    );
    assert_eq!(
        base[first][pose..],
        edit[first][pose..],
        "a blend duration moved the CHARACTER — the machine would be driving \
         the movement, which is backwards"
    );
}

// ── (e) the camera ──────────────────────────────────────────────────────────

/// Run the course with a locomotion camera and answer `(sim trace, camera
/// trace)`.
fn camera_run(steps: u32) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
    let mut sim = pie_sim(None);
    let mut driver = Driver::default();
    let (mut sim_trace, mut cam_trace) = (Vec::new(), Vec::new());
    for i in 0..steps {
        let (z, mode, grounded) = state_of(sim.world());
        let (held, ax) = driver.step(z, mode, grounded);
        // First person for one stretch, so the trace exercises the blend weight
        // and the seat as well as the third-person arm.
        sim.camera_mut().view_mode = if (900..1500).contains(&i) {
            inf_ecs::camera::ViewMode::FirstPerson
        } else {
            inf_ecs::camera::ViewMode::ThirdPerson
        };
        sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
        sim_trace.push(record(sim.world()));
        cam_trace.push(sim.camera().trace_bytes());
    }
    (sim_trace, cam_trace)
}

/// **The camera rides the replay, and is not part of it** (the ViewMode ruling's
/// proof).
///
/// Two claims that have to hold together. The camera trace is **deterministic** —
/// two runs of the same course produce the same bytes, so a camera bug is
/// reproducible from a trace like everything else. And **nothing the camera is
/// told reaches the simulation**, which is Ruling 4 kept literally: `ViewMode`
/// never crosses the sim wire, and there is no camera → sim path at all.
///
/// # "Told", not "stepped" (P29.6 audit, A5)
///
/// The first cut said *byte-identical whether a camera was stepped or not*, and
/// no host can run that experiment: `RuntimeSim::step_once` steps the camera
/// unconditionally, as its last statement, so both sides of every comparison in
/// this file already have one. What is actually falsifiable at this level — and
/// what this arm does — is **perturbation**: drive the same course three times,
/// telling the camera something different each time (view mode, shoulder, a
/// whole different tuning table), and require the sim trace not to move. The
/// stepped-versus-not comparison exists, and it is `inf_physics`'s
/// `stepping_a_camera_changes_nothing_about_the_simulation`, where the two
/// programs really can differ by one line.
#[test]
fn the_camera_trace_is_deterministic_and_is_not_the_sim_trace() {
    let (sim_a, cam_a) = camera_run(CAMERA_STEPS);
    let (sim_b, cam_b) = camera_run(CAMERA_STEPS);
    assert_eq!(cam_a.len() as u32, CAMERA_STEPS);
    assert_eq!(cam_a.len(), cam_b.len());
    assert_eq!(cam_a, cam_b, "the camera trace is not deterministic");

    // …and it is a camera that MOVED, or "deterministic" is a statement about a
    // constant.
    assert!(
        cam_a.iter().any(|r| *r != cam_a[0]),
        "the camera never moved across {CAMERA_STEPS} steps"
    );
    let distinct: BTreeSet<&Vec<u8>> = cam_a.iter().collect();
    assert!(
        distinct.len() > CAMERA_STEPS as usize / 2,
        "the camera produced only {} distinct poses in {CAMERA_STEPS} steps",
        distinct.len()
    );

    // **The ruling.** The same course, told nothing about a view mode, traces
    // identically — and so does the same course under a wholly different camera
    // table, which is the perturbation a leak that ignored `view_mode` would
    // still have to survive.
    let bare = run_trace(pie_sim(None), CAMERA_STEPS);
    assert_eq!(
        bare.len(),
        sim_a.len(),
        "the two runs are different lengths"
    );
    for (i, (a, b)) in sim_a.iter().zip(bare.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step {i}: telling the camera about a view mode changed the \
             simulation — a camera value is reaching the sim, and the ViewMode \
             ruling says none may"
        );
    }
    assert_eq!(sim_a, sim_b);

    // A different camera in every respect the tuning door can express: a longer
    // arm, the other shoulder, a first-person seat all the way through, and lag
    // speeds nothing else in the tree uses.
    let (sim_c, cam_c) = {
        let mut sim = pie_sim(None);
        let mut driver = Driver::default();
        sim.camera_mut().right_shoulder = false;
        sim.camera_mut().view_mode = inf_ecs::camera::ViewMode::FirstPerson;
        let t = &mut sim.camera_mut().tuning;
        assert!(t.set("run.arm_length_m", 6.25));
        assert!(t.set("run.lag_x", 2.5));
        assert!(t.set("collision_radius_m", 0.4));
        assert!(t.set("pivot_height_ratio", 0.55));
        let (mut s, mut c) = (Vec::new(), Vec::new());
        for _ in 0..CAMERA_STEPS {
            let (z, mode, grounded) = state_of(sim.world());
            let (held, ax) = driver.step(z, mode, grounded);
            sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
            s.push(record(sim.world()));
            c.push(sim.camera().trace_bytes());
        }
        (s, c)
    };
    assert_ne!(
        cam_c, cam_a,
        "the retuned camera traced identically, so the perturbation below is \
         about nothing"
    );
    for (i, (a, b)) in sim_c.iter().zip(bare.iter()).enumerate() {
        assert_eq!(
            a, b,
            "step {i}: retuning the camera changed the simulation — a camera \
             value is reaching the sim through a path `view_mode` does not take"
        );
    }
}

/// The camera **collides**: a course that runs a character up against three
/// ledges and into a pool pulls the camera in more than once, and the pull is
/// recorded rather than inferred.
#[test]
fn the_camera_sweeps_against_the_course() {
    let mut sim = pie_sim(None);
    let mut driver = Driver::default();
    let mut pulled = 0u32;
    let mut worst = 0.0f64;
    for _ in 0..CAMERA_STEPS {
        let (z, mode, grounded) = state_of(sim.world());
        let (held, ax) = driver.step(z, mode, grounded);
        sim.step_once(RuntimeInput::with_down(held).with_axes(ax));
        let pull = sim.camera().collision_pull_m;
        if pull > 1e-6 {
            pulled += 1;
            worst = worst.max(pull);
        }
    }
    assert!(
        pulled > 0,
        "the camera never met the course — `cast_shape`'s third consumer is not \
         running, and a third-person camera that cannot be blocked is a camera \
         that ends up inside walls"
    );
    assert!(
        worst > 0.25,
        "the worst pull was {worst:.3} m, which is a rounding error rather than \
         a collision"
    );
}

// ── (f) the content really is derived, and the machine really is proposed ────

/// **The repository's first committed DERIVED content** — the remainder P29.4
/// and P29.5 both wrote down, closed.
///
/// Each clip is read off disk and checked for what only the derivation puts
/// there: a root-motion track, a distance track, foot-plant sync markers,
/// footstep notifies and the `W_Gait` channel. A generated clip has none of
/// them, which is what makes their presence proof rather than decoration.
#[test]
fn the_committed_clips_are_derived_and_the_machine_is_proposed() {
    let dir = sample_dir();
    let ids = samples::phase29_asset_guids();
    for name in [
        "Hero Idle.inf_anim",
        "Hero Walk.inf_anim",
        "Hero Run.inf_anim",
    ] {
        let asset: inf_anim::AnimClipAsset =
            inf_asset::decode(&std::fs::read(dir.join(name)).unwrap()).expect("the clip decodes");
        let clip = &asset.clip;
        assert!(
            clip.root_motion.is_some(),
            "{name} carries no root-motion track"
        );
        assert!(clip.distance.is_some(), "{name} carries no distance track");
        assert!(
            clip.curve(inf_anim::channels::als::W_GAIT).is_some(),
            "{name} carries no `W_Gait` — it was never measured"
        );
        assert!(
            clip.curve(inf_anim::channels::als::FOOT_LOCK_L).is_some(),
            "{name} carries no foot-lock channel"
        );
    }
    // The two cycles plant their feet; the idle does not, and that difference is
    // what the proposal clusters on.
    let plants = |name: &str| -> usize {
        let asset: inf_anim::AnimClipAsset =
            inf_asset::decode(&std::fs::read(dir.join(name)).unwrap()).unwrap();
        asset
            .clip
            .markers
            .iter()
            .filter(|m| m.is_sync() && m.name.starts_with("plant_"))
            .count()
    };
    assert!(plants("Hero Walk.inf_anim") >= 2, "the walk does not cycle");
    assert!(plants("Hero Run.inf_anim") >= 2, "the run does not cycle");

    // The machine is the PROPOSAL — three tiers, in tier order, over the three
    // derived clips, and it validates.
    let sm: inf_anim::StateMachineAsset =
        inf_asset::decode(&std::fs::read(dir.join("Hero Locomotion.inf_sm")).unwrap()).unwrap();
    let names: Vec<&str> = sm.machine.states.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(
        names,
        ["idle", "walk", "run"],
        "the proposal did not separate the tiers — the ladder it was given is \
         not this creature's"
    );
    assert_eq!(
        sm.machine,
        samples::phase29_machine(),
        "the committed machine drifted from the proposal"
    );
    sm.machine.validate().expect("the proposal validates");
    assert_eq!(sm.skeleton, Some(*ids.skeleton.as_bytes()));

    // And the machine really is what the wizard would write: the same shape the
    // wizard's own arm asserts, over the same doors.
    assert!(
        sm.machine
            .params
            .iter()
            .any(|p| p.name == inf_ecs::anim_bridge::params::SPEED),
        "the machine reads no `speed`, so nothing the engine publishes reaches it"
    );
}

// ── (g) the refusals P29.7 owns ─────────────────────────────────────────────

/// `Driving` and `Flying` are **typed refusals**, by name, until P29.7.
///
/// # What this arm is, and what it is not (P29.6 audit, A3)
///
/// A **unit** check on `request_mode`, the one door a mode change goes through,
/// asked from every legal starting mode. The first cut's doc claimed "the course
/// proves it with a live character rather than with a unit fixture" while the
/// body was one hard-coded call — the claim and the code disagreed, and the code
/// was the honest half: a course cannot demonstrate that a mode is *unreachable*
/// by visiting it.
///
/// What ties it to the course is `refused_modes()`: the same `match` that says
/// these two are exempt from the twelve-mode obligation says they refuse here,
/// so the exemption and the refusal cannot drift apart.
#[test]
fn driving_and_flying_are_refusals_with_the_wave_that_owns_them_named() {
    let refused = refused_modes();
    assert_eq!(
        refused.len(),
        2,
        "the exemption list moved: {refused:?} — the twelve-mode obligation and \
         this arm read the same `match`, so it must be Driving and Flying"
    );
    for mode in refused {
        // From EVERY mode a character can actually be in, not just from
        // `Grounded`: a refusal that only holds on one row of the transition
        // table is a refusal with a way in.
        for (_, from) in required_modes() {
            let verdict = inf_ecs::movement::request_mode(from, mode, true, true);
            assert_ne!(
                verdict.mode, mode,
                "{mode:?} was entered from {from:?}; P29.7 owns its mechanics \
                 and this wave ships a refusal"
            );
            assert!(
                matches!(verdict.refusal, MovementRefusal::ModeNotYetImplemented),
                "{mode:?} from {from:?} refused with {:?}, which does not name \
                 the wave that owns it",
                verdict.refusal
            );
        }
    }
}

// ── the fixture's own integrity ─────────────────────────────────────────────

/// The file list this fixture copies is the folder, checked rather than assumed.
#[test]
fn the_fixture_copies_every_committed_sample_file() {
    let listed: BTreeSet<String> = sample_files().into_iter().map(str::to_string).collect();
    let on_disk: BTreeSet<String> = std::fs::read_dir(sample_dir())
        .expect("the sample folder exists")
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .filter(|n| n != "README.md")
        .collect();
    assert_eq!(
        listed, on_disk,
        "the sample folder and this file's list have drifted — a file the cook \
         never sees is a file the shipping side is missing"
    );
}
