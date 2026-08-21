//! **THE PLAYER-CORE GATE** (island wave I5).
//!
//! The wave's headline claim, and it is one sentence: *a session that opens the
//! in-game menu, rebinds a key, closes the menu and then walks, runs, sprints,
//! slides and dives produces the same trace in the shipped player and in a PIE
//! preview, byte for byte.*
//!
//! # What makes this different from every other trace in the tree
//!
//! Every other scripted replay in this repository presses **action names** —
//! `RuntimeInput::with_down(["sprint"])` — which is deliberate for a movement
//! gate and blind to the entire input layer. This one presses **keys**, through
//! the shipped binding table, the shipped `InputState`, the shipped settings
//! dialog and the shipped hold clock, in the same order `PlayerApp::frame` does.
//! So the things it certifies are the ones no name-driven trace can:
//!
//! * the owner's table really binds Shift to sprint and Ctrl to walk;
//! * the C key's click and long press really produce different verbs, from the
//!   *duration* of a real press measured on the sim's own fixed step;
//! * a rebinding really changes which key does what, in a running session;
//! * an open menu really costs the simulation **zero fixed steps**, which is
//!   what makes the whole thing byte-comparable at all — a menu that did not
//!   pause would put the player's reading time into the trace.
//!
//! # The two hosts
//!
//! The shipped side is a **cooked pack**; the PIE side is the payload the editor
//! really builds, through `sim_from_payload`. Both then run the *same* host
//! frame ([`Host::frame`]), because the input layer is the host's and a parity
//! claim about it has to exercise it on both sides.

use std::path::{Path, PathBuf};

use uuid::Uuid;

use inf_editor_core::samples;
use inf_input::{InputEvent, InputMap, InputState};
use inf_player::runtime_sim::RuntimeSim;
use inf_player::ui::PlayerUi;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;

// ── the fixture: the committed phase-29 course ──────────────────────────────

fn hero() -> Uuid {
    samples::phase29_hero()
}

fn sample_dir() -> PathBuf {
    samples::phase29_locomotion_dir()
}

/// Copy the committed sample into a scaffolded project and answer its `Content`.
///
/// The file list is `read_dir`'s rather than a second copy of `phase29_gate`'s
/// nineteen names: that file already pins the list against the folder, so a
/// second literal here would be a second thing to keep in step for no claim.
fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    inf_project::ProjectManifest::new("Phase 29 Locomotion", "blank-3d")
        .save(&proj)
        .expect("the manifest saves");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("mkdir Content");
    for entry in std::fs::read_dir(sample_dir()).expect("the sample directory exists") {
        let entry = entry.expect("a readable entry");
        if entry.file_type().is_ok_and(|t| t.is_file()) {
            std::fs::copy(entry.path(), content.join(entry.file_name())).expect("copy");
        }
    }
    content
}

/// **The shipping side**: a `RuntimeSim` off a cooked pack.
fn pack_sim(tmp: &Path) -> RuntimeSim {
    scaffold(tmp);
    let out = tmp.join("out");
    inf_packager::cook(
        &tmp.join("proj"),
        &out,
        &inf_packager::CookOptions::default(),
    )
    .expect("the course cooks");
    let source = inf_player::level::PackLevelSource::open(&out).expect("pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("pack world builds");
    inf_player::sim_from_built(built)
}

/// **The PIE side**: the payload the editor really builds.
fn pie_sim() -> RuntimeSim {
    let dir = sample_dir();
    let ids = samples::phase29_asset_guids();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Phase29Locomotion.inf_lvl"))
        .expect("the committed course loads");
    let read = |name: &str| std::fs::read(dir.join(name)).expect("the sample file is committed");
    let controller =
        samples::decode_actor(&read("Hero Controller.inf_act")).expect("the controller decodes");
    let skel = read("Hero.inf_skel");
    let mesh = read("Hero Body.inf_mesh");
    let idle = read("Hero Idle.inf_anim");
    let walk = read("Hero Walk.inf_anim");
    let run = read("Hero Run.inf_anim");
    let sm = read("Hero Locomotion.inf_sm");
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
    // comparison of two motionless characters.
    assert_eq!(payload.machines.len(), 1, "the machine must ride the wire");
    assert_eq!(payload.skeletons.len(), 1, "the rig must ride the wire");
    assert_eq!(
        payload.clips.len(),
        3,
        "all three cycles must ride the wire"
    );
    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
}

// ── the host frame, mirrored from `PlayerApp::frame` ─────────────────────────

/// One host: a sim, the input layer in front of it, and the UI session that
/// routes keys away from it.
///
/// **This is `PlayerApp::frame` minus the rendering**, in the same order, and it
/// has to be: a parity claim about the input path is worth nothing if the trace
/// bypasses it.
struct Host {
    sim: RuntimeSim,
    state: InputState,
    map: InputMap,
    ui: PlayerUi,
    down: Vec<String>,
    /// Last frame's vertical velocity, for the launch edge.
    prev_vy: f64,
    /// The settings directory, kept alive for the session.
    _dir: tempfile::TempDir,
}

impl Host {
    fn new(sim: RuntimeSim) -> Self {
        let dir = tempfile::tempdir().expect("a settings directory");
        // Each host gets its own EMPTY settings directory, so both start from the
        // shipped table — which is what makes a divergence a divergence rather
        // than one of them having found a file.
        let (mut ui, map) = PlayerUi::open(dir.path().to_path_buf());
        ui.menu.single_player = true;
        let mut sim = sim;
        sim.set_press_threshold_s(ui.settings.press_threshold_s());
        Self {
            state: InputState::new(map.clone()),
            sim,
            map,
            ui,
            down: Vec::new(),
            prev_vy: 0.0,
            _dir: dir,
        }
    }

    /// One frame, with `keys` held and `mouse_x` counts of horizontal mouse
    /// motion. Answers this step's trace record.
    fn frame(&mut self, keys: &[&str], mouse_x: f32) -> Vec<u8> {
        // The events a window would have delivered: the difference between the
        // held set and the last one.
        let want: Vec<String> = keys.iter().map(|k| (*k).to_string()).collect();
        let mut events = Vec::new();
        for k in &want {
            if !self.down.contains(k) {
                events.push((k.clone(), true));
            }
        }
        for k in &self.down {
            if !want.contains(k) {
                events.push((k.clone(), false));
            }
        }
        self.down = want;

        // 1. The dialog gets each key first, and what it takes never reaches the
        //    game — `PlayerApp::window_event`'s rule.
        let mut forwarded = Vec::new();
        for (code, pressed) in events {
            let mut map = self.map.clone();
            let verdict = self.ui.key(&code, pressed, &mut map);
            if verdict.bindings_changed {
                self.map = map;
                self.state.set_map(self.map.clone());
            }
            if !verdict.consumed {
                forwarded.push(InputEvent::Key { code, pressed });
            }
        }
        // The mouse never reaches the dialog outside a capture, and the dialog
        // does not read motion at all — so it goes straight through, exactly as
        // `PlayerApp::device_event` sends it.
        if mouse_x != 0.0 {
            forwarded.push(InputEvent::MouseMotion {
                delta: [mouse_x, 0.0],
            });
        }
        // 2. Fold the frame.
        self.state.apply_dt(&forwarded, DT);
        // 3. The menu action's edge, read from the RESOLVED state — so a player
        //    who rebound the menu opens it with whatever they bound.
        if self.state.just_pressed(inf_input::actions::MENU) {
            self.ui.toggle();
        }
        // 4. The pause is on the SIM.
        self.sim.set_sim_paused(self.ui.pauses_sim());
        // 5. Step.
        let input = inf_player::input::held_actions(&self.state, DT);
        self.sim.step_once(input);
        self.record()
    }

    /// The trace record: where the character is, what it is doing, and how many
    /// fixed steps have run.
    ///
    /// The **step count** is in the record on purpose: it is what makes the
    /// menu's pause visible to the comparison rather than something the arm has
    /// to ask about separately, and a host that quietly kept stepping through the
    /// menu would diverge on the first paused frame.
    fn record(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let w = self.sim.world();
        let e = w.entity_of(hero()).expect("the hero exists");
        let t = w
            .world()
            .get::<inf_ecs::components::Transform>(e)
            .expect("with a transform");
        let cm = w
            .world()
            .get::<inf_ecs::components::CharacterMovement>(e)
            .expect("with a movement component");
        for v in [
            t.translation.x,
            t.translation.y,
            t.translation.z,
            cm.runtime.velocity.x,
            cm.runtime.velocity.y,
            cm.runtime.velocity.z,
            cm.runtime.mapped_speed,
        ] {
            out.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        out.extend_from_slice(&self.sim.steps().to_le_bytes());
        out.push(cm.mode as u8);
        out.push(cm.gait as u8);
        out.push(cm.runtime.actual_gait as u8);
        out.push(cm.runtime.grounded as u8);
        out.push(self.sim.sim_paused() as u8);
        out
    }

    fn mode(&self) -> inf_ecs::components::MovementMode {
        let w = self.sim.world();
        let e = w.entity_of(hero()).unwrap();
        w.world()
            .get::<inf_ecs::components::CharacterMovement>(e)
            .unwrap()
            .mode
    }

    fn speed(&self) -> f64 {
        let w = self.sim.world();
        let e = w.entity_of(hero()).unwrap();
        let v = w
            .world()
            .get::<inf_ecs::components::CharacterMovement>(e)
            .unwrap()
            .runtime
            .velocity;
        (v.x * v.x + v.z * v.z).sqrt()
    }

    fn z(&self) -> f64 {
        let w = self.sim.world();
        let e = w.entity_of(hero()).unwrap();
        w.world()
            .get::<inf_ecs::components::Transform>(e)
            .unwrap()
            .translation
            .z
    }

    /// **Whether a DIVE launched on this frame.**
    ///
    /// The dive branch overwrites the whole velocity with
    /// `(dive_speed_mps forward, dive_up_speed_mps up)` — so a launch is an
    /// *exact* planar speed with an upward component, and nothing else in the
    /// movement model produces one. That exactness is the point: the sprint the
    /// character was carrying is 6.5 m/s and the dive's is 5.5, so the frame the
    /// verb fires is unmistakable.
    ///
    /// **Why the launch and not the mode.** A dive off flat ground is 4 cm of
    /// clearance in a fixed step, and the ground snap is entitled to take it
    /// back — measured here at exactly that, and recorded the same way by
    /// `inf-physics`' own `a_dive_needs_a_sprint_and_the_refusal_is_a_value`.
    /// How long `MovementMode::Dive` *lasts* is a fact about a dive with
    /// somewhere to go, and `phase29_gate`'s station is what holds it: that
    /// course dives at z = 36 with the whole floor ahead of it, and this one is
    /// pinned inside twenty-one metres of clear ground so it can turn round.
    fn dive_launched(&mut self) -> bool {
        let vy = {
            let w = self.sim.world();
            let e = w.entity_of(hero()).unwrap();
            w.world()
                .get::<inf_ecs::components::CharacterMovement>(e)
                .unwrap()
                .runtime
                .velocity
                .y
        };
        // The RISING edge of "going up fast", so a launch counts once rather
        // than once per frame of the arc it starts. Nothing a character
        // sprinting on flat ground does gives it a metre a second upwards; a
        // dive gives it `dive_up_speed_mps`, which is two and a half.
        let launched = vy > DIVE_UP_FLOOR_MPS && self.prev_vy <= DIVE_UP_FLOOR_MPS;
        self.prev_vy = vy;
        launched
    }
}

/// The upward speed that marks a launch, m/s. Well under
/// `CharacterMovement::dive_up_speed_mps` (2.5) and well over anything running
/// on a floor produces, which is zero.
const DIVE_UP_FLOOR_MPS: f64 = 1.0;

// ── the script ───────────────────────────────────────────────────────────────

// ── the course's shape, and why the script has to know it ────────────────────
//
// The committed phase-29 floor runs `z ∈ [-10, 100]` and its low roof sits at
// `z ∈ [11, 17]`, so the clear open ground is **twenty-one metres** with the
// start at `z = -4`. A sprint station is nineteen metres on its own.
//
// A time-scripted run walks straight into the roof and crouch-shuffles through
// the rest of the course at 1.5 m/s — measured, on this file's first draft, with
// the dive station spending all ninety of its steps stuck under a ceiling. That
// is the phase-29 driver's own lesson met from the other side: **a station is at
// a place, not at a time**. So the script turns the character round when it
// reaches the end of the clear ground, and every station happens somewhere it
// fits.

/// The band the script keeps the character inside, metres of `z`.
const BAND: (f64, f64) = (-8.0, 9.0);
/// How many frames a half-turn takes.
const TURN_FRAMES: u32 = 30;
/// Mouse counts per frame during a turn: 180 degrees over [`TURN_FRAMES`], at
/// the shipped 0.15 deg/count sensitivity.
const TURN_COUNTS: f32 = 180.0 / TURN_FRAMES as f32 / 0.15;

/// One station of the course.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Station {
    /// Settle on the ground.
    Settle,
    /// Run at the default gait — **nothing held**.
    Run,
    /// Ctrl walks.
    Walk,
    /// Shift sprints.
    Sprint,
    /// A C **click** while sprinting: a slide.
    Slide,
    /// Stand back up.
    Stand,
    /// Sprint again, for the dive.
    RunUp,
    /// A C **long press** while sprinting: a dive.
    Dive,
    /// Open the menu and rebind `sprint` from Shift to B.
    Menu,
    /// Sprint again — with the NEW key.
    Rebound,
}

/// The stations and their step budgets, in order.
mod at {
    use super::Station;

    /// The course, in the order a player walks it.
    pub const STATIONS: [Station; 10] = [
        Station::Settle,
        Station::Run,
        Station::Walk,
        Station::Sprint,
        Station::Slide,
        Station::Stand,
        Station::RunUp,
        Station::Dive,
        Station::Menu,
        Station::Rebound,
    ];

    /// How long the menu window is, in frames.
    pub const MENU_FRAMES: u32 = 140;

    /// How many steps each station gets. A turn does **not** spend one, so a
    /// station's budget is time spent doing the thing rather than time spent on
    /// the course.
    pub const BUDGETS: [u32; 10] = [
        40,          // Settle
        120,         // Run
        120,         // Walk
        180,         // Sprint
        60,          // Slide
        40,          // Stand
        180,         // RunUp
        90,          // Dive
        MENU_FRAMES, // Menu
        200,         // Rebound
    ];

    /// The whole course, plus enough slack for the turns it takes on the way.
    pub const TOTAL: u32 = 1600;
}

/// **The script**: a stage machine whose stages advance on a step budget and
/// whose *turns* are a fact about where the character is.
///
/// Deterministic and shared: both hosts run the same one, see the same `z`, and
/// therefore turn on the same step — the phase-29 driver's discipline, and the
/// reason it makes PIE == shipping a stronger claim rather than a weaker one.
#[derive(Debug, Default, Clone)]
struct Script {
    /// Steps spent in the current station.
    held: u32,
    /// Frames left in a half-turn, if one is running.
    turning: u32,
    /// Whether the aim points at `+z`.
    facing_forward: bool,
    /// Which station.
    stage: usize,
}

impl Script {
    fn new() -> Self {
        Self {
            facing_forward: true,
            ..Default::default()
        }
    }

    /// One frame's input: the keys held, and this frame's mouse-x delta.
    fn step(&mut self, z: f64) -> (Vec<&'static str>, f32) {
        // A turn in progress owns the frame: no movement keys, just the mouse.
        if self.turning > 0 {
            self.turning -= 1;
            if self.turning == 0 {
                self.facing_forward = !self.facing_forward;
            }
            return (Vec::new(), TURN_COUNTS);
        }
        // At the end of the clear ground, turn round — the station picks up
        // where it left off, facing the other way.
        let at_edge = if self.facing_forward {
            z >= BAND.1
        } else {
            z <= BAND.0
        };
        if at_edge {
            self.turning = TURN_FRAMES;
            return (Vec::new(), TURN_COUNTS);
        }
        let k = self.held;
        self.held += 1;
        let budget = at::BUDGETS[self.stage.min(at::BUDGETS.len() - 1)];
        if self.held >= budget {
            self.stage += 1;
            self.held = 0;
        }
        match self.stage_of(k) {
            Station::Settle => (Vec::new(), 0.0),
            Station::Run => (vec!["KeyW"], 0.0),
            Station::Walk => (vec!["Control", "KeyW"], 0.0),
            Station::Sprint => (vec!["Shift", "KeyW"], 0.0),
            // **The click**: one frame down, the rest up. The click fires on the
            // release, so a held C would be the other verb entirely.
            Station::Slide => (
                if k == 0 {
                    vec!["Shift", "KeyW", "KeyC"]
                } else {
                    vec!["Shift", "KeyW"]
                },
                0.0,
            ),
            // Out of whatever the slide left behind, with another click.
            Station::Stand => (if k == 0 { vec!["KeyC"] } else { Vec::new() }, 0.0),
            Station::RunUp => (vec!["Shift", "KeyW"], 0.0),
            // **The long press**: held past the threshold, so it is a dive.
            Station::Dive => (vec!["Shift", "KeyW", "KeyC"], 0.0),
            Station::Menu => (menu_keystroke(k), 0.0),
            // The rebound sprint: **B**, which was Shift five seconds ago.
            Station::Rebound => (vec!["KeyB", "KeyW"], 0.0),
        }
    }

    fn stage_of(&self, _k: u32) -> Station {
        at::STATIONS[self.stage.min(at::STATIONS.len() - 1)]
    }

    /// Which station is running right now.
    fn station(&self) -> Station {
        at::STATIONS[self.stage.min(at::STATIONS.len() - 1)]
    }
}

/// The menu window, keystroke by keystroke.
///
/// Every press is one frame and every gap is one frame, because a key held
/// across two frames has no second rising edge. **Escape goes last**, two frames
/// from the end, so the dialog stands open for most of the window and the pause
/// claim is about a hundred frames rather than about twenty.
fn menu_keystroke(k: u32) -> Vec<&'static str> {
    // The row index is derived, so a table that grows a row above Sprint does
    // not silently rebind something else.
    let sprint_row = inf_ui::bindings::rows()
        .iter()
        .position(|r| r.id == "sprint")
        .expect("the table has a Sprint row");
    let plan: Vec<&'static str> = ["Tab", "ArrowRight", "ArrowRight", "ArrowRight"]
        .into_iter()
        .chain(std::iter::repeat_n("ArrowDown", sprint_row + 1))
        .chain(["Enter", "KeyB"])
        .collect();
    let last = at::MENU_FRAMES;
    let slot = (k / 2) as usize;
    if k == last - 2 {
        vec!["Escape"]
    } else if k % 2 == 0 && slot < plan.len() {
        vec![plan[slot]]
    } else {
        Vec::new()
    }
}

/// Run the whole course on one host, answering the trace and the per-station
/// peak speed.
fn run(mut host: Host) -> Run {
    let mut script = Script::new();
    let mut trace = Vec::with_capacity(at::TOTAL as usize);
    // **The speed a station SETTLES at**, not its peak.
    //
    // A peak is the wrong measure for a *slower* tier: the walk station starts
    // at run speed and decelerates into it, so its peak is the run's. Measured,
    // on this file's first draft: `Ctrl must walk: 3.47 against 1.65`, where
    // 3.47 is the run speed the character was carrying into the station. The
    // last sample taken while the movement keys were down is the number the
    // claim is about.
    let mut settled = [0.0f64; at::STATIONS.len()];
    // How many DIVE LAUNCHES each station produced — see `Host::dive_launched`.
    let mut launches = [0usize; at::STATIONS.len()];
    let mut peak = [0.0f64; at::STATIONS.len()];
    let mut modes: Vec<(Station, inf_ecs::components::MovementMode)> = Vec::new();
    let mut open_frames = 0u32;
    let mut steps_while_open = 0u64;
    let mut frozen: Option<Vec<u8>> = None;
    for _ in 0..at::TOTAL {
        let z = host.z();
        let station = script.station();
        let (keys, mouse) = script.step(z);
        let was_open = host.ui.menu.open;
        let before = host.sim.steps();
        let rec = host.frame(&keys, mouse);
        let stage = at::STATIONS.iter().position(|s| *s == station).unwrap_or(0);
        if !keys.is_empty() && mouse == 0.0 {
            settled[stage] = host.speed();
            peak[stage] = peak[stage].max(host.speed());
        }
        modes.push((station, host.mode()));
        if host.dive_launched() {
            launches[stage] += 1;
        }
        if was_open && host.ui.menu.open {
            open_frames += 1;
            steps_while_open += host.sim.steps() - before;
            match &frozen {
                None => frozen = Some(rec.clone()),
                Some(first) => assert_eq!(&rec, first, "the world moved while the menu was open"),
            }
        }
        trace.push(rec);
    }
    Run {
        trace,
        settled,
        peak,
        launches,
        modes,
        open_frames,
        steps_while_open,
        host,
    }
}

/// One host's whole run.
struct Run {
    trace: Vec<Vec<u8>>,
    /// The speed the character settled at in each station.
    settled: [f64; at::STATIONS.len()],
    /// The fastest the character went in each station while it was driving.
    peak: [f64; at::STATIONS.len()],
    /// How many dive launches each station produced.
    launches: [usize; at::STATIONS.len()],
    /// `(station, mode)` per step.
    modes: Vec<(Station, inf_ecs::components::MovementMode)>,
    /// How many frames the menu stood open.
    open_frames: u32,
    /// How many fixed steps ran while it did. Must be zero.
    steps_while_open: u64,
    host: Host,
}

impl Run {
    /// How many steps of `station` were spent in `mode`.
    fn saw(&self, station: Station, mode: inf_ecs::components::MovementMode) -> usize {
        self.modes
            .iter()
            .filter(|(s, m)| *s == station && *m == mode)
            .count()
    }

    fn settled_of(&self, station: Station) -> f64 {
        self.settled[at::STATIONS.iter().position(|s| *s == station).unwrap()]
    }

    /// The fastest `station` got.
    fn peak_of(&self, station: Station) -> f64 {
        self.peak[at::STATIONS.iter().position(|s| *s == station).unwrap()]
    }

    /// How many dive launches `station` produced.
    fn launches_of(&self, station: Station) -> usize {
        self.launches[at::STATIONS.iter().position(|s| *s == station).unwrap()]
    }
}

// ── the arms ─────────────────────────────────────────────────────────────────

/// **THE HEADLINE.** PIE == shipping over the whole course, byte for byte —
/// through the menu, the rebinding and the whole gait ladder.
#[test]
fn pie_equals_shipping_through_the_menu_the_rebinding_and_the_whole_gait_ladder() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let ship = run(Host::new(pack_sim(tmp.path())));
    let pie = run(Host::new(pie_sim()));
    assert_eq!(
        ship.trace.len(),
        pie.trace.len(),
        "the two hosts produced different numbers of steps"
    );
    assert_eq!(ship.trace.len() as u32, at::TOTAL);
    // Not vacuous, on three counts: the character moved, the record width never
    // changed, and the course really visited every station.
    assert_ne!(
        ship.trace[0],
        ship.trace[ship.trace.len() - 1],
        "the course ended in exactly the state it started in"
    );
    let width = ship.trace[0].len();
    assert!(
        ship.trace.iter().all(|r| r.len() == width),
        "the record width changed mid-run"
    );
    for station in at::STATIONS {
        assert!(
            ship.modes.iter().any(|(s, _)| *s == station),
            "the course never reached {station:?}"
        );
    }
    for (i, (s, p)) in ship.trace.iter().zip(pie.trace.iter()).enumerate() {
        assert_eq!(
            s, p,
            "step {i}: the cooked pack and the PIE payload disagree"
        );
    }
}

/// **RUN BY DEFAULT, CTRL WALKS, SHIFT SPRINTS — THROUGH THE KEYS.**
///
/// The owner's ruling, measured end to end: the keys a player presses, through
/// the shipped table, the shipped resolver and the shipped intent, read as three
/// different speeds. Every other gait arm in the tree presses action *names* and
/// is blind to the table.
///
/// The speed each station **settles** at, sampled on the last frame of it where
/// the movement keys were down and no turn was running — a peak would read the
/// *previous* tier for the walk, because the character enters that station at
/// run speed and decelerates into it (measured: 3.47 against 1.65).
#[test]
fn the_shipped_keys_produce_three_gaits_and_the_default_is_run() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let r = run(Host::new(pack_sim(tmp.path())));
    let tune = inf_ecs::components::CharacterMovement::default();
    // **A station that ACCELERATES into its tier is measured by the fastest it
    // got; one that DECELERATES into a slower tier is measured by where it
    // settled.** Both are stated rather than picked: the walk station enters at
    // run speed, so its peak is the run's (measured at 3.47 against 1.65), and
    // the run and sprint stations enter from a stop or a turn, so their last
    // frame can be mid-acceleration.
    let (walk, run_, sprint) = (
        r.settled_of(Station::Walk),
        r.peak_of(Station::Run),
        r.peak_of(Station::Sprint),
    );
    println!(
        "nothing held: {run_:.3} m/s (run is {:.3}) | Ctrl: {walk:.3} (walk is {:.3}) | Shift: {sprint:.3} (sprint is {:.3})",
        tune.run_speed_mps, tune.walk_speed_mps, tune.sprint_speed_mps
    );
    assert!(
        (run_ - tune.run_speed_mps).abs() < 0.25,
        "with NOTHING held the character must run: {run_} against {}",
        tune.run_speed_mps
    );
    assert!(
        (walk - tune.walk_speed_mps).abs() < 0.25,
        "Ctrl must walk: {walk} against {}",
        tune.walk_speed_mps
    );
    assert!(
        (sprint - tune.sprint_speed_mps).abs() < 0.25,
        "Shift must sprint: {sprint} against {}",
        tune.sprint_speed_mps
    );
    assert!(walk < run_ && run_ < sprint, "{walk} / {run_} / {sprint}");
}

/// **THE C KEY'S TWO VERBS, FROM A REAL PRESS.**
///
/// A one-frame press is a slide; a press held past the threshold is a dive. The
/// duration is measured by the sim's own hold clock off the key's held set, so
/// this is the whole discrimination end to end — and the two stations press the
/// **same key** with the **same modifier**, differing only in how long it is
/// down.
#[test]
fn a_click_slides_and_a_long_press_dives() {
    use inf_ecs::components::MovementMode as M;
    let tmp = tempfile::tempdir().expect("tempdir");
    let r = run(Host::new(pack_sim(tmp.path())));
    let slid = r.saw(Station::Slide, M::Slide);
    let launched = r.launches_of(Station::Dive);
    println!(
        "the click station held Slide for {slid} steps and launched {} dives; the long-press station launched {launched} dives and held Slide for {} steps",
        r.launches_of(Station::Slide),
        r.saw(Station::Dive, M::Slide)
    );
    assert!(
        slid >= 5,
        "a C CLICK at sprint speed must slide — it held Slide for {slid} steps"
    );
    assert_eq!(
        launched, 1,
        "a C LONG PRESS at sprint speed must launch exactly one dive — it launched {launched}"
    );
    // …and the two stations really are the same key at two durations: **neither
    // verb appears at the other's station**. Without this pair the arm would
    // pass for a model that produced both verbs for both presses.
    assert_eq!(r.launches_of(Station::Slide), 0, "the click dived");
    assert_eq!(r.saw(Station::Dive, M::Slide), 0, "the long press slid");
    // **Exactly one** launch is itself a claim: the long press is held for
    // ninety frames and must fire on the frame the hold crosses the threshold
    // and on no other. A rule without the `prev < threshold` half fires every
    // frame for the rest of the press.
    assert!(
        at::BUDGETS[at::STATIONS
            .iter()
            .position(|s| *s == Station::Dive)
            .unwrap()]
            > 30,
        "the long-press station is too short for the once-only claim to mean anything"
    );
}

/// **AN OPEN MENU COSTS THE SIMULATION ZERO FIXED STEPS**, and the rebinding
/// made inside it changes which key sprints.
#[test]
fn the_menu_pauses_the_sim_and_the_rebinding_takes_effect() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut r = run(Host::new(pack_sim(tmp.path())));
    println!(
        "the menu stood open for {} frames and cost {} fixed steps",
        r.open_frames, r.steps_while_open
    );
    assert_eq!(
        r.steps_while_open, 0,
        "an open menu advanced the simulation — a player's reading time is in the trace"
    );
    assert!(
        r.open_frames > 80,
        "the menu was only open for {} frames, which is too few for the claim to mean anything",
        r.open_frames
    );
    assert!(!r.host.ui.menu.open, "the script never closed the menu");

    // The rebinding landed: Sprint is B, and the override was recorded as the
    // DIFFERENCE from the shipped table rather than as the whole table.
    let table = inf_ui::bindings::rows();
    let sprint = table.iter().find(|row| row.id == "sprint").unwrap();
    assert_eq!(
        inf_ui::bindings::token_in(&r.host.map, sprint),
        "KeyB",
        "the dialog did not rebind Sprint"
    );
    assert_eq!(
        r.host
            .ui
            .settings
            .bindings
            .get("sprint")
            .map(String::as_str),
        Some("KeyB"),
        "the override was not recorded for the settings file"
    );
    assert_eq!(
        r.host.ui.settings.bindings.len(),
        1,
        "the whole table was stored instead of the difference: {:?}",
        r.host.ui.settings.bindings
    );

    // …and the NEW key sprints.
    let tune = inf_ecs::components::CharacterMovement::default();
    let with_b = r.peak_of(Station::Rebound);
    println!(
        "after the rebinding, B + W reaches {with_b:.3} m/s (sprint is {:.3})",
        tune.sprint_speed_mps
    );
    assert!(
        (with_b - tune.sprint_speed_mps).abs() < 0.25,
        "the rebound key does not sprint: {with_b}"
    );
    // The control: the OLD key no longer does. Held for three seconds, the
    // character falls back to the run tier — which is what says the assertion
    // above is about the rebinding and not about `KeyW` alone.
    let mut speed = 0.0f64;
    for _ in 0..180 {
        r.host.frame(&["Shift", "KeyW"], 0.0);
        speed = r.host.speed();
    }
    println!(
        "…and Shift + W now settles at {speed:.3} m/s (run is {:.3})",
        tune.run_speed_mps
    );
    assert!(
        (speed - tune.run_speed_mps).abs() < 0.25,
        "Shift still sprints after being rebound away: {speed}"
    );
}

/// **A KEY THE DIALOG TOOK NEVER REACHED THE GAME.**
///
/// The menu window's keystrokes include `ArrowDown` (Move Back's second key in
/// the shipped table) and `KeyB` (which the capture ate). If any of them had
/// leaked, the character would have been walking while the menu was open — and
/// the pause arm would not have caught it, because a paused sim does not
/// integrate. This one reads the resolved INPUT rather than the world.
#[test]
fn the_dialogs_keys_never_reach_the_resolved_input() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let mut host = Host::new(pack_sim(tmp.path()));
    let mut script = Script::new();
    let mut leaked: Vec<(u32, String)> = Vec::new();
    let mut pressed_in_window = 0usize;
    for i in 0..at::TOTAL {
        let z = host.z();
        let (keys, mouse) = script.step(z);
        let open_before = host.ui.menu.open;
        host.frame(&keys, mouse);
        if open_before && host.ui.menu.open {
            if !keys.is_empty() {
                pressed_in_window += 1;
            }
            for a in host.state.map().action_names() {
                if host.state.pressed(a) {
                    leaked.push((i, a.to_string()));
                }
            }
            for a in host.state.map().axis_names() {
                if host.state.axis(a).abs() > f32::EPSILON {
                    leaked.push((i, a.to_string()));
                }
            }
        }
    }
    assert!(
        leaked.is_empty(),
        "these controls were live while the menu was open: {leaked:?}"
    );
    // Not vacuous: the window really did contain presses.
    assert!(
        pressed_in_window >= 8,
        "only {pressed_in_window} keys were pressed while the menu was open"
    );
}
