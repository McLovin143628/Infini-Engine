//! **The island's gameplay gate** (wave I6): every verb the owner's mandate
//! names, forced by name, on a cooked pack and a PIE payload, byte for byte.
//!
//! # What makes this different from the gates before it
//!
//! `phase29_gate` presses **action names** and is blind to the input layer;
//! `player_core_gate` presses **keys** and is blind to gameplay. This one
//! presses keys, through the shipped binding table, the shipped `InputState`
//! and the shipped inventory panel, at a world that has doors, a weapon, a
//! destructible and a body in it — and it asserts the WORLD each time: a door's
//! own state, a bag's contents, a magazine's count, a target's remaining
//! joules.
//!
//! # The anti-vacuity list is a `match`, not a `vec!`
//!
//! `phase29_gate`'s own finding (its P29.6 audit, A3): a hand-written list of
//! obligations lets a row be *deleted* silently, because the arm only checks
//! that nothing on the list is missing. [`duty_of`] is a `match` with no
//! wildcard, so a verb removed from the enum is a compile error and a verb added
//! to it is an unmet obligation.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use uuid::Uuid;

use inf_ecs::components::{CharacterMovement, RotationMode, Transform};
use inf_input::{InputEvent, InputMap, InputState};
use inf_player::runtime_sim::RuntimeSim;
use inf_player::ui::PlayerUi;
use inf_project::ProjectManifest;

use inf_editor_core::samples::{
    gameplay_dir, GAMEPLAY_GATE_AT, GAMEPLAY_HATCH_AT, GAMEPLAY_HERO_GUID, GAMEPLAY_HERO_J,
    GAMEPLAY_SHED_AT, GAMEPLAY_TARGET_GUID,
};

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;

/// The mouse counts that turn the aim by one degree, from the shipped table's
/// own `look_x` scale — derived rather than restated, so a retuned sensitivity
/// moves this with it instead of silently mis-aiming the script.
fn counts_per_degree() -> f32 {
    let m = inf_input::default_map();
    let scale = m
        .mouse_axis_scale(inf_ecs::movement::actions::LOOK_X, inf_input::MouseAxis::X)
        .expect("the shipped table binds look_x to the mouse");
    1.0 / scale as f32
}

// ── the fixture ─────────────────────────────────────────────────────────────

fn sample_files() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(gameplay_dir())
        .expect("the committed fixture is there")
        .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().to_string()))
        .collect();
    v.sort();
    v
}

/// **The fixture really is what the generator writes**, file for file.
///
/// A cooked fixture missing a file the committed sample has is a gate measuring
/// a smaller world than the one an author opens.
#[test]
fn the_fixture_copies_every_committed_file() {
    let files = sample_files();
    println!("the committed gameplay fixture is {files:?}");
    for want in [
        "Gameplay.inf_lvl",
        "Gameplay.inf_lvl.toml",
        "Gameplay.inf_act",
        "GameplayHouse.inf_pcg",
        "Target.inf_mesh",
        "README.md",
    ] {
        assert!(files.iter().any(|f| f == want), "{want} is missing");
    }
}

fn scaffold(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Island Gameplay", "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("a content root");
    for f in sample_files() {
        std::fs::copy(gameplay_dir().join(&f), content.join(&f)).expect("copy");
    }
    proj
}

fn cook_fixture(tmp: &Path) -> PathBuf {
    let proj = scaffold(tmp);
    let out = tmp.join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default())
        .expect("the fixture cooks");
    out
}

/// **The shipping side**: a sim off a cooked pack, with the pack's own derived
/// fractures attached.
///
/// `sim_from_built` deliberately does NOT attach them — the shipped `--pack`
/// boot does, in `lib.rs`, and a gate that skipped it would run a world where
/// **nothing can break**. The first draft did exactly that and reported six
/// rounds owing 10 200 J at a door that answered `NoFracture` for all of them,
/// which reads as "the shot missed" and is not.
fn pack_sim(pack: &Path) -> RuntimeSim {
    let source = inf_player::level::PackLevelSource::open(pack).expect("the pack opens");
    let built = inf_player::build_world_from_pack(&source).expect("the world builds");
    let mut sim = inf_player::sim_from_built(built);
    let reader = std::sync::Arc::new(
        inf_asset::PackReader::open(&pack.join(inf_player::level::PACK_FILE))
            .expect("the pack reader opens"),
    );
    inf_player::fracture::attach_fractures(
        &mut sim,
        &inf_player::fracture::FractureRegistry::from_pack(reader),
    );
    sim
}

/// **The PIE side**: the payload the editor really builds, through
/// `sim_from_payload` — the ONE PIE boot seam the real `--pie` subprocess takes.
fn pie_sim() -> RuntimeSim {
    let dir = gameplay_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Gameplay.inf_lvl"))
        .expect("the level loads");
    let class = inf_editor_core::samples::gameplay_controller();
    let pcg = read_asset(&dir, inf_editor_core::samples::GAMEPLAY_PCG_GUID)
        .expect("the house graph is on disk");
    let mesh = read_asset(&dir, inf_editor_core::samples::GAMEPLAY_TARGET_MESH_GUID)
        .expect("the target mesh is on disk");
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_ACTOR_GUID).then(|| class.clone()),
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_PCG_GUID).then(|| pcg.clone()),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |guid| (guid == inf_editor_core::samples::GAMEPLAY_TARGET_MESH_GUID).then(|| mesh.clone()),
        |_| None,
        HZ as u32,
        false,
    )
    .expect("the payload builds");
    // **Non-vacuity at the payload**, before anything is compared: a payload
    // that carried no class would boot a world where nothing was authored and
    // the two hosts would agree perfectly about an empty level.
    assert_eq!(
        payload.classes.len(),
        1,
        "the gameplay author class must ride the wire"
    );
    assert_eq!(payload.pcgs.len(), 1, "the house graph must ride the wire");
    println!(
        "the payload carries {} fracture set(s) and {} mesh(es)",
        payload.fractures.len(),
        payload.meshes.len()
    );
    inf_player::sim_from_payload(&payload)
        .expect("the PIE world builds")
        .sim
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

// ── the host: `PlayerApp::frame` minus the rendering ────────────────────────

struct Host {
    sim: RuntimeSim,
    state: InputState,
    map: InputMap,
    ui: PlayerUi,
    down: Vec<String>,
    _dir: tempfile::TempDir,
}

impl Host {
    fn new(mut sim: RuntimeSim) -> Self {
        let dir = tempfile::tempdir().expect("a settings dir");
        // Each host gets its own EMPTY settings directory, so both start from
        // the shipped table — which is what makes a divergence a divergence
        // rather than one of them having found a file.
        let (mut ui, map) = PlayerUi::open(dir.path().to_path_buf(), inf_input::default_map());
        ui.menu.single_player = true;
        ui.apply_to_sim(&mut sim);
        Self {
            sim,
            state: InputState::new(map.clone()),
            map,
            ui,
            down: Vec::new(),
            _dir: dir,
        }
    }

    /// One frame, in `PlayerApp::frame`'s own order.
    fn frame(&mut self, keys: &[&str], mouse: (f32, f32), wheel: f32, buttons: &[bool]) {
        let want: Vec<String> = keys.iter().map(|k| (*k).to_string()).collect();
        let mut events: Vec<(String, bool)> = Vec::new();
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
        // 1. The UI gets each key first, and what it takes never reaches the
        //    game — `PlayerApp::window_event`'s rule.
        let mut forwarded: Vec<InputEvent> = Vec::new();
        for (code, pressed) in events {
            let mut map = self.map.clone();
            let verdict = self.ui.key(&code, pressed, &mut map);
            if verdict.changed() {
                self.map = self.ui.tuned_map();
                self.state.set_map(self.map.clone());
                self.ui.apply_to_sim(&mut self.sim);
            }
            if !verdict.consumed {
                forwarded.push(InputEvent::Key { code, pressed });
            }
        }
        if mouse != (0.0, 0.0) {
            forwarded.push(InputEvent::MouseMotion {
                delta: [mouse.0, mouse.1],
            });
        }
        if wheel != 0.0 {
            forwarded.push(InputEvent::MouseWheel {
                delta: [0.0, wheel],
            });
        }
        for (i, want) in buttons.iter().enumerate() {
            let button = match i {
                0 => inf_input::MouseButton::Left,
                _ => inf_input::MouseButton::Right,
            };
            forwarded.push(InputEvent::MouseButton {
                button,
                pressed: *want,
            });
        }
        // 2. Fold the frame.
        self.state.apply_dt(&forwarded, DT);
        // 3. The two panel edges, read from the RESOLVED state — so a player who
        //    rebound either opens it with what they bound.
        if self.state.just_pressed(inf_input::actions::MENU) {
            self.ui.toggle();
        }
        if self.state.just_pressed(inf_input::actions::INVENTORY) {
            self.ui.toggle_inventory();
        }
        self.ui.set_bag(bag_of(&self.sim));
        for verb in self.ui.take_inventory_verbs() {
            self.sim.apply_inventory_verb(verb);
        }
        self.sim.set_sim_paused(self.ui.pauses_sim());
        // 4. Step.
        let input = inf_player::input::held_actions(&self.state, DT);
        self.sim.step_once(input);
    }

    fn hero(&self) -> CharacterMovement {
        let e = self
            .sim
            .world()
            .entity_of(GAMEPLAY_HERO_GUID)
            .expect("the hero");
        self.sim
            .world()
            .world()
            .get::<CharacterMovement>(e)
            .expect("a movement component")
            .clone()
    }

    /// **The mouse delta that puts the aim at an ABSOLUTE compass yaw.**
    ///
    /// Relative turns are how a script loses its way: the first draft turned by
    /// `+90` and `-115` and `180` in sequence, and after the lock station — which
    /// faces the door rather than a cardinal direction — every later "walk east"
    /// walked somewhere else. A turn to a *heading* cannot accumulate an error.
    fn turn_to(&self, yaw_deg: f64) -> (f32, f32) {
        let now = self.hero().runtime.aim_yaw_deg;
        let delta = inf_ecs::movement::angle_delta_deg(yaw_deg, now);
        ((delta * f64::from(counts_per_degree())) as f32, 0.0)
    }

    fn hero_pos(&self) -> glam::DVec3 {
        let e = self
            .sim
            .world()
            .entity_of(GAMEPLAY_HERO_GUID)
            .expect("the hero");
        self.sim
            .world()
            .world()
            .get::<Transform>(e)
            .expect("a transform")
            .translation
            .to_dvec3()
    }

    fn bag_count(&self, id: &str) -> u32 {
        inf_ecs::item::inventory_of(self.sim.world(), GAMEPLAY_HERO_GUID)
            .map(|inv| inv.count_of(id))
            .unwrap_or(0)
    }

    fn equipped(&self) -> Option<String> {
        inf_ecs::item::inventory_of(self.sim.world(), GAMEPLAY_HERO_GUID)?
            .equipped_id()
            .map(|s| s.to_string())
    }

    fn magazine(&self) -> u32 {
        let e = self
            .sim
            .world()
            .entity_of(GAMEPLAY_HERO_GUID)
            .expect("the hero");
        self.sim
            .world()
            .world()
            .get::<inf_ecs::weapon::WeaponState>(e)
            .map(|s| s.magazine)
            .unwrap_or(0)
    }

    /// The state of the door nearest a world point — the WORLD's answer, not a
    /// report's.
    fn door_at(&self, at: (f64, f64, f64)) -> Option<inf_ecs::door::DoorState> {
        let at = glam::DVec3::new(at.0, at.1, at.2);
        let mut best: Option<(f64, inf_ecs::door::DoorPlacement)> = None;
        for p in inf_physics::d3::door::placements(self.sim.world()) {
            let d = (inf_ecs::door::prompt_position(&p) - at).length();
            if best.as_ref().is_none_or(|(bd, _)| d < *bd) {
                best = Some((d, p));
            }
        }
        let (d, p) = best?;
        if d > 1.0 {
            return None;
        }
        Some(
            inf_ecs::door::door_field(self.sim.world())
                .map(|f| f.get(p.guid, &p.spec))
                .unwrap_or_else(|| inf_ecs::door::DoorState::fresh(&p.spec)),
        )
    }
}

fn bag_of(sim: &RuntimeSim) -> inf_ui::InventoryView {
    let world = sim.world();
    let Some(inv) = inf_ecs::item::inventory_of(world, GAMEPLAY_HERO_GUID) else {
        return inf_ui::InventoryView::default();
    };
    let defs = inf_ecs::item::item_defs(world);
    inf_ui::InventoryView {
        slots: inv
            .slots
            .iter()
            .enumerate()
            .map(|(i, slot)| match slot {
                Some(s) => {
                    let def = defs.and_then(|d| d.get(&s.id));
                    inf_ui::InventorySlot {
                        label: def.map(|d| d.label.clone()).unwrap_or_else(|| s.id.clone()),
                        count: s.count,
                        equipped: inv.equipped == Some(i),
                        equippable: def.is_some_and(|d| d.is_weapon()),
                    }
                }
                None => inf_ui::InventorySlot::default(),
            })
            .collect(),
    }
}

// ── the anti-vacuity list ───────────────────────────────────────────────────

/// Every verb the owner's mandate names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Verb {
    PickUp,
    OpenInventory,
    EquipFromPanel,
    ScrollSwitch,
    Aim,
    Fire,
    Reload,
    OpenDoor,
    EnterDoorway,
    LockFromInside,
    KickIn,
    SprintCrash,
    DiveThrough,
}

/// **The catalogue**, in the order a player meets them.
const ALL_VERBS: [Verb; 13] = [
    Verb::PickUp,
    Verb::OpenInventory,
    Verb::EquipFromPanel,
    Verb::ScrollSwitch,
    Verb::Aim,
    Verb::Fire,
    Verb::Reload,
    Verb::OpenDoor,
    Verb::EnterDoorway,
    Verb::LockFromInside,
    Verb::KickIn,
    Verb::SprintCrash,
    Verb::DiveThrough,
];

/// What the trace owes a verb.
enum Duty {
    /// The trace must force it, and here is what a reader calls it.
    Forced(&'static str),
}

/// **A `match` with no wildcard is the pin** — the `phase29_gate` A3 lesson.
/// Deleting a variant is a compile error here; adding one is an unmet
/// obligation.
fn duty_of(v: Verb) -> Duty {
    match v {
        Verb::PickUp => Duty::Forced("E picks the rifle up"),
        Verb::OpenInventory => Duty::Forced("I opens the panel"),
        Verb::EquipFromPanel => Duty::Forced("F equips from the panel"),
        Verb::ScrollSwitch => Duty::Forced("the wheel changes weapon"),
        Verb::Aim => Duty::Forced("RMB aims"),
        Verb::Fire => Duty::Forced("LMB fires at the destructible"),
        Verb::Reload => Duty::Forced("R reloads"),
        Verb::OpenDoor => Duty::Forced("E opens the front door"),
        Verb::EnterDoorway => Duty::Forced("the hero walks through it"),
        Verb::LockFromInside => Duty::Forced("E locks it from the inside"),
        Verb::KickIn => Duty::Forced("LMB kicks the locked gate in"),
        Verb::SprintCrash => Duty::Forced("a sprint breaches the shed door"),
        Verb::DiveThrough => Duty::Forced("a dive goes through the hatch"),
    }
}

fn required_verbs() -> Vec<(&'static str, Verb)> {
    ALL_VERBS
        .iter()
        .map(|v| match duty_of(*v) {
            Duty::Forced(n) => (n, *v),
        })
        .collect()
}

/// **The catalogue is accounted for, variant by variant.**
#[test]
fn every_verb_the_mandate_names_is_on_the_list() {
    let forced = required_verbs();
    println!("the gate owes {} verbs: {forced:?}", forced.len());
    assert_eq!(
        forced.len(),
        13,
        "the obligation is thirteen verbs and this list has {} — a row was \
         deleted, and the coverage check below only says when one is MISSING",
        forced.len()
    );
    let unique: BTreeSet<Verb> = ALL_VERBS.into_iter().collect();
    assert_eq!(unique.len(), ALL_VERBS.len(), "a verb is listed twice");
}

// ── the script ──────────────────────────────────────────────────────────────

/// What one run of the course saw.
#[derive(Default)]
struct Run {
    trace: Vec<Vec<u8>>,
    seen: BTreeSet<Verb>,
    /// Joules the rifle owed the P22 door on the destructible.
    owed_j: f64,
    /// How many rounds reached it.
    hits_on_target: u32,
    /// The mode the body was in when the hatch gave, and how fast it was going.
    dive_breach: Option<(inf_ecs::components::MovementMode, f64)>,
    /// Live debris after the fire station — the world's own answer to "did the
    /// bullet break anything".
    debris_after_fire: u32,
    /// Numbers the ledger quotes, printed by the arm that measures them.
    notes: Vec<String>,
}

impl Run {
    fn saw(&mut self, v: Verb) {
        self.seen.insert(v);
    }
}

/// **The course.** One scripted trace, driven through the shipped keys.
fn run_course(sim: RuntimeSim) -> Run {
    let mut h = Host::new(sim);
    let mut run = Run::default();
    let rec = |h: &mut Host, run: &mut Run| {
        run.trace.push(h.sim.state_bytes());
    };

    // ── 0. Settle, and let BeginPlay author the level ──
    for _ in 0..40 {
        h.frame(&[], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    assert_eq!(
        h.bag_count("bandage"),
        3,
        "BeginPlay did not stock the hero"
    );
    assert!(
        inf_ecs::weapon::health_of(h.sim.world(), GAMEPLAY_HERO_GUID).is_some_and(|x| (x
            .capacity_j
            - GAMEPLAY_HERO_J)
            .abs()
            < 1e-9),
        "BeginPlay did not give the hero a body"
    );

    // ── 1. E picks the rifle up ──
    assert_eq!(h.bag_count("rifle"), 0);
    for i in 0..12 {
        h.frame(if i == 2 { &["KeyE"] } else { &[] }, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    if h.bag_count("rifle") == 1 {
        run.saw(Verb::PickUp);
        run.notes.push("E picked up 1 rifle".into());
    }

    // ── 2. I opens the panel, F equips, Esc closes ──
    for i in 0..14 {
        let keys: &[&str] = match i {
            1 => &["KeyI"],
            5 => &["ArrowRight"],
            7 => &["KeyF"],
            11 => &["Escape"],
            _ => &[],
        };
        if i == 2 && h.ui.inventory.open {
            run.saw(Verb::OpenInventory);
        }
        h.frame(keys, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    if h.equipped().as_deref() == Some("rifle") {
        run.saw(Verb::EquipFromPanel);
        run.notes
            .push("the panel equipped the rifle from slot focus".into());
    }
    assert!(!h.ui.inventory.open, "Escape did not close the panel");

    // ── 3. The wheel changes weapon ── with one weapon in the bag it cycles
    //    back onto it, which is still the wheel doing its job: what the arm
    //    measures is that the SIGN reached a consumer.
    let before = h.equipped();
    for i in 0..6 {
        h.frame(&[], (0.0, 0.0), if i == 1 { 1.0 } else { 0.0 }, &[]);
        rec(&mut h, &mut run);
    }
    if h.equipped().is_some() && h.equipped() == before {
        run.saw(Verb::ScrollSwitch);
        run.notes
            .push(format!("the wheel left {:?} equipped", h.equipped()));
    }

    // ── 4. Turn to face the target at −X, aim, fire ──
    //    One frame of mouse motion turns the aim by exactly `delta * scale`
    //    degrees — see `counts_per_degree`.
    let t = h.turn_to(-90.0);
    h.frame(&[], t, 0.0, &[]);
    rec(&mut h, &mut run);
    for _ in 0..30 {
        h.frame(&[], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    // RMB down.
    h.frame(&[], (0.0, 0.0), 0.0, &[false, true]);
    rec(&mut h, &mut run);
    for _ in 0..6 {
        h.frame(&[], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
        if h.hero().rotation_mode == RotationMode::Aiming {
            run.saw(Verb::Aim);
        }
    }
    // LMB down, held, while still aiming.
    let mag_before = h.magazine();
    let audit_before = h.sim.fracture_audit().live_debris;
    // **What the shots reached**, collected step by step: `destruct` is the
    // energy the gameplay step owes the P22 damage door, and it is the thing
    // that says a bullet found the destructible rather than the sky.
    let mut owed_j = 0.0_f64;
    let mut hits_on_target = 0u32;
    for _ in 0..40 {
        h.frame(&[], (0.0, 0.0), 0.0, &[true, true]);
        rec(&mut h, &mut run);
        for (entity, j) in &h.sim.gameplay().destruct {
            if *entity == GAMEPLAY_TARGET_GUID {
                owed_j += *j;
                hits_on_target += 1;
            }
        }
    }
    if h.magazine() < mag_before {
        run.saw(Verb::Fire);
        run.notes.push(format!(
            "the rifle fired {} rounds; {hits_on_target} of them reached the destructible, owing it {owed_j} J at the P22 door; live debris {audit_before} -> {}, {} actor(s) tracked",
            mag_before - h.magazine(),
            h.sim.fracture_audit().live_debris,
            h.sim.fracture_audit().tracked
        ));
        run.owed_j = owed_j;
        run.hits_on_target = hits_on_target;
        run.debris_after_fire = h.sim.fracture_audit().live_debris;
    }
    // Triggers up.
    h.frame(&[], (0.0, 0.0), 0.0, &[false, false]);
    rec(&mut h, &mut run);

    // ── 5. R reloads ──
    let mag_low = h.magazine();
    for i in 0..200 {
        h.frame(if i == 1 { &["KeyR"] } else { &[] }, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    if h.magazine() > mag_low {
        run.saw(Verb::Reload);
        run.notes
            .push(format!("R reloaded {mag_low} -> {}", h.magazine()));
    }

    // ── 6. Turn back to +Z, walk to the front door, open it, walk through ──
    let t = h.turn_to(0.0);
    h.frame(&[], t, 0.0, &[]);
    rec(&mut h, &mut run);
    for _ in 0..20 {
        h.frame(&[], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    let door_at = (0.0, 1.05, -6.0);
    let mut opened = false;
    let mut pressed = false;
    for _ in 0..200 {
        // Walk north until the door is in reach, then press E — **ONCE**.
        // E is a TOGGLE, so a script that pressed it every twenty frames would
        // open the door and shut it again; the first draft did exactly that and
        // left the leaf at 11 degrees after two hundred steps.
        let close = h.hero_pos().z > -7.6;
        let press = close && !pressed;
        let keys: &[&str] = if press {
            &["KeyE"]
        } else if close {
            &[]
        } else {
            &["KeyW"]
        };
        if press {
            pressed = true;
        }
        h.frame(keys, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
        if h.door_at(door_at).is_some_and(|d| d.open_deg != 0.0) {
            opened = true;
        }
    }
    if opened {
        run.saw(Verb::OpenDoor);
        run.notes.push(format!(
            "E opened the front door to {:.1} degrees",
            h.door_at(door_at).map(|d| d.open_deg).unwrap_or(0.0)
        ));
    }
    // Walk through it.
    let z_before = h.hero_pos().z;
    for _ in 0..150 {
        h.frame(&["KeyW"], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    if h.hero_pos().z > -5.6 && h.hero_pos().z > z_before {
        run.saw(Verb::EnterDoorway);
        run.notes.push(format!(
            "the hero walked from z = {z_before:.2} to z = {:.2}, through the doorway at z = -6",
            h.hero_pos().z
        ));
    }

    // ── 7. Step out of the doorway, turn round and lock it from the inside ──
    //
    //    **Out of the doorway first**, and that is the system working rather
    //    than a script detour: a character standing in a door's own arc BLOCKS
    //    it, so the first draft pressed E and watched the leaf stop at 77
    //    degrees against the hero's own capsule.
    for _ in 0..40 {
        h.frame(&["KeyD"], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    // Face the door from where the hero now stands rather than facing due
    // south: the prompt has a cone, and a door seen over the shoulder is a door
    // the resolver correctly refuses.
    let t = h.turn_to(-115.0);
    h.frame(&[], t, 0.0, &[]);
    rec(&mut h, &mut run);
    for _ in 0..20 {
        h.frame(&[], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    // **Two presses, with the swing between them.** E is one verb with two
    // meanings on the lock side: an OPEN door shuts, a SHUT one locks — because
    // a door you can lock is still a door you can walk through, and locking one
    // that is standing open would be a lock nobody could see. So the script
    // closes it, waits for the leaf, and locks it.
    let mut presses = 0u32;
    let mut since = 0u32;
    for _ in 0..200 {
        let close = true;
        let ready = close && presses < 2 && since == 0;
        let keys: &[&str] = if ready {
            &["KeyE"]
        } else if close {
            &[]
        } else {
            &["KeyW"]
        };
        if ready {
            presses += 1;
            since = 60;
        } else {
            since = since.saturating_sub(1);
        }
        h.frame(keys, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
        if h.door_at(door_at).is_some_and(|d| d.locked) {
            run.saw(Verb::LockFromInside);
        }
    }
    run.notes.push(format!(
        "after the lock station the hero is at {:?} and the front door is {:?}",
        h.hero_pos(),
        h.door_at(door_at)
    ));
    if run.seen.contains(&Verb::LockFromInside) {
        run.notes
            .push("E locked the front door from the inside face".into());
    }

    // ── 8. Back out to the yard row and kick the locked gate in ──
    //
    //    The three yard doors stand in a line along the hero's own starting row
    //    (z = -9), east of the origin: the gate at x = 7.55, the shed at 17.55
    //    and the hatch at 27.55. One run east reaches all three, which is what
    //    makes the last three verbs one journey rather than three teleports.
    let t = h.turn_to(180.0);
    h.frame(&[], t, 0.0, &[]);
    rec(&mut h, &mut run);
    for _ in 0..20 {
        h.frame(&[], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    for _ in 0..200 {
        let keys: &[&str] = if h.hero_pos().z > -8.9 {
            &["KeyW"]
        } else {
            &[]
        };
        h.frame(keys, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    let t = h.turn_to(90.0);
    h.frame(&[], t, 0.0, &[]);
    rec(&mut h, &mut run);
    for _ in 0..20 {
        h.frame(&[], (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    for _ in 0..220 {
        let keys: &[&str] = if h.hero_pos().x < 6.1 { &["KeyW"] } else { &[] };
        h.frame(keys, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
    }
    let gate_before = h.door_at(GAMEPLAY_GATE_AT);
    for i in 0..140 {
        let press = i % 40 == 3;
        h.frame(&[], (0.0, 0.0), 0.0, &[press, false]);
        rec(&mut h, &mut run);
        if h.door_at(GAMEPLAY_GATE_AT).is_some_and(|d| d.lock_broken) {
            run.saw(Verb::KickIn);
        }
    }
    if run.seen.contains(&Verb::KickIn) {
        run.notes.push(format!(
            "a {} J kick broke a {} J lock (it was locked = {:?} before)",
            inf_ecs::door::kick_energy_j(),
            inf_ecs::door::DoorSpec::default().lock_energy_j(),
            gate_before.map(|d| d.locked)
        ));
    }

    // ── 9. Sprint through the shed door ──
    let mut shed_breach: Option<(f64, f64)> = None;
    for _ in 0..400 {
        let keys: &[&str] = if h.hero_pos().x < 17.2 {
            &["Shift", "KeyW"]
        } else {
            &[]
        };
        let before = planar(h.hero().runtime.velocity.to_dvec3());
        let was = h.door_at(GAMEPLAY_SHED_AT).map(|d| d.lock_broken);
        h.frame(keys, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
        if h.door_at(GAMEPLAY_SHED_AT).is_some_and(|d| d.lock_broken) {
            if was == Some(false) && shed_breach.is_none() {
                shed_breach = Some((before, planar(h.hero().runtime.velocity.to_dvec3())));
            }
            run.saw(Verb::SprintCrash);
        }
    }
    if let Some((into, out)) = shed_breach {
        run.notes.push(format!(
            "the sprint hit the shed door at {into:.3} m/s and left at {out:.3} m/s - {:.1} % kept",
            100.0 * out / into
        ));
    }
    run.notes.push(format!(
        "at the end of the shed run the hero is at x = {:.2}, and the shed door is {:?}",
        h.hero_pos().x,
        h.door_at(GAMEPLAY_SHED_AT)
    ));

    // ── 10. Dive through the hatch ──
    // **The dive is armed TWO STEPS out**, and the arithmetic is the reason:
    // the breach is decided at the top of the movement step, before the mode
    // table has honoured the dive, so a dive requested on the step it reaches
    // the door is still `Grounded` when the breach is priced. At a sprint a step
    // is 0.108 m and the breach reaches 1.2 m, so the request goes in at
    // x = 26.3 and the leaf is met in `Dive` at about 26.4.
    let mut dived = false;
    let mut dive_mode_at_breach = None;
    for _ in 0..500 {
        let x = h.hero_pos().x;
        let arm = !dived && x >= 26.3;
        let keys: &[&str] = if arm {
            &["Shift", "KeyW", "KeyF"]
        } else if x < 27.2 {
            &["Shift", "KeyW"]
        } else {
            &[]
        };
        if arm {
            dived = true;
        }
        let was = h.door_at(GAMEPLAY_HATCH_AT).map(|d| d.open_deg != 0.0);
        h.frame(keys, (0.0, 0.0), 0.0, &[]);
        rec(&mut h, &mut run);
        let now = h
            .door_at(GAMEPLAY_HATCH_AT)
            .is_some_and(|d| d.open_deg != 0.0);
        if now {
            if was == Some(false) && dive_mode_at_breach.is_none() {
                dive_mode_at_breach =
                    Some((h.hero().mode, planar(h.hero().runtime.velocity.to_dvec3())));
            }
            run.saw(Verb::DiveThrough);
        }
    }
    run.dive_breach = dive_mode_at_breach;
    if let Some((mode, speed)) = dive_mode_at_breach {
        run.notes.push(format!(
            "the hatch gave at {speed:.3} m/s - the DIVE's own launch speed ({} m/s) - and the mode at the end of that step was {mode:?}",
            CharacterMovement::default().dive_speed_mps
        ));
    }
    run.notes.push(format!(
        "at the end of the hatch run the hero is at x = {:.2}, and the hatch is {:?}",
        h.hero_pos().x,
        h.door_at(GAMEPLAY_HATCH_AT)
    ));
    if run.seen.contains(&Verb::DiveThrough) {
        run.notes.push(format!(
            "a dive went through the hatch, ending at x = {:.2}",
            h.hero_pos().x
        ));
    }
    run
}

fn planar(v: glam::DVec3) -> f64 {
    glam::DVec3::new(v.x, 0.0, v.z).length()
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **THE GATE.** Every verb forced, and the cooked pack and the PIE payload
/// byte-identical over the whole course.
#[test]
fn pie_equals_shipping_over_every_gameplay_verb() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pack = cook_fixture(tmp.path());
    let ship = run_course(pack_sim(&pack));
    let pie = run_course(pie_sim());

    for note in &ship.notes {
        println!("{note}");
    }
    // **The anti-vacuity check first**: a trace that forced nothing and matched
    // itself perfectly is two identical empty worlds agreeing.
    let missing: Vec<&str> = required_verbs()
        .into_iter()
        .filter(|(_, v)| !ship.seen.contains(v))
        .map(|(n, _)| n)
        .collect();
    assert!(
        missing.is_empty(),
        "the course did not force: {missing:?} — it certifies a SUBSET of the mandate"
    );
    assert_eq!(pie.seen, ship.seen, "the two hosts forced different verbs");

    assert_eq!(
        ship.trace.len(),
        pie.trace.len(),
        "the two hosts produced different numbers of steps"
    );
    assert!(ship.trace.len() > 1000, "{}", ship.trace.len());
    assert_ne!(
        ship.trace[0],
        ship.trace[ship.trace.len() - 1],
        "the course ended in exactly the state it started in"
    );
    for (i, (s, p)) in ship.trace.iter().zip(pie.trace.iter()).enumerate() {
        assert_eq!(
            s, p,
            "step {i}: the cooked pack and the PIE payload disagree"
        );
    }
}

/// **A BULLET REALLY BREAKS THE WALL** — the joules cross the P22 door and the
/// world answers with chunks.
///
/// The arm the main gate's coverage check cannot make: forcing the `Fire` verb
/// only needs the magazine to move, and a weapon that fired into a world where
/// nothing could break would satisfy it perfectly. Measured: the first round
/// carries the rifle's own 1 700 J to the destructible and takes the whole
/// twelve-chunk block off it, which is why the other five arrive at debris
/// rather than at a wall.
#[test]
fn a_rifle_round_spends_its_joules_at_the_p22_door() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pack = cook_fixture(tmp.path());
    let run = run_course(pack_sim(&pack));
    println!(
        "{} round(s) reached the destructible, owing {} J; {} chunks came off",
        run.hits_on_target, run.owed_j, run.debris_after_fire
    );
    assert!(
        run.hits_on_target >= 1,
        "no round reached the destructible at all"
    );
    let want = inf_ecs::weapon::WeaponDef::default().damage_j;
    assert!(
        (run.owed_j - want * f64::from(run.hits_on_target)).abs() < 1e-9,
        "the joules owed are not the rounds' own damage: {} vs {}",
        run.owed_j,
        want * f64::from(run.hits_on_target)
    );
    assert!(
        run.debris_after_fire > 0,
        "the energy reached the door and nothing came off: a shot into a world where nothing can break satisfies the Fire verb and proves nothing"
    );
    // …and the DIVE really went through at the dive's own launch speed, which is
    // the half of that verb a position alone would not say.
    let (mode, speed) = run.dive_breach.expect("the hatch gave");
    let want = inf_ecs::components::CharacterMovement::default().dive_speed_mps;
    println!(
        "the hatch gave in {mode:?} at {speed} m/s against the dive's own launch speed of {want} — {:e} m/s apart",
        (speed - want).abs()
    );
    // **The tolerance is the portable trigonometry's own error and nothing
    // else.** A dive launches along the character's facing, and that direction
    // goes through `psin64`/`pcos64` — the P14 law — so the planar magnitude
    // lands a few ten-millionths off the nominal. The arm prints what it
    // measured rather than trusting the round number, exactly as
    // `interact::VIEW_CONE_EPSILON_DEG`'s own arm does.
    assert!(
        (speed - want).abs() < 1e-5,
        "the hatch gave at {speed} m/s, which is not the dive's launch speed"
    );
}

/// **Two independent cooks replay bit-identically.**
#[test]
fn the_course_replays_across_two_independent_cooks() {
    let a = tempfile::tempdir().expect("tempdir a");
    let b = tempfile::tempdir().expect("tempdir b");
    let pack_a = cook_fixture(a.path());
    let pack_b = cook_fixture(b.path());
    let ra = run_course(pack_sim(&pack_a));
    let rb = run_course(pack_sim(&pack_b));
    assert_eq!(ra.trace.len(), rb.trace.len());
    assert!(!ra.seen.is_empty(), "the replay forced nothing");
    for (i, (x, y)) in ra.trace.iter().zip(rb.trace.iter()).enumerate() {
        assert_eq!(x, y, "step {i}: two independent cooks diverged");
    }
}

/// **The gameplay traces are part of `state_bytes`**, so the comparison above is
/// about doors, bags, magazines and joules rather than about positions alone.
///
/// Without this the gate would still pass on an engine where none of the four
/// new sections was folded in — two hosts agreeing about where a character is
/// standing, and about nothing this wave built.
#[test]
fn the_trace_carries_the_doors_the_bag_the_magazine_and_the_body() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let pack = cook_fixture(tmp.path());
    let mut sim = pack_sim(&pack);
    for _ in 0..60 {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    }
    let world = sim.world();
    let doors = inf_ecs::door::door_state_bytes(world);
    let items = inf_ecs::item::item_state_bytes(world);
    let health = inf_ecs::weapon::health_state_bytes(world);
    println!(
        "after the level authored itself: {} door bytes, {} inventory bytes, {} health bytes",
        doors.len(),
        items.len(),
        health.len()
    );
    // The bag and the body exist because `BeginPlay` made them; the doors are
    // untouched, so their section is EMPTY — which is the sparseness claim, and
    // it is what keeps every pre-I6 trace byte-identical.
    assert!(!items.is_empty(), "the hero's bag is not in the trace");
    assert!(!health.is_empty(), "the hero's body is not in the trace");
    assert!(
        doors.is_empty(),
        "a level nobody has touched a door in wrote {} door bytes",
        doors.len()
    );
    // …and the whole buffer really contains them.
    let all = sim.state_bytes();
    for (name, part) in [("items", items), ("health", health)] {
        assert!(
            all.windows(part.len()).any(|w| w == part),
            "the {name} section is not inside `state_bytes`"
        );
    }
}
