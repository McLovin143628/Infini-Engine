//! **THE CONTROLS CERTIFICATION** (wave CERT1, clause CP-C6): one arm per row of
//! the owner's binding table, driven at the KEY, asserting the WORLD.
//!
//! # The table this file certifies
//!
//! | key | verb |
//! |---|---|
//! | W / S / A / D | move forward / back / strafe left / strafe right |
//! | Shift | sprint |
//! | Ctrl | walk |
//! | E | interact / pick up / use |
//! | R | reload |
//! | C | crouch or slide on a **click**, prone or dive on a **long press** |
//! | Space | jump — or **dive into water** when there is water to dive into |
//! | LMB | attack / shoot |
//! | RMB | aim |
//! | wheel | change weapon |
//! | I | inventory |
//! | Tab | in-game settings |
//!
//! **The gait default is RUN.**
//!
//! # What makes this file a certification instrument rather than another gate
//!
//! Every arm here presses a **key code**, hands it to the shipped `PlayerUi`
//! first (a key a dialog takes never reaches the game — `PlayerApp::window_event`'s
//! rule), folds it through the shipped `InputMap` and the shipped `InputState`,
//! reduces it with the shipped `inf_player::input::held_actions`, and steps the
//! shipped `RuntimeSim`. Nothing here writes an action name, an intent field or a
//! component; the only thing an arm touches is a keyboard.
//!
//! And every arm then asserts a **world quantity** and prints the number it
//! measured, so the certification memo cites a measurement rather than a claim:
//! a signed velocity component *in the character's own aim frame*, a
//! `MovementMode` variant by name, a `Gait` value, a `RotationMode`, the equipped
//! weapon's id, a magazine count, a bag count, the panel's open flag, or the
//! simulation's own fixed-step counter.
//!
//! ## The keys are LITERALS, deliberately
//!
//! An arm could ask `inf_ui::bindings` which token the table has under
//! `move_x-` and press that. It would be self-consistent and it would be
//! **unfalsifiable**: swap A and D in `inf_input::default_map` and such an arm
//! presses the swapped key, reads the swapped direction and stays green. The
//! owner's table names *keys*, so this file names keys, and swapping the table
//! turns the A and D arms red — which is measured, in this wave's falsification
//! pass, not asserted.
//!
//! ## The fixture is built here, and why
//!
//! `player_core_gate` runs on the committed phase-29 course, which is the right
//! world for a gait ladder and has **no water, no pickups and no weapons** — so a
//! certification of the *whole* table cannot stand on it. This file builds the
//! smallest world each row needs, in `movement_parity`'s own hand-built style
//! (that file's header gives the reason: no committed level carries a
//! `CharacterMovement`, because autostep is opt-in by component presence). The
//! water fixture is `inf-physics`' `water_buoyancy_3d` geometry verbatim — a lake
//! whose near edge is one metre ahead of a character standing on dry ground — so
//! the Space-into-water arm reaches the same outcome that file reaches from an
//! intent, from a key.
//!
//! ## Honest bounds
//!
//! * **Every row is reachable from a key.** No row in the owner's table needed an
//!   action name to drive it, and no row is unimplemented.
//! * `Tab` **closes** the dialog as well as opening it (`inf_ui::menu`'s
//!   `"Escape" | "Tab"` arm), and this file asserts both halves.
//! * The C key's long press is classified on the **sim's fixed step**, not a wall
//!   clock, so the "held for 30 frames" here is half a second of *simulation*.
//!
//! # The falsification record
//!
//! Every mutation below was applied to this file's own harness, the suite run,
//! and the file restored byte for byte. A certification that never watched its
//! own instrument go red is a claim about a passing test, not about an engine.
//!
//! | mutation | arms red |
//! |---|---|
//! | the two strafe arms press each other's key (an A/D swap, from the arm's side) | **2** — `a_strafes_…`, `d_strafes_…`, on the SIGN clause |
//! | both strafe arms press `KeyD` (one direction for two rows) | **2** — the sign clause *and* the opposite-signs clause |
//! | the hold threshold raised to the band ceiling (2 s), past every press here | **2** — `c_held_goes_prone_…`, `c_held_at_sprint_dives`; **both click arms stay green**, so the discrimination is real |
//! | the dialog no longer pauses the sim (`set_sim_paused(false)`) | **1** — `tab_…`, at 60 fixed steps spent behind an open dialog |
//! | the character is not `player_controlled` (the keys reach nobody) | **18 of 20** — everything except `i_…` and `tab_…`, which are the two rows that are decisions about the SESSION rather than about the character |

use glam::DVec2;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, Gait,
    MovementMode, RigidBody3D, RotationMode, Transform, WaterBody,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::EcsWorld;
use inf_input::{InputEvent, InputMap, InputState, MouseButton};
use inf_player::runtime_sim::RuntimeSim;
use inf_player::ui::PlayerUi;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;
/// Real gravity, so the character is really standing on something.
const GRAVITY: DVec2 = DVec2::new(0.0, -9.81);
/// The capsule radius; the half-heights come from the component's own defaults,
/// so the fixture is reading the shipped tuning rather than restating it.
const RADIUS: f64 = 0.3;

const HERO: Uuid = Uuid::from_u128(0xce27_0001);
const GROUND: Uuid = Uuid::from_u128(0xce27_0002);
const LAKE: Uuid = Uuid::from_u128(0xce27_0003);
const PICKUP: Uuid = Uuid::from_u128(0xce27_0004);

/// The item a pickup arm puts on the floor.
const BANDAGE: &str = "bandage";
/// The two weapons the wheel cycles between.
const RIFLE: &str = "rifle";
const PISTOL: &str = "pistol";

// ── the fixture ─────────────────────────────────────────────────────────────

/// What a given arm needs in its world. Everything is off by default, so an arm
/// that does not ask for water runs on a level `water_surface_at` answers `None`
/// for in `O(1)` — which is the level every other arm wants.
#[derive(Clone, Copy, Default)]
struct Fixture {
    /// The `z` of a 200 m-square lake's centre, whose surface sits half a metre
    /// **below** the floor the character stands on. `Some(101.0)` puts its near
    /// edge at `z = 1`, one metre in front of a character at the origin — inside
    /// `movement::DIVE_WATER_REACH_M` (2.5 m) of the feet.
    lake_at_z: Option<f64>,
    /// A bandage lying on the floor a metre in front of the character.
    pickup: bool,
    /// A rifle and a pistol in the bag, the rifle equipped.
    armed: bool,
}

/// The hero: a kinematic capsule with a **player-controlled** `CharacterMovement`,
/// facing `+Z` (yaw 0), which is what makes "forward" and "right" nameable.
fn hero_parts() -> (
    RigidBody3D,
    Collider3D,
    CharacterController3D,
    CharacterMovement,
    Transform,
) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, 0.0);
    // Yaw 0: the aim frame's forward is world `+Z` and its right is world `+X`.
    t.rotation.y = 0.0;
    (
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
            radius: RADIUS,
            ..Default::default()
        },
        CharacterController3D::default(),
        cm,
        t,
    )
}

/// The two weapon definitions the armed fixture carries.
///
/// The rifle is `WeaponDef::default()` — the shipped rifle — and the pistol
/// differs in every field the wheel arm reads about it, so "the equipped id
/// moved" cannot be satisfied by two names for one thing.
fn weapon_catalogue() -> Vec<inf_ecs::item::ItemDef> {
    let rifle = inf_ecs::weapon::WeaponDef::default();
    let pistol = inf_ecs::weapon::WeaponDef {
        automatic: false,
        magazine: 12,
        reserve: 36,
        rounds_per_minute: 300.0,
        ..Default::default()
    };
    vec![
        inf_ecs::item::ItemDef {
            id: RIFLE.into(),
            label: "Rifle".into(),
            stack_max: 1,
            mass_kg: 3.5,
            weapon: Some(rifle),
        },
        inf_ecs::item::ItemDef {
            id: PISTOL.into(),
            label: "Pistol".into(),
            stack_max: 1,
            mass_kg: 0.9,
            weapon: Some(pistol),
        },
        inf_ecs::item::ItemDef {
            id: BANDAGE.into(),
            label: "Bandage".into(),
            stack_max: 5,
            mass_kg: 0.05,
            weapon: None,
        },
    ]
}

fn build_world(f: Fixture) -> EcsWorld {
    let mut w = EcsWorld::new();

    // The floor: top face at y = 0, so the character's feet are at y = 0.
    let e = w.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, -0.5, 0.0);
    w.world_mut().entity_mut(e).insert((
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(60.0, 0.5, 60.0),
            ..Default::default()
        },
        t,
    ));

    if let Some(z) = f.lake_at_z {
        let e = w.spawn_with_guid(LAKE, "Water", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 0.0, z);
        // Amplitude zero, so the surface query is exact and the arm's claim is
        // about the dive rule rather than about a wave's phase.
        let body = WaterBody {
            wave_amplitude_m: 0.0,
            ..WaterBody::lake(-0.5, Vec2d::splat(100.0))
        };
        w.world_mut().entity_mut(e).insert((body, t));
    }

    let e = w.spawn_with_guid(HERO, "Hero", None);
    w.world_mut().entity_mut(e).insert(hero_parts());

    if f.pickup || f.armed {
        let defs = inf_ecs::item::item_defs_mut(&mut w);
        for def in weapon_catalogue() {
            assert!(defs.insert(def), "the catalogue refused a definition");
        }
        assert!(
            inf_ecs::item::give_inventory(&mut w, HERO, 8),
            "the hero got no bag"
        );
    }
    if f.pickup {
        // One metre ahead of the feet and a hand's height off the floor: inside
        // `interact::DEFAULT_REACH_M` (2.5 m) and inside the default view cone
        // of a character facing `+Z`.
        assert_eq!(
            inf_ecs::item::spawn_pickup(&mut w, PICKUP, BANDAGE, 1, Vec3d::new(0.0, 0.2, 1.0)),
            Some(PICKUP),
            "the pickup did not spawn"
        );
    }
    if f.armed {
        assert_eq!(
            inf_ecs::item::give(&mut w, HERO, RIFLE, 1),
            0,
            "the rifle did not fit"
        );
        assert_eq!(
            inf_ecs::item::give(&mut w, HERO, PISTOL, 1),
            0,
            "the pistol did not fit"
        );
        assert!(
            inf_physics::d3::gameplay::equip_weapon(&mut w, HERO, RIFLE),
            "the rifle would not equip"
        );
    }

    w.mark_dirty();
    w.propagate();
    w
}

// ── the host: `PlayerApp::frame` minus the rendering ────────────────────────

/// One host: a sim, the input layer in front of it, and the UI session that
/// routes keys away from it — in `PlayerApp::frame`'s own order.
struct Host {
    sim: RuntimeSim,
    state: InputState,
    map: InputMap,
    ui: PlayerUi,
    down: Vec<String>,
    /// Kept alive for the session; an EMPTY settings directory, so the host
    /// starts from the shipped table rather than from a file it found.
    _dir: tempfile::TempDir,
}

impl Host {
    fn new(f: Fixture) -> Self {
        let mut sim = RuntimeSim::new(build_world(f), Vec::new(), GRAVITY, HZ);
        let dir = tempfile::tempdir().expect("a settings directory");
        let (mut ui, map) = PlayerUi::open(dir.path().to_path_buf(), inf_input::default_map());
        // Single player, so an open dialog really pauses the simulation — which
        // is the world quantity the Tab arm reads.
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

    /// One frame with `keys` held, `mouse_x` counts of horizontal motion,
    /// `wheel` notches, and `buttons` as `[left, right]` **level** changes (an
    /// empty slice leaves the buttons exactly as they were).
    fn frame(&mut self, keys: &[&str], mouse_x: f32, wheel: f32, buttons: &[bool]) {
        // The events a window would have delivered: the difference between the
        // held set and the last one.
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
        if mouse_x != 0.0 {
            forwarded.push(InputEvent::MouseMotion {
                delta: [mouse_x, 0.0],
            });
        }
        if wheel != 0.0 {
            forwarded.push(InputEvent::MouseWheel {
                delta: [0.0, wheel],
            });
        }
        for (i, down) in buttons.iter().enumerate() {
            let button = if i == 0 {
                MouseButton::Left
            } else {
                MouseButton::Right
            };
            forwarded.push(InputEvent::MouseButton {
                button,
                pressed: *down,
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
        self.ui.set_bag(self.bag_view());
        for verb in self.ui.take_inventory_verbs() {
            self.sim.apply_inventory_verb(verb);
        }
        // 4. The pause is on the SIM.
        self.sim.set_sim_paused(self.ui.pauses_sim());
        // 5. Step.
        let input = inf_player::input::held_actions(&self.state, DT);
        self.sim.step_once(input);
    }

    /// What the panel is showing — the character's own bag, projected out of the
    /// sim exactly as the shipped host projects it.
    fn bag_view(&self) -> inf_ui::InventoryView {
        let world = self.sim.world();
        let Some(inv) = inf_ecs::item::inventory_of(world, HERO) else {
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

    // ── the world, read ──

    fn cm(&self) -> CharacterMovement {
        let e = self.sim.world().entity_of(HERO).expect("the hero exists");
        self.sim
            .world()
            .world()
            .get::<CharacterMovement>(e)
            .expect("with a movement component")
            .clone()
    }

    /// **The planar velocity in the character's own aim frame**: `.x` is metres
    /// per second to its RIGHT, `.y` metres per second FORWARD.
    ///
    /// This is what makes `A` and `D` distinguishable: a world-space magnitude is
    /// the same number for both, and a sign-agnostic arm passes for a control
    /// scheme with the two swapped.
    fn local_velocity(&self) -> Vec2d {
        let cm = self.cm();
        inf_ecs::movement::rotate_into_frame(
            Vec2d::new(cm.runtime.velocity.x, cm.runtime.velocity.z),
            cm.runtime.aim_yaw_deg,
        )
    }

    fn speed(&self) -> f64 {
        let v = self.cm().runtime.velocity;
        (v.x * v.x + v.z * v.z).sqrt()
    }

    fn vertical_speed(&self) -> f64 {
        self.cm().runtime.velocity.y
    }

    fn height(&self) -> f64 {
        let e = self.sim.world().entity_of(HERO).expect("the hero exists");
        self.sim
            .world()
            .world()
            .get::<Transform>(e)
            .expect("with a transform")
            .translation
            .y
    }

    fn magazine(&self) -> u32 {
        let e = self.sim.world().entity_of(HERO).expect("the hero exists");
        self.sim
            .world()
            .world()
            .get::<inf_ecs::weapon::WeaponState>(e)
            .map(|s| s.magazine)
            .unwrap_or(0)
    }

    fn equipped(&self) -> Option<String> {
        inf_ecs::item::inventory_of(self.sim.world(), HERO)?
            .equipped_id()
            .map(str::to_string)
    }

    fn bag_count(&self, id: &str) -> u32 {
        inf_ecs::item::inventory_of(self.sim.world(), HERO)
            .map(|inv| inv.count_of(id))
            .unwrap_or(0)
    }

    // ── driving ──

    /// Hold `keys` for `frames` frames, nothing else moving.
    fn hold(&mut self, keys: &[&str], frames: u32) {
        for _ in 0..frames {
            self.frame(keys, 0.0, 0.0, &[]);
        }
    }

    /// Let the character settle onto the floor with nothing held.
    fn settle(&mut self) {
        self.hold(&[], 40);
    }

    /// Every `MovementMode` seen while `keys` are held for `frames` frames.
    fn modes_while(&mut self, keys: &[&str], frames: u32) -> Vec<MovementMode> {
        (0..frames)
            .map(|_| {
                self.frame(keys, 0.0, 0.0, &[]);
                self.cm().mode
            })
            .collect()
    }
}

/// The gait a key set settles the character into, and the speed it settles at.
///
/// **Settled, not peak**: a station that decelerates into a slower tier has the
/// *previous* tier's peak, which is `player_core_gate`'s own measured lesson
/// (3.47 against 1.65 for the walk). Every station here starts from a stop, so
/// the last frame is the honest one either way — and stating it keeps the two
/// arms that share this helper from disagreeing.
fn settled_with(keys: &[&str]) -> (Gait, f64) {
    let mut h = Host::new(Fixture::default());
    h.settle();
    h.hold(keys, 180);
    (h.cm().gait, h.speed())
}

/// The lateral (right-positive) velocity a single strafe key settles at.
fn strafe_lateral(key: &str) -> f64 {
    let mut h = Host::new(Fixture::default());
    h.settle();
    h.hold(&[key], 120);
    h.local_velocity().x
}

// ── W · S · A · D ───────────────────────────────────────────────────────────

/// **W walks the character FORWARD in its own frame.**
#[test]
fn w_drives_the_character_forward_in_its_own_frame() {
    let mut h = Host::new(Fixture::default());
    h.settle();
    h.hold(&["KeyW"], 120);
    let v = h.local_velocity();
    println!("W: forward {:+.3} m/s, lateral {:+.3} m/s", v.y, v.x);
    assert!(
        v.y > 1.0,
        "W must drive the character forward: forward {:+.3} m/s",
        v.y
    );
    assert!(
        v.x.abs() < 0.25,
        "W must not steer: lateral {:+.3} m/s",
        v.x
    );
}

/// **S walks the character BACKWARDS in its own frame** — the opposite sign of
/// W's, on the same axis, so a build that bound both keys to the same half of
/// `move_y` fails here.
#[test]
fn s_walks_the_character_backwards_in_its_own_frame() {
    let mut h = Host::new(Fixture::default());
    h.settle();
    h.hold(&["KeyS"], 120);
    let v = h.local_velocity();
    println!("S: forward {:+.3} m/s, lateral {:+.3} m/s", v.y, v.x);
    assert!(
        v.y < -1.0,
        "S must drive the character backwards: forward {:+.3} m/s",
        v.y
    );
    assert!(
        v.x.abs() < 0.25,
        "S must not steer: lateral {:+.3} m/s",
        v.x
    );
}

/// **A strafes to the character's LEFT** — a NEGATIVE lateral velocity in its own
/// aim frame, and the opposite sign to D's.
///
/// The opposition is the whole arm. A world-space magnitude is the same number
/// for both keys, so an arm that measured speed alone would pass for a build with
/// the two swapped.
#[test]
fn a_strafes_to_the_characters_left() {
    let left = strafe_lateral("KeyA");
    let right = strafe_lateral("KeyD");
    println!("A: lateral {left:+.3} m/s (D's is {right:+.3} m/s)");
    assert!(left < -1.0, "A must strafe LEFT: lateral {left:+.3} m/s");
    assert!(
        left * right < 0.0,
        "A and D must produce OPPOSITE lateral signs: {left:+.3} vs {right:+.3}"
    );
}

/// **D strafes to the character's RIGHT** — a POSITIVE lateral velocity in its own
/// aim frame, and the opposite sign to A's.
#[test]
fn d_strafes_to_the_characters_right() {
    let right = strafe_lateral("KeyD");
    let left = strafe_lateral("KeyA");
    println!("D: lateral {right:+.3} m/s (A's is {left:+.3} m/s)");
    assert!(right > 1.0, "D must strafe RIGHT: lateral {right:+.3} m/s");
    assert!(
        left * right < 0.0,
        "A and D must produce OPPOSITE lateral signs: {left:+.3} vs {right:+.3}"
    );
}

// ── the gait ladder: default RUN · Shift · Ctrl ─────────────────────────────

/// **THE DEFAULT GAIT IS RUN**, with the constant read BY NAME.
///
/// Two halves, and both are needed: the type's own default is `Gait::Run`
/// (a literal `1` here would pass for a renumbered enum), and a character with
/// **nothing but a direction key held** really settles at `run_speed_mps`.
#[test]
fn the_default_gait_is_run_and_nothing_held_runs() {
    assert_eq!(
        Gait::default(),
        Gait::Run,
        "the gait default moved off Run — the owner's table says RUN"
    );
    let tune = CharacterMovement::default();
    let (gait, speed) = settled_with(&["KeyW"]);
    println!(
        "nothing but W held: gait {gait:?} at {speed:.3} m/s (Gait::default() is {:?}, run_speed_mps is {:.3})",
        Gait::default(),
        tune.run_speed_mps
    );
    assert_eq!(gait, Gait::Run, "the default gait is not Run");
    assert!(
        (speed - tune.run_speed_mps).abs() < 0.25,
        "with nothing held the character must RUN: {speed:.3} against {:.3}",
        tune.run_speed_mps
    );
}

/// **SHIFT SPRINTS** — the gait value by name and the speed it implies.
#[test]
fn shift_sprints() {
    let tune = CharacterMovement::default();
    let (gait, speed) = settled_with(&["Shift", "KeyW"]);
    let (_, run) = settled_with(&["KeyW"]);
    println!(
        "Shift + W: gait {gait:?} at {speed:.3} m/s (sprint_speed_mps is {:.3}; W alone is {run:.3})",
        tune.sprint_speed_mps
    );
    assert_eq!(gait, Gait::Sprint, "Shift did not reach the gait");
    assert!(
        (speed - tune.sprint_speed_mps).abs() < 0.25,
        "Shift must SPRINT: {speed:.3} against {:.3}",
        tune.sprint_speed_mps
    );
    assert!(speed > run, "the sprint is not faster than the run");
}

/// **CTRL WALKS** — the *slow* modifier, because the default is Run.
#[test]
fn ctrl_walks() {
    let tune = CharacterMovement::default();
    let (gait, speed) = settled_with(&["Control", "KeyW"]);
    let (_, run) = settled_with(&["KeyW"]);
    println!(
        "Ctrl + W: gait {gait:?} at {speed:.3} m/s (walk_speed_mps is {:.3}; W alone is {run:.3})",
        tune.walk_speed_mps
    );
    assert_eq!(gait, Gait::Walk, "Ctrl did not reach the gait");
    assert!(
        (speed - tune.walk_speed_mps).abs() < 0.25,
        "Ctrl must WALK: {speed:.3} against {:.3}",
        tune.walk_speed_mps
    );
    assert!(speed < run, "the walk is not slower than the run");
}

// ── C: click and long press, standing and sprinting ─────────────────────────

/// **A C CLICK CROUCHES.**
///
/// One frame down, then up: the click fires on the RELEASE, because nothing can
/// know a press is short until it ends.
#[test]
fn c_clicked_crouches() {
    let mut h = Host::new(Fixture::default());
    h.settle();
    h.frame(&["KeyC"], 0.0, 0.0, &[]);
    let modes = h.modes_while(&[], 30);
    let crouched = modes.iter().filter(|m| **m == MovementMode::Crouch).count();
    println!(
        "a C click held Crouch for {crouched} of 30 steps; it ended in {:?}",
        h.cm().mode
    );
    assert!(
        crouched > 20,
        "a C CLICK must crouch — it was in Crouch for {crouched} of 30 steps"
    );
    assert!(
        !modes.contains(&MovementMode::Prone),
        "a C click went PRONE: {modes:?}"
    );
}

/// **A C LONG PRESS GOES PRONE, WHERE THE CLICK ONLY CROUCHED.**
///
/// The same key, with the same nothing else held, differing only in how long it
/// is down — and the crouch control is measured in the same arm so the pair is a
/// discrimination rather than two unrelated numbers.
#[test]
fn c_held_goes_prone_where_c_clicked_only_crouches() {
    let threshold = {
        let h = Host::new(Fixture::default());
        h.sim.press_threshold_s()
    };
    let mut h = Host::new(Fixture::default());
    h.settle();
    // 45 frames is 0.75 s of SIMULATION, three times the shipped threshold.
    let modes = h.modes_while(&["KeyC"], 45);
    let prone = modes.iter().filter(|m| **m == MovementMode::Prone).count();
    let crouch = modes.iter().filter(|m| **m == MovementMode::Crouch).count();

    // The control: the same key, clicked.
    let mut c = Host::new(Fixture::default());
    c.settle();
    c.frame(&["KeyC"], 0.0, 0.0, &[]);
    c.hold(&[], 30);
    let clicked = c.cm().mode;

    println!(
        "C held for 45 steps (threshold {threshold:.3} s): Prone for {prone} steps, Crouch for {crouch}; the same key CLICKED ended in {clicked:?}"
    );
    assert!(
        prone > 20,
        "a C LONG PRESS must go PRONE — it was in Prone for {prone} of 45 steps"
    );
    assert_eq!(
        clicked,
        MovementMode::Crouch,
        "the C click did not crouch, so the pair says nothing"
    );
}

/// **A C CLICK AT SPRINT SPEED SLIDES**, which the same click from a standstill
/// does not.
#[test]
fn c_clicked_at_sprint_slides() {
    let mut h = Host::new(Fixture::default());
    h.settle();
    h.hold(&["Shift", "KeyW"], 180);
    let entry = h.speed();
    h.frame(&["Shift", "KeyW", "KeyC"], 0.0, 0.0, &[]);
    let modes = h.modes_while(&["Shift", "KeyW"], 60);
    let slid = modes.iter().filter(|m| **m == MovementMode::Slide).count();
    println!(
        "a C click at {entry:.3} m/s held Slide for {slid} of 60 steps (slide_entry_speed_mps is {:.3})",
        CharacterMovement::default().slide_entry_speed_mps
    );
    assert!(
        slid >= 5,
        "a C CLICK at sprint speed must SLIDE — it held Slide for {slid} steps"
    );
}

/// **A C LONG PRESS AT SPRINT SPEED DIVES.**
///
/// Read as a **launch**, not as a mode: a dive off a flat floor is four
/// centimetres of clearance in a fixed step and the ground snap is entitled to
/// take it back, so what the verb *is* is the velocity it writes —
/// `(dive_speed_mps forward, dive_up_speed_mps up)`, and nothing else in the
/// movement model produces an upward metre a second on a floor.
#[test]
fn c_held_at_sprint_dives() {
    let tune = CharacterMovement::default();
    let mut h = Host::new(Fixture::default());
    h.settle();
    h.hold(&["Shift", "KeyW"], 180);
    let mut peak_up = 0.0f64;
    for _ in 0..60 {
        h.frame(&["Shift", "KeyW", "KeyC"], 0.0, 0.0, &[]);
        peak_up = peak_up.max(h.vertical_speed());
    }
    // The launch less one fixed step of gravity, derived from the fixture's own
    // gravity rather than absorbed into a tolerance.
    let expected = tune.dive_up_speed_mps + GRAVITY.y * DT;
    println!(
        "a C long press at sprint speed launched at {peak_up:+.6} m/s upward (dive_up_speed_mps {:.3} → {expected:+.6} after one step of gravity)",
        tune.dive_up_speed_mps
    );
    assert!(
        (peak_up - expected).abs() < 1e-6,
        "a C LONG PRESS at sprint speed must DIVE — the peak upward speed was {peak_up:+.6}, not {expected:+.6}"
    );

    // The control: the same key CLICKED at the same speed slides instead, and a
    // slide has no upward launch at all.
    let mut c = Host::new(Fixture::default());
    c.settle();
    c.hold(&["Shift", "KeyW"], 180);
    c.frame(&["Shift", "KeyW", "KeyC"], 0.0, 0.0, &[]);
    let mut click_up = 0.0f64;
    for _ in 0..60 {
        c.frame(&["Shift", "KeyW"], 0.0, 0.0, &[]);
        click_up = click_up.max(c.vertical_speed());
    }
    println!("…and the same key CLICKED peaked at {click_up:+.3} m/s upward");
    assert!(
        click_up < 1.0,
        "the C click launched a dive: {click_up:+.3} m/s upward"
    );
}

// ── Space: the jump, and the dive into water ────────────────────────────────

/// **SPACE JUMPS** — the vertical launch, the airborne mode and the height
/// gained, on dry land.
#[test]
fn space_jumps() {
    let tune = CharacterMovement::default();
    let mut h = Host::new(Fixture::default());
    h.settle();
    let rest = h.height();
    h.frame(&["Space"], 0.0, 0.0, &[]);
    let launch = h.vertical_speed();
    let mode = h.cm().mode;
    let mut peak = h.height();
    for _ in 0..30 {
        h.frame(&[], 0.0, 0.0, &[]);
        peak = peak.max(h.height());
    }
    // The launch less one fixed step of gravity — see
    // `space_at_the_water_is_a_dive_and_not_a_jump` for why that is the honest
    // number rather than a tolerance around the tuning value.
    let expected = tune.jump_speed_mps + GRAVITY.y * DT;
    println!(
        "Space: launched at {launch:+.6} m/s (jump_speed_mps {:.3} → {expected:+.6} after one step of gravity), mode {mode:?}, rose {:.3} m from {rest:.3}",
        tune.jump_speed_mps,
        peak - rest
    );
    assert!(
        (launch - expected).abs() < 1e-6,
        "Space must launch at the jump speed: {launch:+.6} against {expected:+.6}"
    );
    assert_eq!(
        mode,
        MovementMode::FallFree,
        "a deliberate jump must be FallFree (full air control)"
    );
    assert!(
        peak - rest > 0.2,
        "Space did not raise the character: {rest:.3} -> {peak:.3}"
    );
}

/// **SPACE AT THE WATER IS A DIVE AND NOT A JUMP.**
///
/// One key, two verbs, and only the world can tell them apart — the same outcome
/// `inf-physics`' `a_sprinting_jump_at_the_water_is_a_dive_and_on_dry_land_is_a
/// _jump` reaches from an INTENT, reached here from a KEY.
///
/// The dry-land control is what makes the arm able to fail: a step that turned
/// every sprinting jump into a dive would pass without it.
#[test]
fn space_at_the_water_is_a_dive_and_not_a_jump() {
    let tune = CharacterMovement::default();

    // ── the lake's near edge one metre ahead, sprinting: a DIVE ──
    let mut h = Host::new(Fixture {
        lake_at_z: Some(101.0),
        ..Default::default()
    });
    h.settle();
    h.hold(&["Shift"], 10);
    h.frame(&["Shift", "Space"], 0.0, 0.0, &[]);
    let wet = h.cm().runtime.velocity;
    let wet_mode = h.cm().mode;

    // ── the same key, the same sprint, on DRY LAND: a jump ──
    let mut d = Host::new(Fixture::default());
    d.settle();
    d.hold(&["Shift"], 10);
    d.frame(&["Shift", "Space"], 0.0, 0.0, &[]);
    let dry = d.cm().runtime.velocity;

    // **The launch, minus one step of gravity.** The verb writes the vertical
    // speed and the same fixed step then integrates the fall it has just begun,
    // so what is on the component at the end of the step is the launch less
    // `g·dt` — derived from the fixture's own gravity rather than absorbed into
    // a tolerance, which is what lets the two verbs be told apart to the bit.
    let launched_up = |v: f64| v + GRAVITY.y * DT;
    println!(
        "Space at the water: forward {:+.6} m/s, up {:+.6} m/s, mode {wet_mode:?} (dive_speed_mps {:.3}, dive_up_speed_mps {:.3} → {:+.6} after one step of gravity) | on dry land: forward {:+.6} m/s, up {:+.6} m/s (jump_speed_mps {:.3} → {:+.6})",
        wet.z,
        wet.y,
        tune.dive_speed_mps,
        tune.dive_up_speed_mps,
        launched_up(tune.dive_up_speed_mps),
        dry.z,
        dry.y,
        tune.jump_speed_mps,
        launched_up(tune.jump_speed_mps)
    );
    assert!(
        (wet.z - tune.dive_speed_mps).abs() < 1e-6,
        "Space at the water must launch FORWARD at the dive speed: {:+.6} against {:.3}",
        wet.z,
        tune.dive_speed_mps
    );
    assert!(
        (wet.y - launched_up(tune.dive_up_speed_mps)).abs() < 1e-6,
        "…and UP at the dive's own upward speed: {:+.6} against {:+.6}",
        wet.y,
        launched_up(tune.dive_up_speed_mps)
    );
    assert!(
        dry.z.abs() < 1e-6,
        "the same press on dry land threw the character forward: {:+.6} m/s",
        dry.z
    );
    assert!(
        (dry.y - launched_up(tune.jump_speed_mps)).abs() < 1e-6,
        "the same press on dry land must be a JUMP: {:+.6} against {:+.6}",
        dry.y,
        launched_up(tune.jump_speed_mps)
    );
}

// ── E · R · LMB · RMB · wheel ───────────────────────────────────────────────

/// **E PICKS UP THE THING IN FRONT OF THE CHARACTER.**
///
/// The bag count is the world quantity: zero before the press, one after it, and
/// the pickup entity is gone from the world.
#[test]
fn e_picks_up_the_item_in_front_of_the_character() {
    let mut h = Host::new(Fixture {
        pickup: true,
        ..Default::default()
    });
    h.settle();
    let before = h.bag_count(BANDAGE);
    assert_eq!(before, 0, "the fixture started with the item already held");
    assert!(
        h.sim.world().entity_of(PICKUP).is_some(),
        "the pickup is not in the world, so the press has nothing to reach"
    );
    h.frame(&["KeyE"], 0.0, 0.0, &[]);
    h.hold(&[], 10);
    let after = h.bag_count(BANDAGE);
    let still_there = h.sim.world().entity_of(PICKUP).is_some();
    println!(
        "E: bag holds {before} -> {after} {BANDAGE}; the pickup entity remains: {still_there}"
    );
    assert_eq!(after, 1, "E did not pick the item up");
    assert!(!still_there, "the item was taken and left lying there");
}

/// **R RELOADS THE EQUIPPED WEAPON.**
///
/// The magazine is the world quantity, and it is spent first with the trigger —
/// a reload of a full magazine is a refusal, and an arm that never emptied one
/// would be measuring nothing.
#[test]
fn r_reloads_the_equipped_weapon() {
    let mut h = Host::new(Fixture {
        armed: true,
        ..Default::default()
    });
    h.settle();
    let full = h.magazine();
    // Spend some rounds.
    for _ in 0..40 {
        h.frame(&[], 0.0, 0.0, &[true, false]);
    }
    h.frame(&[], 0.0, 0.0, &[false, false]);
    let low = h.magazine();
    assert!(low < full, "the trigger spent nothing: {full} -> {low}");
    // R, then long enough for the reload clock to run out.
    h.frame(&["KeyR"], 0.0, 0.0, &[]);
    h.hold(&[], 200);
    let back = h.magazine();
    println!("R: magazine {full} -> {low} (fired) -> {back} (reloaded)");
    assert!(
        back > low,
        "R did not reload: the magazine stayed at {low} of {full}"
    );
    assert_eq!(back, full, "the reload did not fill the magazine");
}

/// **LMB FIRES THE EQUIPPED WEAPON.**
///
/// Two world quantities, because either alone is weak: the magazine goes down,
/// and the fixed step's own shot counter goes up by exactly what the magazine
/// lost.
#[test]
fn lmb_fires_the_equipped_weapon() {
    let mut h = Host::new(Fixture {
        armed: true,
        ..Default::default()
    });
    h.settle();
    let before = h.magazine();
    let mut shots = 0u32;
    for _ in 0..40 {
        h.frame(&[], 0.0, 0.0, &[true, false]);
        shots += h.sim.gameplay().shots;
    }
    h.frame(&[], 0.0, 0.0, &[false, false]);
    let after = h.magazine();
    println!("LMB: magazine {before} -> {after}, {shots} shot(s) in the gameplay report");
    assert!(shots > 0, "LMB fired nothing");
    assert_eq!(
        before - after,
        shots,
        "the rounds the magazine lost and the shots the step reported disagree"
    );

    // The control: the same forty frames with the trigger UP spend nothing.
    let mut c = Host::new(Fixture {
        armed: true,
        ..Default::default()
    });
    c.settle();
    c.hold(&[], 40);
    println!(
        "…and with the trigger up the magazine stayed at {}",
        c.magazine()
    );
    assert_eq!(
        c.magazine(),
        before,
        "the weapon fired with nothing pressed"
    );
}

/// **RMB PUTS THE CHARACTER IN `RotationMode::Aiming`**, and releasing it takes
/// the character back out.
#[test]
fn rmb_aims() {
    let mut h = Host::new(Fixture::default());
    h.settle();
    let before = h.cm().rotation_mode;
    h.frame(&[], 0.0, 0.0, &[false, true]);
    h.hold(&[], 5);
    let aiming = h.cm().rotation_mode;
    h.frame(&[], 0.0, 0.0, &[false, false]);
    h.hold(&[], 5);
    let released = h.cm().rotation_mode;
    println!("RMB: rotation mode {before:?} -> {aiming:?} (held) -> {released:?} (released)");
    assert_eq!(aiming, RotationMode::Aiming, "RMB did not reach the aim");
    assert_ne!(
        released,
        RotationMode::Aiming,
        "the character stayed in Aiming after RMB came up"
    );
}

/// **THE WHEEL CHANGES WEAPON.**
///
/// Two notches, and the equipped id must MOVE and come back: with one weapon in
/// the bag the wheel cycles onto the weapon it is already on, so "unchanged" is a
/// claim a wheel wired to nothing satisfies perfectly (`phase30_gameplay_gate`
/// measured exactly that). The fixture carries two.
#[test]
fn the_wheel_changes_weapon() {
    let mut h = Host::new(Fixture {
        armed: true,
        ..Default::default()
    });
    h.settle();
    let before = h.equipped();
    h.frame(&[], 0.0, 1.0, &[]);
    h.hold(&[], 5);
    let switched = h.equipped();
    h.frame(&[], 0.0, 1.0, &[]);
    h.hold(&[], 5);
    let back = h.equipped();
    println!("wheel: equipped {before:?} -> {switched:?} -> {back:?}");
    assert_eq!(before.as_deref(), Some(RIFLE), "the fixture is not armed");
    assert_eq!(
        switched.as_deref(),
        Some(PISTOL),
        "one notch did not change the equipped weapon"
    );
    assert_eq!(back, before, "a second notch did not come back round");
}

// ── I · Tab ─────────────────────────────────────────────────────────────────

/// **I OPENS AND CLOSES THE INVENTORY.**
///
/// The second half is the one an open panel's "take every key" rule makes hard:
/// the press cannot reach the host's edge, so the panel itself has to answer it.
#[test]
fn i_opens_and_closes_the_inventory() {
    let mut h = Host::new(Fixture {
        armed: true,
        ..Default::default()
    });
    h.settle();
    assert!(!h.ui.inventory.open, "the panel started open");
    h.frame(&["KeyI"], 0.0, 0.0, &[]);
    h.hold(&[], 2);
    let opened = h.ui.inventory.open;
    // What the panel is actually showing, read off the view the host hands it —
    // so "the bag opened" is a claim about the world's contents and not only
    // about a flag.
    let slots = h.bag_view().slots.iter().filter(|s| s.count > 0).count();
    h.frame(&["KeyI"], 0.0, 0.0, &[]);
    h.hold(&[], 2);
    let closed = h.ui.inventory.open;
    println!("I: panel open {opened} (showing {slots} filled slot(s)) -> open {closed}");
    assert!(opened, "I did not open the inventory");
    assert!(!closed, "I did not close the inventory it opened");
    assert!(
        slots > 0,
        "the panel opened onto an empty bag, so it is not showing the world"
    );
}

/// **TAB OPENS THE IN-GAME SETTINGS, AND AN OPEN DIALOG COSTS THE SIMULATION
/// ZERO FIXED STEPS.**
///
/// The step counter is the world quantity, and it is the right one: a dialog that
/// did not pause would put the player's reading time into every trace this engine
/// compares. Tab closes it again (`inf_ui::menu`'s own `"Escape" | "Tab"` arm), so
/// both halves of the owner's row are here.
#[test]
fn tab_opens_the_in_game_settings_and_freezes_the_sim() {
    let mut h = Host::new(Fixture::default());
    h.settle();
    assert!(!h.ui.menu.open, "the dialog started open");
    let steps_before_open = h.sim.steps();
    h.hold(&[], 30);
    let free_steps = h.sim.steps() - steps_before_open;

    h.frame(&["Tab"], 0.0, 0.0, &[]);
    let opened = h.ui.menu.open;
    let steps_at_open = h.sim.steps();
    h.hold(&[], 60);
    let steps_while_open = h.sim.steps() - steps_at_open;

    h.frame(&["Tab"], 0.0, 0.0, &[]);
    let closed_by_tab = !h.ui.menu.open;
    let steps_at_close = h.sim.steps();
    h.hold(&[], 30);
    let steps_after = h.sim.steps() - steps_at_close;

    println!(
        "Tab: 30 free frames cost {free_steps} fixed steps; opened {opened}; 60 frames with the dialog open cost {steps_while_open}; Tab closed it {closed_by_tab}; 30 frames after cost {steps_after}"
    );
    assert!(opened, "Tab did not open the settings dialog");
    assert_eq!(
        steps_while_open, 0,
        "an open dialog advanced the simulation by {steps_while_open} fixed steps"
    );
    assert!(closed_by_tab, "Tab did not close the dialog it opened");
    // Non-vacuity on both sides: the sim really does step when the dialog is
    // shut, before AND after — a host that had simply stopped stepping would
    // satisfy the zero above perfectly.
    assert_eq!(free_steps, 30, "the sim did not step with the dialog shut");
    assert_eq!(steps_after, 30, "the sim did not resume after the dialog");
}
