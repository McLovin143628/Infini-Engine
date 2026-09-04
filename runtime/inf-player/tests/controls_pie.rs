//! **The controls, in a REAL `--pie` subprocess** (wave FIX1, protocol 3).
//!
//! CERT1's CP-C6 certified twenty bindings and its own audit corrected the row
//! to say which world they ran in: *"a hand-built fixture per row, not the island
//! and not PIE"*. The owner's sentence had been *"in PIE on the island"*, and the
//! reason the row could not honour it was structural rather than lazy — **there
//! was no way to press a key in a PIE session**. `EditorToPlayer` could say
//! *when* to step and the player answered with one `u64`; nothing on the wire
//! carried an input and nothing carried a world.
//!
//! Protocol 3 carries both, and this file is what they are for. Every arm below:
//!
//! 1. builds a `ScenePayload` from a real `SceneDoc` — the same
//!    `build_scene_payload` the Play button calls;
//! 2. spawns the **real `inf-player --pie` binary** through `PieSession`, the
//!    same door `pie_start` takes;
//! 3. presses a **literal key code** into an `InputFrame`, which the player
//!    reduces with the shipped `PlayerUi` → `InputMap` → `InputState` →
//!    `held_actions` → `RuntimeSim::step_once` path;
//! 4. asserts a **world quantity** read back over `Probe` — a signed velocity in
//!    the character's own aim frame, a `MovementMode` by name, a `Gait`, a
//!    magazine, a bag count, `sim.steps()` itself.
//!
//! The keys are literals and **not read from the binding table**, for CP-C6's
//! own reason: an arm that asked the table which key is `move_x−` would press the
//! swapped key after a swap and stay green.
//!
//! # What this world is, plainly
//!
//! One document per fixture: a 120 m floor whose top face is `y = 0`, the
//! **committed starter character** (its real 161-bone rig, its real body mesh and
//! its real `Starter_Locomotion.inf_sm`, read from `samples/starter-character/`),
//! and — where a row needs one — a lake, a catalogue, two weapons and a pickup.
//! The catalogue and the placements are authored the way this engine's own
//! doctrine says they must be, through a **Blueprint class on the hero**
//! (`inf_ecs::item`'s header: "the Blueprint class is the one authoring surface
//! that reaches all three [boot paths]"), so the `.inf_act` bytes ride the
//! payload exactly as a cooked pack carries them.
//!
//! The character is the committed one rather than a bare capsule because the same
//! session then answers the other question the author asked — **is it standing in
//! a T-pose** — off the same probe, on the same content, in the same subprocess.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Duration;

use uuid::Uuid;

use inf_blueprint::{BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Stmt, Ty};
use inf_ecs::components::{
    AnimStateMachine, BodyKind3D, CharacterController3D, CharacterMovement, Collider3D,
    ColliderShape3DKind, RigidBody3D, SkeletalMesh, Transform, WaterBody,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::pie::{build_scene_payload, PieSession};
use inf_editor_core::scene::SceneDoc;
use inf_runtime::pie::{InputFrame, ScenePayload, WorldProbe};

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;
/// The capsule radius; the half-heights come from the component's own defaults,
/// so this fixture reads the shipped tuning rather than restating it.
const RADIUS: f64 = 0.3;

const HERO: Uuid = Uuid::from_u128(0xf1c1_0001);
const GROUND: Uuid = Uuid::from_u128(0xf1c1_0002);
const LAKE: Uuid = Uuid::from_u128(0xf1c1_0003);
const CLASS: Uuid = Uuid::from_u128(0xf1c1_0004);
const WALL: Uuid = Uuid::from_u128(0xf1c1_0005);

const BANDAGE: &str = "bandage";
const RIFLE: &str = "rifle";
const PISTOL: &str = "pistol";

/// How long a probe may take. Generous: the answer is one frame's work, but the
/// player may be mid-load when the first one is asked.
const PROBE_TIMEOUT: Duration = Duration::from_secs(20);

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

/// The committed starter character's directory.
fn starter_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/starter-character")
        .canonicalize()
        .expect("the committed starter character is in the tree")
}

/// **An EMPTY settings directory for every player this file spawns.**
///
/// `PieInputHost::open` reads the player's own settings file, which is right for
/// a session somebody is driving and wrong for a certification: a machine whose
/// author had turned their look sensitivity down would measure different numbers
/// from a fresh one. `settings_dir()` honours this variable, children inherit the
/// environment, and the directory is leaked deliberately so it outlives every
/// session in this binary.
fn isolate_settings() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let dir = tempfile::tempdir().expect("a settings tempdir");
        std::env::set_var(inf_player::ui::SETTINGS_DIR_ENV, dir.path());
        std::mem::forget(dir);
    });
}

// ── the fixture ─────────────────────────────────────────────────────────────

/// What a given arm needs in its world. Everything is off by default — CP-C6's
/// own shape, and for its reason: an arm that does not ask for water runs on a
/// level `water_surface_at` answers `None` for in `O(1)`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct Fixture {
    /// A 200 m-square lake whose surface sits half a metre **below** the floor,
    /// centred at `z = 101` so its near edge is one metre in front of a hero at
    /// the origin — inside `movement::DIVE_WATER_REACH_M`.
    lake: bool,
    /// A bandage on the floor a metre ahead (and a bag to put it in).
    pickup: bool,
    /// A rifle and a pistol in the bag, the rifle equipped.
    armed: bool,
    /// A 6 m wall two metres BEHIND the hero — behind, because that is where a
    /// third-person boom is, and the camera's collision sweep is what this
    /// fixture exists to make fire.
    wall: bool,
}

/// The catalogue, as the TOML `item.define` takes. The rifle is the shipped
/// default and the pistol differs in every field the wheel arm reads, so "the
/// equipped id moved" cannot be satisfied by two names for one thing.
const CATALOGUE: &str = "\
[rifle]
label = \"Rifle\"
stack_max = 1
mass_kg = 3.5
[rifle.weapon]

[pistol]
label = \"Pistol\"
stack_max = 1
mass_kg = 0.9
[pistol.weapon]
automatic = false
magazine = 12
reserve = 36
rounds_per_minute = 300.0

[bandage]
label = \"Bandage\"
stack_max = 5
mass_kg = 0.05
";

fn call(path: &[&str], args: Vec<Expr>) -> Stmt {
    Stmt::ExprStmt(Expr::Call {
        path: path.iter().map(|s| s.to_string()).collect(),
        args,
    })
}

fn str_lit(s: &str) -> Expr {
    Expr::Lit(Lit::Str(s.into()))
}

/// The hero's own entity handle, which the host seeds into the class's `entity`
/// variable — the pattern every generated controller in this tree uses.
fn me() -> Expr {
    Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![str_lit("entity")],
    }
}

/// **The one authoring surface for a catalogue and its placements.** See this
/// file's header, and `inf_ecs::item`'s.
fn outfitter(f: Fixture) -> BlueprintClass {
    let mut body = vec![call(&["item", "define"], vec![str_lit(CATALOGUE)])];
    if f.armed || f.pickup {
        // A rifle first, in both cases: `item::give` creates the bag, and the
        // pickup row needs one to put a bandage in. The pickup row's assertion is
        // about the BANDAGE's count, which starts at zero either way.
        body.push(call(
            &["item", "give"],
            vec![me(), str_lit(RIFLE), Expr::Lit(Lit::Int(1))],
        ));
    }
    if f.armed {
        body.push(call(
            &["item", "give"],
            vec![me(), str_lit(PISTOL), Expr::Lit(Lit::Int(1))],
        ));
        body.push(call(&["item", "equip"], vec![me(), str_lit(RIFLE)]));
    }
    if f.pickup {
        // One metre ahead of the feet and a hand's height off the floor: inside
        // `interact::DEFAULT_REACH_M` and inside the view cone of a hero facing +Z.
        body.push(call(
            &["item", "spawn_pickup"],
            vec![
                str_lit(BANDAGE),
                Expr::Lit(Lit::Float(0.0)),
                Expr::Lit(Lit::Float(0.2)),
                Expr::Lit(Lit::Float(1.0)),
                Expr::Lit(Lit::Int(1)),
            ],
        ));
    }
    let mut class = BlueprintClass::new("fix1.outfitter", "FIX1 Outfitter");
    class.variables = vec![inf_blueprint::Variable {
        // Seeded by the host, exactly as every generated controller's is.
        name: "entity".into(),
        ty: Ty::Int,
        default: Lit::Int(0),
        exposed: false,
    }];
    class.events = vec![EventBinding {
        event: EventKind::BeginPlay,
        body: BlueprintFn {
            id: EventKind::BeginPlay.key(),
            name: EventKind::BeginPlay.key(),
            params: vec![],
            ret: Ty::Unit,
            body,
        },
    }];
    class
}

/// The committed starter character's eight fixed asset ids.
fn starter_ids() -> inf_editor_core::character::CharacterIds {
    inf_editor_core::samples::starter_character_ids()
}

fn document(f: Fixture) -> SceneDoc {
    let ids = starter_ids();
    let asset = |id: Option<inf_asset::AssetId>| id.expect("every starter id is fixed").0;
    let mut doc = SceneDoc::new();

    // The floor: top face at y = 0, so the hero's feet are at y = 0.
    let e = doc.create_with_guid(GROUND, SpawnKind::Empty, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, -0.5, 0.0);
    doc.world_mut().world_mut().entity_mut(e).insert((
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

    if f.wall {
        // Two metres behind a hero facing +Z, 6 m tall and 40 m wide: the boom
        // trails into it at rest, which is what makes `collision_pull_m` a
        // measurement rather than a zero.
        let e = doc.create_with_guid(WALL, SpawnKind::Empty, "Wall", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 3.0, -2.0);
        doc.world_mut().world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(20.0, 3.0, 0.5),
                ..Default::default()
            },
            t,
        ));
    }

    if f.lake {
        let e = doc.create_with_guid(LAKE, SpawnKind::Empty, "Water", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, 0.0, 101.0);
        // Amplitude zero, so the surface query is exact and the arm's claim is
        // about the dive rule rather than about a wave's phase.
        let body = WaterBody {
            wave_amplitude_m: 0.0,
            ..WaterBody::lake(-0.5, Vec2d::splat(100.0))
        };
        doc.world_mut().world_mut().entity_mut(e).insert((body, t));
    }

    // The hero: the committed starter character's rig, body and machine on a
    // kinematic capsule with a player-controlled `CharacterMovement`, facing +Z
    // (yaw 0), which is what makes "forward" and "right" nameable.
    let e = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(0.0, cm.stand_half_height_m + RADIUS, 0.0);
    t.rotation.y = 0.0;
    doc.world_mut().world_mut().entity_mut(e).insert((
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
        SkeletalMesh {
            mesh: Some(asset(ids.mesh)),
            skeleton: Some(asset(ids.skeleton)),
        },
        AnimStateMachine {
            sm: Some(asset(ids.machine)),
            ..Default::default()
        },
        inf_ecs::components::ActorClass(CLASS),
    ));
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    doc
}

fn payload(f: Fixture) -> ScenePayload {
    let dir = starter_dir();
    let ids = starter_ids();
    let asset = |id: Option<inf_asset::AssetId>| id.expect("every starter id is fixed").0;
    let anim: HashMap<Uuid, Vec<u8>> = [
        (asset(ids.skeleton), "Starter.inf_skel"),
        (asset(ids.machine), "Starter_Locomotion.inf_sm"),
        (asset(ids.idle), "Starter_Idle.inf_anim"),
        (asset(ids.walk), "Starter_Walk.inf_anim"),
        (asset(ids.run), "Starter_Run.inf_anim"),
    ]
    .into_iter()
    .map(|(g, file)| {
        (
            g,
            std::fs::read(dir.join(file)).expect("the starter asset is committed"),
        )
    })
    .collect();
    let mesh = std::fs::read(dir.join("Starter_Body.inf_mesh")).expect("the body is committed");
    let mesh_id = asset(ids.mesh);
    let class = outfitter(f);
    build_scene_payload(
        &document(f),
        |g| (g == CLASS).then(|| class.clone()),
        |_| None,
        |g| anim.get(&g).cloned(),
        |_| None,
        |_| None,
        |_| None,
        |g| (g == mesh_id).then(|| mesh.clone()),
        |_| None,
        // tick-hz 0: no per-frame sleep. Every step this file takes is asked for.
        0,
        false,
    )
    .expect("the fixture builds a scene payload")
}

/// Built once per fixture shape: the document, the class and the starter
/// character's bytes are the same every time and encoding them twenty times is
/// twenty seconds nobody gets back.
fn cached(f: Fixture) -> ScenePayload {
    static CACHE: OnceLock<std::sync::Mutex<HashMap<Fixture, ScenePayload>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut cache = cache.lock().expect("payload cache");
    cache.entry(f).or_insert_with(|| payload(f)).clone()
}

// ── the session ─────────────────────────────────────────────────────────────

/// One real subprocess, driven by input frames and read by probes.
struct Pie {
    session: PieSession,
}

impl Pie {
    fn open(f: Fixture) -> Self {
        isolate_settings();
        let session =
            PieSession::spawn_scene(&player_bin(), &cached(f)).expect("the real player spawns");
        Self { session }
    }

    /// Hold `keys` for `frames` fixed steps.
    fn hold(&mut self, keys: &[&str], frames: u32) {
        for _ in 0..frames {
            self.drive(InputFrame::held(keys.iter().copied(), DT));
        }
    }

    /// One frame, and the `Frame` report that answers it — drained so the probe
    /// below does not have to wade through a session's worth of them.
    fn drive(&mut self, frame: InputFrame) {
        self.session.input(frame).expect("the input frame lands");
        self.session
            .wait_for(Duration::from_secs(10), |e| {
                matches!(e, inf_runtime::pie::PlayerToEditor::Frame { .. })
            })
            .expect("the player reports the frame it stepped");
    }

    /// Nothing held, long enough for the character to come to rest.
    fn settle(&mut self) {
        self.hold(&[], 40);
    }

    fn probe(&mut self) -> WorldProbe {
        self.session
            .probe(None, PROBE_TIMEOUT)
            .expect("the player answers a probe")
    }

    fn hero(&mut self) -> inf_runtime::pie::ActorProbe {
        self.probe().hero.expect("the level has a hero")
    }
}

/// A settled hero after holding `keys` for 180 steps — CP-C6's `settled_with`,
/// through the subprocess.
fn settled_with(keys: &[&str]) -> inf_runtime::pie::ActorProbe {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    pie.hold(keys, 180);
    pie.hero()
}

/// The lateral component of a strafe, after 120 steps.
fn strafe_lateral(key: &str) -> f64 {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    pie.hold(&[key], 120);
    pie.hero().local_velocity[0]
}

/// The `MovementMode` seen on each of `frames` steps while `keys` are held.
fn modes_while(pie: &mut Pie, keys: &[&str], frames: u32) -> Vec<String> {
    let mut out = Vec::with_capacity(frames as usize);
    for _ in 0..frames {
        pie.drive(InputFrame::held(keys.iter().copied(), DT));
        out.push(pie.hero().movement_mode);
    }
    out
}

// ── 0. the pair agrees, and the world is not empty ──────────────────────────

/// **The running pair speaks one protocol.** The constant is pinned in
/// `inf-runtime`; this says the binary on disk agrees with the editor library
/// that is about to drive it — which is the half a source constant cannot claim.
#[test]
fn the_real_player_speaks_this_editors_protocol_version() {
    isolate_settings();
    let session = PieSession::spawn_scene(&player_bin(), &cached(Fixture::default()))
        .expect("the real player spawns");
    assert_eq!(
        session.protocol(),
        inf_runtime::pie::PIE_PROTOCOL_VERSION,
        "the spawned player speaks a different protocol"
    );
    assert_eq!(session.protocol(), 3, "wave FIX1's Input + Probe frames");
    let _ = session.stop(Duration::from_secs(5));
}

/// **The subject exists, and the arms below are not measuring an empty world.**
/// The P21.4 lesson, stated first: two empty worlds agree perfectly.
#[test]
fn the_pie_world_carries_the_hero_the_arms_press_keys_at() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    let probe = pie.probe();
    assert!(probe.entities >= 2, "{probe:?}");
    let hero = probe.hero.expect("the level has a player-controlled hero");
    assert_eq!(hero.name, "Hero");
    assert_eq!(hero.guid, *HERO.as_bytes());
    assert!(hero.grounded, "the hero is not standing on the floor");
    assert!(
        probe.steps >= 40,
        "the session did not step: {} steps",
        probe.steps
    );
}

// ── 1–4. the four movement keys, in the character's own frame ───────────────

#[test]
fn w_drives_the_character_forward_in_its_own_frame() {
    let hero = settled_with(&["KeyW"]);
    assert!(
        (hero.local_velocity[1] - 3.75).abs() < 0.05,
        "W forward {:?}",
        hero.local_velocity
    );
    assert!(
        hero.local_velocity[0].abs() < 0.05,
        "{:?}",
        hero.local_velocity
    );
}

#[test]
fn s_walks_the_character_backwards_in_its_own_frame() {
    let hero = settled_with(&["KeyS"]);
    assert!(
        (hero.local_velocity[1] + 3.75).abs() < 0.05,
        "S forward {:?}",
        hero.local_velocity
    );
}

#[test]
fn a_strafes_to_the_characters_left() {
    let a = strafe_lateral("KeyA");
    assert!(a < -1.0, "A lateral {a}");
}

#[test]
fn d_strafes_to_the_characters_right() {
    let d = strafe_lateral("KeyD");
    assert!(d > 1.0, "D lateral {d}");
    let a = strafe_lateral("KeyA");
    assert!(
        a.signum() != d.signum(),
        "A and D strafe the same way: {a} / {d}"
    );
}

// ── 5–7. the gaits ──────────────────────────────────────────────────────────

#[test]
fn the_default_gait_is_run_and_nothing_held_runs() {
    let hero = settled_with(&["KeyW"]);
    assert_eq!(hero.gait, "Run", "{hero:?}");
    assert!((hero.speed - 3.75).abs() < 0.05, "run speed {}", hero.speed);
}

#[test]
fn shift_sprints() {
    let hero = settled_with(&["KeyW", "Shift"]);
    assert_eq!(hero.gait, "Sprint", "{hero:?}");
    assert!(
        (hero.speed - 6.5).abs() < 0.1,
        "sprint speed {}",
        hero.speed
    );
}

#[test]
fn ctrl_walks() {
    let hero = settled_with(&["KeyW", "Control"]);
    assert_eq!(hero.gait, "Walk", "{hero:?}");
    assert!(
        (hero.speed - 1.65).abs() < 0.05,
        "walk speed {}",
        hero.speed
    );
}

// ── 8–11. C: crouch, prone, slide, dive ─────────────────────────────────────

#[test]
fn c_clicked_crouches() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    pie.hold(&["KeyC"], 2);
    let modes = modes_while(&mut pie, &[], 30);
    assert!(
        modes.iter().all(|m| m == "Crouch"),
        "a click did not crouch: {modes:?}"
    );
}

#[test]
fn c_held_goes_prone_where_c_clicked_only_crouches() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    // Held well past the long-press threshold.
    let modes = modes_while(&mut pie, &["KeyC"], 45);
    assert!(
        modes.iter().any(|m| m == "Prone"),
        "a long press did not go prone: {modes:?}"
    );
    let mut click = Pie::open(Fixture::default());
    click.settle();
    click.hold(&["KeyC"], 2);
    let after = modes_while(&mut click, &[], 45);
    assert!(
        !after.iter().any(|m| m == "Prone"),
        "a CLICK went prone, so the arm above is not measuring the hold: {after:?}"
    );
}

#[test]
fn c_clicked_at_sprint_slides() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    pie.hold(&["KeyW", "Shift"], 120);
    pie.hold(&["KeyW", "Shift", "KeyC"], 2);
    let modes = modes_while(&mut pie, &["KeyW", "Shift"], 60);
    assert!(
        modes.iter().any(|m| m == "Slide"),
        "a click at sprint did not slide: {modes:?}"
    );
}

#[test]
fn c_held_at_sprint_dives() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    pie.hold(&["KeyW", "Shift"], 120);
    let mut peak = f64::MIN;
    for _ in 0..45 {
        pie.drive(InputFrame::held(["KeyW", "Shift", "KeyC"], DT));
        peak = peak.max(pie.hero().vertical_speed);
    }
    assert!(
        peak > 1.0,
        "a dive at sprint did not launch: peak {peak} m/s"
    );
}

// ── 12–13. Space ────────────────────────────────────────────────────────────

#[test]
fn space_jumps() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    let before = pie.hero().position[1];
    pie.drive(InputFrame::held(["Space"], DT));
    let launched = pie.hero();
    assert!(
        launched.vertical_speed > 3.0,
        "Space did not launch: {} m/s",
        launched.vertical_speed
    );
    let mut peak = launched.position[1];
    for _ in 0..40 {
        pie.drive(InputFrame::held([] as [&str; 0], DT));
        peak = peak.max(pie.hero().position[1]);
    }
    assert!(
        peak - before > 0.5,
        "the hero rose only {:.4} m",
        peak - before
    );
}

#[test]
fn space_at_the_water_is_a_dive_and_not_a_jump() {
    // **Sprinting**, because that is the verb: a dive is a sprinting jump at
    // water, and a standing one at the same lake is still a jump.
    let mut wet = Pie::open(Fixture {
        lake: true,
        ..Default::default()
    });
    wet.settle();
    wet.hold(&["Shift"], 10);
    wet.drive(InputFrame::held(["Shift", "Space"], DT));
    let wet_hero = wet.hero();

    let mut dry = Pie::open(Fixture::default());
    dry.settle();
    dry.hold(&["Shift"], 10);
    dry.drive(InputFrame::held(["Shift", "Space"], DT));
    let dry_hero = dry.hero();

    assert!(
        wet_hero.local_velocity[1] > 1.0,
        "a dive at water carried no forward speed: {:?}",
        wet_hero.local_velocity
    );
    assert!(
        dry_hero.local_velocity[1].abs() < 0.01,
        "a dry jump carried forward speed: {:?}",
        dry_hero.local_velocity
    );
    assert!(
        dry_hero.vertical_speed > wet_hero.vertical_speed,
        "the dry jump did not go higher than the dive: {} vs {}",
        dry_hero.vertical_speed,
        wet_hero.vertical_speed
    );
}

// ── 14. E ───────────────────────────────────────────────────────────────────

#[test]
fn e_picks_up_the_item_in_front_of_the_character() {
    let mut pie = Pie::open(Fixture {
        pickup: true,
        ..Default::default()
    });
    pie.settle();
    let before = pie.probe();
    let bagged = |p: &inf_runtime::pie::ActorProbe| {
        p.bag
            .iter()
            .find(|(id, _)| id == BANDAGE)
            .map_or(0, |(_, n)| *n)
    };
    assert_eq!(
        bagged(before.hero.as_ref().expect("hero")),
        0,
        "the bandage was already in the bag"
    );
    pie.hold(&["KeyE"], 4);
    pie.hold(&[], 4);
    let after = pie.probe();
    assert_eq!(
        bagged(after.hero.as_ref().expect("hero")),
        1,
        "E did not pick the bandage up"
    );
    assert!(
        after.entities < before.entities,
        "the pickup entity is still in the world: {} -> {}",
        before.entities,
        after.entities
    );
}

// ── 15–18. the weapons ──────────────────────────────────────────────────────

const ARMED: Fixture = Fixture {
    lake: false,
    pickup: false,
    armed: true,
    wall: false,
};

#[test]
fn lmb_fires_the_equipped_weapon() {
    let mut pie = Pie::open(ARMED);
    pie.settle();
    let full = pie.hero().magazine;
    assert!(full > 0, "the rifle is not equipped: {full}");
    let mut fired = 0u32;
    for _ in 0..60 {
        pie.drive(InputFrame::held([] as [&str; 0], DT).with_buttons([0]));
        fired += pie.probe().shots;
    }
    let after = pie.hero();
    assert!(fired > 0, "the left button fired nothing");
    assert!(
        after.magazine < full,
        "the magazine did not empty: {full} -> {}",
        after.magazine
    );
}

#[test]
fn r_reloads_the_equipped_weapon() {
    let mut pie = Pie::open(ARMED);
    pie.settle();
    let full = pie.hero().magazine;
    for _ in 0..60 {
        pie.drive(InputFrame::held([] as [&str; 0], DT).with_buttons([0]));
    }
    let spent = pie.hero().magazine;
    assert!(
        spent < full,
        "nothing was fired, so nothing can be reloaded"
    );
    pie.hold(&["KeyR"], 4);
    pie.hold(&[], 240);
    assert_eq!(
        pie.hero().magazine,
        full,
        "R did not refill the magazine (was {spent})"
    );
}

#[test]
fn rmb_aims() {
    let mut pie = Pie::open(ARMED);
    pie.settle();
    let before = pie.hero().rotation_mode;
    for _ in 0..10 {
        pie.drive(InputFrame::held([] as [&str; 0], DT).with_buttons([1]));
    }
    let aiming = pie.hero().rotation_mode;
    pie.hold(&[], 10);
    let after = pie.hero().rotation_mode;
    assert_eq!(aiming, "Aiming", "RMB did not aim (was {before})");
    assert_ne!(after, "Aiming", "the aim did not release: {after}");
}

#[test]
fn the_wheel_changes_weapon() {
    let mut pie = Pie::open(ARMED);
    pie.settle();
    let first = pie.hero().equipped;
    assert_eq!(first, RIFLE, "the rifle is not equipped");
    pie.drive(InputFrame::held([] as [&str; 0], DT).with_wheel(0.0, 1.0));
    pie.hold(&[], 4);
    let moved = pie.hero().equipped;
    assert_eq!(moved, PISTOL, "the wheel did not change weapon");
    pie.drive(InputFrame::held([] as [&str; 0], DT).with_wheel(0.0, -1.0));
    pie.hold(&[], 4);
    assert_eq!(pie.hero().equipped, RIFLE, "the wheel did not come back");
}

// ── 19–20. the session's own two decisions ──────────────────────────────────

#[test]
fn i_opens_and_closes_the_inventory() {
    let mut pie = Pie::open(ARMED);
    pie.settle();
    assert!(!pie.probe().inventory_open);
    pie.hold(&["KeyI"], 2);
    pie.hold(&[], 2);
    assert!(pie.probe().inventory_open, "I did not open the panel");
    pie.hold(&["KeyI"], 2);
    pie.hold(&[], 2);
    assert!(!pie.probe().inventory_open, "I did not close the panel");
}

#[test]
fn tab_opens_the_in_game_settings_and_freezes_the_sim() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    let before = pie.probe();
    assert!(!before.menu_open);
    // Thirty frames with nothing held: thirty steps.
    pie.hold(&[], 30);
    let ran = pie.probe().steps - before.steps;
    assert_eq!(ran, 30, "thirty frames did not step thirty times");

    pie.hold(&["Tab"], 2);
    pie.hold(&[], 2);
    let opened = pie.probe();
    assert!(opened.menu_open, "Tab did not open the settings");
    pie.hold(&[], 60);
    assert_eq!(
        pie.probe().steps,
        opened.steps,
        "the simulation ran while the settings dialog was open"
    );
    // …and Tab closes it, which is what makes the freeze a state and not a stop.
    pie.hold(&["Tab"], 2);
    pie.hold(&[], 30);
    assert!(!pie.probe().menu_open, "Tab did not close the settings");
    assert!(
        pie.probe().steps > opened.steps,
        "the simulation did not resume"
    );
}

// ── the falsification the whole file rests on ───────────────────────────────

/// **Drop the input frame and the arms go red.**
///
/// The mutation this file is built to survive is the one that would make every
/// arm above vacuous: a player that ignored `Input` and stepped with an empty
/// one. It is measured rather than argued — the same session, driven with
/// `Step` (the protocol-2 door, which steps with `RuntimeInput::default()`)
/// instead of `Input`, must leave the hero exactly where it started while the
/// driven one must not.
#[test]
fn without_the_input_frame_the_hero_does_not_move() {
    isolate_settings();
    let payload = cached(Fixture::default());

    let mut driven = Pie::open(Fixture::default());
    driven.settle();
    let from = driven.hero().position;
    driven.hold(&["KeyW"], 120);
    let to = driven.hero().position;
    let moved = ((to[0] - from[0]).powi(2) + (to[2] - from[2]).powi(2)).sqrt();

    let mut stepped =
        PieSession::spawn_scene(&player_bin(), &payload).expect("the control player spawns");
    stepped.step(160).expect("step");
    let a = stepped
        .probe(None, PROBE_TIMEOUT)
        .expect("probe")
        .hero
        .expect("hero")
        .position;
    stepped.step(120).expect("step");
    let b = stepped
        .probe(None, PROBE_TIMEOUT)
        .expect("probe")
        .hero
        .expect("hero")
        .position;
    let drifted = ((b[0] - a[0]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
    let _ = stepped.stop(Duration::from_secs(5));

    println!("FIX1 driven {moved:.4} m / undriven {drifted:.4} m over 120 steps");
    assert!(
        moved > 5.0,
        "the driven hero moved {moved:.4} m — the input frame reached nothing"
    );
    assert!(
        drifted < 0.01,
        "the UNDRIVEN hero moved {drifted:.4} m, so the arms above prove nothing"
    );
}

// ── the pose, and the camera, in the same subprocess ────────────────────────

/// **The hero is not standing in its bind pose** (wave FIX1).
///
/// The author's report was *"the character is in T-Pose position and will not
/// move"*, and the brief's hypothesis was a resolver failure — no `AnimPlayer`,
/// no clip, or a clip that will not resolve. **The instrument says otherwise and
/// this arm is the instrument**: the machine resolves, the clips resolve, a full
/// 161-joint pose is published, and before this wave the pose it published was
/// the bind pose with a 5.6 mm hip bob on it, because `idle_clip` wrote two
/// tracks and `sample_clip` seeds every other joint from `local_bind`.
///
/// So the assertion is on the *content* of the pose, not on its size: every
/// "the pose store is N × 6 476 bytes" arm in this repository is green while the
/// character stands in a T. `pose_max_delta` is exactly `0.0` for the rest pose
/// by construction, so "not rest" is a bit and not a tolerance.
#[test]
fn the_hero_in_pie_is_posed_and_the_pose_follows_the_input() {
    let mut pie = Pie::open(Fixture::default());
    pie.settle();
    let idle = pie.hero();
    assert_eq!(
        idle.pose_joints, 161,
        "the machine published no pose at all: {idle:?}"
    );
    assert!(
        !idle.pose_is_rest,
        "after 40 steps the hero is drawn in its REST pose (delta {})",
        idle.pose_max_delta
    );

    // …and it CHANGES with input: a running character is not an idling one.
    pie.hold(&["KeyW"], 90);
    let running = pie.hero();
    println!(
        "FIX1 pose: idle delta {:.6} (rest {}), running delta {:.6} (rest {}), {} joints",
        idle.pose_max_delta,
        idle.pose_is_rest,
        running.pose_max_delta,
        running.pose_is_rest,
        running.pose_joints
    );
    assert!(!running.pose_is_rest, "{running:?}");
    assert!(
        (running.pose_max_delta - idle.pose_max_delta).abs() > 0.05,
        "the pose did not change with the input: idle {:.6} against running {:.6}",
        idle.pose_max_delta,
        running.pose_max_delta
    );
}

/// **The camera comes in rather than entering the wall** (clause 4b, in PIE).
///
/// `inf_physics::d3::camera` has swept a sphere from the pivot to the desired
/// camera position since P29.6 and pulled in on contact; what nobody had done is
/// watch it do so in a real session. This puts a 6 m wall two metres behind the
/// hero — where a third-person boom is — and reads `collision_pull_m`, the
/// sweep's own record of how far it came in, every step.
///
/// **The control is the same trace with no wall**, and it is what makes the
/// measurement mean something: on a flat floor the sweep meets nothing and the
/// pull-in is exactly zero, so a session that reported a pull-in there would be
/// reporting a bug rather than a camera. Measured, in that order.
#[test]
fn the_pie_camera_pulls_in_rather_than_entering_the_wall() {
    let sweep = |wall: bool| {
        let mut pie = Pie::open(Fixture {
            wall,
            ..Default::default()
        });
        pie.settle();
        let mut worst = 0.0f64;
        let mut min_eye_y = f64::MAX;
        let mut behind_wall = 0u32;
        for _ in 0..90 {
            pie.drive(InputFrame::held(["KeyS"], DT));
            let p = pie.probe();
            worst = worst.max(p.camera_pull_in_m);
            min_eye_y = min_eye_y.min(p.camera_eye[1]);
            // The wall's near face is at z = −1.5; the eye must never be past it.
            if wall && p.camera_eye[2] < -1.5 {
                behind_wall += 1;
            }
        }
        let end = pie.probe();
        let hero = end.hero.expect("hero");
        let boom = ((end.camera_eye[0] - hero.position[0]).powi(2)
            + (end.camera_eye[1] - hero.position[1]).powi(2)
            + (end.camera_eye[2] - hero.position[2]).powi(2))
        .sqrt();
        (worst, min_eye_y, boom, behind_wall, end.camera_eye)
    };

    let (walled, wall_eye_y, wall_boom, behind, wall_eye) = sweep(true);
    let (open, open_eye_y, open_boom, _, open_eye) = sweep(false);
    println!(
        "FIX1 camera: with a wall, worst pull-in {walled:.4} m, lowest eye y {wall_eye_y:.4} m, \
         final boom {wall_boom:.4} m, eye {wall_eye:?}; the CONTROL with no wall, pull-in \
         {open:.4} m, lowest eye y {open_eye_y:.4} m, boom {open_boom:.4} m, eye {open_eye:?}"
    );
    assert_eq!(
        behind, 0,
        "the camera was on the far side of the wall on {behind} of 90 steps"
    );
    assert!(
        walled > 0.05,
        "the sweep never fired against a wall two metres behind the hero: {walled:.4} m"
    );
    assert!(
        open < 1.0e-9,
        "the CONTROL pulled the camera in with nothing in the way: {open:.4} m"
    );
    assert!(
        wall_eye_y > 0.0 && open_eye_y > 0.0,
        "the camera went under the floor: {wall_eye_y:.4} / {open_eye_y:.4}"
    );
    assert!(
        open_boom > 0.5,
        "the camera collapsed onto the hero with nothing in the way: {open_boom:.4} m"
    );
}

// ── the determinism of the door this wave opened ────────────────────────────

/// **Two driven sessions, one input trace, the same simulation** (audit FIX1).
///
/// Protocol 3 opened a second way into `RuntimeSim::step_once` — the first since
/// P9.4 — and the wave armed what an input frame *does* without arming that it
/// does the same thing twice. Determinism is the house's oldest law and the
/// PIE == shipping doctrine rests on it: a driven trace that is not reproducible
/// cannot be compared with anything, so every control arm above would be
/// measuring one roll of a die.
///
/// The assertion is on the **state hash the player reports for each step**, in
/// order and to the bit, over a trace that exercises the parts most likely to
/// carry state across a frame: a gait change, a strafe, a jump, a mode change and
/// a release. Two real subprocesses, spawned from one payload, so a divergence in
/// process-local state (a hash-map iteration order, a clock read, an
/// uninitialised accumulator) is visible where a single-process arm would share
/// it and see nothing.
#[test]
fn one_input_trace_drives_two_sessions_to_the_same_simulation() {
    isolate_settings();
    /// The trace, as (held keys, how many steps).
    const TRACE: [(&[&str], u32); 7] = [
        (&[], 30),
        (&["KeyW"], 40),
        (&["KeyW", "Shift"], 40),
        (&["KeyW", "KeyA"], 30),
        (&["KeyW", "Space"], 20),
        (&["KeyW", "KeyC"], 30),
        (&[], 30),
    ];

    let run = || -> (Vec<u64>, [f64; 3], [f64; 2]) {
        let mut pie = Pie::open(Fixture::default());
        let mut hashes = Vec::new();
        for (keys, frames) in TRACE {
            for _ in 0..frames {
                pie.session
                    .input(InputFrame::held(keys.iter().copied(), DT))
                    .expect("the input frame lands");
                let ev = pie
                    .session
                    .wait_for(Duration::from_secs(10), |e| {
                        matches!(e, inf_runtime::pie::PlayerToEditor::Frame { .. })
                    })
                    .expect("the player reports the frame it stepped");
                if let inf_runtime::pie::PlayerToEditor::Frame { state_hash, .. } = ev {
                    hashes.push(state_hash);
                }
            }
        }
        let hero = pie.hero();
        (hashes.clone(), hero.position, hero.local_velocity)
    };

    let (a_hashes, a_pos, a_vel) = run();
    let (b_hashes, b_pos, b_vel) = run();

    let steps: u32 = TRACE.iter().map(|(_, n)| n).sum();
    println!(
        "FIX1 audit determinism — {steps} driven steps, {} hashes; \
         first {:#018x} last {:#018x}; hero ({:.6}, {:.6}, {:.6})",
        a_hashes.len(),
        a_hashes.first().copied().unwrap_or_default(),
        a_hashes.last().copied().unwrap_or_default(),
        a_pos[0],
        a_pos[1],
        a_pos[2]
    );
    assert_eq!(
        a_hashes.len(),
        steps as usize,
        "one Frame per driven step is the protocol's own shape"
    );
    // The hashes are the ordered claim; naming the first divergence is what makes
    // a failure diagnosable rather than a wall of u64s.
    let split = a_hashes
        .iter()
        .zip(&b_hashes)
        .position(|(x, y)| x != y)
        .unwrap_or(a_hashes.len());
    assert_eq!(
        split,
        a_hashes.len(),
        "two sessions driven by the same trace diverged at step {split}: \
         {:#018x} vs {:#018x}",
        a_hashes[split],
        b_hashes[split]
    );
    assert_eq!(a_hashes, b_hashes, "the driven state hashes differ");
    // …and the world the hashes summarise, read through the other door.
    assert_eq!(
        a_pos, b_pos,
        "the hero landed somewhere else the second time"
    );
    assert_eq!(a_vel, b_vel, "the hero's aim-frame velocity differs");
}
