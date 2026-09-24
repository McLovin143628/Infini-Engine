//! **WAVE VEH3e — VEHICLE AUDIO.** The gate.
//!
//! Every claim of the wave, each arm **mutation-verified** and each carrying an
//! **engagement count**, so "the arm ran" and "something happened" are two
//! different facts and the second one is asserted.
//!
//! # What every arm in this file reads
//!
//! The COMMAND STREAM the hosts queued — `audio_command_log`, sliced per step,
//! folded into what each voice key was last told — and, for a generated grain,
//! the committed clip's own BYTES through an independent WAV parser and an
//! independent period estimator. Never a report's opinion of itself, never the
//! planner's intent read back off its memory.
//!
//! # Would VEH1a's ONE pitched loop pass it?
//!
//! The audit brief's question for every arm. "fails" means the pre-wave fence
//! (one `Play` on the chassis's own key, then `SetPitch` + `SetVolume` +
//! `SetPosition` every step from `engine_cue`) reds it.
//!
//! | arm | reads | mutation that reds it | engagement | the old loop? |
//! |---|---|---|---|---|
//! | `the_stack_is_eight_loops_on_salted_keys_in_a_pinned_order` | the Plays at the engine's start: keys, clips, order (the rolling road the eighth, VEH3e audit) | two layers swapped in `car_loops`; two salts made equal | 8 Plays, 8 distinct keys | **fails** — one Play, on the chassis key |
//! | `the_grain_period_is_in_the_committed_bytes` | the COMMITTED `.inf_audio` bytes, parsed and measured here; then every emitted grain pitch x the measured period against `(rpm/60)(cyl/2)` | `GRAIN_REF_RPM` in `inf_ecs` moved off the baked 2400 | 15 clips, driven steps with a grain pitch | **fails** — its pitch is `0.65..2.05` of revs |
//! | `the_squeal_follows_slip_and_not_speed` | the squeal key's commands at two speeds with one slip, and over the course against `squeal_voice(slip)` | the squeal fed `slip x speed/10` | loud squeal steps below 5 m/s AND above 15 m/s | **fails** — no squeal key |
//! | `a_squeal_changes_its_clip_with_the_surface` | the squeal key's re-`Play` clip on the gravel strip | the surface voice ignored (always sealed) | re-Plays onto gravel | **fails** |
//! | `the_load_crossfade_moves_between_the_three_grains` | the three grain keys' volumes each step against `level x weights(load)` | `load_weights` pinned to the mid grain | steps at load 0, 1 and under a fuel cut | **fails** — one key |
//! | `the_whine_steps_down_on_every_upshift` | the whine key's pitch across each gear change | the whine pitched by the final drive alone | upshifts seen | **fails** — no whine key |
//! | `the_turbo_spools_and_blows_off_on_the_lift` | the turbo key's pitch vs boost; one blow-off Play on the lift | the blow-off edge deleted | boost steps; one lift | **fails** |
//! | `the_kerb_thumps_front_then_rear` | the impulse keys' Plays at the kerb | the rising-edge guard dropped (an impulse every step of a spike) | two thumps | **fails** |
//! | `a_parked_car_is_silent_until_somebody_gets_in` | the car keys before boarding, at entry, after exit | `running()` answering true | steps silent, Plays at entry, Stops at exit | **fails** — a Play on the first step it is seen |
//! | `the_door_slams_on_the_shut_step_and_the_motor_rows_are_silent` | the door keys on a RIGGED shipped host against `door_deg` and `handle_weight` | the handle gate dropped; the shut edge dropped | slam rows, hand rows, motor rows | **fails** — no door sound at all |
//! | `a_door_held_open_through_the_close_does_not_slam` (audit) | the slam key on a shipped exit whose door is held open through `ClosingDoor`, against a control that shuts; the joint's angle off the damage row | the end-step hinge read reverted to the machine's reset `door_deg` | one timed-out close, one shut close | **passes** (no door sound at all) -- the false slam was VEH3e's own |
//! | `the_course_render_does_not_clip` (audit) | the WAV BYTES of the shipped course rendered through the render-to-file door, with the master track and without it | the limiter taken off the master track | the control's peak within 1 dB of full scale | n/a -- a property of the mix, not the stream |
//! | `a_traffic_car_sings_the_near_stack_on_its_own_cadence` (audit) | the planner's cues for a traffic car beside an authored one: which keys Play, how often a loop is re-told, the grain pitch sent | `NEAR_EVERY` 1; the traffic test answering false | two Plays, >10 pitches | **fails** -- traffic was voiceless |
//! | `the_road_rolls_by_speed_on_its_surface` (audit) | the roll key's folded volume/pitch each step against `roll_voice(speed)`; its re-`Play` clip on gravel | `roll_voice` reading the throttle; the roll clip pinned to sealed | loud >15 m/s, silent stopped, a gravel re-Play | **fails** -- no roll key |
//! | `a_bail_out_lands_with_a_thud` | the thud key on a moving exit | the thud branch deleted | one bail | **fails** |
//! | `pie_equals_shipping_on_the_audio_course` | both hosts' per-step command slices | the planner called with `dt * 2.0` in one host | every kind of cue seen | passes — it compares, it does not judge |
//! | `the_audio_log_holds_the_drive_and_the_count_is_stated` | `dropped_audio_commands`, the per-step counts | `AUDIO_LOG_CAPACITY` cut to 4096 | the whole course | passes (3 a step) |
//! | `sixty_four_cars_cost_what_they_print` | the audio phase's own clock at 64 cars, 1 and 64 voiced, and the whole step against a voiceless control | n/a — a COST arm | cars voiced, commands a step | n/a |
//! | `a_cooked_pack_carries_every_vehicle_clip` | the cooked pack's `.inf_audio` index | the engine-clip closure deleted from the cook | 31 clips | **fails** — none in the pack |
//! | `the_planner_is_a_function_of_the_stream_so_far` | two runs of the course; the planner's source | a clock or an RNG in `vehicle_audio.rs` | whole course | passes |
//! | `the_shipped_host_draws_the_audio_row_and_logs_the_columns` | `window.rs` / `pie_drive.rs` source + the Ring-0 row | the call deleted | one row | n/a |

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use glam::DVec3;
use uuid::Uuid;

use inf_audio::AudioCommand;
use inf_ecs::boarding::{BoardPhase, DOOR_SHUT_DEG};
use inf_ecs::components::{
    AudioSource, BodyKind3D, CharacterController3D, CharacterMovement, Collider3D,
    ColliderShape3DKind, MovementMode, RigidBody3D, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{SurfaceClass, VehicleControls, VehicleDef};
use inf_ecs::vehicle_audio::{
    self as va, AxleVoice, DoorLayer, EngineLoad, GrainFamily, SurfaceVoice, VoiceLayer,
    VoiceMemory, VoiceTelemetry,
};
use inf_ecs::EcsWorld;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;
const RADIUS: f64 = 0.3;
const CHASSIS: Uuid = Uuid::from_u128(0x5E3E_0001);
const GROUND: Uuid = Uuid::from_u128(0x5E3E_0002);
const HERO: Uuid = Uuid::from_u128(0x5E3E_0003);
const KERB: Uuid = Uuid::from_u128(0x5E3E_0004);
const GRAVEL: Uuid = Uuid::from_u128(0x5E3E_0005);
const SKEL_GUID: Uuid = Uuid::from_u128(0x5E3E_0006);
const SM_GUID: Uuid = Uuid::from_u128(0x5E3E_0007);
/// Beside the car's driver's (`+X`) flank, a couple of metres back.
const HERO_AT: DVec3 = DVec3::new(2.6, 0.0, -1.6);
/// Where the kerb crosses the road, metres up `+Z` from the car.
const KERB_Z: f64 = 70.0;
/// Its height, metres.
const KERB_H: f64 = 0.12;

// ── the fixture ─────────────────────────────────────────────────────────────

fn catalogue_def(id: &str) -> VehicleDef {
    *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .unwrap_or_else(|| panic!("the catalogue has no `{id}` row"))
}

/// **The course car**: the catalogue's sports row, re-voiced as a turbocharged
/// cross-plane V8 — every change through `VehicleClass::set`, the door a roster
/// row will use.
fn course_def() -> VehicleDef {
    let mut def = catalogue_def("sports");
    for (k, v) in [
        ("cylinders", 8.0),
        ("engine_voice_kind", 0.0),
        ("firing_order_variant", 1.0),
        ("turbo_boost_max", 0.8),
    ] {
        assert!(def.class.set(k, v), "the class has no `{k}`");
    }
    def
}

type SlabRow = (Uuid, DVec3, DVec3, f64);

/// The ground, the kerb and a gravel strip.
fn course_slabs() -> Vec<SlabRow> {
    vec![
        (
            GROUND,
            DVec3::new(0.0, -0.5, 100.0),
            DVec3::new(120.0, 0.5, 220.0),
            0.9,
        ),
        (
            KERB,
            DVec3::new(0.0, KERB_H * 0.5, KERB_Z),
            DVec3::new(20.0, KERB_H * 0.5, 0.25),
            0.9,
        ),
        (
            GRAVEL,
            DVec3::new(0.0, 0.0025, 150.0),
            DVec3::new(40.0, 0.0025, 60.0),
            SurfaceClass::Gravel.dry_mu(),
        ),
    ]
}

fn slab_bits(
    centre: DVec3,
    half: DVec3,
    friction: f64,
) -> (Transform, Visibility, RigidBody3D, Collider3D) {
    (
        Transform {
            translation: Vec3d::from_dvec3(centre),
            ..Default::default()
        },
        Visibility::default(),
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::from_dvec3(half),
            friction,
            ..Default::default()
        },
    )
}

fn hero_bits() -> (
    Transform,
    RigidBody3D,
    Collider3D,
    CharacterController3D,
    CharacterMovement,
) {
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(HERO_AT.x, cm.stand_half_height_m + RADIUS, HERO_AT.z);
    (
        t,
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
    )
}

fn car_at(def: &VehicleDef) -> DVec3 {
    DVec3::new(
        0.0,
        inf_ecs::vehicle::resting_origin_y(def, 0.0) + 0.15,
        0.0,
    )
}

fn spawn_car(world: &mut EcsWorld, guid: Uuid, at: DVec3, def: &VehicleDef, voice: bool) {
    let spawn = inf_ecs::vehicle::RigSpawn {
        name: "Car".into(),
        at,
        yaw_deg: 0.0,
        paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
        clip: None,
        engine_voice: voice,
        livery: None,
    };
    inf_ecs::vehicle::spawn_rig(world, guid, def, &spawn);
}

fn shipped_world(def: &VehicleDef) -> EcsWorld {
    let mut world = EcsWorld::new();
    for (g, c, h, mu) in course_slabs() {
        let e = world.spawn_with_guid(g, "Slab", None);
        world.world_mut().entity_mut(e).insert(slab_bits(c, h, mu));
    }
    spawn_car(&mut world, CHASSIS, car_at(def), def, true);
    let h = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(h).insert(hero_bits());
    world.propagate();
    world
}

/// The same course as a shipped host with a SKELETON on the hero, so the
/// boarding's hand pass runs and `handle_weight` is a real request — the
/// VEH3d gate's `rigged_runtime_sim`, on this course.
fn rigged_sim(def: &VehicleDef) -> inf_player::runtime_sim::RuntimeSim {
    use inf_player::runtime_sim::RuntimeSim;
    const IDLE: inf_anim::ClipRef = [0xe3; 16];
    let mut world = shipped_world(def);
    let e = world.entity_of(HERO).expect("the hero");
    world.world_mut().entity_mut(e).insert((
        inf_ecs::components::AnimStateMachine {
            sm: Some(SM_GUID),
            ..Default::default()
        },
        inf_ecs::components::SkeletalMesh {
            mesh: None,
            skeleton: Some(SKEL_GUID),
        },
    ));
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
    let skeleton = inf_anim::build_template(
        inf_anim::BodyPlan::Biped,
        &inf_anim::BodyParams {
            height_m: 1.8,
            ..Default::default()
        },
    )
    .expect("the mannequin builds");
    sim.set_skeletons([(SKEL_GUID, skeleton)].into_iter().collect());
    sim.set_state_machines(
        [(
            SM_GUID,
            inf_anim::StateMachine {
                states: vec![inf_anim::SmState::clip("idle", IDLE)],
                entry: 0,
                ..Default::default()
            },
        )]
        .into_iter()
        .collect(),
    );
    sim.set_pose_clips(
        [(
            Uuid::from_bytes(IDLE),
            inf_anim::AnimClip::new("idle", Vec::new()),
        )]
        .into_iter()
        .collect(),
    );
    sim
}

// ── the course ──────────────────────────────────────────────────────────────

/// What the driver does on one step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Intent {
    press: bool,
    throttle: f32,
    steer: f32,
    handbrake: bool,
}

impl Intent {
    fn throttle(v: f32) -> Self {
        Self {
            throttle: v,
            ..Default::default()
        }
    }
    fn press() -> Self {
        Self {
            press: true,
            ..Default::default()
        }
    }
}

/// **The drive**, `k` steps after the hero reached the wheel: sit and idle a
/// second; a full-throttle launch (the tyres let go off the line — the
/// burnout) up through the gears and over the kerb; lift (the blow-off); a
/// handbrake turn at speed (the slide), sliding onto the gravel; hold the
/// handbrake to a stop; get out; the door is shut behind.
fn course(k: u32) -> Intent {
    match k {
        0..=59 => Intent::default(),
        60..=299 => Intent::throttle(1.0),
        300..=329 => Intent::default(),
        330..=389 => Intent {
            steer: 1.0,
            handbrake: true,
            ..Default::default()
        },
        390..=599 => Intent {
            handbrake: true,
            ..Default::default()
        },
        640 => Intent::press(),
        _ => Intent::default(),
    }
}

/// Steps the course runs for after the hero is at the wheel.
const DRIVE_STEPS: u32 = 780;
/// The step the E is pressed on to board.
const BOARD_PRESS: u32 = 30;
/// A ceiling on the boarding, steps.
const BOARD_MAX: u32 = 900;

/// **One step of a host's record.**
#[derive(Clone, Debug)]
struct Step {
    /// The commands the step queued, in order.
    cmds: Vec<AudioCommand>,
    /// The voice the vehicle door published for the car, if any.
    voice: Option<VoiceTelemetry>,
    /// The hero's boarding: phase, hinge degrees, handle weight.
    board: (BoardPhase, f64, f64),
    /// The hero's movement mode.
    mode: MovementMode,
    /// Steps since the hero reached the wheel, or `None` before.
    k: Option<u32>,
}

fn board_of(world: &EcsWorld) -> (BoardPhase, f64, f64, bool, MovementMode) {
    let e = world.entity_of(HERO).expect("the hero");
    let cm = world.world().get::<CharacterMovement>(e).expect("a mover");
    let b = cm.runtime.boarding;
    (
        b.phase,
        b.door_deg,
        b.handle_weight,
        cm.runtime.seat.is_seated(),
        cm.mode,
    )
}

fn at_wheel(phase: BoardPhase, seated: bool) -> bool {
    phase == BoardPhase::Driving || (phase == BoardPhase::Idle && seated)
}

/// A host, stepped by the same script.
trait Host {
    fn step(&mut self, i: Intent);
    fn log(&self) -> &[AudioCommand];
    fn dropped(&self) -> u64;
    fn world(&self) -> &EcsWorld;
    fn voice(&self) -> Option<VoiceTelemetry>;
}

struct Shipped(inf_player::runtime_sim::RuntimeSim);

impl Host for Shipped {
    fn step(&mut self, i: Intent) {
        use inf_ecs::movement::actions::{HANDBRAKE, INTERACT, MOVE_X, MOVE_Y};
        use inf_player::runtime_sim::RuntimeInput;
        let mut input = RuntimeInput::default();
        if i.press {
            input = input.press(INTERACT);
        }
        if i.handbrake {
            input = input.press(HANDBRAKE);
        }
        if i.throttle != 0.0 {
            input = input.axis_at(MOVE_Y, i.throttle);
        }
        if i.steer != 0.0 {
            input = input.axis_at(MOVE_X, i.steer);
        }
        self.0.step_once(input);
    }
    fn log(&self) -> &[AudioCommand] {
        self.0.audio_command_log()
    }
    fn dropped(&self) -> u64 {
        self.0.dropped_audio_commands()
    }
    fn world(&self) -> &EcsWorld {
        self.0.world()
    }
    fn voice(&self) -> Option<VoiceTelemetry> {
        self.0
            .vehicles()
            .iter()
            .find(|o| o.chassis == CHASSIS)
            .and_then(|o| o.voice)
    }
}

struct Preview {
    doc: inf_editor_core::scene::SceneDoc,
    session: inf_editor_core::simulate::SimSession,
}

impl Preview {
    fn new(def: &VehicleDef) -> Self {
        use inf_editor_core::ipc::SpawnKind;
        use inf_editor_core::scene::SceneDoc;
        use inf_editor_core::simulate::SimSession;
        let mut doc = SceneDoc::new();
        for (g, c, h, mu) in course_slabs() {
            let e = doc.create_with_guid(g, SpawnKind::Empty, "Slab", None);
            doc.world_mut()
                .world_mut()
                .entity_mut(e)
                .insert(slab_bits(c, h, mu));
        }
        inf_editor_core::vehicle::spawn_vehicle(
            &mut doc,
            CHASSIS,
            def,
            inf_editor_core::vehicle::VehicleSpawn {
                name: "Car",
                at: car_at(def),
                yaw_deg: 0.0,
                paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
                clip: None,
                engine_voice: true,
                livery: None,
            },
        );
        let h = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
        doc.world_mut()
            .world_mut()
            .entity_mut(h)
            .insert(hero_bits());
        doc.world_mut().propagate();
        let session = SimSession::enter(&mut doc, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
        Self { doc, session }
    }
}

impl Host for Preview {
    fn step(&mut self, i: Intent) {
        use inf_ecs::movement::actions::{HANDBRAKE, INTERACT, MOVE_X, MOVE_Y};
        use inf_editor_core::simulate::SimInput;
        let mut down: Vec<&str> = Vec::new();
        if i.press {
            down.push(INTERACT);
        }
        if i.handbrake {
            down.push(HANDBRAKE);
        }
        let mut axes: BTreeMap<String, f32> = BTreeMap::new();
        if i.throttle != 0.0 {
            axes.insert(MOVE_Y.to_string(), i.throttle);
        }
        if i.steer != 0.0 {
            axes.insert(MOVE_X.to_string(), i.steer);
        }
        self.session
            .step_once(&mut self.doc, SimInput::with_down(down).with_axes(axes));
    }
    fn log(&self) -> &[AudioCommand] {
        self.session.audio_command_log()
    }
    fn dropped(&self) -> u64 {
        self.session.dropped_audio_commands()
    }
    fn world(&self) -> &EcsWorld {
        self.doc.world()
    }
    fn voice(&self) -> Option<VoiceTelemetry> {
        self.session
            .vehicles()
            .iter()
            .find(|o| o.chassis == CHASSIS)
            .and_then(|o| o.voice)
    }
}

/// Board, run `drive` from the wheel for `steps`, and record every step.
fn run(host: &mut dyn Host, drive: fn(u32) -> Intent, steps: u32) -> Vec<Step> {
    let mut out = Vec::new();
    let mut wheel: Option<u32> = None;
    for step in 0..(BOARD_MAX + steps) {
        let k = wheel.map(|w| step - w);
        if k.is_some_and(|k| k >= steps) {
            break;
        }
        let i = match k {
            None if step == BOARD_PRESS => Intent::press(),
            None => Intent::default(),
            Some(k) => drive(k),
        };
        let before = host.log().len();
        host.step(i);
        assert_eq!(host.dropped(), 0, "the audio log evicted at step {step}");
        let (phase, deg, hw, seated, mode) = board_of(host.world());
        out.push(Step {
            cmds: host.log()[before..].to_vec(),
            voice: host.voice(),
            board: (phase, deg, hw),
            mode,
            k,
        });
        if wheel.is_none() && at_wheel(phase, seated) {
            wheel = Some(step + 1);
        }
    }
    assert!(
        wheel.is_some(),
        "the hero never reached the wheel, so the course is a parked car"
    );
    out
}

fn shipped_course() -> Vec<Step> {
    let mut host = Shipped(inf_player::runtime_sim::RuntimeSim::new(
        shipped_world(&course_def()),
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        HZ,
    ));
    run(&mut host, course, DRIVE_STEPS)
}

// ── reading the stream ──────────────────────────────────────────────────────

/// Every key the car's voice stack plays on.
fn car_keys() -> BTreeMap<u64, VoiceLayer> {
    let key = va::entity_key(CHASSIS);
    VoiceLayer::ALL
        .iter()
        .map(|l| (va::voice_key(key, *l), *l))
        .collect()
}

fn door_keys() -> BTreeMap<u64, DoorLayer> {
    let key = va::entity_key(HERO);
    [
        DoorLayer::Latch,
        DoorLayer::Creak,
        DoorLayer::Slam,
        DoorLayer::Thud,
    ]
    .iter()
    .map(|d| (va::door_key(key, *d), *d))
    .collect()
}

fn source_of(c: &AudioCommand) -> Option<u64> {
    match c {
        AudioCommand::Play(p) => Some(p.source),
        AudioCommand::Stop { source }
        | AudioCommand::SetVolume { source, .. }
        | AudioCommand::SetPitch { source, .. }
        | AudioCommand::SetPosition { source, .. }
        | AudioCommand::SetOcclusion { source, .. } => Some(*source),
        AudioCommand::SetListener(_) => None,
    }
}

/// **What the mixer holds for one key** after a step: the fold of every
/// command that addressed it so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Voice {
    clip: Uuid,
    volume: f64,
    pitch: f64,
    playing: bool,
    looping: bool,
}

/// Fold the stream per step into each key's state.
fn fold(steps: &[Step]) -> Vec<BTreeMap<u64, Voice>> {
    let mut state: BTreeMap<u64, Voice> = BTreeMap::new();
    let mut out = Vec::with_capacity(steps.len());
    for s in steps {
        for c in &s.cmds {
            match c {
                AudioCommand::Play(p) => {
                    state.insert(
                        p.source,
                        Voice {
                            clip: p.clip,
                            volume: p.volume,
                            pitch: p.pitch,
                            playing: true,
                            looping: p.looping,
                        },
                    );
                }
                AudioCommand::Stop { source } => {
                    state.remove(source);
                }
                AudioCommand::SetVolume { source, volume } => {
                    if let Some(v) = state.get_mut(source) {
                        v.volume = *volume;
                    }
                }
                AudioCommand::SetPitch { source, pitch } => {
                    if let Some(v) = state.get_mut(source) {
                        v.pitch = *pitch;
                    }
                }
                _ => {}
            }
        }
        out.push(state.clone());
    }
    out
}

fn key(layer: VoiceLayer) -> u64 {
    va::voice_key(va::entity_key(CHASSIS), layer)
}

fn plays_on(s: &Step, source: u64) -> Vec<&inf_audio::PlayCommand> {
    s.cmds
        .iter()
        .filter_map(|c| match c {
            AudioCommand::Play(p) if p.source == source => Some(p),
            _ => None,
        })
        .collect()
}

fn car_cmds(s: &Step) -> usize {
    let keys = car_keys();
    s.cmds
        .iter()
        .filter(|c| source_of(c).is_some_and(|k| keys.contains_key(&k)))
        .count()
}

// ── 1. THE LAYER SET ────────────────────────────────────────────────────────

/// **SEVEN LOOPS ON SALTED KEYS, IN A PINNED ORDER.** The engine starts as
/// the driver climbs in: one `Play` per loop — three grains of the car's own
/// family, the whine, the turbo, a squeal per axle — on seven distinct keys,
/// none of them the chassis's own, in the planner's order. Nothing ever
/// addresses the bare chassis key.
///
/// Also prints **the per-step command-count table** for the one hero car.
///
/// **Mutations → red**: two layers swapped in `car_loops`; two
/// `VOICE_SALTS` made equal (one key, so one Play replaces another).
#[test]
fn the_stack_is_eight_loops_on_salted_keys_in_a_pinned_order() {
    let steps = shipped_course();
    let start = steps
        .iter()
        .position(|s| plays_on(s, key(VoiceLayer::GrainIdle)).len() == 1)
        .expect("the engine never started");
    let want = [
        (
            VoiceLayer::GrainIdle,
            va::grain_clip(GrainFamily::P8Cross, EngineLoad::Idle),
        ),
        (
            VoiceLayer::GrainMid,
            va::grain_clip(GrainFamily::P8Cross, EngineLoad::Mid),
        ),
        (
            VoiceLayer::GrainFull,
            va::grain_clip(GrainFamily::P8Cross, EngineLoad::Full),
        ),
        (VoiceLayer::Whine, va::whine_clip()),
        (VoiceLayer::Turbo, va::turbo_clip()),
        (
            VoiceLayer::SquealFront,
            va::squeal_clip(SurfaceVoice::Sealed),
        ),
        (
            VoiceLayer::SquealRear,
            va::squeal_clip(SurfaceVoice::Sealed),
        ),
        (VoiceLayer::Roll, va::roll_clip(SurfaceVoice::Sealed)),
    ];
    let plays: Vec<&inf_audio::PlayCommand> = steps[start]
        .cmds
        .iter()
        .filter_map(|c| match c {
            AudioCommand::Play(p) if car_keys().contains_key(&p.source) => Some(p),
            _ => None,
        })
        .collect();
    println!(
        "the engine starts on step {start} ({:?}): {} Plays",
        steps[start].board.0,
        plays.len()
    );
    assert_eq!(plays.len(), want.len(), "the stack is eight loops");
    let mut distinct = BTreeSet::new();
    for (p, (layer, clip)) in plays.iter().zip(want) {
        println!(
            "  {layer:?}: key {:016x} clip {} vol {:.3} pitch {:.3} looping {}",
            p.source, p.clip, p.volume, p.pitch, p.looping
        );
        assert_eq!(p.source, key(layer), "{layer:?} is not on its salted key");
        assert_eq!(p.clip, clip, "{layer:?} plays the wrong clip");
        assert!(p.looping, "{layer:?} is a loop");
        distinct.insert(p.source);
    }
    assert_eq!(distinct.len(), 8, "two loops share a key");
    let bare = va::entity_key(CHASSIS);
    let on_bare = steps
        .iter()
        .flat_map(|s| s.cmds.iter())
        .filter(|c| source_of(c) == Some(bare))
        .count();
    assert_eq!(on_bare, 0, "a command addressed the chassis's own key");

    // THE PER-STEP COUNT TABLE, for the one hero car, by what it is doing.
    let mut rows: BTreeMap<&str, (usize, usize, usize)> = BTreeMap::new();
    for s in &steps {
        let Some(v) = s.voice else { continue };
        let what = match s.k {
            None => "parked / boarding",
            Some(_) if !v.running() => "engine off",
            Some(k) if (60..300).contains(&k) && v.speed_mps < 4.0 => "launch (burnout)",
            Some(k) if (60..300).contains(&k) => "full throttle",
            Some(k) if (300..330).contains(&k) => "lift / coast",
            Some(k) if (330..390).contains(&k) => "handbrake slide",
            Some(k) if (390..600).contains(&k) => "sliding to a stop",
            _ => "idle / stopped",
        };
        let n = car_cmds(s);
        let r = rows.entry(what).or_insert((0, 0, usize::MAX));
        r.0 += 1;
        r.1 += n;
        r.2 = r.2.min(n);
    }
    println!("\nPER-STEP COMMAND COUNT, one hero car (car keys only):");
    println!("  {:20} {:>6} {:>8} {:>6}", "state", "steps", "mean", "min");
    for (what, (n, sum, min)) in &rows {
        println!(
            "  {what:20} {n:6} {:8.2} {min:6}",
            *sum as f64 / (*n).max(1) as f64
        );
    }
    let max = steps.iter().map(car_cmds).max().unwrap_or(0);
    println!("  worst single step: {max} commands");
    assert!(
        max <= 8 * 3 + 3,
        "a step queued {max} car commands; eight loops x three plus three one-shots is the ceiling"
    );
}

// ── 2. THE GRAINS ───────────────────────────────────────────────────────────

/// Parse a WAV's first `data` chunk as 16-bit mono PCM — HERE, not through the
/// generator or kira, so the bytes are read by something that did not write
/// them.
fn wav_pcm(bytes: &[u8]) -> (u32, Vec<f64>) {
    assert_eq!(&bytes[0..4], b"RIFF");
    assert_eq!(&bytes[8..12], b"WAVE");
    let mut i = 12;
    let mut rate = 0u32;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32::from_le_bytes(bytes[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = &bytes[i + 8..(i + 8 + len).min(bytes.len())];
        if id == b"fmt " {
            assert_eq!(u16::from_le_bytes([body[2], body[3]]), 1, "mono");
            rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
            assert_eq!(u16::from_le_bytes([body[14], body[15]]), 16, "16-bit");
        }
        if id == b"data" {
            let pcm = body
                .chunks_exact(2)
                .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])) / 32_768.0)
                .collect();
            return (rate, pcm);
        }
        i += 8 + len + (len & 1);
    }
    panic!("no data chunk");
}

/// The WAV inside a committed `.inf_audio`: the `RIFF` magic, found.
fn committed_wav(file: &Path) -> Vec<u8> {
    let raw = std::fs::read(file).unwrap_or_else(|e| panic!("{}: {e}", file.display()));
    let at = raw
        .windows(4)
        .position(|w| w == b"RIFF")
        .unwrap_or_else(|| panic!("{} holds no RIFF", file.display()));
    raw[at..].to_vec()
}

/// **An independent period estimator**: rectify, smooth with a 2 ms boxcar,
/// remove the mean, and take the FIRST lag from 1 to 40 ms that is a local
/// maximum of the circular autocorrelation within 60 % of the window's best —
/// written here so the gate does not grade the generator with the generator's
/// own `grain_period_s`.
fn period_off_the_bytes(rate: u32, pcm: &[f64]) -> f64 {
    let n = pcm.len();
    let w = (0.002 * f64::from(rate)) as usize;
    let rect: Vec<f64> = pcm.iter().map(|s| s.abs()).collect();
    let env: Vec<f64> = (0..n)
        .map(|i| (0..w).map(|j| rect[(i + n - j) % n]).sum::<f64>() / w as f64)
        .collect();
    let mean = env.iter().sum::<f64>() / n as f64;
    let e: Vec<f64> = env.iter().map(|v| v - mean).collect();
    let r = |lag: usize| -> f64 { (0..n).map(|i| e[i] * e[(i + lag) % n]).sum() };
    let lo = (0.001 * f64::from(rate)) as usize;
    let hi = (0.040 * f64::from(rate)) as usize;
    let rs: Vec<f64> = (lo - 1..=hi + 1).map(r).collect();
    let at = |l: usize| rs[l + 1 - lo];
    let best = (lo..=hi).map(at).fold(f64::MIN, f64::max);
    let lag = (lo..=hi)
        .find(|&l| at(l) >= 0.6 * best && at(l) >= at(l - 1) && at(l) >= at(l + 1))
        .expect("a repeat");
    let (a, b, c) = (at(lag - 1), at(lag), at(lag + 1));
    let d = a - 2.0 * b + c;
    let shift = if d.abs() > 1e-18 {
        (0.5 * (a - c) / d).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    (lag as f64 + shift) / f64::from(rate)
}

/// **THE FIRING PERIOD IS IN THE COMMITTED BYTES, AND THE PITCH COMMAND
/// DELIVERS `(rpm/60)·(cyl/2)`.**
///
/// Half one: each of the fifteen committed grain files is opened, its WAV
/// parsed here, and its period measured here, against `1 / ((2400/60) ·
/// (cyl/2))` seconds — so a four and a V8 at the same reference differ by exactly 2:1
/// in their BYTES. Half two: on every driven step of the course, the firing
/// rate the listener hears — the grain's measured frequency times the pitch the
/// host SENT — is the firing rate of the telemetry's own rpm, to 1 %.
///
/// **Mutation → red**: `inf_ecs::vehicle_audio::GRAIN_REF_RPM` moved to 3000
/// (the bytes still say 2400, so half two is off by 25 %).
#[test]
fn the_grain_period_is_in_the_committed_bytes() {
    let dir = inf_editor_core::samples::vehicle_audio_dir();
    assert_eq!(
        va::VEHICLE_CLIP_NAMES,
        inf_audio::vehicle_synth::VEHICLE_CLIP_NAMES,
        "the engine and the generator number the clips differently"
    );
    assert_eq!(va::GRAIN_REF_RPM, inf_audio::vehicle_synth::GRAIN_REF_RPM);
    println!("GRAIN PERIODS, measured off the committed bytes:");
    println!(
        "  {:18} {:>5} {:>7} {:>10} {:>10} {:>8}",
        "file", "cyl", "bytes", "period ms", "expect ms", "err %"
    );
    let mut period_of: BTreeMap<(GrainFamily, EngineLoad), f64> = BTreeMap::new();
    for family in GrainFamily::ALL {
        for load in EngineLoad::ALL {
            let idx = (family.index() * 3 + load.index()) as usize;
            let name = va::VEHICLE_CLIP_NAMES[idx];
            let file = dir.join(format!("Vehicle_{name}.inf_audio"));
            let wav = committed_wav(&file);
            let (rate, pcm) = wav_pcm(&wav);
            assert_eq!(rate, 22_050);
            let p = period_off_the_bytes(rate, &pcm);
            // The firing PERIOD is the reciprocal of the firing RATE
            // `(rpm / 60) * (cyl / 2)`, in seconds. (The audit brief writes it
            // `60 / ((rpm/60)(cyl/2))`, which is a rate's reciprocal times a
            // stray sixty: 0.75 s for a four at 2 400 rpm, not 12.5 ms.)
            let expect = 1.0 / ((va::GRAIN_REF_RPM / 60.0) * (family.cylinders() / 2.0));
            let err = 100.0 * (p - expect) / expect;
            println!(
                "  {name:18} {:5} {:7} {:10.4} {:10.4} {err:8.3}",
                family.cylinders(),
                std::fs::metadata(&file).unwrap().len(),
                p * 1000.0,
                expect * 1000.0
            );
            assert!(err.abs() < 1.0, "{name}: {p} s against {expect}");
            period_of.insert((family, load), p);
        }
    }
    let four = period_of[&(GrainFamily::P4, EngineLoad::Mid)];
    let eight = period_of[&(GrainFamily::P8Cross, EngineLoad::Mid)];
    println!(
        "  a four against a V8 at the same revs: {:.4}",
        four / eight
    );
    assert!((four / eight - 2.0).abs() < 0.02);

    // Half two, off the shipped host's stream.
    let steps = shipped_course();
    let states = fold(&steps);
    let (mut checked, mut worst) = (0usize, 0.0f64);
    for (s, st) in steps.iter().zip(&states) {
        let Some(v) = s.voice else { continue };
        if !v.running() || v.rpm < 1_000.0 {
            continue;
        }
        // Every AUDIBLE grain: a silent one is not re-pitched (the planner
        // does not address a voice nobody can hear), so its folded pitch is
        // the last one it was sent, from a step when it was heard.
        for (load, layer) in [
            (EngineLoad::Idle, VoiceLayer::GrainIdle),
            (EngineLoad::Mid, VoiceLayer::GrainMid),
            (EngineLoad::Full, VoiceLayer::GrainFull),
        ] {
            let Some(g) = st.get(&key(layer)).filter(|g| g.volume > 0.0) else {
                continue;
            };
            let heard_hz = g.pitch / period_of[&(GrainFamily::P8Cross, load)];
            let firing_hz = (v.rpm / 60.0) * (v.cylinders / 2.0);
            worst = worst.max((heard_hz - firing_hz).abs() / firing_hz);
            checked += 1;
        }
    }
    println!(
        "  {checked} driven steps: the heard firing rate against (rpm/60)(cyl/2), worst {:.4} %",
        worst * 100.0
    );
    assert!(checked > 200, "only {checked} steps had a running grain");
    assert!(
        worst < 0.01,
        "the heard firing rate is {:.2} % off",
        worst * 100.0
    );
}

// ── 3. THE SQUEAL ───────────────────────────────────────────────────────────

fn telemetry(slip: f64, speed: f64, surface: SurfaceClass) -> VoiceTelemetry {
    VoiceTelemetry {
        rpm: 3_000.0,
        idle_rpm: 900.0,
        redline_rpm: 7_600.0,
        throttle: 0.3,
        boost: 0.0,
        turbocharged: false,
        fuel_cut: false,
        gear: 2,
        shaft_rpm: 3_000.0,
        cylinders: 8.0,
        voice_kind: 0.0,
        firing_order: 1.0,
        occupied: true,
        quiet: false,
        speed_mps: speed,
        axles: [
            AxleVoice {
                slip: 0.0,
                surface,
                compression_m: 0.05,
                grounded: true,
            },
            AxleVoice {
                slip,
                surface,
                compression_m: 0.05,
                grounded: true,
            },
        ],
    }
}

/// A world with one voiced car standing still, for driving the planner
/// directly with made-up telemetry.
fn planner_world() -> EcsWorld {
    let mut world = EcsWorld::new();
    spawn_car(
        &mut world,
        CHASSIS,
        DVec3::new(0.0, 1.0, 0.0),
        &course_def(),
        true,
    );
    world.propagate();
    world
}

/// The squeal commands one planner step answers for `t`, after a first step at
/// zero slip so the loops exist.
fn squeal_cmds(t: VoiceTelemetry) -> Vec<va::VoiceCue> {
    let world = planner_world();
    let mut mem = VoiceMemory::new();
    let mut quiet = t;
    quiet.axes_zero();
    let _ = mem.plan(&world, &[(CHASSIS, quiet)], DT);
    mem.plan(&world, &[(CHASSIS, t)], DT)
        .into_iter()
        .filter(|c| c.source() == key(VoiceLayer::SquealRear))
        .collect()
}

trait AxesZero {
    fn axes_zero(&mut self);
}

impl AxesZero for VoiceTelemetry {
    fn axes_zero(&mut self) {
        for a in &mut self.axles {
            a.slip = 0.0;
        }
    }
}

/// **THE SQUEAL FOLLOWS SLIP AND NOT SPEED** — the research doc's own test: a
/// car sliding at 10 km/h squeals exactly as a car sliding at 100 km/h does at
/// the same slip.
///
/// Half one drives the shipped PLANNER with the two telemetries (identical but
/// for `speed_mps`) and compares the squeal key's commands. Half two reads the
/// course's own stream: on every step, the rear squeal key's volume is
/// `squeal_voice(slip)` of that step's own slip — and the loud steps span a
/// launch below 5 m/s and a handbrake slide above 15 m/s.
///
/// **Mutation → red**: the planner feeding `squeal_voice(a.slip *
/// t.speed_mps.abs() / 10.0)`.
#[test]
fn the_squeal_follows_slip_and_not_speed() {
    let slow = squeal_cmds(telemetry(1.6, 10.0 / 3.6, SurfaceClass::Asphalt));
    let fast = squeal_cmds(telemetry(1.6, 100.0 / 3.6, SurfaceClass::Asphalt));
    println!("slip 1.6 at 10 km/h: {slow:?}");
    println!("slip 1.6 at 100 km/h: {fast:?}");
    assert!(!slow.is_empty(), "the squeal said nothing at slip 1.6");
    assert_eq!(slow, fast, "the squeal is a function of speed");
    let (vol, _) = va::squeal_voice(1.6);
    assert!(vol > 0.3);

    let steps = shipped_course();
    let states = fold(&steps);
    let (mut slow_loud, mut fast_loud, mut checked) = (0usize, 0usize, 0usize);
    let mut worst = 0.0f64;
    println!("\nTHE SQUEAL TABLE (rear axle, the course):");
    println!(
        "  {:>6} {:>8} {:>8} {:>8}",
        "step", "speed", "|slip|", "volume"
    );
    for (i, (s, st)) in steps.iter().zip(&states).enumerate() {
        let Some(v) = s.voice else { continue };
        let Some(q) = st.get(&key(VoiceLayer::SquealRear)) else {
            continue;
        };
        let want = va::squeal_voice(v.axles[1].slip).0;
        let want = if want >= va::VOICE_FLOOR { want } else { 0.0 };
        worst = worst.max((q.volume - want).abs());
        checked += 1;
        if q.volume > 0.3 {
            if v.speed_mps.abs() < 5.0 {
                if slow_loud == 0 {
                    println!(
                        "  {i:6} {:8.2} {:8.3} {:8.3}  (slow)",
                        v.speed_mps, v.axles[1].slip, q.volume
                    );
                }
                slow_loud += 1;
            } else if v.speed_mps.abs() > 15.0 {
                if fast_loud == 0 {
                    println!(
                        "  {i:6} {:8.2} {:8.3} {:8.3}  (fast)",
                        v.speed_mps, v.axles[1].slip, q.volume
                    );
                }
                fast_loud += 1;
            }
        }
    }
    println!("  {checked} steps checked, worst |volume - squeal_voice(slip)| = {worst:.2e}; loud below 5 m/s: {slow_loud}, above 15 m/s: {fast_loud}");
    assert!(checked > 300);
    assert!(worst < 1e-12, "a squeal volume is not squeal_voice(slip)");
    assert!(
        slow_loud > 5 && fast_loud > 5,
        "the course did not slide at both speeds"
    );
}

/// **THE ROAD ROLLS BY SPEED, ON ITS SURFACE** (VEH3e audit, (g')). Over the
/// shipped course, on every step the roll key is live, its folded volume and
/// pitch are `roll_voice` of that step's own telemetry (speed; no wheel down,
/// no roll), and the clip is the rear axle's surface: the slide onto the
/// gravel strip re-`Play`s the roll with `Roll_Gravel`.
///
/// **Anti-vacuity**: loud steps above 15 m/s AND silent steps stopped with the
/// engine running; a gravel re-Play.
///
/// **Mutations → red**: `roll_voice` reading the throttle instead of the
/// speed; the roll clip pinned to the sealed surface.
#[test]
fn the_road_rolls_by_speed_on_its_surface() {
    let steps = shipped_course();
    let states = fold(&steps);
    let roll = key(VoiceLayer::Roll);
    let (mut checked, mut loud_fast, mut quiet_stopped) = (0usize, 0usize, 0usize);
    let mut worst = 0.0f64;
    for (s, st) in steps.iter().zip(&states) {
        let Some(v) = s.voice else { continue };
        let Some(q) = st.get(&roll) else { continue };
        let (wv, wp) = va::roll_voice(&v);
        let wv = if wv >= va::VOICE_FLOOR { wv } else { 0.0 };
        worst = worst.max((q.volume - wv).abs());
        if wv > 0.0 {
            worst = worst.max((q.pitch - wp).abs());
        }
        checked += 1;
        if v.speed_mps.abs() > 15.0 && q.volume > 0.1 {
            loud_fast += 1;
        }
        if v.speed_mps.abs() < 0.3 && v.running() && q.volume == 0.0 {
            quiet_stopped += 1;
        }
    }
    let gravel: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            plays_on(s, roll)
                .iter()
                .any(|p| p.clip == va::roll_clip(SurfaceVoice::Loose))
        })
        .map(|(i, _)| i)
        .collect();
    println!(
        "THE ROLLING ROAD: {checked} live steps, worst |sent - roll_voice| {worst:.2e}; loud above 15 m/s on {loud_fast}, silent at a standstill on {quiet_stopped}; re-Played onto gravel on steps {gravel:?}"
    );
    assert!(checked > 300);
    assert!(
        worst < 1e-12,
        "a roll volume or pitch is not roll_voice(speed)"
    );
    assert!(
        loud_fast > 10 && quiet_stopped > 10,
        "the course never rolled fast and stood still"
    );
    assert!(!gravel.is_empty(), "the roll never changed onto the gravel");
}

/// **A SQUEAL CHANGES ITS CLIP WITH THE SURFACE**: the handbrake slide carries
/// the car onto the gravel strip, and the rear squeal is re-`Play`ed with the
/// LOOSE clip (the hiss) — and, off the planner, a soft surface gets the scrub.
///
/// **Mutation → red**: `SurfaceVoice::of` answering `Sealed` for everything.
#[test]
fn a_squeal_changes_its_clip_with_the_surface() {
    let steps = shipped_course();
    let replays: Vec<(usize, Uuid)> = steps
        .iter()
        .enumerate()
        .skip(1)
        .flat_map(|(i, s)| {
            plays_on(s, key(VoiceLayer::SquealRear))
                .into_iter()
                .map(move |p| (i, p.clip))
        })
        .collect();
    println!("rear squeal Plays: {replays:?}");
    assert!(
        replays
            .iter()
            .any(|(_, c)| *c == va::squeal_clip(SurfaceVoice::Loose)),
        "the slide reached the gravel and the squeal never became the hiss"
    );
    assert_ne!(
        va::squeal_clip(SurfaceVoice::Sealed),
        va::squeal_clip(SurfaceVoice::Loose)
    );
    // Soft ground, off the planner.
    let world = planner_world();
    let mut mem = VoiceMemory::new();
    let cues = mem.plan(
        &world,
        &[(CHASSIS, telemetry(1.6, 5.0, SurfaceClass::Mud))],
        DT,
    );
    assert!(cues.iter().any(|c| matches!(c, va::VoiceCue::Play { clip, .. } if *clip == va::squeal_clip(SurfaceVoice::Soft))));
}

// ── 4. THE LOAD CROSSFADE ───────────────────────────────────────────────────

/// **THE LOAD CROSSFADE**: on every running step of the course, the three grain
/// keys' volumes are the engine's level times `load_weights(load)` — the
/// no-load grain on the lift and at idle, the full-load grain under the
/// throttle, and the full-load grain dropping OUT on every fuel-cut step (a
/// limiter bounce is heard as a stutter).
///
/// **Mutation → red**: `load_weights` pinned to `[0, 1, 0]`.
#[test]
fn the_load_crossfade_moves_between_the_three_grains() {
    let steps = shipped_course();
    let states = fold(&steps);
    let src = AudioSource::default();
    let (mut at0, mut at1, mut cut, mut checked) = (0usize, 0usize, 0usize, 0usize);
    let mut worst = 0.0f64;
    for (s, st) in steps.iter().zip(&states) {
        let Some(v) = s.voice else { continue };
        if !v.running() {
            continue;
        }
        let level = src.volume * va::engine_level(&v);
        // The doc's three loads as TRIANGLES written out here (0 %, 50 %,
        // 100 %) -- not `va::load_weights`, which is the function under test:
        // an arm that asked it would pass whatever it answered (the audit's
        // first mutation run: `load_weights` pinned to the mid grain stayed
        // GREEN).
        let l = v.load();
        let w = [
            (1.0 - 2.0 * l).max(0.0),
            1.0 - (2.0 * l - 1.0).abs(),
            (2.0 * l - 1.0).max(0.0),
        ];
        for (i, layer) in [
            VoiceLayer::GrainIdle,
            VoiceLayer::GrainMid,
            VoiceLayer::GrainFull,
        ]
        .into_iter()
        .enumerate()
        {
            let Some(g) = st.get(&key(layer)) else {
                continue;
            };
            let want = level * w[i];
            let want = if want >= va::VOICE_FLOOR { want } else { 0.0 };
            // A silent voice that stays silent is not re-sent, so its folded
            // volume is the last one it was sent — zero.
            worst = worst.max((g.volume - want).abs());
        }
        checked += 1;
        let audible = |layer| st.get(&key(layer)).is_some_and(|g: &Voice| g.volume > 0.0);
        if v.load() == 0.0 && !v.fuel_cut {
            at0 += 1;
            assert!(audible(VoiceLayer::GrainIdle) && !audible(VoiceLayer::GrainFull));
        } else if v.load() == 1.0 {
            at1 += 1;
            assert!(audible(VoiceLayer::GrainFull) && !audible(VoiceLayer::GrainIdle));
        }
        if v.fuel_cut && v.throttle > 0.5 {
            cut += 1;
        }
    }
    println!(
        "{checked} running steps: {at0} at no load, {at1} at full load, {cut} under a fuel cut at full throttle; worst |volume - level x weight| = {worst:.2e}"
    );
    assert!(
        at0 > 30 && at1 > 30,
        "the course did not cover both ends of the fade"
    );
    assert!(worst < 1e-12, "a grain volume is not the crossfade");
}

// ── 5. THE WHINE ────────────────────────────────────────────────────────────

/// **THE WHINE STEPS DOWN ON EVERY UPSHIFT** — by the ratio of the two gears,
/// because it is pitched by the input shaft (the driven wheels times this
/// gear's ratio), and a shift changes the ratio in one step while the wheels do
/// not change speed at all.
///
/// Prints the SHIFT TABLE: gear, whine pitch before and after, the ratio of the
/// two against the ratio of the gears.
///
/// **Mutation → red**: the whine pitched by `final_drive` alone.
#[test]
fn the_whine_steps_down_on_every_upshift() {
    let def = course_def();
    let t = def.class.to_tuning();
    let steps = shipped_course();
    let states = fold(&steps);
    let mut shifts = 0usize;
    println!("THE SHIFT TABLE:");
    println!(
        "  {:>5} {:>6} {:>9} {:>9} {:>9} {:>9}",
        "step", "gears", "pitch -", "pitch +", "ratio", "gears r."
    );
    for i in 1..steps.len() {
        let (Some(a), Some(b)) = (steps[i - 1].voice, steps[i].voice) else {
            continue;
        };
        if b.gear != a.gear + 1 || a.gear < 1 {
            continue;
        }
        let (Some(pa), Some(pb)) = (
            states[i - 1].get(&key(VoiceLayer::Whine)),
            states[i].get(&key(VoiceLayer::Whine)),
        ) else {
            continue;
        };
        let got = pb.pitch / pa.pitch;
        let want = t.drive_ratio(b.gear) / t.drive_ratio(a.gear);
        println!(
            "  {i:5} {:>2}->{:<2} {:9.4} {:9.4} {got:9.4} {want:9.4}",
            a.gear, b.gear, pa.pitch, pb.pitch
        );
        assert!(
            (got - want).abs() < 0.05 * want,
            "the whine stepped by {got} across {}->{}, the gears by {want}",
            a.gear,
            b.gear
        );
        shifts += 1;
    }
    assert!(shifts >= 2, "the course made {shifts} upshifts");
}

// ── 6. THE TURBO ────────────────────────────────────────────────────────────

/// **THE TURBO SPOOLS AND BLOWS OFF ON THE LIFT**: the turbo key's pitch is
/// `0.6 + 1.2 × boost` on every audible step, and the blow-off key is `Play`ed
/// exactly once — on the step the throttle came off with boost up. A car with
/// no turbo (the CONTROL) plays neither.
///
/// **Mutation → red**: the blow-off branch deleted.
#[test]
fn the_turbo_spools_and_blows_off_on_the_lift() {
    let steps = shipped_course();
    let states = fold(&steps);
    let mut spooled = 0usize;
    for (s, st) in steps.iter().zip(&states) {
        let (Some(v), Some(tb)) = (s.voice, st.get(&key(VoiceLayer::Turbo))) else {
            continue;
        };
        if tb.volume > 0.0 {
            assert!((tb.pitch - (0.6 + 1.2 * v.boost)).abs() < 1e-12);
            spooled += 1;
        }
    }
    let blow: Vec<(usize, f64, f64)> = steps
        .iter()
        .enumerate()
        .flat_map(|(i, s)| {
            plays_on(s, key(VoiceLayer::BlowOff))
                .into_iter()
                .map(move |p| (i, p.volume, s.voice.map(|v| v.boost).unwrap_or(0.0)))
        })
        .collect();
    println!("{spooled} audible turbo steps; blow-offs (step, volume, boost): {blow:?}");
    assert!(spooled > 60);
    assert_eq!(blow.len(), 1, "one lift, one blow-off");
    let lift = steps
        .iter()
        .position(|s| s.k == Some(300))
        .expect("the lift step");
    assert_eq!(blow[0].0, lift, "the blow-off is not on the lift step");
    assert!(blow[0].2 >= va::BLOW_OFF_MIN_BOOST);

    // THE CONTROL: the same car with no turbo.
    let mut def = course_def();
    assert!(def.class.set("turbo_boost_max", 0.0));
    let mut host = Shipped(inf_player::runtime_sim::RuntimeSim::new(
        shipped_world(&def),
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        HZ,
    ));
    let na = run(&mut host, course, 400);
    let turbo_cmds = na
        .iter()
        .flat_map(|s| s.cmds.iter())
        .filter(|c| {
            matches!(source_of(c), Some(k) if k == key(VoiceLayer::Turbo) || k == key(VoiceLayer::BlowOff))
        })
        .count();
    assert_eq!(turbo_cmds, 0, "a car with no turbo whistled");
}

// ── 7. THE KERB ─────────────────────────────────────────────────────────────

/// **THE KERB THUMPS, FRONT THEN REAR**: crossing the 12 cm kerb at speed, the
/// front impulse key and then the rear one are `Play`ed once each with the
/// SEALED impulse clip — and nothing else on the course's smooth road raises
/// one after the car first settled.
///
/// **Mutation → red**: the rising-edge guard dropped (`!mem.hot[i]` → `true`),
/// which fires every step of a spike.
#[test]
fn the_kerb_thumps_front_then_rear() {
    let steps = shipped_course();
    let mut hits: Vec<(usize, VoiceLayer, Uuid, f64)> = Vec::new();
    for (i, s) in steps.iter().enumerate() {
        if s.k.is_none() {
            continue;
        }
        for layer in [VoiceLayer::ImpulseFront, VoiceLayer::ImpulseRear] {
            for p in plays_on(s, key(layer)) {
                hits.push((
                    i,
                    layer,
                    p.clip,
                    s.voice.map(|v| v.speed_mps).unwrap_or(0.0),
                ));
            }
        }
    }
    println!("impulses while driving (step, axle, clip, speed): {hits:?}");
    assert_eq!(hits.len(), 2, "one kerb, two axles");
    assert_eq!(hits[0].1, VoiceLayer::ImpulseFront);
    assert_eq!(hits[1].1, VoiceLayer::ImpulseRear);
    assert!(hits[0].0 < hits[1].0);
    for h in &hits {
        assert_eq!(h.2, va::impulse_clip(SurfaceVoice::Sealed));
        assert!(h.3 > 5.0);
    }
    // THE RISING EDGE, driven through the shipped planner: a strut closing
    // above the onset for FOUR steps in a row (a long ramp, a landing that
    // keeps compressing) is ONE thump, not four. The course's own kerb spike
    // lasts a single step, so it cannot see this (the first mutation run:
    // the edge guard dropped stayed GREEN on the course alone).
    let world = planner_world();
    let mut mem = VoiceMemory::new();
    let mut t = telemetry(0.0, 10.0, SurfaceClass::Gravel);
    let mut thumps = 0usize;
    for step in 0..8 {
        t.axles[0].compression_m = 0.05 + 0.03 * f64::from(step.clamp(1, 5) - 1);
        let cues = mem.plan(&world, &[(CHASSIS, t)], DT);
        thumps += cues
            .iter()
            .filter(|c| matches!(c, va::VoiceCue::Play { source, clip, .. } if *source == key(VoiceLayer::ImpulseFront) && *clip == va::impulse_clip(SurfaceVoice::Loose)))
            .count();
    }
    println!("a strut closing at 1.8 m/s for four steps: {thumps} thump(s), with the gravel clip");
    assert_eq!(thumps, 1, "a sustained spike thumped {thumps} times");
}

// ── 8. ENGINE ON / OFF ──────────────────────────────────────────────────────

/// **A PARKED CAR IS SILENT UNTIL SOMEBODY GETS IN, AND STOPS WHEN THEY
/// LEAVE**: no loop key is addressed before the hero is at the car, the seven
/// loops start while they climb in, and all seven are `Stop`ped once the car is
/// empty and its drivetrain quiet.
///
/// **Mutation → red**: `VoiceTelemetry::running` answering `true`.
#[test]
fn a_parked_car_is_silent_until_somebody_gets_in() {
    let steps = shipped_course();
    let loops: BTreeSet<u64> = [
        VoiceLayer::GrainIdle,
        VoiceLayer::GrainMid,
        VoiceLayer::GrainFull,
        VoiceLayer::Whine,
        VoiceLayer::Turbo,
        VoiceLayer::SquealFront,
        VoiceLayer::SquealRear,
        VoiceLayer::Roll,
    ]
    .into_iter()
    .map(key)
    .collect();
    let first = steps
        .iter()
        .position(|s| {
            s.cmds
                .iter()
                .any(|c| source_of(c).is_some_and(|k| loops.contains(&k)))
        })
        .expect("the engine never spoke");
    let phase_then = steps[first].board.0;
    let silent_before = first;
    let stops: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| {
            s.cmds
                .iter()
                .any(|c| matches!(c, AudioCommand::Stop { source } if loops.contains(source)))
        })
        .map(|(i, _)| i)
        .collect();
    println!(
        "silent for {silent_before} steps; the engine starts in `{}`; loops stopped on steps {stops:?} ({:?})",
        phase_then.name(),
        stops.first().map(|i| steps[*i].board.0)
    );
    assert!(
        silent_before > 60,
        "the parked car spoke before anybody reached it"
    );
    assert!(matches!(
        phase_then,
        BoardPhase::EnteringIK | BoardPhase::Seated | BoardPhase::Driving
    ));
    let stopped: usize = stops
        .iter()
        .map(|i| {
            steps[*i]
                .cmds
                .iter()
                .filter(|c| matches!(c, AudioCommand::Stop { source } if loops.contains(source)))
                .count()
        })
        .sum();
    assert_eq!(
        stopped, 8,
        "the engine switched off {stopped} of its eight loops"
    );
    assert!(stops.iter().all(|i| steps[*i].k.is_some_and(|k| k > 640)));
}

// ── 9. THE DOOR ─────────────────────────────────────────────────────────────

/// **THE DOOR SLAMS ON THE SHUT STEP, AND THE MOTOR-ONLY ROWS ARE SILENT**, on
/// a RIGGED shipped host (so the hand pass runs and `handle_weight` is real).
///
/// * the LATCH is `Play`ed on the step the door phase's mark is set;
/// * the SLAM is `Play`ed exactly once per closing, on the step `door_deg`
///   first reads shut after reading open;
/// * on the Seated rows where the door is still open and NO hand is on the
///   inner handle (`handle_weight == 0`, the motor-only first part of the
///   pull) no door key is addressed at all; the creak speaks on the rows
///   where the hand IS on it.
///
/// **Mutations → red**: the Seated creak gate dropped (a creak on the motor
/// rows); the slam's edge dropped (a slam on every shut step).
#[test]
fn the_door_slams_on_the_shut_step_and_the_motor_rows_are_silent() {
    let mut host = Shipped(rigged_sim(&course_def()));
    let steps = run(&mut host, course, 700);
    let doors = door_keys();
    let dk = |l: DoorLayer| va::door_key(va::entity_key(HERO), l);
    let (mut motor, mut hand, mut hand_creak) = (0usize, 0usize, 0usize);
    let mut slams: Vec<usize> = Vec::new();
    let mut latches: Vec<usize> = Vec::new();
    let mut crossings: Vec<usize> = Vec::new();
    for i in 1..steps.len() {
        let s = &steps[i];
        let (phase, deg, hw) = s.board;
        let door_cmds = s
            .cmds
            .iter()
            .filter(|c| source_of(c).is_some_and(|k| doors.contains_key(&k)))
            .count();
        if !plays_on(s, dk(DoorLayer::Slam)).is_empty() {
            slams.push(i);
        }
        if !plays_on(s, dk(DoorLayer::Latch)).is_empty() {
            latches.push(i);
        }
        let (pphase, pdeg, _) = steps[i - 1].board;
        if matches!(pphase, BoardPhase::Seated | BoardPhase::ClosingDoor)
            && pdeg > DOOR_SHUT_DEG
            && deg <= DOOR_SHUT_DEG
        {
            crossings.push(i);
        }
        if phase == BoardPhase::Seated && deg > DOOR_SHUT_DEG {
            if hw == 0.0 {
                motor += 1;
                assert_eq!(
                    door_cmds, 0,
                    "step {i}: a door sound on a motor-only row (door {deg:.1} deg, no hand)"
                );
            } else {
                hand += 1;
                if s.cmds
                    .iter()
                    .any(|c| source_of(c) == Some(dk(DoorLayer::Creak)))
                {
                    hand_creak += 1;
                }
            }
        }
    }
    println!(
        "THE DOOR ROWS: {motor} motor-only Seated rows (silent), {hand} hand rows ({hand_creak} with a creak command); latches {latches:?}; shut crossings {crossings:?}; slams {slams:?}"
    );
    assert!(motor > 5, "the pull had no motor-only rows to be silent on");
    assert!(hand > 5 && hand_creak > 0, "the hand rows never creaked");
    assert!(!crossings.is_empty());
    assert_eq!(slams, crossings, "a slam is not on its shut step");
    assert_eq!(latches.len(), 2, "one latch in, one out");

    // THE SHUT EDGE, driven through the shipped planner: a door that is
    // pulled shut EARLY and then sits shut for several `Seated` steps (the
    // machine leaves `Seated` only after `SEATED_MIN_S`) slams ONCE. The
    // course's door shut after the minimum, so it left `Seated` on the very
    // step it shut and could not see this (the first mutation run: the edge
    // dropped stayed GREEN on the course alone).
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert(hero_bits());
    world.propagate();
    let door = Uuid::from_u128(0x5E3E_00D0);
    let mut mem = VoiceMemory::new();
    let mut slams_direct = 0usize;
    for (i, deg) in [40.0, 20.0, 8.0, 2.0, 1.0, 0.5, 0.5, 0.5]
        .into_iter()
        .enumerate()
    {
        {
            let mut cm = world
                .world_mut()
                .get_mut::<CharacterMovement>(e)
                .expect("a mover");
            let b = &mut cm.runtime.boarding;
            b.phase = BoardPhase::Seated;
            b.door = door;
            b.door_deg = deg;
            b.handle_weight = 1.0;
            b.time_s = i as f64 * DT;
        }
        slams_direct += mem
            .plan(&world, &[], DT)
            .iter()
            .filter(|c| c.source() == dk(DoorLayer::Slam))
            .count();
    }
    println!("a door shut early and held shut for five Seated steps: {slams_direct} slam(s)");
    assert_eq!(
        slams_direct, 1,
        "a door that stayed shut slammed {slams_direct} times"
    );
}

/// One stationary exit on the shipped host, the hero's door HELD OPEN through
/// the whole `ClosingDoor` phase when `hold` — the damage row's motor target
/// rewritten to the open angle every step, which is a hand (or a bollard) on
/// the door: the joint re-aims whenever its command moves. Answers the slam
/// steps, the step the boarding ended on, and the door's MEASURED hinge angle
/// (off the joint, through the damage row) on that step.
fn exit_with_the_door(hold: bool) -> (Vec<usize>, Option<usize>, f64, usize) {
    fn exit(k: u32) -> Intent {
        if k == 60 {
            Intent::press()
        } else {
            Intent::default()
        }
    }
    let mut host = Shipped(inf_player::runtime_sim::RuntimeSim::new(
        shipped_world(&course_def()),
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        HZ,
    ));
    let slam = va::door_key(va::entity_key(HERO), DoorLayer::Slam);
    let mut wheel: Option<u32> = None;
    let mut slams = Vec::new();
    let mut ended: Option<usize> = None;
    let mut end_deg = f64::NAN;
    let mut door = Uuid::nil();
    let mut open_deg = 0.0f64;
    let mut closing_steps = 0usize;
    let mut last = BoardPhase::Idle;
    for step in 0..(BOARD_MAX + 400) {
        let k = wheel.map(|w| step - w);
        if k.is_some_and(|k| k >= 400) {
            break;
        }
        let i = match k {
            None if step == BOARD_PRESS => Intent::press(),
            None => Intent::default(),
            Some(k) => exit(k),
        };
        let before = host.0.audio_command_log().len();
        host.step(i);
        let sim = &mut host.0;
        assert_eq!(sim.dropped_audio_commands(), 0);
        let cmds = &sim.audio_command_log()[before..];
        if cmds
            .iter()
            .any(|c| matches!(c, AudioCommand::Play(p) if p.source == slam))
        {
            slams.push(step as usize);
        }
        let (phase, _, _, seated, _) = board_of(sim.world());
        {
            let e = sim.world().entity_of(HERO).expect("the hero");
            let b = sim
                .world()
                .world()
                .get::<CharacterMovement>(e)
                .expect("a mover")
                .runtime
                .boarding;
            if !b.door.is_nil() {
                door = b.door;
            }
        }
        let part = |sim: &inf_player::runtime_sim::RuntimeSim| {
            inf_ecs::bodywork::damage_row(sim.world(), CHASSIS)
                .and_then(|r| r.parts.get(&door).copied())
        };
        if phase == BoardPhase::Exiting {
            // The open target, SIGNED: a left-hand door opens to a negative
            // hinge angle.
            if let Some(p) = part(sim) {
                if p.target_deg.abs() > open_deg.abs() {
                    open_deg = p.target_deg;
                }
            }
        }
        if phase == BoardPhase::ClosingDoor {
            closing_steps += 1;
            if hold && open_deg != 0.0 {
                if let Some(r) = inf_ecs::bodywork::damage_mut(sim.world_mut())
                    .rows
                    .get_mut(&CHASSIS)
                {
                    if let Some(s) = r.parts.get_mut(&door) {
                        s.target_deg = open_deg;
                    }
                }
            }
        }
        if last == BoardPhase::ClosingDoor && phase != BoardPhase::ClosingDoor && ended.is_none() {
            ended = Some(step as usize);
            end_deg = part(sim).map(|p| p.angle_deg.abs()).unwrap_or(f64::NAN);
        }
        last = phase;
        if wheel.is_none() && at_wheel(phase, seated) {
            wheel = Some(step + 1);
        }
    }
    (slams, ended, end_deg, closing_steps)
}

/// **A DOOR HELD OPEN THROUGH THE CLOSE DOES NOT SLAM** (VEH3e audit, carried
/// 5 — the false slam). A `ClosingDoor` that times out resets the boarding
/// machine, `door_deg` included, to zero whether or not the door shut; the
/// planner read that reset as the hinge crossing shut and slammed a door that
/// was standing open. On the SHIPPED host, one stationary exit twice:
///
/// * the control — the door shuts on its motor: the phase ends early (well
///   inside `CLOSING_MAX_S`), the joint reads shut, and the slam lands on that
///   end step, once;
/// * held — the door's motor is re-aimed open every `ClosingDoor` step (a
///   hand, a bollard): the phase runs the whole `CLOSING_MAX_S` and TIMES OUT,
///   the joint reads OPEN on the end step, and **no slam** is played.
///
/// **Mutation → red**: the planner's end-step hinge read back to the
/// machine's own reset `door_deg` (the pre-audit code) slams the held door.
#[test]
fn a_door_held_open_through_the_close_does_not_slam() {
    // Only the EXIT's slams: the entry shut its door long before.
    let after = |r: (Vec<usize>, Option<usize>, f64, usize)| {
        let from = r.1.unwrap_or(0).saturating_sub(120);
        (
            r.0.into_iter().filter(|s| *s > from).collect::<Vec<_>>(),
            r.1,
            r.2,
            r.3,
        )
    };
    let (c_slams, c_end, c_deg, c_len) = after(exit_with_the_door(false));
    let (h_slams, h_end, h_deg, h_len) = after(exit_with_the_door(true));
    println!(
        "THE CLOSE: shut on its motor -- {c_len} ClosingDoor steps, ended at {c_end:?} with the joint at {c_deg:.2} deg, slams {c_slams:?}; HELD OPEN -- {h_len} steps, ended at {h_end:?} with the joint at {h_deg:.2} deg, slams {h_slams:?}"
    );
    let limit = (inf_ecs::boarding::CLOSING_MAX_S * HZ).round() as usize;
    assert!(c_end.is_some() && h_end.is_some(), "a close never ended");
    assert!(
        c_len < limit,
        "the control's door did not shut on its own ({c_len} steps)"
    );
    assert!(
        c_deg <= DOOR_SHUT_DEG,
        "the control ended with the door at {c_deg}"
    );
    assert_eq!(
        c_slams,
        vec![c_end.unwrap()],
        "the control's slam is not on its end step"
    );
    assert!(
        h_len + 1 >= limit,
        "the held close did not time out ({h_len} steps)"
    );
    assert!(
        h_deg > 20.0,
        "the held door was not open at the end ({h_deg} deg)"
    );
    assert!(
        h_slams.is_empty(),
        "a door standing open slammed: {h_slams:?}"
    );
}

/// **A BAIL-OUT LANDS WITH A THUD**: a press at speed throws the hero out of
/// the moving car, and the thud key is `Play`ed once — on the step the body
/// starts its roll.
///
/// **Mutation → red**: the thud branch deleted.
#[test]
fn a_bail_out_lands_with_a_thud() {
    fn bail(k: u32) -> Intent {
        match k {
            0..=239 => Intent::throttle(1.0),
            240 => Intent {
                press: true,
                throttle: 1.0,
                ..Default::default()
            },
            _ => Intent::default(),
        }
    }
    let mut host = Shipped(inf_player::runtime_sim::RuntimeSim::new(
        shipped_world(&course_def()),
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        HZ,
    ));
    let steps = run(&mut host, bail, 420);
    let thud = va::door_key(va::entity_key(HERO), DoorLayer::Thud);
    let thuds: Vec<usize> = steps
        .iter()
        .enumerate()
        .filter(|(_, s)| !plays_on(s, thud).is_empty())
        .map(|(i, _)| i)
        .collect();
    let rolls: Vec<usize> = (1..steps.len())
        .filter(|&i| steps[i].mode == MovementMode::Roll && steps[i - 1].mode != MovementMode::Roll)
        .collect();
    println!("thuds {thuds:?}, roll starts {rolls:?}");
    assert!(!rolls.is_empty(), "the moving exit never rolled");
    assert_eq!(thuds, rolls[..1].to_vec());
}

/// Every 16-bit sample of a stereo WAV's first `data` chunk, read HERE — to
/// the end of the file, not to the header's size (a capture patches its sizes
/// every 60 steps, and the bytes after the last patch are still the drive).
fn wav_samples(bytes: &[u8]) -> Vec<i16> {
    let at = bytes
        .windows(4)
        .position(|w| w == b"data")
        .expect("a data chunk");
    bytes[at + 8..]
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// The shipped course rendered through the render-to-file door into `path`:
/// `limited` is the shipped master track, `!limited` the measurement control.
fn render_course(path: &Path, limited: bool) -> Vec<i16> {
    // The car's emitter at 2.4: a roster row's `AudioSource.volume` scales its
    // whole stack, and a loud row is exactly the case a ceiling exists for.
    // (At the default 1.0 this host's course peaks at 0.82 and never needs one.)
    let mut world = shipped_world(&course_def());
    let car = world.entity_of(CHASSIS).expect("the car");
    world
        .world_mut()
        .get_mut::<AudioSource>(car)
        .expect("a voiced car")
        .volume = 2.4;
    let mut sim = inf_player::runtime_sim::RuntimeSim::new(
        world,
        Vec::new(),
        glam::DVec2::new(0.0, -9.81),
        HZ,
    );
    // The 31 COMMITTED clips, decoded from their `.inf_audio` payloads --
    // what a cooked pack resolves them to.
    let dir = inf_editor_core::samples::vehicle_audio_dir();
    let mut clips = BTreeMap::new();
    for (i, name) in va::VEHICLE_CLIP_NAMES.iter().enumerate() {
        let bytes = std::fs::read(dir.join(format!("Vehicle_{name}.inf_audio"))).expect("a clip");
        let asset: inf_audio::AudioAsset = inf_asset::decode(&bytes).expect("an audio payload");
        clips.insert(va::vehicle_clip(i as u8), asset);
    }
    sim.set_audio_clips(clips);
    if limited {
        sim.capture_audio_to(path, 48_000).expect("a capture");
    } else {
        sim.capture_audio_unlimited_to(path, 48_000)
            .expect("a capture");
    }
    let mut host = Shipped(sim);
    let steps = run(&mut host, course, DRIVE_STEPS);
    assert!(steps.len() > 900);
    drop(host);
    wav_samples(&std::fs::read(path).expect("the capture"))
}

/// **THE COURSE'S BYTES DO NOT CLIP** (VEH3e audit, carried 6 — the mix had no
/// limiter). The shipped course (the burnout, the pull, the kerb, the slide,
/// the doors) rendered through the render-to-file door twice — through the
/// shipped master track and through the same mixer with NO limiter — and the
/// WAV BYTES counted: samples at 16-bit full scale, and the peak.
///
/// The car's emitter is authored at volume 2.4 (a roster row's knob): at 1.0
/// this host's course peaks at 0.82 and a ceiling has nothing to hold.
///
/// **Anti-vacuity**: the control must CLIP (the course is loud enough for the
/// ceiling to have work), and the limited render must carry the drive (its
/// RMS within 1 dB of the control's).
///
/// **Mutation → red**: the limiter taken off the master track
/// (`master_track` ignoring `limit`).
#[test]
fn the_course_render_does_not_clip() {
    let tmp = tempfile::tempdir().expect("a temp dir");
    let raw = render_course(&tmp.path().join("raw.wav"), false);
    let lim = render_course(&tmp.path().join("limited.wav"), true);
    let clipped = |x: &[i16]| x.iter().filter(|v| **v >= 32_767 || **v <= -32_767).count();
    let peak =
        |x: &[i16]| x.iter().map(|v| i32::from(*v).abs()).max().unwrap_or(0) as f64 / 32_767.0;
    let rms = |x: &[i16]| {
        (x.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
            / 32_767.0
    };
    let ceiling = f64::from(inf_audio::limiter::CEILING);
    println!(
        "THE COURSE RENDER ({} samples): NO LIMITER peak {:.4} clipped {} rms {:.4}; SHIPPED peak {:.4} clipped {} rms {:.4} (ceiling {ceiling})",
        raw.len(),
        peak(&raw),
        clipped(&raw),
        rms(&raw),
        peak(&lim),
        clipped(&lim),
        rms(&lim)
    );
    assert!(
        clipped(&raw) > 0,
        "the control never clipped: nothing for a ceiling to hold"
    );
    assert!(rms(&lim) > rms(&raw) * 0.89, "the limiter ate the drive");
    assert_eq!(clipped(&lim), 0, "the shipped mix clipped");
    assert!(peak(&lim) < ceiling + 1.0 / 32_767.0, "over the ceiling");
}

/// **A TRAFFIC CAR SINGS THE NEAR STACK, ON ITS OWN CADENCE** (VEH3e audit,
/// (d')). Two cars side by side through the shipped planner, the same
/// telemetry every step (revs climbing 1 000 -> 5 000 rpm, then a slide): the
/// level's own car and a TRAFFIC car (a record of the population resource).
///
/// * the traffic car starts exactly TWO loops — the half-load grain of its
///   own family and one squeal — and never the idle / full grains, the whine
///   or the turbo;
/// * each of its loops is re-told at most once per `NEAR_EVERY` steps, and the
///   grain pitch it is told is the revs' own `(rpm/2400)(cyl/family cyl)`;
/// * the authored car gets the full stack and more than twice the commands.
///
/// **Mutations → red**: `NEAR_EVERY` 1 (every step re-told); the traffic test
/// in `plan` answering false (the full stack for traffic).
#[test]
fn a_traffic_car_sings_the_near_stack_on_its_own_cadence() {
    const TRAFFIC: Uuid = Uuid::from_u128(0x5E3E_0071);
    let mut world = EcsWorld::new();
    let def = course_def();
    spawn_car(&mut world, CHASSIS, DVec3::new(0.0, 1.0, 0.0), &def, true);
    spawn_car(&mut world, TRAFFIC, DVec3::new(6.0, 1.0, 0.0), &def, true);
    let mut pop = inf_ecs::traffic::TrafficPopulationRes::default();
    pop.records.insert(
        TRAFFIC,
        inf_ecs::traffic::TrafficRecord::parked(
            def.clone(),
            inf_ecs::math::Color::new(0.5, 0.5, 0.5, 1.0),
            DVec3::new(6.0, 1.0, 0.0),
            0.0,
        ),
    );
    world.world_mut().insert_resource(pop);
    world.propagate();
    let family = GrainFamily::P8Cross;
    let mut mem = VoiceMemory::new();
    let tk = |l: VoiceLayer| va::voice_key(va::entity_key(TRAFFIC), l);
    let traffic_keys: BTreeSet<u64> = VoiceLayer::ALL.iter().map(|l| tk(*l)).collect();
    let mut plays: Vec<u64> = Vec::new();
    let mut last_told: BTreeMap<u64, usize> = BTreeMap::new();
    let (mut t_updates, mut h_updates) = (0usize, 0usize);
    let mut pitches_checked = 0usize;
    const STEPS: usize = 160;
    for i in 0..STEPS {
        let mut t = telemetry(
            if i >= 120 { 1.6 } else { 0.0 },
            20.0,
            SurfaceClass::Asphalt,
        );
        t.rpm = 1_000.0 + 4_000.0 * (i.min(119) as f64 / 119.0);
        t.throttle = (i as f64 / 80.0).min(1.0);
        let cues = mem.plan(&world, &[(CHASSIS, t), (TRAFFIC, t)], DT);
        for c in &cues {
            let k = c.source();
            if traffic_keys.contains(&k) {
                match c {
                    va::VoiceCue::Play { .. } => plays.push(k),
                    _ => {
                        t_updates += 1;
                        if let Some(prev) = last_told.insert(k, i).filter(|p| *p != i) {
                            assert!(
                                i - prev >= va::NEAR_EVERY as usize,
                                "step {i}: a NEAR loop re-told {} step(s) after the last",
                                i - prev
                            );
                        }
                    }
                }
                if let va::VoiceCue::Pitch { pitch, .. } = c {
                    if k == tk(VoiceLayer::GrainMid) {
                        let want = va::grain_pitch(&t, family);
                        assert!((pitch - want).abs() < 1e-12, "step {i}: {pitch} vs {want}");
                        pitches_checked += 1;
                    }
                }
            } else if !matches!(c, va::VoiceCue::Play { .. }) {
                h_updates += 1;
            }
        }
    }
    println!(
        "THE NEAR STACK: traffic Plays on {:?}; {t_updates} traffic updates vs {h_updates} for the level's car over {STEPS} steps; {pitches_checked} grain pitches checked; near voiced {} of {}",
        plays
            .iter()
            .map(|k| VoiceLayer::ALL.iter().find(|l| tk(**l) == *k).copied())
            .collect::<Vec<_>>(),
        mem.near_voiced_cars(),
        mem.voiced_cars()
    );
    assert_eq!(
        plays,
        vec![
            tk(VoiceLayer::GrainMid),
            tk(VoiceLayer::SquealRear),
            tk(VoiceLayer::Roll)
        ],
        "the traffic car's loops"
    );
    assert!(
        t_updates > 0 && pitches_checked > 10,
        "the NEAR voice was never re-told"
    );
    assert!(
        h_updates > 2 * t_updates,
        "the NEAR stack is not cheaper: {t_updates} vs {h_updates}"
    );
    assert_eq!((mem.near_voiced_cars(), mem.voiced_cars()), (1, 2));

    // THE PLAYER TAKES IT: a player-controlled body seated in the traffic car
    // makes it the player's car -- the full stack starts on the keys the NEAR
    // stack did not have -- and getting out drops it back, STOPPING them.
    let h = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(h).insert(hero_bits());
    let seat = |world: &mut EcsWorld, car: Uuid| {
        world
            .world_mut()
            .get_mut::<CharacterMovement>(h)
            .expect("a mover")
            .runtime
            .seat
            .vehicle = car;
    };
    let t = telemetry(0.0, 20.0, SurfaceClass::Asphalt);
    seat(&mut world, TRAFFIC);
    let took: Vec<va::VoiceCue> = mem
        .plan(&world, &[(TRAFFIC, t)], DT)
        .into_iter()
        .filter(|c| traffic_keys.contains(&c.source()))
        .collect();
    let started: BTreeSet<u64> = took
        .iter()
        .filter(|c| matches!(c, va::VoiceCue::Play { .. }))
        .map(|c| c.source())
        .collect();
    seat(&mut world, Uuid::nil());
    let left: BTreeSet<u64> = mem
        .plan(&world, &[(TRAFFIC, t)], DT)
        .into_iter()
        .filter(|c| matches!(c, va::VoiceCue::Stop { .. }) && traffic_keys.contains(&c.source()))
        .map(|c| c.source())
        .collect();
    println!(
        "THE PLAYER TAKES A TRAFFIC CAR: {} loop(s) started, {} stopped on getting out",
        started.len(),
        left.len()
    );
    let full: BTreeSet<u64> = [
        VoiceLayer::GrainIdle,
        VoiceLayer::GrainFull,
        VoiceLayer::Whine,
        VoiceLayer::SquealFront,
    ]
    .iter()
    .map(|l| tk(*l))
    .collect();
    assert!(
        full.is_subset(&started),
        "the player's traffic car did not get the full stack"
    );
    assert_eq!(
        started, left,
        "getting out did not drop back to the NEAR stack"
    );
}

// ── 10. BOTH HOSTS ──────────────────────────────────────────────────────────

/// **PIE == SHIPPING ON THE AUDIO COURSE** — the editor's `SimSession` and the
/// shipped `RuntimeSim` over the same board-drive-kerb-slide-exit course, each
/// step's command slice compared WHOLE: count, order, keys, clips, volumes,
/// pitches, positions.
///
/// **Anti-vacuity**: the shipped stream carries every kind of vehicle cue — the
/// seven loops' Plays, a blow-off, both impulses, a gravel re-Play, the
/// latch, the creak, the slam, the engine's Stops.
///
/// **Mutation → red**: one host's fence calling `voices.plan(world, &voiced,
/// dt * 2.0)` (the door rates halve, so the creak's volumes differ).
#[test]
fn pie_equals_shipping_on_the_audio_course() {
    let shipped = shipped_course();
    let mut preview = Preview::new(&course_def());
    let pie = run(&mut preview, course, DRIVE_STEPS);
    let keys = car_keys();
    let doors = door_keys();
    let mut kinds: BTreeSet<String> = BTreeSet::new();
    for s in &shipped {
        for c in &s.cmds {
            let Some(k) = source_of(c) else { continue };
            let what = keys
                .get(&k)
                .map(|l| format!("{l:?}"))
                .or_else(|| doors.get(&k).map(|d| format!("{d:?}")));
            if let Some(w) = what {
                let verb = match c {
                    AudioCommand::Play(_) => "Play",
                    AudioCommand::Stop { .. } => "Stop",
                    AudioCommand::SetVolume { .. } => "Volume",
                    AudioCommand::SetPitch { .. } => "Pitch",
                    AudioCommand::SetPosition { .. } => "Move",
                    _ => "other",
                };
                kinds.insert(format!("{w}:{verb}"));
            }
        }
    }
    println!(
        "the shipped stream carries {} kinds: {kinds:?}",
        kinds.len()
    );
    for want in [
        "GrainIdle:Play",
        "GrainFull:Volume",
        "Whine:Pitch",
        "Turbo:Pitch",
        "SquealRear:Play",
        "BlowOff:Play",
        "ImpulseFront:Play",
        "ImpulseRear:Play",
        "Latch:Play",
        "Creak:Play",
        "Creak:Stop",
        "Slam:Play",
        "GrainIdle:Stop",
        "GrainIdle:Move",
    ] {
        assert!(kinds.contains(want), "the course never produced `{want}`");
    }
    let total: usize = shipped.iter().map(|s| s.cmds.len()).sum();
    println!(
        "{} steps shipped, {} preview; {total} commands",
        shipped.len(),
        pie.len()
    );
    assert_eq!(
        shipped.len(),
        pie.len(),
        "the two hosts ran different courses"
    );
    for (i, (a, b)) in shipped.iter().zip(&pie).enumerate() {
        assert_eq!(
            a.cmds, b.cmds,
            "step {i}: the editor's preview and the shipped player queued different audio"
        );
    }
}

/// **`dropped == 0`, and the stream's rate stated.** The course runs through
/// the shipped host's own bounded log with nothing evicted (asserted every step
/// in `run`); this prints what the drive costs the log a second and the
/// horizon the ring buys at that rate.
///
/// **Mutation → red**: `AUDIO_LOG_CAPACITY` cut to 2 048 (the course queues
/// more than that, so `run`'s per-step `dropped == 0` reds).
#[test]
fn the_audio_log_holds_the_drive_and_the_count_is_stated() {
    let steps = shipped_course();
    let total: usize = steps.iter().map(|s| s.cmds.len()).sum();
    let car: usize = steps.iter().map(car_cmds).sum();
    let secs = steps.len() as f64 / HZ;
    let rate = total as f64 / secs;
    let horizon = inf_audio::AUDIO_LOG_CAPACITY as f64 / rate;
    println!(
        "{} steps ({secs:.1} s): {total} commands ({car} on the car's keys), {rate:.1}/s — the {}-entry ring holds {horizon:.0} s at this rate",
        steps.len(),
        inf_audio::AUDIO_LOG_CAPACITY
    );
    assert!(total > 2_048, "the course is too short to test the ring");
}

// ── 11. THE COST ────────────────────────────────────────────────────────────

/// Sixty-four cars on a grid, `voiced` of them with an emitter, every one
/// driven at part throttle with a gentle weave so all 64 are running.
fn grid_sim(voiced: usize) -> inf_player::runtime_sim::RuntimeSim {
    let def = course_def();
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    world.world_mut().entity_mut(e).insert(slab_bits(
        DVec3::new(0.0, -0.5, 200.0),
        DVec3::new(300.0, 0.5, 400.0),
        0.9,
    ));
    for i in 0..64usize {
        let (x, z) = ((i % 8) as f64 * 8.0 - 28.0, (i / 8) as f64 * 12.0);
        spawn_car(
            &mut world,
            Uuid::from_u128(0x5E3E_1000 + i as u128),
            DVec3::new(x, car_at(&def).y, z),
            &def,
            i < voiced,
        );
    }
    world.propagate();
    inf_player::runtime_sim::RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ)
}

fn drive_grid(sim: &mut inf_player::runtime_sim::RuntimeSim, step: u32) {
    for i in 0..64u128 {
        if let Some(v) = sim
            .bridge3d_mut()
            .vehicle_mut(Uuid::from_u128(0x5E3E_1000 + i))
        {
            v.control(VehicleControls {
                throttle: 0.6,
                steer: if (step / 45).is_multiple_of(2) {
                    0.15
                } else {
                    -0.15
                },
                occupied: true,
                ..Default::default()
            });
        }
    }
}

/// **SIXTY-FOUR CARS COST WHAT THEY PRINT** — the audio phase's own clock
/// (`set_step_profiling`) and the whole step, at 64 driven cars with **1**
/// voiced (the hero's car among traffic, which is voiceless by tier) and with
/// **all 64** voiced (the worst case the tier rule prevents), against the same
/// 64 with NONE voiced (the CONTROL). MIN of five rounds of 60 steps after 60
/// warm-up steps; reported everywhere, asserted against
/// `AUDIO_STEP_BUDGET_MS` in a release build off CI only.
#[test]
fn sixty_four_cars_cost_what_they_print() {
    let idx = |name: &str| {
        inf_player::step_profile::STEP_PHASE_NAMES
            .iter()
            .position(|n| *n == name)
            .unwrap_or_else(|| panic!("the `{name}` phase exists"))
    };
    let (audio_i, vehicle_i) = (idx("audio"), idx("vehicle"));
    let mut rows: Vec<(usize, f64, f64, f64, f64)> = Vec::new();
    for voiced in [0usize, 1, 64] {
        let mut sim = grid_sim(voiced);
        sim.set_step_profiling(true);
        let mut step = 0u32;
        for _ in 0..60 {
            drive_grid(&mut sim, step);
            sim.step_once(Default::default());
            step += 1;
        }
        let (mut audio, mut vehicle, mut total) = (f64::MAX, f64::MAX, f64::MAX);
        let queued = |s: &inf_player::runtime_sim::RuntimeSim| {
            s.audio_command_log().len() as u64 + s.dropped_audio_commands()
        };
        let before = queued(&sim);
        let mut n_steps = 0usize;
        for _ in 0..5 {
            let (mut a, mut v, mut t) = (0.0, 0.0, 0.0);
            for _ in 0..60 {
                drive_grid(&mut sim, step);
                sim.step_once(Default::default());
                step += 1;
                n_steps += 1;
                let p = sim.step_profile();
                a += p.ms[audio_i];
                v += p.ms[vehicle_i];
                t += p.total_ms();
            }
            audio = audio.min(a / 60.0);
            vehicle = vehicle.min(v / 60.0);
            total = total.min(t / 60.0);
        }
        // The ring evicts in the worst case, which is the tier rule's reason;
        // the count is the log PLUS what fell off it, so it is still exact.
        if voiced <= 1 {
            assert_eq!(sim.dropped_audio_commands(), 0, "{voiced} voiced evicted");
        }
        let cmds = (queued(&sim) - before) as f64 / n_steps as f64;
        println!(
            "  {voiced} voiced: the {}-entry ring holds {:.0} s at this rate",
            inf_audio::AUDIO_LOG_CAPACITY,
            inf_audio::AUDIO_LOG_CAPACITY as f64 / (cmds * HZ).max(1.0)
        );
        rows.push((voiced, cmds, audio, vehicle, total));
    }
    println!(
        "\nVEH3e COST TABLE ({} build), 64 cars driven, MIN of 5 rounds of 60 steps:",
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        }
    );
    println!(
        "  {:>7} {:>10} {:>10} {:>10} {:>10}",
        "voiced", "cmds/step", "audio ms", "vehicle ms", "step ms"
    );
    for r in &rows {
        println!(
            "  {:>7} {:10.2} {:10.4} {:10.4} {:10.4}",
            r.0, r.1, r.2, r.3, r.4
        );
    }
    let (control, one, all) = (rows[0], rows[1], rows[2]);
    println!(
        "  the stack's marginal step cost: {:.4} ms at 1 voiced, {:.4} ms at 64",
        one.4 - control.4,
        all.4 - control.4
    );
    assert!(
        all.1 > one.1 && one.1 > control.1,
        "the voiced cars queued nothing"
    );
    assert!(
        all.2 > 0.0,
        "the audio clock read zero, so it was not armed"
    );
    if cfg!(debug_assertions) {
        eprintln!("dev build: the cost is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the cost is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        all.2 <= inf_player::budget::AUDIO_STEP_BUDGET_MS,
        "the audio phase cost {:.4} ms at 64 voiced cars against a {} ms ceiling {}",
        all.2,
        inf_player::budget::AUDIO_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

// ── 12. THE PACK ────────────────────────────────────────────────────────────

/// **A COOKED PACK CARRIES EVERY VEHICLE CLIP** — the WPN2c audit's H2 law: the
/// command stream cannot see whether its clip is in the pack, so an arm reads
/// the PACK. The gameplay fixture is scaffolded, the vehicle library copied in,
/// cooked, and the pack's own audio index read: all twenty-eight GUIDs, each
/// decoding at 22 050 Hz.
///
/// **Mutation → red**: `vehicle_clips()` dropped from `engine_spawned_clips`.
#[test]
fn a_cooked_pack_carries_every_vehicle_clip() {
    let lib = inf_editor_core::samples::vehicle_audio_dir();
    let tmp = tempfile::tempdir().expect("a temp dir");
    let proj = tmp.path().join("proj");
    inf_project::ProjectManifest::new("Vehicle Audio", "blank-3d")
        .save(&proj)
        .expect("the project scaffolds");
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).expect("a content root");
    for dir in [inf_editor_core::samples::gameplay_dir(), lib] {
        for entry in std::fs::read_dir(&dir).expect("a folder") {
            let path = entry.expect("an entry").path();
            if let Some(name) = path.file_name() {
                std::fs::copy(&path, content.join(name)).expect("copy");
            }
        }
    }
    let out: PathBuf = tmp.path().join("out");
    inf_packager::cook(&proj, &out, &inf_packager::CookOptions::default()).expect("it cooks");
    let source = inf_player::level::PackLevelSource::open(&out).expect("the pack opens");
    let carried = source.audio_assets().expect("the pack's audio index");
    let mut bytes = 0usize;
    for (i, guid) in va::vehicle_clips().iter().enumerate() {
        let a = carried
            .get(guid)
            .unwrap_or_else(|| panic!("the pack does not carry {}", va::VEHICLE_CLIP_NAMES[i]));
        let s = a.decode().expect("it decodes");
        assert_eq!(s.sample_rate(), 22_050);
        bytes += a.bytes.len();
    }
    println!(
        "the cooked pack carries all {} vehicle clips, {bytes} bytes",
        va::vehicle_clips().len()
    );
}

// ── 13. DETERMINISM ─────────────────────────────────────────────────────────

/// **THE PLANNER IS A FUNCTION OF THE STREAM SO FAR**: two runs of the course
/// queue byte-identical streams, and the planner's source reads no clock and
/// draws no random number.
#[test]
fn the_planner_is_a_function_of_the_stream_so_far() {
    let a = shipped_course();
    let b = shipped_course();
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x.cmds, y.cmds, "step {i} differs between two runs");
    }
    let src = include_str!("../../../crates/inf-ecs/src/vehicle_audio.rs");
    for banned in [
        "Instant",
        "SystemTime",
        "rand::",
        "thread_rng",
        ".sin(",
        ".cos(",
        ".exp(",
        ".powf(",
    ] {
        assert!(
            !src.contains(banned),
            "vehicle_audio.rs reaches for `{banned}`"
        );
    }
    let n: usize = a.iter().map(|s| s.cmds.len()).sum();
    println!("two runs, {} steps, {n} commands each, identical", a.len());
}

// ── 14. THE INSTRUMENTS ─────────────────────────────────────────────────────

/// **THE SHIPPED HOST DRAWS THE AUDIO ROW AND LOGS THE COLUMNS** — the Ring-0
/// row formats what it is handed, and the two host files hand it the ENGINE's
/// own voice state (`AudioEngine::voice_params`), never the planner's.
#[test]
fn the_shipped_host_draws_the_audio_row_and_logs_the_columns() {
    let t = telemetry(1.6, 12.0, SurfaceClass::Gravel);
    let row = va::voice_readout(&t, Some(1.25), None, [Some(0.0), Some(0.61)], 9);
    println!("{row}");
    assert!(row.starts_with("AUDIO P8X"));
    assert!(row.contains("WHINE 1.25") && row.contains("TURBO -") && row.contains("R0.61 loose"));
    let window = include_str!("../src/window.rs");
    let drive = include_str!("../src/pie_drive.rs");
    for (src, what) in [(window, "window.rs"), (drive, "pie_drive.rs")] {
        assert!(
            src.contains("inf_ecs::vehicle_audio::"),
            "{what} does not read the vehicle voice"
        );
        assert!(
            src.contains("voice_params("),
            "{what} reads no engine voice state"
        );
    }
    assert!(window.contains("voice_readout("));
    assert!(
        window.contains("[drive, row, damage, audio]"),
        "the audio row is built and never drawn"
    );
}
