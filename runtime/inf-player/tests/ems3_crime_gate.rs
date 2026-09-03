//! **DITCH THE CAR, CHANGE THE COAT, WALK PAST THE CRUISER** (wave EMS3).
//!
//! # What this gate is for
//!
//! The mandate is one paragraph and every clause of it is a verb here:
//!
//! > *"There will be 'Criminal Profiles' that will be dynamically built through
//! > the game. The police will remember the clothing that a user was wearing
//! > when committing a crime or the vehicle they were driving — and the user
//! > will realistically have to ditch their car and/or their clothes to evade
//! > the police and drop their wanted level."*
//!
//! So the gate runs **two traces of the same crime**, in the same town, at the
//! same hour, past the same officer — and the *only* difference between them is
//! whether the hero opened a wardrobe.
//!
//! * **the escape** — carjack a car in outfit A, drive, ditch it, change at a
//!   wardrobe, walk past the searching cruiser: **not recognised**, and the heat
//!   falls to cold;
//! * **the failure** — the identical trace with the wardrobe skipped:
//!   **spotted**, and the search re-anchors onto the hero.
//!
//! # The seven arms
//!
//! * **(a)** the escape;
//! * **(b)** the failure — the same crime, the same officer, no swap;
//! * **(c)** the recognition table at the checkpoints, outfit-vs-swapped and
//!   day-vs-night, printed as the numbers the design rests on;
//! * **(d)** PIE == shipping, byte for byte, over both traces;
//! * **(e)** **THE LAW ARM** — the police search converges on *last-seen* and
//!   never on the player's true transform;
//! * **(f)** the budget — what recognition costs inside the `dispatch` phase;
//! * **(g)** the falsifier — the same town where nobody commits anything files
//!   nothing, draws no stars and casts no rays.
//!
//! Rush hour is `ems2_dispatch_gate::TRAFFIC_HOUR`'s **14:00** rather than 08:00,
//! and for that gate's measured reason: at the commuter peak a cross-town
//! response can be blocked behind a civilian car VEH2b's `drive_intent` stopped
//! in lane. The number is inherited deliberately so the two EMS gates drive the
//! same town.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    MovementMode, PcgVolume, ResidentSlot, RigidBody3D, ScatteredInstance, SlotRole,
    StreamingSource, TimeOfDay, Transform,
};
use inf_ecs::crime::{self, Channel, Description, Response};
use inf_ecs::crowd::{appearance_of, Appearance};
use inf_ecs::math::{Color, Vec2d, Vec3d};
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_physics::WorldGravity;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;

/// A 3×3 grid of 80 m blocks on a 100 m pitch — `ems2_dispatch_gate`'s town.
const PITCH: f64 = 100.0;
const STREET: f64 = 20.0;

/// The hour the town is driven at — see the module header.
const TRAFFIC_HOUR: f64 = 14.0;

/// Steps run before the crime, so the fleet derives and the carriageway settles.
const WARMUP: u32 = 240;

/// Steps run after it. Long enough for the cruiser to be dispatched and to
/// drive, and for `HEAT_DECAY_STEPS` to bite twice on a two-point file.
const RUN: u32 = 2400;

const HERO: Uuid = Uuid::from_u128(0x0E55_0001);
const GROUND: Uuid = Uuid::from_u128(0x0E55_0002);
const SKY: Uuid = Uuid::from_u128(0x0E55_0003);
const CRUISER: Uuid = Uuid::from_u128(0x0E55_0010);
const GETAWAY: Uuid = Uuid::from_u128(0x0E55_0011);
const VICTIM: Uuid = Uuid::from_u128(0x0E55_0012);
const WITNESS: Uuid = Uuid::from_u128(0x0E55_0013);

/// Where the crime happens — a street corner in the middle of the town.
const CRIME_AT: DVec3 = DVec3::new(50.0, 0.0, 0.0);

/// **Where the act happens** — beside the driver's door and at chest height,
/// which is where `d3::carjack::try_carjack` raises one (`door_point`) and is
/// not a coincidence: an act recorded at the chassis ORIGIN is a point INSIDE
/// the car, and the line-of-sight ray to it grazes the roof.
///
/// That marginality is measured rather than assumed. The first cut put the act
/// at `CRIME_AT + Y` — 1.0 m up, over a chassis whose settled roof is about
/// 1.0 m — and the two hosts recorded **different observer lists**: 3 927 bytes
/// against 3 911, which is exactly one `Uuid`. Two hosts whose cars settle a
/// hair differently flip a grazing ray, and the trace says so on the step it
/// happens. Beside the door at 1.4 m the ray is a clear horizontal line and the
/// two hosts agree byte for byte for 2 400 steps.
const DOOR_AT: DVec3 = DVec3::new(52.5, 1.4, 0.0);

/// Where the wardrobe is. On the same street, forty metres along, so getting to
/// it is a walk rather than a teleport.
const WARDROBE_AT: DVec3 = DVec3::new(90.0, 0.0, 0.0);

/// Where the hero stands to be looked at. Eight metres from the officer, which
/// is inside every row of `Channel::weight`'s measured table — so if the swap
/// did not work, this is a recognition.
const CHECKPOINT: DVec3 = DVec3::new(120.0, 0.0, 0.0);

fn blocks(world: &mut EcsWorld) {
    let half = (PITCH - STREET) * 0.5;
    for row in 0..3i32 {
        for col in 0..3i32 {
            let c = DVec2::new(f64::from(col) * PITCH, f64::from(row) * PITCH);
            let guid = Uuid::from_u64_pair(0x0E55_0F00, (row as u64) << 32 | col as u64);
            let e = world.spawn_with_guid(guid, "block", None);
            world.world_mut().entity_mut(e).insert(Transform {
                translation: Vec3d::new(c.x, 0.0, c.y),
                rotation: Vec3d::ZERO,
                scale: Vec3d::ONE,
            });
            let mut v = PcgVolume {
                extent: Vec2d::new(half, half),
                ..Default::default()
            };
            v.residents = vec![ResidentSlot {
                role: SlotRole::Home,
                at: DVec3::new(c.x, 0.0, c.y),
                room: 0,
                building: 0,
                floor: 0,
                index: 0,
                node: 0,
                posture: inf_ecs::components::SlotPosture::Stand,
                shift: inf_ecs::components::SlotShift::Day,
                face: DVec3::ZERO,
            }];
            // **THE WARDROBE**, in the block nearest the crime. One instance in
            // the volume's own derived list, exactly as `inf-pcg`'s assembler
            // leaves one in a bedroom — the same field `inf_ecs::wardrobe`
            // reads and the same mesh GUID the palette draws it under.
            if row == 0 && col == 1 {
                v.evaluated = vec![ScatteredInstance {
                    position: WARDROBE_AT,
                    rotation: glam::DQuat::IDENTITY,
                    scale: 1.0,
                    kind: 0,
                    mesh: Some(inf_ecs::wardrobe::WARDROBE_MESH_GUID),
                    extent: None,
                    glow: 0.0,
                    surface: Default::default(),
                }];
            }
            world.world_mut().entity_mut(e).insert(v);
        }
    }
}

fn ground(world: &mut EcsWorld) {
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(100.0, -0.5, 100.0);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(500.0, 0.5, 500.0),
            ..Default::default()
        },
    ));
}

/// Park the island's own cruiser row, wearing its own livery — the same door
/// `ems2_dispatch_gate` parks a fleet through, so what this gate dispatches is
/// the content the island parks.
fn park_cruiser(world: &mut EcsWorld, at: DVec3) {
    let defs = inf_editor_core::vehicle::island_vehicles();
    let def = defs.get("cruiser").expect("no `cruiser` row");
    let livery = inf_editor_core::vehicle::island_vehicle_livery("cruiser").expect("no livery");
    let rest_y =
        inf_editor_core::vehicle::resting_origin_y(def, 0.0) + inf_editor_core::island::CAR_LIFT_M;
    inf_ecs::vehicle::spawn_rig_at(
        world,
        CRUISER,
        def,
        &inf_ecs::vehicle::RigSpawn {
            name: "Station cruiser".to_string(),
            at: DVec3::new(at.x, rest_y, at.z),
            yaw_deg: 0.0,
            paint: Color::new(0.35, 0.36, 0.38, 1.0),
            clip: None,
            engine_voice: false,
            livery: Some(livery),
        },
        true,
    );
}

/// A character at `at` — the shape both the hero and the carjack victim take.
fn person(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, player: bool) {
    let e = world.spawn_with_guid(guid, name, None);
    world.world_mut().entity_mut(e).insert((
        Transform::from_translation(at),
        CharacterController3D::default(),
        CharacterMovement {
            player_controlled: player,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            radius: 0.35,
            half_extents: Vec3d::new(0.35, 0.9, 0.35),
            ..Default::default()
        },
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
    ));
}

/// The getaway car, with somebody in it — a plain civilian sedan, parked at the
/// crime scene with a driver whose seat the hero is about to take.
fn getaway(world: &mut EcsWorld) {
    let mut def = inf_ecs::vehicle::VehicleDef::default();
    inf_ecs::traffic::size_the_suspension(&mut def);
    let sag = def.class.travel_m * inf_ecs::traffic::STATIC_SAG_FRAC;
    let rest_y = -def.wheel_drop_m + def.wheel_radius_m - sag;
    inf_ecs::vehicle::spawn_rig_at(
        world,
        GETAWAY,
        &def,
        &inf_ecs::vehicle::RigSpawn {
            name: "Getaway".to_string(),
            at: DVec3::new(CRIME_AT.x, rest_y, CRIME_AT.z),
            yaw_deg: 0.0,
            // A red sedan — one half of the description the police will keep.
            paint: Color::new(0.82, 0.12, 0.10, 1.0),
            clip: None,
            engine_voice: false,
            livery: None,
        },
        true,
    );
    person(world, VICTIM, "Driver", CRIME_AT + DVec3::Y, false);
    let e = world.entity_of(VICTIM).expect("the driver exists");
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.mode = MovementMode::Driving;
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: GETAWAY,
            entering: false,
            time_s: 9.0,
            start: Vec3d::from_dvec3(CRIME_AT),
            start_yaw_deg: 0.0,
        };
    }
}

/// **The one fixture**, so the two hosts of arm (d) cannot be given two towns.
fn build(world: &mut EcsWorld) {
    blocks(world);
    ground(world);
    park_cruiser(world, DVec3::new(-46.0, 0.0, 0.0));
    getaway(world);
    person(
        world,
        HERO,
        "Hero",
        CRIME_AT + DVec3::new(2.2, 1.0, 0.0),
        true,
    );
    let s = world.spawn_with_guid(SKY, "Sky", None);
    world.world_mut().entity_mut(s).insert(TimeOfDay {
        seconds: TRAFFIC_HOUR * 3600.0,
        rate: 0.0,
        ..Default::default()
    });
    let e = world.entity_of(HERO).expect("the hero exists");
    world
        .world_mut()
        .entity_mut(e)
        .insert(StreamingSource { radius_m: 1024.0 });
    world.mark_dirty();
    world.propagate();
}

/// **Commit the crime** — through `witness::raise_act`, which is the ONE door
/// `d3::carjack::try_carjack` itself raises a carjack through.
///
/// # Why the door and not the whole verb, stated
///
/// `try_carjack` needs a `PhysicsBridge3D` and the editor's `SimSession` does
/// not expose one, so arm (d) could not run the same script on both hosts
/// through it. What a carjack *adds* on top of this call is ejecting the victim
/// and marking the car taken — VEH2b's clause, certified end to end by
/// `traffic_3d`'s own carjack arm, and unchanged by this wave. What THIS wave
/// owns starts here: the act is observed by the gameplay phase's own pass, with
/// its own line-of-sight rays, its own observer list and the real `look_digest`
/// and `actor_vehicle` of whoever did it.
///
/// The hero is put in the seat first, so the vehicle channel is live at the
/// moment the description is taken — which is the state a carjack leaves a
/// player in and the whole of the "or the vehicle they were driving" clause.
fn commit(world: &mut EcsWorld) {
    take_the_wheel(world);
    inf_ecs::witness::raise_act(world, inf_ecs::witness::ActKind::Carjack, HERO, DOOR_AT);
}

/// Put the hero in the seat.
fn take_the_wheel(world: &mut EcsWorld) {
    let Some(e) = world.entity_of(HERO) else {
        return;
    };
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.mode = MovementMode::Driving;
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: GETAWAY,
            entering: false,
            time_s: 9.0,
            start: Vec3d::from_dvec3(CRIME_AT),
            start_yaw_deg: 0.0,
        };
    }
}

/// …and out of it again — the ditch.
fn leave_the_wheel(world: &mut EcsWorld) {
    let Some(e) = world.entity_of(HERO) else {
        return;
    };
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.mode = MovementMode::Grounded;
        cm.runtime.seat = inf_ecs::components::SeatState::default();
    }
}

/// Stand somewhere and face something.
fn stand(world: &mut EcsWorld, at: DVec3, facing: DVec3) {
    let Some(e) = world.entity_of(HERO) else {
        return;
    };
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::from_dvec3(at);
    }
    let yaw = inf_ecs::traffic::yaw_of_dir(facing - at);
    if let Some(mut cm) = world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.runtime.aim_yaw_deg = yaw;
        cm.runtime.body_yaw_deg = yaw;
    }
    world.propagate();
}

/// Where the officer is standing, or `None` while the cruiser is in its bay.
fn officer_at(world: &EcsWorld) -> Option<DVec3> {
    let crew = inf_ecs::dispatch::crew_guid(CRUISER);
    let e = world.entity_of(crew)?;
    world
        .world()
        .get::<Transform>(e)
        .map(|t| t.translation.to_dvec3())
}

/// **A BYSTANDER ON THE CORNER** — and it is seeded AFTER the host exists, which
/// is a finding rather than a style.
///
/// # Two reasons, and the second one cost this gate a red PIE arm
///
/// It is a **crowd record** and not a character entity, because that is what
/// `witness::candidates_near` reads: an observer of an act is a pedestrian, and
/// a pedestrian in this engine is a row in `CrowdPopulationRes`. A person built
/// out of components is a *character*, which is a different thing and is
/// invisible to the pass. This town's blocks declare `ResidentSlot`s and its
/// society never derives an agent from them — it has no interior nav to walk —
/// so without this a crime in the middle of it is a crime **nobody saw**, and
/// `crime::report_act` refuses it.
///
/// And it is installed **after the host is built**, because a population is a
/// RESOURCE and `SimSession::enter_with_gravity` calls `clear_crowd` on the way
/// in — deliberately, so an editor session starts from nothing. Seeding it in
/// `build` therefore gave the shipped player a witness and the editor an empty
/// street, and the two hosts diverged on the exact step of the crime: the
/// witness section folded **98 bytes against 82**, which is one `Uuid` of
/// observer list. It looked like a line-of-sight divergence and was not; the
/// trace named the step and the section, which is what that section is for.
fn seed_crowd(world: &mut EcsWorld) {
    let mut crowd = std::collections::BTreeMap::new();
    crowd.insert(
        WITNESS,
        inf_ecs::crowd::CrowdRecord::standing(
            inf_ecs::crowd::CrowdArchetype::humanoid(None, None, None),
            DVec3::new(DOOR_AT.x, 0.0, 10.0),
        ),
    );
    inf_ecs::crowd::set_population(world, crowd);
}

// ── the script ──────────────────────────────────────────────────────────────

/// What the hero does, and when.
///
/// One table, so the two traces differ in exactly one boolean and the two hosts
/// run the same script by construction rather than by two copies of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Beat {
    /// Nothing. The world runs.
    Wait,
    /// Take the car in front of a witness — the crime.
    Commit,
    /// Get out, walk away from it, and stand at the wardrobe.
    Ditch,
    /// Press E on the wardrobe.
    Change,
    /// Stand in front of the officer.
    Checkpoint,
}

/// The step each beat lands on. Spread out, because the dispatcher assigns at
/// most one incident a step and a cruiser has to drive out of its bay.
const SCRIPT: [(u32, Beat); 4] = [
    (10, Beat::Commit),
    (300, Beat::Ditch),
    (360, Beat::Change),
    (600, Beat::Checkpoint),
];

fn beat_at(step: u32) -> Beat {
    SCRIPT
        .iter()
        .find(|(s, _)| *s == step)
        .map(|(_, b)| *b)
        .unwrap_or(Beat::Wait)
}

/// What one host's whole run produced.
#[derive(Debug, Clone, PartialEq)]
struct Run {
    /// Heat, step by step.
    heat: Vec<u32>,
    /// Where the police believed the hero was, step by step.
    believed: Vec<Option<DVec3>>,
    /// The highest rung the ladder ever reached.
    peak: Response,
    /// Recognitions over the run.
    recognised: usize,
    /// Rays cast, and rays a wall stopped.
    rays: (usize, usize),
    /// Clear views that still did not clear the threshold.
    unrecognised: usize,
    /// The most officers ever on the beat at once.
    officers: usize,
    /// The outfit at the crime, and at the checkpoint.
    outfits: (u8, u8),
    /// Heat at the checkpoint.
    at_checkpoint: u32,
    /// Whether the wardrobe press changed the description.
    changed: bool,
    /// Incidents opened, and how many scenes were linked to a file.
    dispatched: (u64, usize),
    /// The trace, step by step.
    trace: Vec<Vec<u8>>,
    /// Per-section lengths, step by step.
    ///
    /// **A diagnostic that earned its place.** The PIE arm's first red run said
    /// only "3 927 bytes against 3 911", which is a number nobody can act on;
    /// with this it said `[0, 98, 49, 3780]` against `[0, 82, 49, 3780]`, which
    /// is one `Uuid` of observer list and named the system in one line.
    sections: Vec<[usize; 4]>,
}

impl Run {
    fn new() -> Self {
        Self {
            heat: Vec::with_capacity(RUN as usize),
            believed: Vec::with_capacity(RUN as usize),
            peak: Response::Cold,
            recognised: 0,
            rays: (0, 0),
            unrecognised: 0,
            officers: 0,
            outfits: (0, 0),
            at_checkpoint: 0,
            changed: false,
            dispatched: (0, 0),
            trace: Vec::with_capacity(RUN as usize),
            sections: Vec::with_capacity(RUN as usize),
        }
    }

    fn end_heat(&self) -> u32 {
        self.heat.last().copied().unwrap_or(0)
    }
}

/// **The four sections this gate compares.**
///
/// Not `RuntimeSim::state_bytes`, for `ems1_station_gate`'s reason: the editor
/// host has no such method, so the two sides would be comparing two different
/// functions. These are the Ring-0 folds both hosts read out of the same world,
/// and they are the ones this wave moves — the ledger, what the street saw, the
/// dispatcher (whose `searches` map this wave appended to) and the traffic.
fn sections(world: &EcsWorld) -> [usize; 4] {
    [
        inf_ecs::crime::profile_state_bytes(world).len(),
        inf_ecs::witness::witness_state_bytes(world).len(),
        inf_ecs::dispatch::dispatch_state_bytes(world).len(),
        inf_ecs::traffic::traffic_state_bytes(world).len(),
    ]
}

fn trace_of(world: &EcsWorld) -> Vec<u8> {
    let mut out = inf_ecs::crime::profile_state_bytes(world);
    out.extend_from_slice(&inf_ecs::witness::witness_state_bytes(world));
    out.extend_from_slice(&inf_ecs::dispatch::dispatch_state_bytes(world));
    out.extend_from_slice(&inf_ecs::traffic::traffic_state_bytes(world));
    out
}

/// Apply this step's beat, and say whether the host should press `interact`.
///
/// World-only, so **both** hosts call this one function.
fn apply(beat: Beat, swap: bool, run: &mut Run, world: &mut EcsWorld) -> bool {
    match beat {
        Beat::Wait => {}
        Beat::Commit => {
            run.outfits.0 = appearance_of(world, HERO).outfit;
            commit(world);
        }
        Beat::Ditch => {
            leave_the_wheel(world);
            stand(
                world,
                WARDROBE_AT + DVec3::new(-1.4, 1.0, 0.0),
                WARDROBE_AT + DVec3::Y,
            );
        }
        Beat::Change => {
            if swap {
                // **THE REAL PRESS.** The `interact` edge goes onto the camera
                // subject's `MovementRuntime`, `step_character_movement`
                // resolves it through the ONE interaction site, and the verb
                // match dispatches `InteractVerb::Change`. Nothing in this file
                // calls `change_clothes`.
                return true;
            }
        }
        Beat::Checkpoint => {
            let officer = officer_at(world).unwrap_or(CHECKPOINT);
            let at = DVec3::new(officer.x + 8.0, 1.0, officer.z);
            stand(world, at, officer);
            run.outfits.1 = appearance_of(world, HERO).outfit;
            run.at_checkpoint = crime::heat_of(world, HERO);
        }
    }
    false
}

/// Read the ledger into the run — the same reader on both hosts.
fn observe(run: &mut Run, world: &EcsWorld, stats: inf_physics::d3::RecognitionStats) {
    run.heat.push(crime::heat_of(world, HERO));
    run.believed
        .push(crime::profile_of(world, HERO).map(|p| p.last_seen()));
    if let Some(p) = crime::profile_of(world, HERO) {
        run.peak = run.peak.max(p.response());
    }
    run.recognised += stats.recognised;
    run.rays.0 += stats.rays;
    run.rays.1 += stats.blocked;
    run.unrecognised += stats.unrecognised;
    run.officers = run.officers.max(stats.officers);
    if let Some(d) = inf_ecs::dispatch::dispatch_of(world) {
        run.dispatched = (d.opened, d.searches.len().max(run.dispatched.1));
    }
    run.trace.push(trace_of(world));
    run.sections.push(sections(world));
}

// ── the two hosts ───────────────────────────────────────────────────────────

fn player_run(swap: bool) -> Run {
    let mut world = EcsWorld::new();
    build(&mut world);
    // EARTH, and it is load-bearing — `ems2_dispatch_gate`'s own measured note:
    // the default fixture spelling gives a level with no 3D gravity, where a
    // parked car never falls onto its springs and a fully-steered unit sits at
    // its station looking exactly like a dispatcher that wrote no stick.
    let mut sim = RuntimeSim::with_gravity(world, Vec::new(), WorldGravity::EARTH, HZ);
    seed_crowd(sim.world_mut());
    for _ in 0..WARMUP {
        sim.step_once(RuntimeInput::default());
    }
    let mut run = Run::new();
    for step in 0..RUN {
        let press = apply(beat_at(step), swap, &mut run, sim.world_mut());
        let input = if press {
            RuntimeInput::with_down(["interact"])
        } else {
            RuntimeInput::default()
        };
        sim.step_once(input);
        if press {
            run.changed = appearance_of(sim.world(), HERO).outfit != run.outfits.0;
        }
        observe(&mut run, sim.world(), sim.dispatch_stats().recognition);
    }
    run
}

fn editor_run(swap: bool) -> Run {
    let mut doc = SceneDoc::new();
    build(doc.world_mut());
    let mut session = SimSession::enter_with_gravity(&mut doc, Vec::new(), WorldGravity::EARTH, HZ);
    seed_crowd(doc.world_mut());
    for _ in 0..WARMUP {
        session.step_once(&mut doc, SimInput::default());
    }
    let mut run = Run::new();
    for step in 0..RUN {
        let press = apply(beat_at(step), swap, &mut run, doc.world_mut());
        let input = if press {
            SimInput::with_down(["interact"])
        } else {
            SimInput::default()
        };
        session.step_once(&mut doc, input);
        if press {
            run.changed = appearance_of(doc.world(), HERO).outfit != run.outfits.0;
        }
        observe(&mut run, doc.world(), session.dispatch_stats().recognition);
    }
    session.exit(&mut doc);
    run
}

// ── (a) the escape ──────────────────────────────────────────────────────────

/// **CARJACK IN ONE COAT, WALK PAST IN ANOTHER, AND THE HEAT FALLS TO COLD.**
///
/// The mandate's whole sentence, as one trace. Every claim is a *world* fact
/// rather than a counter: the file exists and holds both channels, the officer
/// is on the road, the hero stands eight metres in front of them in plain
/// daylight, and the ledger's answer is that this is not the man.
#[test]
fn the_hero_changes_at_a_wardrobe_and_walks_past_the_cruiser() {
    let run = player_run(true);
    println!(
        "the escape: outfit {} -> {}, peak {}, heat at the checkpoint {}, at the end {}",
        run.outfits.0,
        run.outfits.1,
        run.peak.name(),
        run.at_checkpoint,
        run.end_heat()
    );
    println!(
        "  {} officer(s) on the beat, {} ray(s) cast ({} blocked), \
         {} pair(s) considered and NOT recognised, {} recognition(s)",
        run.officers, run.rays.0, run.rays.1, run.unrecognised, run.recognised
    );
    // THE CRIME LANDED — armed, so nothing below is a statement about an empty
    // ledger.
    assert!(
        run.heat.iter().any(|h| *h > 0),
        "the carjack never opened a file — every arm below is vacuous"
    );
    assert_eq!(
        run.peak,
        Response::Patrol,
        "one carjack should bring a patrol car and nothing more"
    );
    // THE WARDROBE WORKED, and it worked through the real press.
    assert!(run.changed, "the E press did not change the description");
    assert_ne!(
        run.outfits.0, run.outfits.1,
        "the hero stood at the checkpoint in the coat they did the job in"
    );
    // THE POLICE WERE THERE AND WERE LOOKING.
    assert!(
        run.officers > 0,
        "no officer ever left the station — the checkpoint was watched by nobody"
    );
    // `unrecognised` counts a pair the pass CONSIDERED — it is only reached past
    // the range gate, so this is the claim that the officer was within
    // `RECOGNITION_RANGE_M` of the hero and scored them at zero. A swapped coat
    // and a ditched car match on no channel, so the pass spends no ray at all
    // and `rays` is legitimately zero here; the disjunction is what makes the
    // arm true of both outcomes rather than of a fixture that never met.
    assert!(
        run.rays.0 > 0 || run.unrecognised > 0,
        "the recognition pass never considered the hero at all"
    );
    // …AND DID NOT RECOGNISE HIM.
    assert_eq!(
        run.recognised, 0,
        "the swapped coat was recognised anyway — the evasion route does not work"
    );
    // …SO THE HEAT FELL TO COLD.
    assert_eq!(
        run.end_heat(),
        0,
        "the heat never decayed: {:?}",
        &run.heat[run.heat.len().saturating_sub(5)..]
    );
}

// ── (b) the failure ─────────────────────────────────────────────────────────

/// **THE SAME CRIME WITHOUT THE WARDROBE IS A RECOGNITION**, and the search
/// re-anchors onto the hero.
///
/// The control for arm (a), and the reason the two are one fixture with one
/// boolean between them: if this passed as well, arm (a) would be certifying a
/// recognition pass that never recognises anybody.
#[test]
fn the_same_crime_without_the_swap_is_spotted() {
    let run = player_run(false);
    println!(
        "the failure: outfit {} -> {} (unchanged), {} recognition(s), \
         {} ray(s) ({} blocked)",
        run.outfits.0, run.outfits.1, run.recognised, run.rays.0, run.rays.1
    );
    assert!(!run.changed, "the control run changed its clothes");
    assert_eq!(
        run.outfits.0, run.outfits.1,
        "the hero was not in the same coat at the checkpoint"
    );
    assert!(
        run.recognised > 0,
        "an unchanged description in plain view at eight metres was not recognised — \
         which would make the escape arm a statement about a pass that never fires"
    );
    // THE SEARCH FOLLOWED. The ledger's last-seen ends up away from the crime
    // scene, at a place an officer actually looked at.
    let believed = run
        .believed
        .iter()
        .rev()
        .find_map(|b| *b)
        .expect("a file was open at some point");
    println!("  the police last believed the hero was at {believed:?}");
    let crime_dist = (believed - DOOR_AT).length();
    assert!(
        crime_dist > 10.0,
        "the search never moved off the crime scene ({crime_dist:.1} m)"
    );
    // …and the dispatcher has the scene on file as a SEARCH.
    assert!(
        run.dispatched.1 > 0,
        "no crime scene was ever linked to a criminal file"
    );
}

// ── (c) the table ───────────────────────────────────────────────────────────

/// **THE RECOGNITION SCORES AT THE CHECKPOINT**, printed as the numbers the
/// design rests on, and asserted where the design has a claim.
///
/// Pure arithmetic over the same `match_score` and `sight_factor` the pass
/// calls, at the distance the checkpoint stands at, for the four cases the
/// mandate distinguishes.
#[test]
fn the_recognition_table_at_the_checkpoint() {
    let mut w = EcsWorld::new();
    let act = inf_ecs::witness::WitnessedAct {
        kind: inf_ecs::witness::ActKind::Carjack,
        actor: HERO,
        at: CRIME_AT,
        step: 0,
        observers: vec![WITNESS],
        actor_look: Appearance { outfit: 2 }.digest(),
        actor_vehicle: Some(GETAWAY),
    };
    crime::report_act(&mut w, &act, Some(0x1234)).expect("a file");
    let file = crime::profile_of(&w, HERO).expect("a file").clone();
    let cases = [
        (
            "in the car, same coat",
            Description {
                outfit: Appearance { outfit: 2 }.digest(),
                vehicle: Some(0x1234),
            },
        ),
        (
            "on foot, same coat",
            Description {
                outfit: Appearance { outfit: 2 }.digest(),
                vehicle: None,
            },
        ),
        (
            "in the car, swapped coat",
            Description {
                outfit: Appearance { outfit: 5 }.digest(),
                vehicle: Some(0x1234),
            },
        ),
        (
            "on foot, swapped coat",
            Description {
                outfit: Appearance { outfit: 5 }.digest(),
                vehicle: None,
            },
        ),
    ];
    println!(
        "at 8 m, threshold {:.2} — day / night",
        crime::RECOGNIZE_SCORE
    );
    let mut day: Vec<f64> = Vec::new();
    let mut night: Vec<f64> = Vec::new();
    for (name, seen) in cases {
        let channels = crime::match_score(&file, seen, 0);
        let d = channels * crime::sight_factor(8.0, false);
        let n = channels * crime::sight_factor(8.0, true);
        println!(
            "  {name:<24} {d:.3} {} / {n:.3} {}",
            if d >= crime::RECOGNIZE_SCORE {
                "SEEN"
            } else {
                "----"
            },
            if n >= crime::RECOGNIZE_SCORE {
                "SEEN"
            } else {
                "----"
            }
        );
        day.push(d);
        night.push(n);
    }
    // The four claims the table is FOR.
    assert!(
        day[0] >= crime::RECOGNIZE_SCORE,
        "the getaway car in daylight"
    );
    assert!(
        day[1] >= crime::RECOGNIZE_SCORE,
        "the same coat in daylight"
    );
    assert!(
        day[3] < crime::RECOGNIZE_SCORE,
        "ditching both did not defeat the description in daylight"
    );
    assert!(
        day[2] >= crime::RECOGNIZE_SCORE,
        "keeping the CAR should still give you away, whatever you wear"
    );
    // The night is never harder to hide in, in any row.
    for (i, (d, n)) in day.iter().zip(night.iter()).enumerate() {
        assert!(n <= d, "row {i} scored higher at night than in daylight");
    }
    assert!(
        night[1] < crime::RECOGNIZE_SCORE,
        "a coat at eight metres at night should not be enough"
    );
    // …and the two channels are what they say they are.
    assert_eq!(file.evidence.len(), 2);
    for ch in [Channel::Outfit, Channel::Vehicle] {
        assert!(
            file.evidence.contains_key(&ch),
            "{} is not on file",
            ch.name()
        );
    }
}

// ── (d) PIE == shipping ─────────────────────────────────────────────────────

/// **PIE == SHIPPING, BYTE FOR BYTE, OVER BOTH TRACES.**
///
/// The editor's Simulate session and the shipped player run the same script over
/// the same town and fold the same four sections on every one of `RUN` steps.
/// A divergence in the ledger, in what the street saw, in the dispatcher's
/// searches or in the traffic shows up on the step it happened.
#[test]
fn pie_equals_shipping_over_both_traces() {
    for swap in [true, false] {
        let ship = player_run(swap);
        let pie = editor_run(swap);
        assert_eq!(ship.trace.len(), pie.trace.len());
        for (i, (a, b)) in ship.trace.iter().zip(pie.trace.iter()).enumerate() {
            assert!(
                a == b,
                "swap={swap}: the two hosts diverged on step {i}. ship {:?} against \
                 pie {:?} — profile / witness / dispatch / traffic, in bytes. The \
                 section whose length moved is the system that disagreed; if the \
                 lengths match, the disagreement is inside one of them.",
                ship.sections[i],
                pie.sections[i]
            );
        }
        assert_eq!(ship.heat, pie.heat, "swap={swap}: the heat curves differ");
        assert_eq!(
            ship.believed, pie.believed,
            "swap={swap}: the two hosts believed the hero was in different places"
        );
        assert_eq!(ship.recognised, pie.recognised);
        assert_eq!(ship.outfits, pie.outfits);
        // ARMED: a trace of nothing would compare equal on both hosts.
        assert!(
            ship.trace.iter().any(|t| !t.is_empty()),
            "swap={swap}: every step folded an empty trace"
        );
        println!(
            "swap={swap}: {} steps compared, the longest fold is {} bytes",
            ship.trace.len(),
            ship.trace.iter().map(|t| t.len()).max().unwrap_or(0)
        );
    }
}

// ── (e) THE LAW ARM ─────────────────────────────────────────────────────────

/// **THE POLICE SEARCH CONVERGES ON LAST-SEEN, AND NEVER ON THE PLAYER.**
///
/// The wave's signature claim, as a measurement over a whole run.
///
/// The hero commits the crime, changes at the wardrobe and then walks three
/// hundred metres. From that point on the ledger's `last_seen` must be
/// **frozen** — while the hero's own transform moves every step — and every
/// crime scene the dispatcher is holding for that file must be at the frozen
/// point rather than under the hero's feet.
///
/// The falsifier is built in: the two positions are printed side by side and
/// asserted to be far apart. A system that read the transform would have them
/// equal, and every other arm in this file would still pass.
#[test]
fn the_police_search_converges_on_last_seen_and_never_on_the_player() {
    let mut world = EcsWorld::new();
    build(&mut world);
    let mut sim = RuntimeSim::with_gravity(world, Vec::new(), WorldGravity::EARTH, HZ);
    seed_crowd(sim.world_mut());
    for _ in 0..WARMUP {
        sim.step_once(RuntimeInput::default());
    }
    let mut run = Run::new();
    apply(Beat::Commit, true, &mut run, sim.world_mut());
    for _ in 0..300 {
        sim.step_once(RuntimeInput::default());
    }
    apply(Beat::Ditch, true, &mut run, sim.world_mut());
    apply(Beat::Change, true, &mut run, sim.world_mut());
    sim.step_once(RuntimeInput::with_down(["interact"]));
    assert!(
        appearance_of(sim.world(), HERO).outfit != run.outfits.0,
        "the swap did not happen, so this arm would be measuring a recognisable man"
    );
    let frozen = crime::profile_of(sim.world(), HERO)
        .expect("a file is open")
        .last_seen();
    // …and now the hero walks. A metre a step, three hundred metres, in full
    // view of anything that cared to look — but describing nobody.
    let mut moved = 0usize;
    let mut followed = 0usize;
    for i in 0..300 {
        let at = DVec3::new(WARDROBE_AT.x + i as f64, 1.0, 0.0);
        stand(sim.world_mut(), at, at + DVec3::X);
        sim.step_once(RuntimeInput::default());
        let Some(p) = crime::profile_of(sim.world(), HERO) else {
            continue;
        };
        if (p.last_seen() - frozen).length() > 1e-9 {
            moved += 1;
        }
        let Some(d) = inf_ecs::dispatch::dispatch_of(sim.world()) else {
            continue;
        };
        for (incident, suspect) in &d.searches {
            if *suspect != HERO {
                continue;
            }
            let Some(scene) = d.incidents.get(incident) else {
                continue;
            };
            if (scene.at - at).length() < 1.0 {
                followed += 1;
            }
        }
    }
    let truth = sim
        .world()
        .entity_of(HERO)
        .and_then(|e| sim.world().world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .expect("the hero is in the world");
    println!("after 300 steps of walking: the hero is at {truth:?}, the police believe {frozen:?}");
    assert_eq!(moved, 0, "the ledger learned a position no witness gave it");
    assert_eq!(
        followed, 0,
        "a crime scene followed the player without a single sighting"
    );
    assert!(
        (truth - frozen).length() > 100.0,
        "the fixture did not actually separate the two ({:.1} m)",
        (truth - frozen).length()
    );
}

// ── (f) the budget ──────────────────────────────────────────────────────────

/// **WHAT RECOGNITION COSTS**, inside the `dispatch` phase's own ceiling.
///
/// The wave mints no constant: the pass runs inside `step_dispatch`, so
/// `inf_player::budget::DISPATCH_STEP_BUDGET_MS` is the number it has to live
/// under, and it may only ever DECREASE (§8). The table is printed on every run,
/// in dev and in CI, and asserted on a real machine in release —
/// `ems2_dispatch_gate`'s own conditioning, for its reasons.
#[test]
fn recognition_rides_the_dispatch_budget() {
    let mut world = EcsWorld::new();
    build(&mut world);
    let mut sim = RuntimeSim::with_gravity(world, Vec::new(), WorldGravity::EARTH, HZ);
    seed_crowd(sim.world_mut());
    for _ in 0..WARMUP {
        sim.step_once(RuntimeInput::default());
    }
    let mut run = Run::new();
    apply(Beat::Commit, false, &mut run, sim.world_mut());
    let mut rays = 0usize;
    let mut looked = 0usize;
    let mut worst_files = 0usize;
    for _ in 0..1200 {
        // **THE HERO STANDS IN FRONT OF THE OFFICER, EVERY STEP** (EMS3 audit).
        // Without this the arm measured a `dispatch` row containing a
        // recognition pass that cast **zero rays**: the cruiser's search takes
        // it a hundred metres from a hero who never moves off the crime scene,
        // the range gate rejects every pair before a ray is spent, and the
        // ceiling assertion below was `0 <= 899 * 32`. A budget over the cheap
        // half of a pass is not a budget for the pass.
        apply(Beat::Checkpoint, false, &mut run, sim.world_mut());
        sim.step_once(RuntimeInput::default());
        let r = sim.dispatch_stats().recognition;
        rays += r.rays;
        looked += usize::from(r.officers > 0);
        worst_files = worst_files.max(r.files);
    }
    // MIN of three rounds — `ems2_dispatch_gate`'s own discipline, because a
    // step profile is one step and a single one of them is a fact about a
    // scheduler rather than about this pass.
    sim.set_step_profiling(true);
    let (rounds, per_round) = (3u32, 240u32);
    let mut best: Option<inf_player::step_profile::StepProfile> = None;
    for _ in 0..rounds {
        let mut mean = inf_player::step_profile::StepProfile::default();
        for _ in 0..per_round {
            // Still in front of the officer, so the profiled steps are the ones
            // that cast rays rather than the ones that reject on range.
            apply(Beat::Checkpoint, false, &mut run, sim.world_mut());
            sim.step_once(RuntimeInput::default());
            mean.accumulate(&sim.step_profile());
        }
        mean.scale(1.0 / f64::from(per_round));
        if best.as_ref().is_none_or(|b| mean.total_ms() < b.total_ms()) {
            best = Some(mean);
        }
    }
    let mean = best.expect("three rounds");
    let idx = inf_player::step_profile::STEP_PHASE_NAMES
        .iter()
        .position(|n| *n == "dispatch")
        .expect("the `dispatch` phase exists");
    let dispatch = mean.ms[idx];
    println!(
        "
EMS3 STEP TABLE ({} build), {:.4} ms total, MIN of {rounds} rounds of {per_round}:",
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        },
        mean.total_ms()
    );
    for (name, ms) in mean.dearest_first() {
        if ms > 0.0005 {
            println!("  {name:>18}  {ms:.4} ms");
        }
    }
    println!(
        "  {rays} recognition ray(s) over {looked} step(s) with an officer out, at most \
         {worst_files} open file(s); the ceiling is {} rays a step",
        inf_physics::d3::crime::MAX_RECOGNITION_RAYS
    );
    // **ARMED**, because a ceiling assertion over zero work is satisfied by a
    // pass that never ran: the file has to exist and an officer has to have been
    // out looking for the ceiling to be a claim about anything.
    assert!(
        worst_files > 0,
        "no file was ever open — the ceiling is vacuous"
    );
    assert!(
        looked > 0,
        "no officer was ever on the beat — the ceiling is vacuous"
    );
    // …and the pass actually LOOKED. `rays > 0` is the third arming clause and
    // the one the audit added: an officer on the beat and a file open still buy
    // nothing if every pair fails the range gate, and a ceiling over zero rays
    // is satisfied by a pass that measured nothing.
    assert!(
        rays > 0,
        "not one recognition ray was cast in {looked} steps with an officer out \
         — the budget row below contains no line-of-sight work at all"
    );
    assert!(
        rays <= looked * inf_physics::d3::crime::MAX_RECOGNITION_RAYS,
        "{rays} rays over {looked} steps is past the per-step ceiling"
    );
    // A clock, so: release only, real machine only.
    if cfg!(debug_assertions) || std::env::var_os("CI").is_some() {
        return;
    }
    assert!(
        dispatch <= inf_player::budget::DISPATCH_STEP_BUDGET_MS,
        "the dispatch phase costs {dispatch:.4} ms against a ceiling of {:.4}",
        inf_player::budget::DISPATCH_STEP_BUDGET_MS
    );
}

// ── (g) the falsifier ───────────────────────────────────────────────────────

/// **A TOWN WHERE NOBODY DOES ANYTHING FILES NOTHING.**
///
/// The same fixture, the same officer, the same wardrobe, the same number of
/// steps — and no crime. Every arm above is therefore about a wanted system
/// rather than about a fixture that would have produced those numbers anyway.
#[test]
fn a_town_with_no_crime_in_it_has_no_wanted_level() {
    let mut world = EcsWorld::new();
    build(&mut world);
    let mut sim = RuntimeSim::with_gravity(world, Vec::new(), WorldGravity::EARTH, HZ);
    let mut rays = 0usize;
    let mut files = 0usize;
    for _ in 0..(WARMUP + 600) {
        sim.step_once(RuntimeInput::default());
        let r = sim.dispatch_stats().recognition;
        rays += r.rays;
        files = files.max(r.files);
    }
    println!("no crime: {files} file(s), {rays} ray(s)");
    assert_eq!(files, 0, "a file opened with nothing to open it");
    assert_eq!(rays, 0, "a ray was cast with nobody to look for");
    assert!(
        inf_ecs::crime::profile_state_bytes(sim.world()).is_empty(),
        "the ledger folded bytes on a level where nobody is wanted — which would \
         move every trace committed before this wave"
    );
    assert_eq!(crime::heat_of(sim.world(), HERO), 0);
    assert!(
        inf_ecs::crime::wanted_readout(sim.world(), HERO).is_none(),
        "the HUD would draw stars over an innocent man"
    );
    // …and the clock the whole feed rides IS moving, so no arm above is
    // certified against a frozen counter. See `d3::crime`'s header for why a
    // level with no streets has no clock at all.
    assert!(
        inf_ecs::traffic::steps(sim.world()) > 0,
        "the traffic clock never advanced — every `step` in this gate is 0 and the \
         crime feed's forward read can never fire"
    );
}
