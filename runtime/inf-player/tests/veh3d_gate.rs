//! **WAVE VEH3d — BOARDING.** The gate.
//!
//! Every claim of the wave, in one file, each arm **mutation-verified** and each
//! carrying an **engagement count** — so "the arm ran" and "something happened"
//! are two different facts and the second one is asserted.
//!
//! # What every arm in this file reads
//!
//! The JOINTS and the WORLD — never a state name or a table on its own. A hand
//! on a handle is the evaluated pose's hand joint against the live handle (the
//! door body's own pose times the handle's offset in the door frame); a seated
//! pelvis is the pelvis joint against the cushion and the chassis roof; the
//! wheel is the rim angle read off the wheels' own steer and the hands' joints
//! against the rim grips; a pedal is the foot joint's travel against the input;
//! a door is the hinge angle read off VEH3c's revolute through
//! `part_angle_rad`; an exit and a pull-out are the capsule's own clearance
//! against every collider in the world. The phase trace is read too, and it is
//! read as ORDER and DURATION, which the old warp could not fake.
//!
//! # Would P29.7's 0.55 s warp, with no IK, pass it?
//!
//! The audit brief's question for every arm, answered in the table. "fails"
//! means the old seat step, run through this arm, reds it.
//!
//! | arm | reads | mutation that reds it | engagement | the old warp? |
//! |---|---|---|---|---|
//! | `the_machine_walks_its_phases_in_order_with_their_durations` | the phase trace, the root's path and yaw | the approach clocked at `UNLOCK_MIN_S` (no walk) | steps per phase, path metres | **fails** — one phase, on the press step |
//! | `the_hand_takes_the_outer_handle_within_two_centimetres` | the hand JOINT vs the live handle | the reach dip zeroed | steps at weight 1 before the latch | **fails** — no hand request |
//! | `the_door_opens_on_its_revolute_motor_as_the_hand_reaches` | the hinge angle off the joint, the part body | the motor call deleted from the door phase | degrees travelled | **fails** — the door never moves |
//! | `the_seat_is_inside_the_cabin_on_every_family` | the pelvis/head/foot JOINTS vs the cushion, roof and pedals | `SEAT_FLOOR_FRAC_Y` back to the roof | six families | **fails** — feet on the roof |
//! | `the_hands_follow_the_rim_through_a_full_lock` | the hand JOINTS vs the rim grips, the rim angle | `vehicle_steer` answering zero | rim degrees swept, metres the hands moved | **fails** — no hand request |
//! | `the_feet_press_the_pedals_with_the_inputs` | the foot JOINTS' travel vs the inputs | the pedal inputs zeroed | millimetres pressed | **fails** — no foot request |
//! | `a_passenger_rides_its_own_seat_and_does_not_drive` | the passenger's joints vs its sockets, the car's travel | the passenger's intent reaching the car | metres driven, joints placed | **fails** — no seat index |
//! | `an_occupied_seat_is_never_entered` | the prompt, the census, the mode | the `Occupied` candidate deleted | presses refused | **fails** — no prompt, no refusal |
//! | `the_carjack_plays_the_same_pipeline` | both bodies' phase traces, the census, the victim's landing | the pull skipped (victim left seated) | steps, bodies per seat | **fails** — a one-frame eject |
//! | `the_victim_is_never_put_down_inside_a_wall` | the victim's capsule vs the colliders | `clear_exit` taking the preferred point unchecked | the candidate index taken | **fails** — `door_point` into the wall |
//! | `the_exit_is_the_reverse_and_never_lands_in_geometry` | the phase trace, the hinge, the capsule's clearance | `capsule_clear` answering true | steps, degrees, the candidate index | **fails** — the exit point is fixed |
//! | `a_moving_exit_is_a_roll` | the mode trace and the landing kind | the bail flag never set | metres per second at the door | **fails** — a soft landing |
//! | `a_door_torn_off_is_boarded_through_the_opening` | the door phase's length, the hand request | `door_for_seat` ignoring the latch | the door's latch | passes — the warp has no door to wait for |
//! | `the_boarding_camera_rides_the_director` | the director's holder, the drive pivot | the claim not pushed | steps held | **fails** — no claim |
//! | `the_boarding_section_is_empty_until_somebody_boards` | the nineteenth section's bytes | `is_quiet` answering false | bytes per phase | **fails** — no section |
//! | `pie_equals_shipping_on_a_board_drive_exit_course` | both hosts' boarding bytes, step by step | either host's fold deleted (the anti-vacuity half) | steps with bytes | **fails** — nothing to compare |
//! | `a_boarding_costs_what_it_costs` | the step's own milliseconds, control vs boarding | n/a — a COST arm | the boarding really ran | n/a |
//! | `sixty_four_seated_drivers_cost_what_they_cost` | the post-solve pass's milliseconds at 64 rigs | n/a — a COST arm | 64 hands placed | n/a |
//! | `the_level_camera_table_round_trips_through_the_write_half` | the file on disk, read back | `to_toml_changed` writing the whole table | keys written | n/a — a camera arm |
//! | `the_shipped_host_draws_the_boarding_row` | `window.rs` and the Ring-0 row | the call deleted from `window.rs` | one row | n/a |
//! | `this_wave_moved_no_schema` | the tunable count and the derive lists | a `Serialize` on `BoardingState` | 100 tunables | n/a |
//! | `the_island_census_has_no_doubly_occupied_seat` | the island's seats over ten minutes | a second driver seated | seats counted | n/a — local-only |
//!
//! # PIE == shipping, the mirrors, and the wire
//!
//! `pie_equals_shipping_on_a_board_drive_exit_course` drives the editor's
//! `SimSession` and the shipped player's `RuntimeSim` through their own front
//! doors, pressing E through the INPUT door on both, over a board, a drive and
//! an exit, and compares `boarding::boarding_state_bytes` step by step. The
//! machine needs no MIRROR fence of its own: it runs inside
//! `step_character_movement` (the ground phases and the seat phases) and inside
//! the bridge's write-back (the hands and feet), both of which each host calls
//! once through the fences `fixed_step_mirror` already pins. The nineteenth
//! trace section is pinned in `projector_mirror`'s `SECTIONS` in the commit
//! that folded it. No persisted field lands: `BoardingState` rides
//! `MovementRuntime`, which is `#[serde(skip)]`.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::boarding::{self as board, BoardPhase, SeatIndex};
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    MovementMode, RigidBody3D, Transform, Visibility,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::vehicle::VehicleDef;
use inf_ecs::EcsWorld;
use inf_physics::d3::{self, PhysicsBridge3D};

const DT: f64 = 1.0 / 60.0;
const RADIUS: f64 = 0.3;
const CHASSIS: Uuid = Uuid::from_u128(0x5E3D_0001);
const GROUND: Uuid = Uuid::from_u128(0x5E3D_0002);
const HERO: Uuid = Uuid::from_u128(0x5E3D_0003);
const VICTIM: Uuid = Uuid::from_u128(0x5E3D_0004);
const RIDER: Uuid = Uuid::from_u128(0x5E3D_0005);
const WALL: Uuid = Uuid::from_u128(0x5E3D_0006);
const SKEL_GUID: Uuid = Uuid::from_u128(0x5E3D_0007);
const SM_GUID: Uuid = Uuid::from_u128(0x5E3D_0008);
/// Where the hero is put down: beside the saloon's DRIVER's (`+X`) flank, a
/// couple of metres back, facing up the car — the showcase's own picture.
const HERO_AT: DVec3 = DVec3::new(2.6, 0.0, -1.6);

// ── the fixture ─────────────────────────────────────────────────────────────

struct Yard {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    rig: Option<(inf_anim::SkeletonAsset, inf_anim::StateMachine)>,
    clips: std::collections::BTreeMap<inf_anim::ClipRef, inf_anim::AnimClip>,
}

fn catalogue_def(id: &str) -> VehicleDef {
    *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .unwrap_or_else(|| panic!("the catalogue has no `{id}` row"))
}

fn ground(world: &mut EcsWorld) {
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    world.world_mut().entity_mut(e).insert((
        Transform {
            translation: Vec3d::new(0.0, -0.5, 0.0),
            ..Default::default()
        },
        Visibility::default(),
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(200.0, 0.5, 200.0),
            friction: 0.9,
            ..Default::default()
        },
    ));
}

/// A static slab standing on the ground: `centre` and `half` in metres.
fn wall(world: &mut EcsWorld, guid: Uuid, centre: DVec3, half: DVec3) {
    let e = world.spawn_with_guid(guid, "Wall", None);
    world.world_mut().entity_mut(e).insert((
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
            friction: 0.8,
            ..Default::default()
        },
    ));
}

fn car(world: &mut EcsWorld, guid: Uuid, at: DVec3, yaw_deg: f64, def: &VehicleDef) {
    let spawn = inf_ecs::vehicle::RigSpawn {
        name: "Car".into(),
        at,
        yaw_deg,
        paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
        clip: None,
        engine_voice: false,
        livery: None,
    };
    inf_ecs::vehicle::spawn_rig(world, guid, def, &spawn);
}

/// Stand one capsule character up with its FEET at `at`.
fn stand(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, yaw_deg: f64, player: bool) {
    let cm = CharacterMovement {
        player_controlled: player,
        ..Default::default()
    };
    let e = world.spawn_with_guid(guid, name, None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(at.x, at.y + cm.stand_half_height_m + RADIUS, at.z);
    t.rotation.y = yaw_deg;
    world.world_mut().entity_mut(e).insert((
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
    ));
}

/// One step's record of a body, for the phase and path arms.
#[derive(Clone, Copy, Debug)]
struct Sample {
    phase: BoardPhase,
    at: DVec3,
    yaw: f64,
    door_deg: f64,
}

impl Yard {
    fn new(row: &str) -> Self {
        Self::build(row, |_| {})
    }

    /// A catalogue row settled on its wheels at the origin heading `+Z`, the
    /// hero at [`HERO_AT`] with the mannequin rig, and whatever `extra` adds.
    fn build(row: &str, extra: impl FnOnce(&mut EcsWorld)) -> Self {
        let def = catalogue_def(row);
        let mut world = EcsWorld::default();
        ground(&mut world);
        let y = inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15;
        car(&mut world, CHASSIS, DVec3::new(0.0, y, 0.0), 0.0, &def);
        stand(&mut world, HERO, "Hero", HERO_AT, 0.0, true);
        extra(&mut world);
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        let mut y = Self {
            world,
            bridge,
            rig: None,
            clips: std::collections::BTreeMap::new(),
        };
        y.install_rig(&[HERO]);
        y.step(90);
        y
    }

    /// The mannequin and a one-state machine, on every body named — the
    /// wpn2b gate's own fixture rig.
    fn install_rig(&mut self, who: &[Uuid]) {
        const IDLE: inf_anim::ClipRef = [0xd3; 16];
        if self.rig.is_none() {
            let skeleton = inf_anim::build_template(
                inf_anim::BodyPlan::Biped,
                &inf_anim::BodyParams {
                    height_m: 1.8,
                    ..Default::default()
                },
            )
            .expect("the mannequin builds");
            self.clips
                .insert(IDLE, inf_anim::AnimClip::new("idle", Vec::new()));
            self.rig = Some((
                skeleton,
                inf_anim::StateMachine {
                    states: vec![inf_anim::SmState::clip("idle", IDLE)],
                    entry: 0,
                    ..Default::default()
                },
            ));
        }
        for g in who {
            let e = self.world.entity_of(*g).expect("the character");
            self.world.world_mut().entity_mut(e).insert((
                inf_ecs::components::AnimStateMachine {
                    sm: Some(SM_GUID),
                    ..Default::default()
                },
                inf_ecs::components::SkeletalMesh {
                    mesh: None,
                    skeleton: Some(SKEL_GUID),
                },
            ));
        }
        self.world.mark_dirty();
    }

    /// One fixed step, in the hosts' own order: movement, vehicles, gameplay,
    /// the solver, the write-back (where the boarding's hands and feet are
    /// placed), the pose.
    fn step(&mut self, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
            d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            let _ = d3::step_gameplay(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
            if let Some((skeleton, machine)) = &self.rig {
                let machines = |g: Uuid| (g == SM_GUID).then_some(machine);
                let skels = |g: Uuid| (g == SKEL_GUID).then_some(skeleton);
                let clips = |c: inf_anim::ClipRef| self.clips.get(&c);
                let vars = |_: Uuid| std::collections::BTreeMap::new();
                inf_ecs::pose::step_pose_evaluation(
                    &mut self.world,
                    DT,
                    &machines,
                    &skels,
                    &clips,
                    &vars,
                );
            }
        }
    }

    fn cm(&self, who: Uuid) -> CharacterMovement {
        let e = self.world.entity_of(who).expect("the character");
        self.world
            .world()
            .get::<CharacterMovement>(e)
            .expect("a mover")
            .clone()
    }

    fn with_cm(&mut self, who: Uuid, f: impl FnOnce(&mut CharacterMovement)) {
        let e = self.world.entity_of(who).expect("the character");
        if let Some(mut cm) = self.world.world_mut().get_mut::<CharacterMovement>(e) {
            f(&mut cm);
        }
    }

    fn press(&mut self, who: Uuid) {
        self.with_cm(who, |cm| cm.runtime.press_interact = true);
    }

    /// Hold a stick for the next steps — the runtime intent the input door
    /// writes.
    fn stick(&mut self, who: Uuid, x: f64, y: f64) {
        self.with_cm(who, |cm| cm.runtime.intent_move = Vec2d::new(x, y));
    }

    fn at(&self, who: Uuid) -> DVec3 {
        d3::boarding::capsule_centre(&self.world, who).expect("placed")
    }

    fn sample(&self, who: Uuid) -> Sample {
        let cm = self.cm(who);
        let b = cm.runtime.boarding;
        Sample {
            phase: b.phase,
            at: self.at(who),
            yaw: cm.runtime.body_yaw_deg,
            door_deg: b.door_deg,
        }
    }

    /// Press E and step until the body reaches the wheel, recording each step.
    fn board(&mut self, who: Uuid) -> Vec<Sample> {
        self.press(who);
        let mut out = Vec::new();
        for _ in 0..900 {
            self.step(1);
            let s = self.sample(who);
            out.push(s);
            if s.phase == BoardPhase::Driving {
                return out;
            }
        }
        panic!(
            "`{who}` never reached the wheel in fifteen seconds; last {:?}",
            out.last()
        )
    }

    /// A joint's world position off the evaluated pose.
    fn joint(
        &self,
        who: Uuid,
        role: inf_anim::BoneRoleKind,
        side: inf_anim::BoneSide,
    ) -> Option<DVec3> {
        let (rig, _) = self.rig.as_ref()?;
        let j = rig.role_index().first(role, side)?;
        let posed = inf_ecs::pose::evaluated_pose(&self.world, who)?;
        let to_world = inf_ecs::pose::model_to_world_of(&self.world, who)?;
        let g = inf_anim::pose::global_transforms(&rig.skeleton, &posed.pose);
        let p = g.get(j as usize)?.to_scale_rotation_translation().2;
        Some(to_world.transform_point3(DVec3::new(
            f64::from(p.x),
            f64::from(p.y),
            f64::from(p.z),
        )))
    }

    /// `[left, right]` hand joints.
    fn hands(&self, who: Uuid) -> [Option<DVec3>; 2] {
        [
            self.joint(who, inf_anim::BoneRoleKind::Hand, inf_anim::BoneSide::Left),
            self.joint(who, inf_anim::BoneRoleKind::Hand, inf_anim::BoneSide::Right),
        ]
    }

    /// `[left, right]` foot joints.
    fn feet(&self, who: Uuid) -> [Option<DVec3>; 2] {
        [
            self.joint(who, inf_anim::BoneRoleKind::Foot, inf_anim::BoneSide::Left),
            self.joint(who, inf_anim::BoneRoleKind::Foot, inf_anim::BoneSide::Right),
        ]
    }

    /// The feet in the CAR's frame — a pedal stroke is a travel against the
    /// floor pan, and a car under throttle carries both feet forward with it.
    fn feet_in_car(&self, who: Uuid) -> [Option<DVec3>; 2] {
        let car = self.frame();
        self.feet(who).map(|f| f.map(|p| car.local(p).to_dvec3()))
    }

    /// What the hand pass was asked for: `[left, right]` targets and weights.
    fn hand_asks(&self, who: Uuid) -> [Option<(DVec3, f64)>; 2] {
        match inf_ecs::pose::hand_ik(&self.world, who) {
            Some(h) => [
                h.reach[0].map(|r| (r.target.to_dvec3(), f64::from(r.weight))),
                h.reach[1].map(|r| (r.target.to_dvec3(), f64::from(r.weight))),
            ],
            None => [None, None],
        }
    }

    /// What the foot pass was asked for.
    fn foot_asks(&self, who: Uuid) -> [Option<(DVec3, f64)>; 2] {
        let goals = self
            .world
            .world()
            .get_resource::<inf_ecs::anim_bridge::AnimBridgeRes>()
            .and_then(|r| r.foot_ik.get(&who).copied());
        match goals {
            Some(g) => [
                g[0].map(|x| (x.target.to_dvec3(), f64::from(x.weight))),
                g[1].map(|x| (x.target.to_dvec3(), f64::from(x.weight))),
            ],
            None => [None, None],
        }
    }

    fn frame(&self) -> d3::boarding::CarFrame {
        d3::boarding::car_frame(&self.world, &self.bridge, CHASSIS, true).expect("the car")
    }

    /// The driver's front door on this car.
    fn door(&self) -> Uuid {
        board::door_for_seat(&self.world, CHASSIS, SeatIndex::Driver)
            .expect("the saloon has a driver's door")
    }

    fn car_speed(&self) -> f64 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_linvel(b))
            .map(|v| DVec3::new(v.x, 0.0, v.z).length())
            .unwrap_or(0.0)
    }

    fn car_at(&self) -> DVec3 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_translation(b))
            .unwrap_or(DVec3::ZERO)
    }

    /// Put a non-player body in a seat of the car, the way the traffic tier
    /// does (`Driving`, the warp already finished, the collider parked).
    fn seat_npc(&mut self, who: Uuid, seat: SeatIndex) {
        let at = DVec3::new(-12.0, 0.0, -12.0 - 3.0 * f64::from(seat.as_u8()));
        stand(&mut self.world, who, "Npc", at, 0.0, false);
        self.with_cm(who, |cm| {
            cm.mode = MovementMode::Driving;
            cm.runtime.seat = inf_ecs::components::SeatState {
                vehicle: CHASSIS,
                entering: false,
                time_s: 0.0,
                start: Vec3d::from_dvec3(at),
                start_yaw_deg: 0.0,
                seat: seat.as_u8(),
            };
        });
        self.world.mark_dirty();
        self.world.propagate();
        self.bridge.sync_from_world(&self.world);
        d3::vehicle::park_collider(&mut self.bridge, who, true);
        self.install_rig(&[who]);
        self.step(3);
    }

    /// Whether `who`'s capsule, standing where it is, overlaps anything solid.
    fn clear(&mut self, who: Uuid) -> bool {
        let cm = self.cm(who);
        let feet = self.at(who) - DVec3::Y * (cm.stand_half_height_m + RADIUS);
        let mut own = std::collections::BTreeSet::new();
        if let Some(c) = self.bridge.collider_of(who) {
            own.insert(c);
        }
        d3::boarding::capsule_clear(&mut self.bridge, feet, cm.stand_half_height_m, RADIUS, &own)
    }
}

/// Seconds a phase lasted in a trace — the steps it held, times the step.
fn durations(trace: &[Sample]) -> Vec<(BoardPhase, f64, usize)> {
    let mut out: Vec<(BoardPhase, f64, usize)> = Vec::new();
    for s in trace {
        match out.last_mut() {
            Some((p, t, n)) if *p == s.phase => {
                *t += DT;
                *n += 1;
            }
            _ => out.push((s.phase, DT, 1)),
        }
    }
    out
}

fn print_trace(label: &str, trace: &[Sample]) {
    println!("=== {label}: the phases, in order, with their durations ===");
    for (p, t, n) in durations(trace) {
        println!("  {:<10} {:>6.3} s  ({n} steps)", p.name(), t);
    }
}

fn dist(a: Option<DVec3>, b: DVec3) -> f64 {
    a.map(|a| (a - b).length()).unwrap_or(f64::INFINITY)
}

// ── (b) THE MACHINE ─────────────────────────────────────────────────────────

/// **The machine walks its phases in order, with their durations** — read on
/// the phase trace, the root's path and the body's yaw.
///
/// * the six enter phases, in order, each for at least one step;
/// * `Locked` lasts [`board::LOCKED_S`]; `EnteringIK` lasts the car's OWN
///   `seat_warp()` window — P29.7's 0.55 s, unchanged — which is the claim
///   that the old warp BECAME a phase;
/// * `Unlocking` is ROOT MOTION at walk speed: the path's own length over the
///   phase's own length is [`board::APPROACH_MPS`] within a third, and no step
///   moves the body further than a walking stride allows;
/// * the facing arrives at the flank's inward normal and is LOCKED there
///   through the door phase.
///
/// **The mutation**: the approach clocked at `UNLOCK_MIN_S` whatever its
/// length (`begin`'s `b.mark_s = board::approach_s(len)`) — the body crosses
/// two metres in a fifth of a second, 10 m/s, and the walk-speed band reds.
#[test]
fn the_machine_walks_its_phases_in_order_with_their_durations() {
    let mut y = Yard::new("sedan");
    let start = y.at(HERO);
    let trace = y.board(HERO);
    print_trace("the saloon, from 2 m back", &trace);
    let d = durations(&trace);
    let order: Vec<BoardPhase> = d.iter().map(|(p, _, _)| *p).collect();
    assert_eq!(
        order,
        BoardPhase::ENTER_ORDER.to_vec(),
        "the machine did not walk its six phases in order: {order:?}"
    );
    let len = |p: BoardPhase| d.iter().find(|(q, _, _)| *q == p).map(|(_, t, _)| *t).unwrap();
    // `Locked`: the check, one clock.
    assert!(
        (len(BoardPhase::Locked) - board::LOCKED_S).abs() <= 1.5 * DT,
        "`Locked` lasted {:.3} s against {}",
        len(BoardPhase::Locked),
        board::LOCKED_S
    );
    // `EnteringIK`: P29.7's own window, from the car.
    let (enter_s, _) = y
        .bridge
        .vehicle_of(CHASSIS)
        .expect("the car")
        .seat_warp();
    assert!(
        (len(BoardPhase::EnteringIK) - enter_s).abs() <= 1.5 * DT,
        "`EnteringIK` lasted {:.3} s and the car's own seat warp is {enter_s:.3} s — the old warp is not this phase",
        len(BoardPhase::EnteringIK)
    );
    // `Unlocking`: ROOT MOTION at walk speed, measured on the path.
    let walk: Vec<&Sample> = trace
        .iter()
        .filter(|s| s.phase == BoardPhase::Unlocking)
        .collect();
    let mut path_m = 0.0;
    let mut worst_step = 0.0f64;
    let mut prev = walk[0].at;
    for s in &walk[1..] {
        let step = DVec3::new(s.at.x - prev.x, 0.0, s.at.z - prev.z).length();
        path_m += step;
        worst_step = worst_step.max(step);
        prev = s.at;
    }
    let speed = path_m / len(BoardPhase::Unlocking);
    println!(
        "  the approach: {path_m:.3} m in {:.3} s = {speed:.3} m/s (walk {} m/s), the longest step {worst_step:.4} m; started {:.3} m from the take",
        len(BoardPhase::Unlocking),
        board::APPROACH_MPS,
        (DVec3::new(start.x, 0.0, start.z)
            - DVec3::new(walk[walk.len() - 1].at.x, 0.0, walk[walk.len() - 1].at.z))
        .length()
    );
    assert!(
        path_m > 1.0,
        "the approach covered {path_m:.3} m — the fixture put the hero at the door and this arm measures nothing"
    );
    assert!(
        (speed - board::APPROACH_MPS).abs() < board::APPROACH_MPS / 3.0,
        "the approach walked at {speed:.3} m/s against the walk's {} — it is a slide or a teleport, not root motion",
        board::APPROACH_MPS
    );
    assert!(
        worst_step < 2.0 * board::APPROACH_MPS * DT,
        "one step of the approach moved the body {worst_step:.4} m — a jump, not a stride"
    );
    // The facing: the flank's inward normal, locked through the door.
    let car = y.frame();
    let inward = car.dir(DVec3::new(-1.0, 0.0, 0.0));
    let want = inf_ecs::movement::planar_yaw_deg(Vec2d::new(inward.x, inward.z));
    let door: Vec<&Sample> = trace
        .iter()
        .filter(|s| s.phase == BoardPhase::OpeningDoor)
        .collect();
    let off = door
        .iter()
        .map(|s| inf_ecs::movement::angle_delta_deg(s.yaw, want).abs())
        .fold(0.0f64, f64::max);
    println!("  the facing through the door phase: at most {off:.4} deg off the flank's inward normal ({want:.2} deg)");
    assert!(
        off < 0.5,
        "the body turned {off:.3} deg away from the flank while it had its hand on the door"
    );
}

// ── (d) THE HANDS ───────────────────────────────────────────────────────────

/// **The hand takes the OUTER handle, within two centimetres, at weight 1** —
/// the hand JOINT off the evaluated pose against the handle where the DOOR is
/// (VEH3c's own geometry, through `d3::boarding::handle_world`).
///
/// The weight ramps `0 → 1` over [`board::HAND_REACH_S`] (the doc's *"Blend IK
/// Weight (0.0 -> 1.0) over 0.2s"*), measured at every step of it; the two
/// centimetres are read on every step at weight 1 before the latch is pulled,
/// which is the take. The body DIPS to put the handle in reach — the pelvis
/// offset is read too, because a hand that reached only because the handle was
/// already at shoulder height is a claim about the car and not about the arm.
///
/// **The mutation**: the reach dip zeroed (`begin`'s `dip` answering `0.0`) —
/// the arm cannot reach a handle 0.76 m off the road from a standing shoulder
/// and the hand stops 4.5 cm short.
#[test]
fn the_hand_takes_the_outer_handle_within_two_centimetres() {
    let mut y = Yard::new("sedan");
    y.press(HERO);
    let mut ramp: Vec<(f64, f64)> = Vec::new();
    let mut takes: Vec<f64> = Vec::new();
    let mut deepest = 0.0f64;
    for _ in 0..600 {
        y.step(1);
        let cm = y.cm(HERO);
        let b = cm.runtime.boarding;
        if b.phase == BoardPhase::EnteringIK {
            break;
        }
        if b.phase != BoardPhase::OpeningDoor {
            continue;
        }
        deepest = deepest.min(cm.runtime.pelvis_offset.y);
        let asks = y.hand_asks(HERO);
        let side = usize::from(b.hand_side != 0);
        let Some((_, w)) = asks[side] else { continue };
        if b.mark_s < 0.0 {
            ramp.push((b.time_s, w));
        }
        if b.mark_s < 0.0 && (w - 1.0).abs() < 1e-9 {
            let car = y.frame();
            let handle = d3::boarding::handle_world(&y.world, &y.bridge, &car, b.door, false)
                .expect("the driver's door has a handle");
            let hand = y.hands(HERO)[side];
            takes.push(dist(hand, handle));
        }
    }
    println!("=== the hand on the outer handle ===");
    for (t, w) in &ramp {
        println!("  t {:.3} s  weight {w:.3}  (owed {:.3})", t, (t / board::HAND_REACH_S).min(1.0));
    }
    println!(
        "  at weight 1 before the latch: {} steps, worst {:.2} mm, pelvis dipped {:.3} m",
        takes.len(),
        takes.iter().copied().fold(0.0f64, f64::max) * 1000.0,
        deepest
    );
    // The ramp IS the doc's: linear over 0.2 s.
    for (t, w) in &ramp {
        let owed = (t / board::HAND_REACH_S).clamp(0.0, 1.0);
        assert!(
            (w - owed).abs() < 1e-6,
            "at t {t:.3} s the hand weight was {w:.4} and the ramp owes {owed:.4}"
        );
    }
    assert!(
        ramp.iter().any(|(_, w)| *w > 0.0 && *w < 1.0),
        "the weight never passed through the ramp — it snapped"
    );
    assert!(
        takes.len() >= 5,
        "only {} steps at weight 1 before the latch — the take is not being measured",
        takes.len()
    );
    let worst = takes.iter().copied().fold(0.0f64, f64::max);
    assert!(
        worst <= 0.02,
        "the hand joint ended {:.2} mm from the handle at weight 1 — the take is not on the handle",
        worst * 1000.0
    );
    assert!(
        deepest < -0.05,
        "the pelvis never dipped ({deepest:.3} m) — the handle was not below a standing reach, so this arm proves nothing about the arm"
    );
}

/// **The door opens on its REVOLUTE, through the MOTOR, as the hand reaches**
/// — the hinge angle read off VEH3c's joint (`part_angle_rad`, portable), the
/// part body and its joint, and the order: nothing moves until the hand has
/// held the handle for [`board::HANDLE_HOLD_S`], then the angle climbs past
/// [`board::DOOR_BOARD_DEG`] before the body goes in, and the inner pull shuts
/// it again once the body is seated.
///
/// **The mutation**: the motor call deleted from the door phase
/// (`set_part_open(.., true)` in `step_ground`) — the door stays at 0 deg, the
/// phase waits out [`board::OPENING_MAX_S`] and boards through a shut door,
/// and the angle assertion reds.
#[test]
fn the_door_opens_on_its_revolute_motor_as_the_hand_reaches() {
    let mut y = Yard::new("sedan");
    let door = y.door();
    assert!(
        y.bridge.part_body(door).is_none(),
        "a parked car's door is already a body — the lazy edge VEH3c built is gone"
    );
    y.press(HERO);
    let mut trace: Vec<(BoardPhase, f64, f64, f64)> = Vec::new();
    let mut closed_at: Option<f64> = None;
    for _ in 0..900 {
        y.step(1);
        let b = y.cm(HERO).runtime.boarding;
        let angle = y
            .bridge
            .part_angle_rad(door)
            .map(|a| a.to_degrees().abs())
            .unwrap_or(0.0);
        trace.push((b.phase, b.time_s, b.hand_weight, angle));
        if b.phase == BoardPhase::Driving {
            closed_at = Some(angle);
            break;
        }
    }
    let p = y
        .bridge
        .part_body(door)
        .expect("the opened door is a body on a hinge");
    assert!(p.joint.is_some(), "the door has a body and no joint");
    let before_latch = trace
        .iter()
        .filter(|(ph, t, _, _)| {
            *ph == BoardPhase::OpeningDoor && *t < board::HAND_REACH_S + board::HANDLE_HOLD_S
        })
        .map(|(_, _, _, a)| *a)
        .fold(0.0f64, f64::max);
    let peak_before_seat = trace
        .iter()
        .take_while(|(ph, _, _, _)| *ph != BoardPhase::Seated)
        .map(|(_, _, _, a)| *a)
        .fold(0.0f64, f64::max);
    let at_enter = trace
        .iter()
        .find(|(ph, _, _, _)| *ph == BoardPhase::EnteringIK)
        .map(|(_, _, _, a)| *a)
        .unwrap_or(0.0);
    println!("=== the hinge, read off the joint ===");
    for (ph, t, w, a) in trace.iter().step_by(6) {
        println!("  {:<10} t {:.2}  hand {:.2}  hinge {:.1} deg", ph.name(), t, w, a);
    }
    println!(
        "  before the latch {before_latch:.2} deg; {at_enter:.1} deg when the body went in; peak {peak_before_seat:.1}; {:.2} deg at the wheel",
        closed_at.unwrap_or(-1.0)
    );
    assert!(
        before_latch < 1.0,
        "the door moved {before_latch:.2} deg before the hand had held the handle — it is opening on a clock, not on the hand"
    );
    assert!(
        at_enter >= board::DOOR_BOARD_DEG - 0.5,
        "the body went in with the door at {at_enter:.1} deg — the door phase did not wait for the hinge"
    );
    assert!(
        peak_before_seat <= inf_ecs::vehicle::DOOR_OPEN_DEG + 1.0,
        "the door swung to {peak_before_seat:.1} deg, past its own {} deg limit",
        inf_ecs::vehicle::DOOR_OPEN_DEG
    );
    assert!(
        closed_at.is_some_and(|a| a <= board::DOOR_SHUT_DEG + 0.5),
        "the door was at {closed_at:?} deg when the hero reached the wheel — the inner pull did not shut it"
    );
    // The engagement: it TRAVELLED, both ways, on the joint.
    let travelled: f64 = trace
        .windows(2)
        .map(|w| (w[1].3 - w[0].3).abs())
        .sum();
    println!("  the hinge travelled {travelled:.1} deg in all");
    assert!(travelled > 80.0, "the hinge travelled {travelled:.1} deg");
}

// ── (a) THE SEAT IS INSIDE THE CAR ──────────────────────────────────────────

/// **The seat is inside the cabin, on every family** — carried 160 closed,
/// read on the JOINTS: the pelvis on the cushion socket, the head under the
/// chassis roof, the feet on the pedals; and the seat-height table the report
/// quotes, old roof against new floor and cushion, per family.
///
/// Measured against VEH3b's RE-SPRUNG rows: each car settles on its own
/// suspension before the hero boards it, so the heights are the ones a player
/// sees, not the ones the authored half-extents imply.
///
/// **The mutation**: `SEAT_FLOOR_FRAC_Y` back to `+1.0` (the collider's top
/// face) — the feet land on the roof and every pelvis sits above it.
#[test]
fn the_seat_is_inside_the_cabin_on_every_family() {
    println!("=== where a driver sits, per family (metres above the road) ===");
    println!(
        "  {:<8} {:>6} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8}",
        "row", "half.y", "old seat", "floor", "cushion", "pelvis", "pelv-roof", "head-roof", "feet"
    );
    let mut rows = 0;
    for row in ["sedan", "sports", "suv", "van", "truck", "cruiser"] {
        let mut y = Yard::new(row);
        y.board(HERO);
        y.step(30);
        let car = y.frame();
        let up = |local: Vec3d| car.world(local).y;
        let top = up(Vec3d::new(car.offset.x, car.offset.y + car.half.y, car.offset.z));
        let floor = car.world(car.sockets.seat_floor(SeatIndex::Driver, car.floor_y())).y;
        let cushion = car.world(car.sockets.seat(SeatIndex::Driver));
        let pelvis = y
            .joint(HERO, inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
            .expect("a pelvis");
        let head = y
            .joint(HERO, inf_anim::BoneRoleKind::Head, inf_anim::BoneSide::Center)
            .expect("a head");
        let asks = y.foot_asks(HERO);
        let feet = y.feet(HERO);
        let foot_err = (0..2)
            .map(|i| match asks[i] {
                Some((t, _)) => dist(feet[i], t),
                None => f64::INFINITY,
            })
            .fold(0.0f64, f64::max);
        println!(
            "  {:<8} {:>6.2} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>+9.3} {:>+9.3} {:>6.1}mm",
            row,
            car.half.y,
            top,
            floor,
            cushion.y,
            pelvis.y,
            pelvis.y - top,
            head.y - top,
            foot_err * 1000.0
        );
        assert!(
            (pelvis.y - cushion.y).abs() < 0.03,
            "{row}: the pelvis joint is at {:.3} and the cushion at {:.3} — the body is not sitting on the seat",
            pelvis.y,
            cushion.y
        );
        assert!(
            pelvis.y < top - 0.3,
            "{row}: the pelvis is {:+.3} m from the roof — the driver is not inside the car",
            pelvis.y - top
        );
        assert!(
            head.y < top,
            "{row}: the head joint is {:+.3} m above the roof — the driver's head is through it",
            head.y - top
        );
        assert!(
            foot_err <= 0.02,
            "{row}: a foot is {:.1} mm from its pedal",
            foot_err * 1000.0
        );
        rows += 1;
    }
    assert_eq!(rows, 6, "the table did not walk every family");
}

/// **The hands stay on the rim through a FULL LOCK, both ways** — the hand
/// JOINTS against the rim grips the rack puts them at, and the rim angle read
/// off the wheels' own steer (`rim_angle_deg`, 450 deg of rim at full lock).
///
/// **The mutation**: `vehicle_steer` answering zero — the rim never turns,
/// the grips never move and the "the hands travelled" assertion reds; the
/// rack-angle sweep is read off the WHEELS, so it cannot be faked by the
/// same mutation.
#[test]
fn the_hands_follow_the_rim_through_a_full_lock() {
    let mut y = Yard::new("sedan");
    y.board(HERO);
    y.step(20);
    let max = y
        .world
        .entity_of(CHASSIS)
        .and_then(|e| y.world.world().get::<inf_ecs::components::VehicleClass>(e).copied())
        .map(|c| c.max_steer_deg)
        .expect("the row's class");
    let rim = |y: &Yard| -> f64 {
        let v = y.bridge.vehicle_of(CHASSIS).expect("the car");
        let (mut s, mut n) = (0.0f64, 0);
        for w in v.wheels() {
            if w.steer_deg.abs() > 1e-9 {
                s += w.steer_deg;
                n += 1;
            }
        }
        if n == 0 {
            0.0
        } else {
            board::rim_angle_deg(s / n as f64, max)
        }
    };
    let (mut lo, mut hi) = (0.0f64, 0.0f64);
    let mut worst = 0.0f64;
    let mut placed = 0usize;
    let start = y.hands(HERO);
    let mut far = [0.0f64; 2];
    for (x, n) in [(1.0, 120), (-1.0, 200), (0.0, 120)] {
        y.stick(HERO, x, 0.0);
        for _ in 0..n {
            y.step(1);
            let r = rim(&y);
            lo = lo.min(r);
            hi = hi.max(r);
            let asks = y.hand_asks(HERO);
            let hands = y.hands(HERO);
            for i in 0..2 {
                if let Some((t, w)) = asks[i] {
                    if (w - 1.0).abs() < 1e-9 {
                        worst = worst.max(dist(hands[i], t));
                        placed += 1;
                    }
                }
                if let (Some(a), Some(b)) = (start[i], hands[i]) {
                    far[i] = far[i].max((a - b).length());
                }
            }
        }
    }
    println!(
        "=== the hands on the rim ===\n  the rim swept {lo:.1} .. {hi:.1} deg (a lock is {} deg); {placed} hand-steps at weight 1, worst {:.2} mm; the hands travelled {:.3} / {:.3} m",
        board::WHEEL_LOCK_DEG,
        worst * 1000.0,
        far[0],
        far[1]
    );
    assert!(
        hi > 0.9 * board::WHEEL_LOCK_DEG && lo < -0.9 * board::WHEEL_LOCK_DEG,
        "the rim only swept {lo:.1} .. {hi:.1} deg — this is not a full lock either way"
    );
    assert!(placed > 400, "only {placed} hand-steps were measured");
    assert!(
        worst <= 0.02,
        "a hand was {:.2} mm off its rim grip — the hands are not following the wheel",
        worst * 1000.0
    );
    // The hands RIDE the rim `RIM_RIDE_DEG` either way and the rim slides
    // through them past it (push-pull), so a hand's travel is the chord of a
    // 100-degree arc of a 0.19 m rim: measured 0.150 / 0.165 m, and 0.073 m
    // with the rim frozen (`vehicle_steer` answering zero).
    assert!(
        far[0] > 0.11 && far[1] > 0.11,
        "the hands travelled {:.3} / {:.3} m through a full lock — the rim is not turning under them",
        far[0],
        far[1]
    );
}

/// **The feet press the pedals with the inputs** — the right foot JOINT's
/// travel along the pedal's own arc with the throttle held, the left's with
/// the brake held, each against the pedal target the car's own controls
/// produced (`BoardingState::throttle_in` / `brake_in`, recorded from the
/// `VehicleControls` the car was given).
///
/// **The mutation**: the pedal inputs zeroed (`pedal_inputs` answering
/// `(0, 0)`) — neither foot moves and both travel assertions red.
#[test]
fn the_feet_press_the_pedals_with_the_inputs() {
    let mut y = Yard::new("sedan");
    y.board(HERO);
    y.stick(HERO, 0.0, 0.0);
    y.step(30);
    let rest = y.feet_in_car(HERO);
    // The throttle: held from a standstill.
    y.stick(HERO, 0.0, 1.0);
    y.step(30);
    let throttle = y.cm(HERO).runtime.boarding.throttle_in;
    let pressed = y.feet_in_car(HERO);
    let pressed_world = y.feet(HERO);
    let asks = y.foot_asks(HERO);
    let on = |i: usize, f: [Option<DVec3>; 2]| match asks[i] {
        Some((t, _)) => dist(f[i], t),
        None => f64::INFINITY,
    };
    let right_travel = (pressed[1].unwrap() - rest[1].unwrap()).length();
    let left_still = (pressed[0].unwrap() - rest[0].unwrap()).length();
    let right_on = on(1, pressed_world);
    // The brake: the stick pulled back while rolling forward is a brake, by
    // `VehicleControls::from_intent`'s own rule.
    let before_brake = y.feet_in_car(HERO);
    y.stick(HERO, 0.0, -1.0);
    y.step(6);
    let brake = y.cm(HERO).runtime.boarding.brake_in;
    let braked = y.feet_in_car(HERO);
    let braked_world = y.feet(HERO);
    let asks = y.foot_asks(HERO);
    let left_on = match asks[0] {
        Some((t, _)) => dist(braked_world[0], t),
        None => f64::INFINITY,
    };
    let left_travel = (braked[0].unwrap() - before_brake[0].unwrap()).length();
    println!(
        "=== the feet on the pedals ===\n  throttle {throttle:.2}: the right foot pressed {:.1} mm (the left moved {:.1} mm), {:.2} mm off its pedal\n  brake {brake:.2}: the left foot pressed {:.1} mm, {:.2} mm off its pedal (travel at full input {} mm)",
        right_travel * 1000.0,
        left_still * 1000.0,
        right_on * 1000.0,
        left_travel * 1000.0,
        left_on * 1000.0,
        board::PEDAL_TRAVEL_M * 1000.0
    );
    assert!(throttle > 0.9, "the throttle the car was given is {throttle:.2}");
    assert!(
        right_travel > 0.8 * board::PEDAL_TRAVEL_M * throttle,
        "the throttle foot moved {:.1} mm with the throttle at {throttle:.2}",
        right_travel * 1000.0
    );
    assert!(
        left_still < 0.01,
        "the BRAKE foot moved {:.1} mm on the throttle",
        left_still * 1000.0
    );
    assert!(right_on <= 0.02 && left_on <= 0.02, "a foot is off its pedal");
    assert!(brake > 0.5, "the brake the car was given is {brake:.2}");
    assert!(
        left_travel > 0.8 * board::PEDAL_TRAVEL_M * brake,
        "the brake foot moved {:.1} mm with the brake at {brake:.2}",
        left_travel * 1000.0
    );
}

// ── (d) THE SEATS HAVE AN INDEX ─────────────────────────────────────────────

/// **The hero drives and an NPC rides** — VEH2b's carried 4, closed. The
/// passenger is seated in `SeatIndex::Passenger`, its pelvis on ITS cushion,
/// its hands on the dashboard grab bar and its feet on the floor (the doc's
/// *"Dashboard Grab Bar (Passenger)"* / *"Floor Sockets (Passenger)"*), all
/// read on its joints; and the car obeys the DRIVER's stick and not the
/// passenger's.
///
/// **The mutation**: the seat step's `if seat_idx.drives()` guard made `true`
/// — the passenger's own (reversing) stick reaches the car too, and the
/// "the car went where the driver pointed it" assertion reds.
#[test]
fn a_passenger_rides_its_own_seat_and_does_not_drive() {
    let mut y = Yard::new("sedan");
    y.seat_npc(RIDER, SeatIndex::Passenger);
    y.board(HERO);
    y.step(30);
    // The passenger pushes the stick BACK; the driver pushes it forward.
    y.stick(RIDER, 0.0, -1.0);
    let z0 = y.car_at().z;
    y.stick(HERO, 0.0, 1.0);
    y.step(120);
    let moved = y.car_at().z - z0;
    y.stick(HERO, 0.0, 0.0);
    y.step(60);
    let car = y.frame();
    let cushion = car.world(car.sockets.seat(SeatIndex::Passenger));
    let pelvis = y
        .joint(RIDER, inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
        .expect("a passenger pelvis");
    let asks = y.hand_asks(RIDER);
    let hands = y.hands(RIDER);
    let fasks = y.foot_asks(RIDER);
    let feet = y.feet(RIDER);
    let grips = board::passenger_grips(&car.sockets, SeatIndex::Passenger);
    let floor = board::floor_feet(&car.sockets, SeatIndex::Passenger, car.floor_y());
    let mut hand_err = 0.0f64;
    let mut foot_err = 0.0f64;
    let mut on_grips = 0;
    for i in 0..2 {
        let (t, _) = asks[i].expect("both passenger hands are asked for");
        hand_err = hand_err.max(dist(hands[i], t));
        if grips.iter().any(|g| (car.world(*g) - t).length() < 1e-6) {
            on_grips += 1;
        }
        let (f, _) = fasks[i].expect("both passenger feet are asked for");
        foot_err = foot_err.max(dist(feet[i], f));
        assert!(
            floor.iter().any(|g| (car.world(*g) - f).length() < 1e-6),
            "a passenger foot was sent somewhere that is not the floor"
        );
    }
    let census = d3::boarding::seat_census(&y.world);
    println!(
        "=== the passenger ===\n  the car moved {moved:.2} m forward under the DRIVER's stick with the passenger pulling back\n  the passenger's pelvis {:.3} against its cushion {:.3}; hands {:.2} mm off the grab bar, feet {:.2} mm off the floor\n  the census: {:?}",
        pelvis.y,
        cushion.y,
        hand_err * 1000.0,
        foot_err * 1000.0,
        census
    );
    assert!(
        moved > 2.0,
        "the car moved {moved:.2} m under the driver's full throttle — the passenger's stick is reaching it"
    );
    assert_eq!(on_grips, 2, "the passenger's hands are not on the grab bar");
    assert!(hand_err <= 0.02 && foot_err <= 0.02);
    assert!(
        (pelvis.y - cushion.y).abs() < 0.03,
        "the passenger is not sitting on its seat"
    );
    assert_eq!(census.get(&(CHASSIS, 0)), Some(&1));
    assert_eq!(census.get(&(CHASSIS, 1)), Some(&1));
    // …and the traffic tier's share is what it says.
    let drawn = (0..4000u128)
        .filter(|i| inf_ecs::traffic::carries_passenger(Uuid::from_u128(0xCA00_0000 + i)))
        .count();
    let share = drawn as f64 / 4000.0;
    println!("  the traffic draw: {drawn} of 4000 cars carry a passenger ({share:.3})");
    assert!(
        (share - inf_ecs::traffic::PASSENGER_SHARE).abs() < 0.03,
        "the passenger draw is {share:.3} against {}",
        inf_ecs::traffic::PASSENGER_SHARE
    );
}

// ── THE OCCUPIED SEAT (inherited from the VEH3b audit) ──────────────────────

/// **An occupied driver's seat is NEVER entered** — it is a carjack from the
/// driver's door and a refusal BY NAME from anywhere else. Read on the prompt
/// the player is shown, the refusal counter, the census (never two bodies in
/// one seat, over every step), and the door `begin` itself.
///
/// **The mutation**: the refusal-by-name candidate deleted from
/// `carjack::candidates` — the far side of an occupied car shows no prompt and
/// the press is not refused, and the prompt assertion reds. The direct half:
/// `begin`'s occupied check deleted — `Begin::Occupied` becomes `Started` and
/// the census sees two drivers.
#[test]
fn an_occupied_seat_is_never_entered() {
    let mut y = Yard::new("sedan");
    y.seat_npc(VICTIM, SeatIndex::Driver);
    // (i) From the FAR side: the prompt names the refusal.
    let far = DVec3::new(-2.2, 0.0, -0.5);
    let e = y.world.entity_of(HERO).unwrap();
    y.world.world_mut().get_mut::<Transform>(e).unwrap().translation =
        Vec3d::new(far.x, far.y + 0.9, far.z);
    y.bridge.sync_from_world(&y.world);
    y.step(10);
    let feet = y.at(HERO) - DVec3::Y * 0.9;
    let hit = d3::interact::resolve(&y.world, &y.bridge, feet, 0.0, &Default::default())
        .expect("the one door answers something beside an occupied car");
    let prompt = inf_ecs::interact::prompt_text(hit.verb, &hit.label, "E");
    println!("=== the occupied seat ===\n  from the far side the prompt reads {prompt:?}");
    assert_eq!(hit.guid, CHASSIS);
    assert_eq!(hit.verb, inf_ecs::interact::InteractVerb::Occupied);
    assert_eq!(prompt, "[E] Occupied vehicle");
    let refusals0 = y.cm(HERO).runtime.refusals;
    let mut worst_census = 0u32;
    for _ in 0..6 {
        y.press(HERO);
        for _ in 0..40 {
            y.step(1);
            worst_census = worst_census.max(
                d3::boarding::seat_census(&y.world)
                    .values()
                    .copied()
                    .max()
                    .unwrap_or(0),
            );
            assert_ne!(
                y.cm(HERO).mode,
                MovementMode::Driving,
                "the hero is in an occupied car"
            );
        }
    }
    let refused = y.cm(HERO).runtime.refusals - refusals0;
    println!("  six presses from the far side: {refused} refused, the busiest seat held {worst_census}");
    assert!(refused >= 6, "only {refused} of six presses were refused");
    assert_eq!(worst_census, 1, "a seat held {worst_census} bodies");
    assert_eq!(y.cm(HERO).runtime.boarding.phase, BoardPhase::Idle);
    // (ii) The door itself: an Enter into an occupied seat is refused by value.
    let mut cm = y.cm(HERO);
    let pos = y.at(HERO);
    let verdict =
        d3::boarding::begin(&mut y.world, &mut y.bridge, HERO, &mut cm, pos, RADIUS, CHASSIS, false);
    println!("  `begin` asked for an ENTER into the occupied seat answers {verdict:?}");
    assert_eq!(verdict, d3::boarding::Begin::Occupied);
}

// ── (e) THE CARJACK ─────────────────────────────────────────────────────────

/// **The carjack plays the same pipeline** — the hero walks the same spline,
/// takes the same handle and opens the same door on the same motor, and the
/// driver is pulled out THROUGH it on its own reverse pipeline, forced
/// (`Jacked → Exiting`), landing in `FallControlled` at a point the collide
/// check passed. Read on both bodies' phase traces and on the census at every
/// step — never two bodies in the driver's seat.
///
/// The reference is `frames/steal-car/0010`–`0035`: the approach, the yank,
/// the driver dragged out, the hero in.
///
/// **The mutation**: the pull skipped (the door phase treats a `Jacked` victim
/// as an empty seat) — the hero sits down with the driver still in the seat,
/// and the census reds at two.
#[test]
fn the_carjack_plays_the_same_pipeline() {
    let mut y = Yard::new("sedan");
    y.seat_npc(VICTIM, SeatIndex::Driver);
    // Press until the driver lets go (the resist draw is per step).
    let mut presses = 0;
    for _ in 0..24 {
        y.press(HERO);
        y.step(1);
        presses += 1;
        if y.cm(HERO).runtime.boarding.phase != BoardPhase::Idle {
            break;
        }
        y.step(1);
    }
    assert!(
        y.cm(HERO).runtime.boarding.carjack,
        "twenty-four presses and the carjack never began"
    );
    let mut hero: Vec<Sample> = Vec::new();
    let mut victim: Vec<Sample> = Vec::new();
    let mut worst_census = 0u32;
    let mut braked_speed = 0.0f64;
    let mut landed: Option<(DVec3, MovementMode)> = None;
    for _ in 0..900 {
        y.step(1);
        hero.push(y.sample(HERO));
        victim.push(y.sample(VICTIM));
        worst_census = worst_census.max(
            d3::boarding::seat_census(&y.world)
                .values()
                .copied()
                .max()
                .unwrap_or(0),
        );
        if y.cm(VICTIM).runtime.boarding.phase == BoardPhase::Jacked {
            braked_speed = braked_speed.max(y.car_speed());
        }
        if landed.is_none() && !y.cm(VICTIM).runtime.seat.is_seated() {
            landed = Some((y.at(VICTIM), y.cm(VICTIM).mode));
        }
        if y.cm(HERO).runtime.boarding.phase == BoardPhase::Driving {
            break;
        }
    }
    print_trace("the hero, carjacking", &hero);
    print_trace("the driver, being pulled out", &victim);
    let hero_order: Vec<BoardPhase> = durations(&hero).iter().map(|(p, _, _)| *p).collect();
    let victim_order: Vec<BoardPhase> = durations(&victim).iter().map(|(p, _, _)| *p).collect();
    assert_eq!(
        hero_order,
        BoardPhase::ENTER_ORDER[..].to_vec(),
        "the carjacker did not walk the enter pipeline"
    );
    assert_eq!(
        victim_order.iter().take(2).copied().collect::<Vec<_>>(),
        vec![BoardPhase::Jacked, BoardPhase::Exiting],
        "the driver was not pulled out on its own reverse pipeline: {victim_order:?}"
    );
    // The pull started while the hero was at the open door.
    let pull_at = victim
        .iter()
        .position(|s| s.phase == BoardPhase::Exiting)
        .expect("the pull");
    assert_eq!(
        hero[pull_at].phase,
        BoardPhase::OpeningDoor,
        "the driver was pulled out while the hero was in `{}`",
        hero[pull_at].phase.name()
    );
    assert!(
        hero[pull_at].door_deg >= board::DOOR_BOARD_DEG - 0.5,
        "the driver was pulled through a door open {:.1} deg",
        hero[pull_at].door_deg
    );
    let (at, mode) = landed.expect("the driver never left the seat");
    let car = y.frame();
    let local = car.local(at);
    println!(
        "  {presses} press(es); the driver landed {mode:?} at {:.3?} in the car's frame; the car stood at {braked_speed:.3} m/s while it was held; the busiest seat held {worst_census}",
        local
    );
    assert_eq!(mode, MovementMode::FallControlled, "pulled out, not stepped out");
    assert!(
        local.x > car.half.x + 0.3,
        "the driver landed {:.3} m from the car's centreline — inside the bodywork",
        local.x
    );
    assert_eq!(worst_census, 1, "two bodies in one seat during the carjack");
    assert!(
        braked_speed < 0.5,
        "the victim's car rolled at {braked_speed:.3} m/s while it was being carjacked"
    );
}

/// **The victim is never put down inside a wall** — VEH2b's carried
/// collide-check, closed, measured against the colliders. A wall stands where
/// the pull-out point is; the driver lands at the next candidate the
/// point-in-collider check passes, and its capsule is clear there. A car
/// wedged on every side is a REFUSED carjack, not a body in a wall.
///
/// **The mutation**: `clear_exit` taking the preferred point unchecked — the
/// driver lands inside the slab and the clearance assertion reds.
#[test]
fn the_victim_is_never_put_down_inside_a_wall() {
    // A slab along the driver's flank, from the rear of the seat back, where
    // the pull-out point is.
    let slab_at = DVec3::new(2.05, 1.0, -0.3);
    let slab_half = DVec3::new(0.35, 1.0, 0.45);
    let mut y = Yard::build("sedan", |w| wall(w, WALL, slab_at, slab_half));
    y.seat_npc(VICTIM, SeatIndex::Driver);
    let car = y.frame();
    let preferred = car.world(board::pull_out_point(
        &car.sockets,
        SeatIndex::Driver,
        car.half,
        car.offset,
    ));
    let inside = |p: DVec3| {
        (p.x - slab_at.x).abs() < slab_half.x + RADIUS && (p.z - slab_at.z).abs() < slab_half.z + RADIUS
    };
    assert!(
        inside(preferred),
        "the fixture's wall does not cover the pull-out point {preferred:?} — this arm measures nothing"
    );
    for _ in 0..24 {
        y.press(HERO);
        y.step(1);
        if y.cm(HERO).runtime.boarding.phase != BoardPhase::Idle {
            break;
        }
        y.step(1);
    }
    let mut landed = None;
    for _ in 0..900 {
        y.step(1);
        if landed.is_none() && !y.cm(VICTIM).runtime.seat.is_seated() {
            landed = Some(y.at(VICTIM));
        }
        if y.cm(HERO).runtime.boarding.phase == BoardPhase::Driving {
            break;
        }
    }
    let at = landed.expect("the driver was never pulled out");
    let clear = y.clear(VICTIM);
    println!(
        "=== the pull-out against a wall ===\n  the preferred point {preferred:.3?} is inside the wall; the driver landed at {at:.3?}, clear: {clear}"
    );
    assert!(!inside(at), "the driver was put down inside the wall at {at:?}");
    assert!(clear, "the driver's capsule overlaps something where it landed");

    // WEDGED: walls on every side the candidates look. The carjack is refused
    // and the driver stays at the wheel.
    let mut y = Yard::new("sedan");
    // A ring of slabs round the car, leaving the hero's approach lane open.
    for (i, (c, h)) in [
        (DVec3::new(2.4, 1.0, -0.6), DVec3::new(0.5, 1.0, 1.2)),
        (DVec3::new(-2.4, 1.0, 0.0), DVec3::new(0.5, 1.0, 3.0)),
        (DVec3::new(0.0, 1.0, 3.6), DVec3::new(3.0, 1.0, 0.5)),
        (DVec3::new(0.0, 1.0, -3.6), DVec3::new(3.0, 1.0, 0.5)),
    ]
    .into_iter()
    .enumerate()
    {
        wall(&mut y.world, Uuid::from_u128(0x5E3D_0100 + i as u128), c, h);
    }
    y.world.mark_dirty();
    y.world.propagate();
    y.bridge.sync_from_world(&y.world);
    y.seat_npc(VICTIM, SeatIndex::Driver);
    let car = y.frame();
    let preferred = board::pull_out_point(&car.sockets, SeatIndex::Driver, car.half, car.offset);
    let stand_half = y.cm(VICTIM).stand_half_height_m;
    let own: std::collections::BTreeSet<_> = y.bridge.collider_of(VICTIM).into_iter().collect();
    let refusal = d3::boarding::clear_exit(&mut y.bridge, &car, preferred, stand_half, RADIUS, &own);
    println!("  wedged on every side, `clear_exit` answers {refusal:?}");
    assert!(
        refusal.is_none(),
        "a car walled in on every side still found somewhere to put a body: {refusal:?}"
    );
}

// ── (f) THE EXIT ────────────────────────────────────────────────────────────

/// **The exit is the reverse, and it never lands the body inside geometry** —
/// `Exiting` (the door opens on its motor, the body warps out of the seat past
/// the door) → `ClosingDoor` (the door is shut again) → `Idle`, read on the
/// phase trace, the hinge and the capsule's own clearance where it stands.
/// With a wall against the driver's flank the body leaves by the other side.
///
/// **The mutation**: `clear_exit`'s two locks deleted (the capsule-clear test
/// AND the path sweep from inside the car) — the walled exit lands the body in
/// the wall and the clearance assertion reds. `capsule_clear` answering `true`
/// ALONE survives, measured: a point inside a wall is a point the path sweep
/// ends inside, so the sweep refuses every candidate the capsule test would —
/// two locks on one door, and the arm is armed against losing both.
#[test]
fn the_exit_is_the_reverse_and_never_lands_in_geometry() {
    for walled in [false, true] {
        let mut y = if walled {
            Yard::build("sedan", |w| {
                wall(w, WALL, DVec3::new(1.55, 1.0, -0.4), DVec3::new(0.2, 1.0, 0.9));
            })
        } else {
            Yard::new("sedan")
        };
        if walled {
            // Seat the hero directly: the wall is where the approach would
            // walk, and this half is about the way OUT.
            let e = y.world.entity_of(HERO).unwrap();
            y.world.world_mut().get_mut::<Transform>(e).unwrap().translation =
                Vec3d::new(-2.4, 0.9, 0.2);
            y.bridge.sync_from_world(&y.world);
            y.with_cm(HERO, |cm| {
                cm.mode = MovementMode::Driving;
                cm.runtime.seat = inf_ecs::components::SeatState {
                    vehicle: CHASSIS,
                    entering: false,
                    time_s: 0.0,
                    start: Vec3d::ZERO,
                    start_yaw_deg: 0.0,
                    seat: 0,
                };
            });
            d3::vehicle::park_collider(&mut y.bridge, HERO, true);
            y.step(20);
        } else {
            y.board(HERO);
            y.step(20);
        }
        y.press(HERO);
        let mut trace: Vec<Sample> = Vec::new();
        let mut peak = 0.0f64;
        for _ in 0..400 {
            y.step(1);
            let s = y.sample(HERO);
            peak = peak.max(s.door_deg);
            trace.push(s);
            if s.phase == BoardPhase::Idle {
                break;
            }
        }
        print_trace(if walled { "the exit, walled" } else { "the exit" }, &trace);
        let order: Vec<BoardPhase> = durations(&trace).iter().map(|(p, _, _)| *p).collect();
        assert_eq!(
            order,
            vec![BoardPhase::Exiting, BoardPhase::ClosingDoor, BoardPhase::Idle],
            "the exit is not the reverse pipeline"
        );
        let clear = y.clear(HERO);
        let car = y.frame();
        let local = car.local(y.at(HERO));
        let shut = y
            .bridge
            .part_angle_rad(y.door())
            .map(|a| a.to_degrees().abs())
            .unwrap_or(0.0);
        println!(
            "  out at {local:.3?} in the car's frame, clear {clear}; the door peaked {peak:.1} deg and stands at {shut:.2}"
        );
        assert!(clear, "the body stepped out into something");
        assert_eq!(y.cm(HERO).mode, MovementMode::Grounded);
        assert!(
            local.x.abs() > car.half.x + 0.2 || local.z.abs() > car.half.z + 0.2,
            "the body is still inside the car's footprint at {local:?}"
        );
        if walled {
            assert!(
                local.x < 0.0,
                "with a wall at the driver's door the body left by it, at {local:?}"
            );
        } else {
            assert!(peak >= board::DOOR_BOARD_DEG - 0.5, "the door opened only {peak:.1} deg");
            assert!(shut <= board::DOOR_SHUT_DEG + 0.5, "the door was left at {shut:.2} deg");
        }
    }
}

/// **A moving exit is a ROLL** — above [`board::EXIT_ROLL_MPS`] the body
/// leaves the car in `FallControlled` with the car's velocity, its first
/// grounded step ROLLS (through the roll key's own request, so the capsule is
/// resized once), which is the mode CHAR1b.2's `land_roll` plays in
/// (`inf_anim::als`'s `roll` slot, `LocoMode::Roll`), and the roll comes up
/// STANDING, where the stance key's roll ends crouched. The mode trace is
/// read; that the ROLL CLIP plays on the shipped hero is read on the island by
/// `the_islands_hero_boards_drives_and_rolls_out`.
///
/// **The mutation**: the bail flag never set — the landing is soft by the
/// vertical classifier and no `Roll` appears.
#[test]
fn a_moving_exit_is_a_roll() {
    let mut y = Yard::new("sedan");
    y.board(HERO);
    y.stick(HERO, 0.0, 1.0);
    for _ in 0..600 {
        y.step(1);
        if y.car_speed() > 7.0 {
            break;
        }
    }
    let speed = y.car_speed();
    y.stick(HERO, 0.0, 0.0);
    y.press(HERO);
    let mut modes: Vec<MovementMode> = Vec::new();
    let mut landing = inf_ecs::components::LandingKind::None;
    let mut left_at = None;
    for _ in 0..240 {
        y.step(1);
        let cm = y.cm(HERO);
        if modes.last() != Some(&cm.mode) {
            modes.push(cm.mode);
        }
        if left_at.is_none() && !cm.runtime.seat.is_seated() {
            left_at = Some(cm.runtime.velocity.to_dvec3().length());
        }
        if cm.runtime.landing != inf_ecs::components::LandingKind::None {
            landing = cm.runtime.landing;
        }
        if cm.mode == MovementMode::Grounded && modes.contains(&MovementMode::Roll) {
            break;
        }
    }
    println!(
        "=== the bail-out ===\n  the car at {speed:.2} m/s; the body left at {:.2} m/s; modes {modes:?}; landing {landing:?}",
        left_at.unwrap_or(0.0)
    );
    assert!(speed > board::EXIT_ROLL_MPS * 2.0, "the car only reached {speed:.2} m/s");
    assert_eq!(modes.first(), Some(&MovementMode::FallControlled));
    assert!(
        modes.contains(&MovementMode::Roll),
        "no roll after bailing out at {speed:.2} m/s: {modes:?}"
    );
    assert_eq!(
        modes.last(),
        Some(&MovementMode::Grounded),
        "the bail-out roll did not come up standing: {modes:?}"
    );
}

/// **A car whose door was torn off is boarded THROUGH THE OPENING** — the
/// door phase has no hinge to wait for and no handle to take, so it lasts the
/// reach and the step back and nothing more, and no hand is asked for.
///
/// **The mutation**: `door_for_seat` ignoring the latch (answering the shed
/// door) — the door phase waits out [`board::OPENING_MAX_S`] for a hinge that
/// is lying in the road, and the length assertion reds.
#[test]
fn a_door_torn_off_is_boarded_through_the_opening() {
    let mut y = Yard::new("sedan");
    let door = y.door();
    {
        let r = inf_ecs::bodywork::damage_mut(&mut y.world)
            .rows
            .get_mut(&CHASSIS)
            .expect("the car's row");
        r.parts.get_mut(&door).expect("the door").latch = inf_ecs::bodywork::PartLatch::Shed;
        r.refresh_parts();
    }
    assert!(board::door_for_seat(&y.world, CHASSIS, SeatIndex::Driver).is_none());
    let mut asked = 0usize;
    y.press(HERO);
    let mut trace = Vec::new();
    for _ in 0..600 {
        y.step(1);
        let s = y.sample(HERO);
        if s.phase == BoardPhase::OpeningDoor && y.hand_asks(HERO).iter().any(Option::is_some) {
            asked += 1;
        }
        trace.push(s);
        if s.phase == BoardPhase::Driving {
            break;
        }
    }
    print_trace("through the opening", &trace);
    let opening = durations(&trace)
        .iter()
        .find(|(p, _, _)| *p == BoardPhase::OpeningDoor)
        .map(|(_, t, _)| *t)
        .unwrap_or(f64::INFINITY);
    assert_eq!(y.cm(HERO).runtime.boarding.phase, BoardPhase::Driving);
    assert_eq!(asked, 0, "a hand reached for a handle on a door that is not there");
    assert!(
        opening < board::HAND_REACH_S + board::HANDLE_HOLD_S + board::STEP_BACK_S + 0.1,
        "the door phase lasted {opening:.3} s with no door to open"
    );
}

// ── CAMERA ──────────────────────────────────────────────────────────────────

/// **The boarding camera rides the CHAR1c director** — an `Override` claim
/// under [`inf_ecs::camera::CAMERA_TAG_BOARDING`] for every step of the
/// ground choreography, released when the body reaches the wheel, and never a
/// second camera (the director has one output). And the drive pivot hangs off
/// the chassis ROOF, where the seated subject's feet were until the seat moved
/// into the cabin — so the drive block frames the car as it did.
///
/// **The mutation**: the claim not pushed (`boarding_camera_pose` answering
/// `None`) — the holder is the gameplay rig through the whole choreography and
/// the held-steps assertion reds.
#[test]
fn the_boarding_camera_rides_the_director() {
    let mut y = Yard::new("sedan");
    let mut cam = inf_ecs::camera::LocomotionCamera::default();
    for _ in 0..20 {
        d3::step_camera_with_requests(&mut y.world, &mut y.bridge, &mut cam, HERO, DT);
    }
    y.press(HERO);
    let (mut held, mut ground_steps, mut after_driving_held) = (0usize, 0usize, 0usize);
    let mut driving_for = 0usize;
    for _ in 0..900 {
        y.step(1);
        d3::step_camera_with_requests(&mut y.world, &mut y.bridge, &mut cam, HERO, DT);
        let phase = y.cm(HERO).runtime.boarding.phase;
        let holder = cam.director.holder();
        if phase.is_on_the_ground() {
            ground_steps += 1;
            if holder
                == Some((
                    inf_ecs::camera::CameraLayer::Override,
                    0,
                    inf_ecs::camera::CAMERA_TAG_BOARDING,
                ))
            {
                held += 1;
            }
        }
        if phase == BoardPhase::Driving {
            driving_for += 1;
            if holder.map(|h| h.2) == Some(inf_ecs::camera::CAMERA_TAG_BOARDING) {
                after_driving_held += 1;
            }
            if driving_for > 60 {
                break;
            }
        }
    }
    // The drive pivot: the roof over the chassis centre, plus the tuning's
    // ratio of the body's height — what P29.6 derived from feet ON the roof.
    let car = y.frame();
    let roof = car.world(Vec3d::new(car.offset.x, car.offset.y + car.half.y, car.offset.z));
    let cm = y.cm(HERO);
    let standing = cm.stand_half_height_m + RADIUS;
    let want = roof + DVec3::Y * (2.0 * standing * cam.tuning.pivot_height_ratio);
    let pivot_off = (cam.pivot.to_dvec3() - want).length();
    println!(
        "=== the boarding camera ===\n  the claim held {held} of {ground_steps} ground steps; {after_driving_held} steps after the wheel; the drive pivot is {pivot_off:.4} m from the roof-derived one"
    );
    assert!(ground_steps > 60, "the choreography took {ground_steps} ground steps");
    assert!(
        held >= ground_steps - 2,
        "the boarding claim held the camera for {held} of {ground_steps} ground steps"
    );
    assert_eq!(
        after_driving_held, 0,
        "the boarding claim outlived the boarding"
    );
    assert!(
        pivot_off < 0.25,
        "the drive camera's pivot is {pivot_off:.3} m off the roof — it followed the seat into the cabin"
    );
}

// ── (g) DETERMINISM ─────────────────────────────────────────────────────────

/// **The nineteenth section is EMPTY until somebody boards**, carries one row
/// per boarding body while it runs, and is empty again at the wheel — which is
/// what keeps every trace committed before this wave byte-identical.
///
/// **The mutation**: `BoardingState::is_quiet` answering `false` — a standing
/// hero folds a row and the first assertion reds.
#[test]
fn the_boarding_section_is_empty_until_somebody_boards() {
    let mut y = Yard::new("sedan");
    assert!(
        board::boarding_state_bytes(&y.world).is_empty(),
        "a level where nobody has pressed E folds {} boarding bytes",
        board::boarding_state_bytes(&y.world).len()
    );
    y.press(HERO);
    let mut lens: Vec<(BoardPhase, usize)> = Vec::new();
    for _ in 0..900 {
        y.step(1);
        let b = board::boarding_state_bytes(&y.world);
        let phase = y.cm(HERO).runtime.boarding.phase;
        lens.push((phase, b.len()));
        if phase == BoardPhase::Driving {
            break;
        }
    }
    let loud = lens.iter().filter(|(_, n)| *n > 0).count();
    println!(
        "=== the nineteenth section ===\n  {loud} steps folded {} bytes each; at the wheel it folds {}",
        board::BOARDING_TRACE_BYTES,
        board::boarding_state_bytes(&y.world).len()
    );
    for (p, n) in &lens {
        if *p == BoardPhase::Driving || *p == BoardPhase::Idle {
            assert_eq!(*n, 0, "`{}` folded {n} bytes", p.name());
        } else {
            assert_eq!(
                *n,
                board::BOARDING_TRACE_BYTES,
                "`{}` folded {n} bytes, not one row",
                p.name()
            );
        }
    }
    assert!(loud > 60, "only {loud} steps folded anything");
}

/// **THE EDITOR'S PREVIEW AND THE SHIPPED PLAYER BOARD, DRIVE AND LEAVE THE
/// SAME CAR** — `SimSession` and `RuntimeSim` driven through their own front
/// doors, the E press and the throttle through the INPUT door on both, and
/// `boarding_state_bytes` plus the hero's own transform compared step by step
/// — twice: an empty car, and the same car with its own driver at the wheel,
/// which the same press CARJACKS (the victim's transform and phase are on the
/// row too, so a pull-out that landed the driver somewhere else in one host
/// is a divergence).
///
/// **Anti-vacuity**: two recordings of nothing are equal too, so the shipped
/// trace has to have folded boarding bytes for the whole choreography, the
/// hero has to have reached the wheel, driven, and got out.
///
/// **The mutations**: the nineteenth fold made `let _ =` in
/// `RuntimeSim::state_bytes` — the tail check reds on the first boarding step
/// (`projector_mirror`'s order pin is a SUBSTRING search and passes it,
/// measured); `begin` answering `NoCar` — the anti-vacuity half reds (the
/// hero never reaches the wheel).
#[test]
fn pie_equals_shipping_on_a_board_drive_exit_course() {
    use inf_editor_core::scene::SceneDoc;
    use inf_editor_core::simulate::{SimInput, SimSession};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    use std::collections::BTreeMap;

    const HZ: f64 = 60.0;
    const INTERACT: &str = inf_ecs::movement::actions::INTERACT;
    const MOVE_Y: &str = inf_ecs::movement::actions::MOVE_Y;

    const HANDBRAKE: &str = inf_ecs::movement::actions::HANDBRAKE;
    /// The course, as (interact edge, move_y, handbrake held) per step: board,
    /// drive two seconds, HANDBRAKE to a stop (a press at speed would be the
    /// bail-out, which `a_moving_exit_is_a_roll` owns), get out on foot.
    ///
    /// The CARJACK course presses every fourth step for a second from step
    /// 60, because the victim's resist draw is a function of the step and a
    /// player presses until the door gives; a press made while the machine
    /// runs is ignored by it, and the drive starts long after the last one.
    fn course(step: u32, jack: bool) -> (bool, f32, bool) {
        match step {
            60 => (true, 0.0, false),
            61..=120 if jack && step % 4 == 0 => (true, 0.0, false),
            400..=520 => (false, 1.0, false),
            521..=700 => (false, 0.0, true),
            720 => (true, 0.0, false),
            _ => (false, 0.0, false),
        }
    }
    const STEPS: u32 = 900;

    fn slab() -> (Transform, RigidBody3D, Collider3D) {
        (
            Transform {
                translation: Vec3d::new(0.0, -0.5, 0.0),
                ..Default::default()
            },
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(80.0, 0.5, 80.0),
                friction: 0.9,
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
    /// The car's own driver, for the CARJACK course: seated at the wheel
    /// before either host steps, exactly as a traffic driver is.
    fn victim_bits() -> (
        Transform,
        RigidBody3D,
        Collider3D,
        CharacterController3D,
        CharacterMovement,
    ) {
        let (mut t, body, col, ctl, mut cm) = hero_bits();
        cm.player_controlled = false;
        cm.mode = MovementMode::Driving;
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: CHASSIS,
            entering: false,
            time_s: 0.0,
            start: Vec3d::new(-12.0, 0.0, -12.0),
            start_yaw_deg: 0.0,
            seat: SeatIndex::Driver.as_u8(),
        };
        t.translation = Vec3d::new(0.0, t.translation.y, 0.0);
        (t, body, col, ctl, cm)
    }
    let def = catalogue_def("sedan");
    let at = DVec3::new(0.0, inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15, 0.0);

    type Row = (Vec<u8>, [u64; 3], u8, [u64; 3], u8);
    let hero_row = |world: &EcsWorld| -> Row {
        let e = world.entity_of(HERO).expect("the hero");
        let t = world.world().get::<Transform>(e).expect("placed").translation;
        let cm = world.world().get::<CharacterMovement>(e).expect("a mover");
        let (vt, vp) = world
            .entity_of(VICTIM)
            .map(|v| {
                let t = world.world().get::<Transform>(v).expect("placed").translation;
                let cm = world.world().get::<CharacterMovement>(v).expect("a mover");
                (
                    [t.x.to_bits(), t.y.to_bits(), t.z.to_bits()],
                    cm.runtime.boarding.phase.as_u8(),
                )
            })
            .unwrap_or(([0; 3], 0));
        (
            board::boarding_state_bytes(world),
            [t.x.to_bits(), t.y.to_bits(), t.z.to_bits()],
            cm.runtime.boarding.phase.as_u8(),
            vt,
            vp,
        )
    };
    let run = |jack: bool| -> (Vec<Row>, Vec<Row>) {

    let shipped: Vec<Row> = {
        let mut world = EcsWorld::new();
        let g = world.spawn_with_guid(GROUND, "slab", None);
        world.world_mut().entity_mut(g).insert(slab());
        car(&mut world, CHASSIS, at, 0.0, &def);
        let h = world.spawn_with_guid(HERO, "Hero", None);
        world.world_mut().entity_mut(h).insert(hero_bits());
        if jack {
            let v = world.spawn_with_guid(VICTIM, "Driver", None);
            world.world_mut().entity_mut(v).insert(victim_bits());
        }
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
        (0..STEPS)
            .map(|i| {
                let (press, y, brake) = course(i, jack);
                let mut input = RuntimeInput::default();
                if press {
                    input = input.press(INTERACT);
                }
                if brake {
                    input = input.press(HANDBRAKE);
                }
                if y != 0.0 {
                    input = input.axis_at(MOVE_Y, y);
                }
                sim.step_once(input);
                // The nineteenth section is the TAIL of the shipped trace: a
                // fold turned into `let _ =` keeps the name `projector_mirror`
                // searches for, and is read here instead.
                let folded = board::boarding_state_bytes(sim.world());
                if !folded.is_empty() {
                    assert!(
                        sim.state_bytes().ends_with(&folded),
                        "step {i}: the shipped trace does not END with the boarding section"
                    );
                }
                hero_row(sim.world())
            })
            .collect()
    };

    let preview: Vec<Row> = {
        use inf_editor_core::ipc::SpawnKind;
        let mut doc = SceneDoc::new();
        let g = doc.create_with_guid(GROUND, SpawnKind::Empty, "slab", None);
        doc.world_mut().world_mut().entity_mut(g).insert(slab());
        inf_editor_core::vehicle::spawn_vehicle(
            &mut doc,
            CHASSIS,
            &def,
            inf_editor_core::vehicle::VehicleSpawn {
                name: "Car",
                at,
                yaw_deg: 0.0,
                paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
                clip: None,
                engine_voice: false,
                livery: None,
            },
        );
        let h = doc.create_with_guid(HERO, SpawnKind::Empty, "Hero", None);
        doc.world_mut().world_mut().entity_mut(h).insert(hero_bits());
        if jack {
            let v = doc.create_with_guid(VICTIM, SpawnKind::Empty, "Driver", None);
            doc.world_mut().world_mut().entity_mut(v).insert(victim_bits());
        }
        doc.world_mut().propagate();
        let mut session =
            SimSession::enter(&mut doc, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
        let out = (0..STEPS)
            .map(|i| {
                let (press, y, brake) = course(i, jack);
                let mut down: Vec<&str> = if press { vec![INTERACT] } else { Vec::new() };
                if brake {
                    down.push(HANDBRAKE);
                }
                let mut axes: BTreeMap<String, f32> = BTreeMap::new();
                if y != 0.0 {
                    axes.insert(MOVE_Y.to_string(), y);
                }
                session.step_once(&mut doc, SimInput::with_down(down).with_axes(axes));
                hero_row(doc.world())
            })
            .collect();
        session.exit(&mut doc);
        out
    };
    (shipped, preview)
    };

    for jack in [false, true] {
    let (shipped, preview) = run(jack);
    let what = if jack { "carjack" } else { "board" };
    let loud = shipped.iter().filter(|r| !r.0.is_empty()).count();
    let phases: std::collections::BTreeSet<u8> = shipped.iter().map(|r| r.2).collect();
    let victim: std::collections::BTreeSet<u8> = shipped.iter().map(|r| r.4).collect();
    println!(
        "=== PIE == shipping on a {what} / drive / exit course ===\n  {STEPS} steps; {loud} folded boarding bytes; phases seen {phases:?}; the driver's {victim:?}"
    );
    if jack {
        assert!(
            victim.contains(&BoardPhase::Jacked.as_u8()),
            "the carjack course never pulled the driver out — it compares a boarding"
        );
        assert!(
            shipped.iter().any(|r| r.0.len() >= 2 * board::BOARDING_TRACE_BYTES),
            "the carjack never folded two rows at once (the hero and the victim)"
        );
    }
    for p in [
        BoardPhase::Unlocking,
        BoardPhase::OpeningDoor,
        BoardPhase::EnteringIK,
        BoardPhase::Seated,
        BoardPhase::Driving,
        BoardPhase::Exiting,
        BoardPhase::ClosingDoor,
    ] {
        assert!(
            phases.contains(&p.as_u8()),
            "the shipped course never reached `{}` — this arm compares less than it claims",
            p.name()
        );
    }
    assert!(loud > 120, "only {loud} steps folded boarding bytes");
    if let Some(i) = (0..shipped.len()).find(|i| shipped[*i] != preview[*i]) {
        panic!(
            "PIE and shipping diverged on the {what} course at step {i}: shipped phase {} at {:?}, preview phase {} at {:?}",
            shipped[i].2, shipped[i].1, preview[i].2, preview[i].1
        );
    }
    }
}

/// **A press of E is seen by exactly one step, whatever the frame rate** —
/// `RuntimeSim::run_frame`, the shipped window's own door. A frame that runs
/// no fixed step (a display faster than 60 Hz, or the demo's slow motion)
/// used to drop the press it carried: the edge was the difference against the
/// previous FRAME's keys and nobody stepped to see it. Found by this wave's own
/// demo at 0.3x — seven taps of E before a boarding began.
///
/// **The mutation**: the carry deleted (`frame_edges` never filled) — the
/// press on the zero-step frame is lost and no boarding begins.
#[test]
fn a_press_on_a_frame_that_runs_no_step_still_boards() {
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    let def = catalogue_def("sedan");
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(
        &mut world,
        CHASSIS,
        DVec3::new(0.0, inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15, 0.0),
        0.0,
        &def,
    );
    stand(&mut world, HERO, "Hero", HERO_AT, 0.0, true);
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
    for _ in 0..60 {
        sim.step_once(RuntimeInput::default());
    }
    let phase = |sim: &RuntimeSim| {
        let e = sim.world().entity_of(HERO).unwrap();
        sim.world().world().get::<CharacterMovement>(e).unwrap().runtime.boarding.phase
    };
    assert_eq!(phase(&sim), BoardPhase::Idle);
    // The press lands on a frame too short to run a step, and is let go on the
    // next frame, which runs one.
    let before = sim.steps();
    let ran = sim.run_frame(0.001, RuntimeInput::default().press(inf_ecs::movement::actions::INTERACT));
    assert_eq!(ran, 0, "the first frame was meant to run no step");
    assert_eq!(sim.steps(), before);
    let ran = sim.run_frame(1.0 / 60.0, RuntimeInput::default());
    let mut steps = u32::from(ran > 0);
    while phase(&sim) == BoardPhase::Idle && steps < 3 {
        sim.run_frame(1.0 / 60.0, RuntimeInput::default());
        steps += 1;
    }
    println!(
        "=== a tap on a zero-step frame ===\n  the next frame ran {ran} step(s); the hero is `{}`",
        phase(&sim).name()
    );
    assert_ne!(
        phase(&sim),
        BoardPhase::Idle,
        "a press on a frame that ran no step never reached the simulation"
    );
}

// ── (h) COST ────────────────────────────────────────────────────────────────

/// Whether a clock assert may run here: a RELEASE build, off CI — the house
/// conditioning every budget arm since the VEH3a audit carries. Everywhere
/// else the number is REPORTED and nothing is asserted.
fn clock_may_assert() -> bool {
    cfg!(not(debug_assertions)) && std::env::var("CI").is_err()
}

/// **What one boarding costs** — the whole fixed step (movement, vehicles,
/// gameplay, the solver, the write-back with its hands and feet, the pose)
/// with a body boarding, against the same step with the body standing still.
/// Min of five rounds of a warmed step per phase; never a cold first step.
#[test]
fn a_boarding_costs_what_it_costs() {
    use std::time::Instant;
    // The control: nobody boards.
    let control = {
        let mut y = Yard::new("sedan");
        y.step(30);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            y.step(1);
            best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        best
    };
    // The measured: the same world, phase by phase through a boarding.
    let mut y = Yard::new("sedan");
    y.step(30);
    y.press(HERO);
    y.step(1);
    let mut per_phase: std::collections::BTreeMap<u8, (BoardPhase, f64, usize)> =
        std::collections::BTreeMap::new();
    for _ in 0..900 {
        let phase = y.cm(HERO).runtime.boarding.phase;
        let t = Instant::now();
        y.step(1);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let e = per_phase
            .entry(phase.as_u8())
            .or_insert((phase, f64::INFINITY, 0));
        e.1 = e.1.min(ms);
        e.2 += 1;
        if phase == BoardPhase::Driving && e.2 > 30 {
            break;
        }
    }
    println!("=== the per-boarding cost (min per phase, the whole fixed step) ===");
    println!("  control (nobody boarding): {control:.4} ms");
    let mut worst = 0.0f64;
    for (p, ms, n) in per_phase.values() {
        println!(
            "  {:<10} {ms:.4} ms  ({n} steps)  over the control {:+.4} ms",
            p.name(),
            ms - control
        );
        worst = worst.max(ms - control);
    }
    assert!(per_phase.len() >= 6, "the boarding did not run its phases");
    assert!(control > 0.0, "the clock measured nothing");
    if clock_may_assert() {
        assert!(
            worst <= d3::boarding::BOARDING_STEP_BUDGET_MS,
            "a boarding step costs {worst:.4} ms over a standing one, against {} ms",
            d3::boarding::BOARDING_STEP_BUDGET_MS
        );
    } else {
        println!(
            "  (the {} ms ceiling is asserted under release off CI only)",
            d3::boarding::BOARDING_STEP_BUDGET_MS
        );
    }
}

/// **Sixty-four cars with drivers seated** — what the post-solve hands-and-
/// feet pass costs at the population the vehicle budget is kept at
/// (`VEHICLE_STEP_BUDGET_MS` is named at 64 cars), and what the seat
/// derivation at spawn costs. Min of five warmed rounds.
#[test]
fn sixty_four_seated_drivers_cost_what_they_cost() {
    use std::time::Instant;
    const CARS: usize = 64;
    let def = catalogue_def("sedan");
    let mut world = EcsWorld::default();
    ground(&mut world);
    let t_spawn = Instant::now();
    for i in 0..CARS {
        let c = Uuid::from_u128(0x5E3D_2000 + i as u128);
        let x = ((i % 8) as f64 - 4.0) * 4.0;
        let z = ((i / 8) as f64 - 4.0) * 7.0;
        car(
            &mut world,
            c,
            DVec3::new(x, inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15, z),
            0.0,
            &def,
        );
    }
    let spawn_ms = t_spawn.elapsed().as_secs_f64() * 1000.0;
    world.mark_dirty();
    world.propagate();
    let mut y = Yard {
        world,
        bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
        rig: None,
        clips: std::collections::BTreeMap::new(),
    };
    let t_derive = Instant::now();
    y.bridge.sync_from_world(&y.world);
    let derive_ms = t_derive.elapsed().as_secs_f64() * 1000.0;
    let drivers: Vec<Uuid> = (0..CARS)
        .map(|i| Uuid::from_u128(0x5E3D_3000 + i as u128))
        .collect();
    for (i, d) in drivers.iter().enumerate() {
        let c = Uuid::from_u128(0x5E3D_2000 + i as u128);
        stand(&mut y.world, *d, "Driver", DVec3::new(0.0, 0.0, 60.0 + i as f64), 0.0, false);
        y.with_cm(*d, |cm| {
            cm.mode = MovementMode::Driving;
            cm.runtime.seat = inf_ecs::components::SeatState {
                vehicle: c,
                entering: false,
                time_s: 0.0,
                start: Vec3d::ZERO,
                start_yaw_deg: 0.0,
                seat: 0,
            };
        });
    }
    y.world.mark_dirty();
    y.world.propagate();
    y.bridge.sync_from_world(&y.world);
    for d in &drivers {
        d3::vehicle::park_collider(&mut y.bridge, *d, true);
    }
    y.install_rig(&drivers);
    y.step(30);
    let mut best = f64::INFINITY;
    let mut report = d3::boarding::BoardingReport::default();
    for _ in 0..5 {
        let t = Instant::now();
        report = d3::boarding::follow_boarding(&mut y.bridge, &mut y.world);
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
        y.step(1);
    }
    println!(
        "=== {CARS} seated drivers ===\n  the hands-and-feet pass: {best:.4} ms ({:.2} us a driver); {} hands and {} feet placed over {} seated bodies\n  spawning the {CARS} rigs {spawn_ms:.3} ms; deriving their seats in the first sync {derive_ms:.3} ms",
        best * 1000.0 / CARS as f64,
        report.hands,
        report.feet,
        report.seated
    );
    assert_eq!(report.seated as usize, CARS, "not every driver was walked");
    assert_eq!(report.hands as usize, CARS, "not every driver's hands were placed");
    assert_eq!(report.feet as usize, CARS, "not every driver's feet were placed");
    if clock_may_assert() {
        assert!(
            best <= d3::boarding::SEATED_POSTURE_BUDGET_MS,
            "{CARS} seated drivers cost {best:.4} ms against {} ms",
            d3::boarding::SEATED_POSTURE_BUDGET_MS
        );
    } else {
        println!(
            "  (the {} ms ceiling is asserted under release off CI only)",
            d3::boarding::SEATED_POSTURE_BUDGET_MS
        );
    }
}

// ── THE CAMERA TABLE'S WRITE HALF (CHAR1c carried 159) ──────────────────────

/// **The level's camera table round-trips through the write half** — a tuned
/// table written beside a level names only what it changed and reads back as
/// exactly that table; an untouched one writes no keys; and the island's
/// committed `camera.toml` is what the write half writes for the table it
/// ships with.
///
/// **The mutation**: `to_toml_changed` writing the whole table — the one-key
/// file becomes a two-hundred-line one and the key-count assertion reds.
#[test]
fn the_level_camera_table_round_trips_through_the_write_half() {
    let dir = std::env::temp_dir().join(format!("veh3d-camera-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temp dir");
    let level = dir.join("Level.inf_lvl");
    let mut t = inf_ecs::camera::CameraTuning::default();
    assert!(t.set("run.arm_length_m", 5.25));
    assert!(t.set("aim_blend_speed", 14.5));
    let path = inf_ecs::camera::write_camera_beside(&level, &t).expect("written");
    let text = std::fs::read_to_string(&path).expect("read back");
    let keys: Vec<&str> = text
        .lines()
        .filter(|l| !l.starts_with('#') && l.contains('='))
        .collect();
    let back = inf_ecs::camera::read_camera_beside(&level)
        .expect("parses")
        .expect("present");
    println!(
        "=== the camera table beside a level ===\n  {} key lines written: {keys:?}",
        keys.len()
    );
    assert_eq!(back, t, "the table did not read back as it was written");
    // Two knobs are THREE leaves: a gait block's key lives in both rotation-
    // mode tables (`velocity_direction` and `looking_direction`), which `set`
    // keeps in step, so `run.arm_length_m` changed two of them.
    assert_eq!(keys.len(), 3, "the file names more than what changed: {keys:?}");
    // Untouched writes nothing but the header.
    let plain = inf_ecs::camera::write_camera_beside(&level, &Default::default()).unwrap();
    let plain_text = std::fs::read_to_string(&plain).unwrap();
    assert_eq!(plain_text, inf_ecs::camera::CAMERA_TOML_HEADER);
    // The island's committed file is that, byte for byte.
    const ISLAND: &str = include_str!("../../../samples/island/camera.toml");
    assert_eq!(
        ISLAND, plain_text,
        "the island's committed camera.toml is not what the write half writes"
    );
    // …and the recipe copies it into the project, beside the level.
    const RECIPE: &str = include_str!("../../../samples/island/island.toml");
    assert!(
        RECIPE.contains("\"camera.toml\","),
        "the island recipe does not copy its camera table"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

// ── INSTRUMENTS ─────────────────────────────────────────────────────────────

/// **The shipped host draws the boarding row**, and the row reads what the
/// machine is doing.
#[test]
fn the_shipped_host_draws_the_boarding_row() {
    const SRC: &str = include_str!("../src/window.rs");
    assert!(
        SRC.contains("inf_ecs::boarding::boarding_readout"),
        "`window.rs` no longer calls `boarding_readout` — the boarding is simulated and nothing draws it"
    );
    let mut b = inf_ecs::boarding::BoardingState::default();
    assert_eq!(inf_ecs::boarding::boarding_readout(&b), None);
    b.enter(BoardPhase::OpeningDoor, 0.0);
    b.hand_weight = 1.0;
    b.hand_err_m = 0.0123;
    b.door = Uuid::from_u128(9);
    b.door_deg = 32.4;
    let row = inf_ecs::boarding::boarding_readout(&b).expect("a row");
    println!("the boarding row reads: {row}");
    assert_eq!(row, "BOARDING opening  SEAT driver  HAND 0.012 m  DOOR 32 deg");
}

/// **Every new number this wave added is a RUNTIME one** — no persisted field
/// landed, and VEH3a's spent window stays spent.
#[test]
fn this_wave_moved_no_schema() {
    assert_eq!(
        inf_ecs::vehicle::VehicleTuning::names().len(),
        100,
        "the tunable count moved"
    );
    // The machine rides `MovementRuntime`, which `CharacterMovement` skips.
    const COMPONENTS: &str = include_str!("../../../crates/inf-ecs/src/components.rs");
    let at = COMPONENTS
        .find("pub boarding: crate::boarding::BoardingState,")
        .expect("the machine is on `MovementRuntime`");
    let runtime = COMPONENTS
        .find("pub struct MovementRuntime {")
        .expect("`MovementRuntime`");
    assert!(at > runtime, "the machine is not on the runtime block");
    assert!(
        COMPONENTS.contains("#[serde(skip)]\n    #[reflect(ignore)]\n    pub runtime: MovementRuntime,"),
        "`CharacterMovement::runtime` is no longer `#[serde(skip)]` — the machine would be on the wire"
    );
    const BOARDING: &str = include_str!("../../../crates/inf-ecs/src/boarding.rs");
    let at = BOARDING.find("pub struct BoardingState {").expect("the machine");
    let head = &BOARDING[at.saturating_sub(300)..at];
    assert!(
        !head.contains("Serialize"),
        "`BoardingState` grew a `Serialize`"
    );
    let at = BOARDING.find("pub struct VehicleSockets {").expect("the sockets");
    let head = &BOARDING[at.saturating_sub(300)..at];
    assert!(
        !head.contains("Serialize"),
        "`VehicleSockets` grew a `Serialize` — the sockets are DERIVED, VEH3a's ruling"
    );
}

// ── ON THE ISLAND: the shipped hero, the shipped host ───────────────────────
//
// Fixture arms are not the shipped host (the house law since FIX1). These
// load the island through the player's own loader — the MetaHuman hero, its
// own `.inf_skel` and its own ALS state machine, the island's traffic — and
// measure the same things on it. Local-only: the island is not committed
// content, and CI skips these by name.

fn island_project() -> Option<std::path::PathBuf> {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../island-build/project/Content");
    (p.join("VancouverIsland.inf_lvl").is_file()).then(|| p.canonicalize().unwrap_or(p))
}

fn island_sim(content: &std::path::Path) -> inf_player::runtime_sim::RuntimeSim {
    let source = inf_player::level::DevDirLevelSource::new(content.join("VancouverIsland.inf_lvl"));
    let terrains = inf_player::level::terrain_paths_by_guid_from_dir(content);
    let pcg_terrains = terrains.clone();
    let (skeletons, clips, machines) = inf_player::level::load_anim_assets_from_dir(content);
    let builder = inf_player::level::InfSceneWorldBuilder::with_defaults(
        inf_player::level::load_actor_classes_from_dir(content),
    )
    .with_pcgs(inf_player::level::load_pcg_payloads_by_guid_from_dir(
        content,
    ))
    .with_biome_sets(inf_player::level::load_biome_sets_by_guid_from_dir(content))
    .with_anim_assets(skeletons, clips, machines)
    .with_audio(inf_player::level::load_audio_assets_from_dir(content))
    .with_terrain_resolver(std::sync::Arc::new(move |g| {
        inf_player::level::terrain_source_from_file(pcg_terrains.get(&g)?).ok()
    }));
    let mut built = inf_player::level::load(&source, &builder).expect("the island builds");
    let partition = built.take_partition();
    let pcg = built.pcg_context();
    let mut sim = inf_player::sim_from_built(built);
    inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
    inf_player::attach_terrain_streaming(&mut sim, &inf_player::TerrainContent::Dir(terrains));
    sim
}

/// Move the hero (capsule centre) — the transform and the kinematic body.
fn set_hero(sim: &mut inf_player::runtime_sim::RuntimeSim, hero: Uuid, at: DVec3) {
    if let Some(e) = sim.world().entity_of(hero) {
        if let Some(mut t) = sim.world_mut().world_mut().get_mut::<Transform>(e) {
            t.translation = Vec3d::from_dvec3(at);
        }
    }
    if let Some(b) = sim.bridge3d().body_of(hero) {
        sim.bridge3d_mut().world_mut().set_body_translation(b, at);
    }
}

/// A joint of a sim's character, off its own skeleton.
fn sim_joint(
    sim: &inf_player::runtime_sim::RuntimeSim,
    who: Uuid,
    role: inf_anim::BoneRoleKind,
    side: inf_anim::BoneSide,
) -> Option<DVec3> {
    // `skeleton_of` is keyed by the SKELETON asset, which the entity names on
    // its `SkeletalMesh`.
    let skel = sim
        .world()
        .entity_of(who)
        .and_then(|e| sim.world().world().get::<inf_ecs::components::SkeletalMesh>(e))
        .and_then(|m| m.skeleton)?;
    let rig = sim.skeleton_of(skel)?;
    let j = rig.role_index().first(role, side)?;
    let posed = inf_ecs::pose::evaluated_pose(sim.world(), who)?;
    let to_world = inf_ecs::pose::model_to_world_of(sim.world(), who)?;
    let g = inf_anim::pose::global_transforms(&rig.skeleton, &posed.pose);
    let p = g.get(j as usize)?.to_scale_rotation_translation().2;
    Some(to_world.transform_point3(DVec3::new(
        f64::from(p.x),
        f64::from(p.y),
        f64::from(p.z),
    )))
}

/// **THE ISLAND'S OWN HERO boards a parked car, drives it, and bails out of
/// it** — on the MetaHuman, its own skeleton and its own ALS machine, through
/// the shipped host's own input door. Read on its joints: the hand on the
/// handle at the take, the pelvis inside the car, the hands on the rim, the
/// feet on the pedals — and after the bail, the `roll` STATE of its own
/// machine (`ALS_N_LandRoll_F`, CHAR1b.2's clip) actually entered.
///
/// **The mutations**: the bail flag never set — the machine never enters
/// `roll`; the reach dip zeroed — the MetaHuman's hand stops short of the
/// handle.
#[test]
fn the_islands_hero_boards_drives_and_rolls_out() {
    use inf_player::runtime_sim::RuntimeInput;
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content, CI has none");
        return;
    };
    let mut sim = island_sim(&content);
    for _ in 0..240 {
        sim.step_once(RuntimeInput::default());
    }
    let hero = inf_ecs::movement::camera_subject(sim.world()).expect("the island's hero");
    // The nearest EMPTY car with a live rig: nobody at its wheel.
    let here = d3::boarding::capsule_centre(sim.world(), hero).expect("placed");
    let occupied = d3::carjack::occupied_chassis(sim.world());
    let emergency: Vec<inf_ecs::components::VehicleClass> = {
        let defs = inf_editor_core::vehicle::island_vehicles();
        let mut out = Vec::new();
        for a in inf_pcg::ArchetypeId::ALL {
            for id in inf_editor_core::island::station_fleet(a) {
                if let Some(d) = defs.get(id) {
                    if !out.contains(&d.class) {
                        out.push(d.class);
                    }
                }
            }
        }
        out
    };
    let mut cars: Vec<(f64, Uuid)> = sim
        .bridge3d()
        .vehicle_guids()
        .into_iter()
        .filter(|g| !occupied.contains(g))
        // A CIVILIAN car: a station's appliance or cruiser is dispatch's, and
        // a parked one is held for its crew (island_gate's `civilian_cars`
        // rule, for the same reason).
        .filter(|g| {
            sim.world()
                .entity_of(*g)
                .and_then(|e| sim.world().world().get::<inf_ecs::components::VehicleClass>(e).copied())
                .is_none_or(|c| !emergency.contains(&c))
        })
        .filter_map(|g| {
            let (seat, _, v) = d3::vehicle::seat_pose(sim.bridge3d(), g)?;
            (v.length() < 0.2).then(|| ((seat - here).length(), g))
        })
        .collect();
    cars.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    // …with ROAD ahead of it. The nearest one was measured parked nose-in: its
    // throttle held 4 s reached 1.5 m/s and then 0.1 against whatever is in
    // front of it, which is a car nobody can bail out of at speed. A ray from
    // its front bumper at bonnet height, 20 m, clear of everything.
    let clear_ahead = |sim: &mut inf_player::runtime_sim::RuntimeSim, g: Uuid| -> bool {
        let Some(car) = d3::boarding::car_frame(sim.world(), sim.bridge3d(), g, false) else {
            return false;
        };
        // Two rays: at bonnet height, and at kerb height (0.12 m off the pan's
        // underside — a kerb stops a car a bonnet-height ray flies over).
        //
        // …and the car's whole width swept ahead, from half a metre past its
        // own nose: a car chosen by the two rays alone was measured stopping
        // 0.8 m on against something off its centreline, both driven wheels
        // spinning.
        let ahead = car.dir(DVec3::Z);
        let rays = [car.offset.y + 0.2, car.offset.y - car.half.y + 0.12].iter().all(|y| {
            let from = car.world(Vec3d::new(car.offset.x, *y, car.offset.z + car.half.z + 0.3));
            sim.bridge3d_mut().world_mut().cast_ray(from, ahead, 20.0).is_none()
        });
        let from = car.world(Vec3d::new(car.offset.x, car.offset.y + 0.1, car.offset.z + car.half.z + 0.5));
        let body = d3::ColliderShape3D::Box {
            half_extents: DVec3::new(car.half.x + 0.1, 0.3, 0.2),
        };
        let swept = sim
            .bridge3d_mut()
            .world_mut()
            .cast_shape(&body, from, car.rot, ahead, 20.0, &Default::default())
            .is_none();
        rays && swept
    };
    let chassis = cars
        .iter()
        .take(12)
        .map(|(_, g)| *g)
        .find(|g| clear_ahead(&mut sim, *g))
        .expect("an empty parked car on the island with road ahead of it");
    println!("  candidates {}, chose {chassis}", cars.len());
    let car = d3::boarding::car_frame(sim.world(), sim.bridge3d(), chassis, true).expect("its frame");
    let lift = {
        let e = sim.world().entity_of(hero).unwrap();
        let cm = sim.world().world().get::<CharacterMovement>(e).unwrap();
        let r = sim.world().world().get::<Collider3D>(e).map(|c| c.radius).unwrap_or(0.3);
        cm.stand_half_height_m + r
    };
    let beside = car.world(Vec3d::new(car.half.x + 1.6, 0.0, -1.2));
    let ground = sim.terrain_height_at(beside.x, beside.z);
    set_hero(&mut sim, hero, DVec3::new(beside.x, ground + lift + 0.05, beside.z));
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }
    let mode_at_press = {
        let e = sim.world().entity_of(hero).unwrap();
        sim.world().world().get::<CharacterMovement>(e).unwrap().mode
    };
    sim.step_once(RuntimeInput::default().press(inf_ecs::movement::actions::INTERACT));
    let mut takes: Vec<f64> = Vec::new();
    let mut phases: Vec<BoardPhase> = Vec::new();
    for _ in 0..900 {
        sim.step_once(RuntimeInput::default());
        let b = {
            let e = sim.world().entity_of(hero).unwrap();
            let cm = sim.world().world().get::<CharacterMovement>(e).unwrap();
            cm.runtime.boarding
        };
        if phases.last() != Some(&b.phase) {
            phases.push(b.phase);
        }
        if b.phase == BoardPhase::OpeningDoor && b.mark_s < 0.0 && b.hand_weight >= 1.0 - 1e-9 {
            let car = d3::boarding::car_frame(sim.world(), sim.bridge3d(), chassis, true).unwrap();
            if let Some(h) = d3::boarding::handle_world(sim.world(), sim.bridge3d(), &car, b.door, false)
            {
                let side = if b.hand_side == 0 {
                    inf_anim::BoneSide::Left
                } else {
                    inf_anim::BoneSide::Right
                };
                let hand = sim_joint(&sim, hero, inf_anim::BoneRoleKind::Hand, side);
                takes.push(hand.map(|p| (p - h).length()).unwrap_or(f64::INFINITY));
            }
        }
        if b.phase == BoardPhase::Driving {
            break;
        }
    }
    println!(
        "=== the island's hero, boarding car {chassis} ===\n  mode at the press {mode_at_press:?}; phases {:?}\n  at weight 1 before the latch: {} steps, worst {:.2} mm",
        phases.iter().map(|p| p.name()).collect::<Vec<_>>(),
        takes.len(),
        takes.iter().copied().fold(0.0f64, f64::max) * 1000.0
    );
    assert_eq!(
        mode_at_press,
        MovementMode::Grounded,
        "the hero was not standing when E was pressed — this arm would be measuring a ragdoll"
    );
    assert_eq!(phases.last(), Some(&BoardPhase::Driving), "the hero never reached the wheel");
    assert!(!takes.is_empty(), "the hand was never measured on the handle");
    let worst_take = takes.iter().copied().fold(0.0f64, f64::max);
    // Seated: the pelvis inside, the hands on the rim, the feet on the pedals.
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }
    let car = d3::boarding::car_frame(sim.world(), sim.bridge3d(), chassis, true).unwrap();
    let roof = car.world(Vec3d::new(car.offset.x, car.offset.y + car.half.y, car.offset.z)).y;
    let pelvis = sim_joint(&sim, hero, inf_anim::BoneRoleKind::Pelvis, inf_anim::BoneSide::Center)
        .expect("a pelvis");
    let asks = inf_ecs::pose::hand_ik(sim.world(), hero).cloned().unwrap_or_default();
    let mut rim_err = 0.0f64;
    for (i, side) in [inf_anim::BoneSide::Left, inf_anim::BoneSide::Right].iter().enumerate() {
        if let Some(r) = asks.reach[i] {
            let j = sim_joint(&sim, hero, inf_anim::BoneRoleKind::Hand, *side);
            rim_err = rim_err.max(j.map(|p| (p - r.target.to_dvec3()).length()).unwrap_or(f64::INFINITY));
        }
    }
    println!(
        "  seated: the pelvis {:+.3} m from the roof; the hands {:.2} mm off the rim",
        pelvis.y - roof,
        rim_err * 1000.0
    );
    // Drive, then bail.
    for _ in 0..240 {
        sim.step_once(
            RuntimeInput::default().axis_at(inf_ecs::movement::actions::MOVE_Y, 1.0),
        );
        let v = d3::boarding::car_frame(sim.world(), sim.bridge3d(), chassis, false)
            .map(|c| c.vel.length())
            .unwrap_or(0.0);
        if v > 7.0 {
            break;
        }
    }
    let speed = d3::boarding::car_frame(sim.world(), sim.bridge3d(), chassis, false)
        .map(|c| c.vel.length())
        .unwrap_or(0.0);
    sim.step_once(RuntimeInput::default().press(inf_ecs::movement::actions::INTERACT));
    let mut states: Vec<String> = Vec::new();
    let mut modes: Vec<MovementMode> = Vec::new();
    for _ in 0..120 {
        sim.step_once(RuntimeInput::default());
        let e = sim.world().entity_of(hero).unwrap();
        let mode = sim.world().world().get::<CharacterMovement>(e).unwrap().mode;
        if modes.last() != Some(&mode) {
            modes.push(mode);
        }
        if let Some(s) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) {
            if states.last() != Some(&s.name) {
                states.push(s.name.clone());
            }
        }
    }
    println!(
        "  bailed out at {speed:.2} m/s: modes {modes:?}; machine states {states:?}"
    );
    assert!(
        worst_take <= 0.02,
        "the island hero's hand ended {:.2} mm from the handle at weight 1",
        worst_take * 1000.0
    );
    assert!(pelvis.y < roof - 0.3, "the island hero is not inside the car");
    assert!(rim_err <= 0.02, "the island hero's hands are {:.2} mm off the rim", rim_err * 1000.0);
    assert!(speed > 2.0 * board::EXIT_ROLL_MPS, "the car only reached {speed:.2} m/s");
    assert!(
        modes.contains(&MovementMode::Roll),
        "no roll mode after bailing out at {speed:.2} m/s: {modes:?}"
    );
    assert!(
        states.iter().any(|s| s == "roll"),
        "the hero's own machine never entered `roll` — CHAR1b.2's roll clip did not play: {states:?}"
    );
}

/// **THE ISLAND CENSUS: ten minutes, and no seat ever holds two bodies** —
/// the VEH3b audit's inherited finding, closed as a measurement over the
/// island's own traffic, crews and passengers.
#[test]
fn the_island_census_has_no_doubly_occupied_seat() {
    use inf_player::runtime_sim::RuntimeInput;
    let Some(content) = island_project() else {
        eprintln!("SKIP: no island project — local-only content, CI has none");
        return;
    };
    let mut sim = island_sim(&content);
    const STEPS: u32 = 36_000;
    let (mut worst, mut samples, mut seats_seen, mut passengers) = (0u32, 0u32, 0usize, 0usize);
    for i in 0..STEPS {
        sim.step_once(RuntimeInput::default());
        if i % 30 == 0 {
            let census = d3::boarding::seat_census(sim.world());
            worst = worst.max(census.values().copied().max().unwrap_or(0));
            seats_seen = seats_seen.max(census.len());
            passengers = passengers.max(census.keys().filter(|(_, s)| *s == 1).count());
            samples += 1;
        }
    }
    println!(
        "=== the island census ===\n  {STEPS} steps ({:.1} min), {samples} samples: at most {seats_seen} seats held at once, {passengers} of them passengers; the busiest seat ever held {worst}",
        f64::from(STEPS) / 3600.0
    );
    assert!(seats_seen > 0, "nobody sat in anything on the island — the census is about nothing");
    assert!(worst <= 1, "a seat on the island held {worst} bodies");
}
