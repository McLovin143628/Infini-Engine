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
//! | `the_shipped_host_measures_the_posed_joints_against_the_sockets` (audit) | `RuntimeSim::boarding_residuals`: the POSED hand/foot joints vs the live outer/inner handle, rim grips, pedals | the reach solve's weight forced to 0 (723 / 494 / 796 mm); the foot pass skipped (1068 mm); the inner pull back on the door's centre (0 rows held) | rows at weight 1 per socket, the rim's 450 deg | **fails** — no hand request |
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
        Some(to_world.transform_point3(DVec3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z))))
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

/// Two JOINTS against two SOCKETS (VEH3d audit): the better of the two
/// pairings, its worse joint. The sockets are computed by the arm from the car
/// (never read back off the IK request, which is the target the code set).
fn pair(j: [Option<DVec3>; 2], at: [DVec3; 2]) -> f64 {
    let straight = dist(j[0], at[0]).max(dist(j[1], at[1]));
    let crossed = dist(j[0], at[1]).max(dist(j[1], at[0]));
    straight.min(crossed)
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
    let len = |p: BoardPhase| {
        d.iter()
            .find(|(q, _, _)| *q == p)
            .map(|(_, t, _)| *t)
            .unwrap()
    };
    // `Locked`: the check, one clock.
    assert!(
        (len(BoardPhase::Locked) - board::LOCKED_S).abs() <= 1.5 * DT,
        "`Locked` lasted {:.3} s against {}",
        len(BoardPhase::Locked),
        board::LOCKED_S
    );
    // `EnteringIK`: P29.7's own window, from the car.
    let (enter_s, _) = y.bridge.vehicle_of(CHASSIS).expect("the car").seat_warp();
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
        println!(
            "  t {:.3} s  weight {w:.3}  (owed {:.3})",
            t,
            (t / board::HAND_REACH_S).min(1.0)
        );
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
        println!(
            "  {:<10} t {:.2}  hand {:.2}  hinge {:.1} deg",
            ph.name(),
            t,
            w,
            a
        );
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
    let travelled: f64 = trace.windows(2).map(|w| (w[1].3 - w[0].3).abs()).sum();
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
        let top = up(Vec3d::new(
            car.offset.x,
            car.offset.y + car.half.y,
            car.offset.z,
        ));
        let floor = car
            .world(car.sockets.seat_floor(SeatIndex::Driver, car.floor_y()))
            .y;
        let cushion = car.world(car.sockets.seat(SeatIndex::Driver));
        let pelvis = y
            .joint(
                HERO,
                inf_anim::BoneRoleKind::Pelvis,
                inf_anim::BoneSide::Center,
            )
            .expect("a pelvis");
        let head = y
            .joint(
                HERO,
                inf_anim::BoneRoleKind::Head,
                inf_anim::BoneSide::Center,
            )
            .expect("a head");
        // The feet against the PEDALS the arm lays itself, at the inputs the
        // car was given (VEH3d audit: this read the foot request's own target).
        let feet = y.feet(HERO);
        let (tp, bp) = {
            let b = y.cm(HERO).runtime.boarding;
            board::pedal_faces(&car.sockets, b.throttle_in, b.brake_in)
        };
        let foot_err = pair(feet, [car.world(bp), car.world(tp)]);
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
        .and_then(|e| {
            y.world
                .world()
                .get::<inf_ecs::components::VehicleClass>(e)
                .copied()
        })
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
            // The grips the arm lays itself, off the rim angle IT read off the
            // wheels (VEH3d audit: this read the hand request's own target).
            let car = y.frame();
            let g = board::wheel_grips(&car.sockets, r);
            let held = asks
                .iter()
                .all(|a| a.is_some_and(|(_, w)| (w - 1.0).abs() < 1e-9));
            if held {
                worst = worst.max(pair(hands, [car.world(g[0]), car.world(g[1])]));
                placed += 2;
            }
            for i in 0..2 {
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
    // The pedal faces the ARM lays at the input it read (VEH3d audit: this
    // read the foot request's own target).
    let pedals = |y: &Yard| {
        let b = y.cm(HERO).runtime.boarding;
        let car = y.frame();
        let (tp, bp) = board::pedal_faces(&car.sockets, b.throttle_in, b.brake_in);
        (car.world(tp), car.world(bp))
    };
    let right_travel = (pressed[1].unwrap() - rest[1].unwrap()).length();
    let left_still = (pressed[0].unwrap() - rest[0].unwrap()).length();
    let right_on = dist(pressed_world[1], pedals(&y).0);
    // The brake: the stick pulled back while rolling forward is a brake, by
    // `VehicleControls::from_intent`'s own rule.
    let before_brake = y.feet_in_car(HERO);
    y.stick(HERO, 0.0, -1.0);
    y.step(6);
    let brake = y.cm(HERO).runtime.boarding.brake_in;
    let braked = y.feet_in_car(HERO);
    let braked_world = y.feet(HERO);
    let left_on = dist(braked_world[0], pedals(&y).1);
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
    assert!(
        throttle > 0.9,
        "the throttle the car was given is {throttle:.2}"
    );
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
    assert!(
        right_on <= 0.02 && left_on <= 0.02,
        "a foot is off its pedal"
    );
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
        .joint(
            RIDER,
            inf_anim::BoneRoleKind::Pelvis,
            inf_anim::BoneSide::Center,
        )
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
        if grips.iter().any(|g| (car.world(*g) - t).length() < 1e-6) {
            on_grips += 1;
        }
        let (f, _) = fasks[i].expect("both passenger feet are asked for");
        assert!(
            floor.iter().any(|g| (car.world(*g) - f).length() < 1e-6),
            "a passenger foot was sent somewhere that is not the floor"
        );
    }
    // The JOINTS against the sockets the arm laid (VEH3d audit: these read
    // the requests' own targets).
    hand_err = hand_err.max(pair(hands, [car.world(grips[0]), car.world(grips[1])]));
    foot_err = foot_err.max(pair(feet, [car.world(floor[0]), car.world(floor[1])]));
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
    y.world
        .world_mut()
        .get_mut::<Transform>(e)
        .unwrap()
        .translation = Vec3d::new(far.x, far.y + 0.9, far.z);
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
    println!(
        "  six presses from the far side: {refused} refused, the busiest seat held {worst_census}"
    );
    assert!(refused >= 6, "only {refused} of six presses were refused");
    assert_eq!(worst_census, 1, "a seat held {worst_census} bodies");
    assert_eq!(y.cm(HERO).runtime.boarding.phase, BoardPhase::Idle);
    // (ii) The door itself: an Enter into an occupied seat is refused by value.
    let mut cm = y.cm(HERO);
    let pos = y.at(HERO);
    let verdict = d3::boarding::begin(
        &mut y.world,
        &mut y.bridge,
        HERO,
        &mut cm,
        pos,
        RADIUS,
        CHASSIS,
        false,
    );
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
    assert_eq!(
        mode,
        MovementMode::FallControlled,
        "pulled out, not stepped out"
    );
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
        (p.x - slab_at.x).abs() < slab_half.x + RADIUS
            && (p.z - slab_at.z).abs() < slab_half.z + RADIUS
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
    assert!(
        !inside(at),
        "the driver was put down inside the wall at {at:?}"
    );
    assert!(
        clear,
        "the driver's capsule overlaps something where it landed"
    );

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
    let refusal =
        d3::boarding::clear_exit(&mut y.bridge, &car, preferred, stand_half, RADIUS, &own);
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
                wall(
                    w,
                    WALL,
                    DVec3::new(1.55, 1.0, -0.4),
                    DVec3::new(0.2, 1.0, 0.9),
                );
            })
        } else {
            Yard::new("sedan")
        };
        if walled {
            // Seat the hero directly: the wall is where the approach would
            // walk, and this half is about the way OUT.
            let e = y.world.entity_of(HERO).unwrap();
            y.world
                .world_mut()
                .get_mut::<Transform>(e)
                .unwrap()
                .translation = Vec3d::new(-2.4, 0.9, 0.2);
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
        print_trace(
            if walled {
                "the exit, walled"
            } else {
                "the exit"
            },
            &trace,
        );
        let order: Vec<BoardPhase> = durations(&trace).iter().map(|(p, _, _)| *p).collect();
        assert_eq!(
            order,
            vec![
                BoardPhase::Exiting,
                BoardPhase::ClosingDoor,
                BoardPhase::Idle
            ],
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
            assert!(
                peak >= board::DOOR_BOARD_DEG - 0.5,
                "the door opened only {peak:.1} deg"
            );
            assert!(
                shut <= board::DOOR_SHUT_DEG + 0.5,
                "the door was left at {shut:.2} deg"
            );
        }
    }
}

/// **`capsule_clear` is a lock of its own: the car's OWN parts** (VEH3d audit,
/// priority j'). The implementer's one mutation survivor was `capsule_clear`
/// answering `true`, "because the path sweep is a second lock on the same
/// candidates". It is not the same lock: the path sweep EXCLUDES the car's own
/// colliders (it starts inside the car), and `capsule_clear` does not — it is
/// the only thing standing between an exit point and the car's own open door.
///
/// The driver's door opened on its motor; the exit asked for with the door
/// LEAF itself as the preferred point. The leaf is a collider, the capsule at
/// candidate 0 is inside it, the sweep from inside the cabin excludes it — so
/// only `capsule_clear` can refuse candidate 0.
///
/// **The mutation** (run in the audit): `capsule_clear` answering `true` — the
/// exit takes candidate 0 and the body is put down inside its own car door.
#[test]
fn an_exit_is_never_put_down_inside_the_cars_own_open_door() {
    let mut y = Yard::new("sedan");
    let door = y.door();
    assert!(d3::bodywork::set_part_open(
        &mut y.world,
        &mut y.bridge,
        CHASSIS,
        door,
        true
    ));
    y.step(90);
    let car = y.frame();
    let leaf = y
        .bridge
        .part_body(door)
        .and_then(|p| y.bridge.world().body_translation(p.body))
        .expect("the open door is a live body");
    let deg = d3::boarding::door_angle_deg(&y.world, &y.bridge, CHASSIS, door);
    let preferred = {
        let l = car.local(leaf);
        Vec3d::new(l.x, 0.0, l.z)
    };
    let cm = y.cm(HERO);
    let own: std::collections::BTreeSet<_> = y.bridge.collider_of(HERO).into_iter().collect();
    // Candidate 0, standing on the road: inside the leaf.
    let at0 = {
        let mut p = car.world(preferred);
        p.y = 0.0 + d3::boarding::PLACE_SKIN_M;
        p
    };
    let inside_leaf =
        !d3::boarding::capsule_clear(&mut y.bridge, at0, cm.stand_half_height_m, RADIUS, &own);
    let people = d3::boarding::people(&y.world, &y.bridge);
    let got = d3::boarding::clear_exit(
        &mut y.bridge,
        &car,
        preferred,
        cm.stand_half_height_m,
        RADIUS,
        &people,
    );
    println!(
        "=== the car's own door ===\n  the door open {deg:.1} deg; candidate 0 (the leaf) inside it: {inside_leaf}; the exit took {:?}",
        got.map(|(_, i)| i)
    );
    assert!(deg > 30.0, "the door only opened {deg:.1} deg");
    assert!(
        inside_leaf,
        "a capsule at the door leaf is clear of it — this arm tests nothing"
    );
    let (feet, i) = got.expect("a sedan in an empty yard has somewhere to get out");
    assert_ne!(
        i, 0,
        "the exit put the body down inside its own car's open door"
    );
    assert!(
        d3::boarding::capsule_clear(&mut y.bridge, feet, cm.stand_half_height_m, RADIUS, &own),
        "the exit point it chose is inside something"
    );
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
    assert!(
        speed > board::EXIT_ROLL_MPS * 2.0,
        "the car only reached {speed:.2} m/s"
    );
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
    assert_eq!(
        asked, 0,
        "a hand reached for a handle on a door that is not there"
    );
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
    let roof = car.world(Vec3d::new(
        car.offset.x,
        car.offset.y + car.half.y,
        car.offset.z,
    ));
    let cm = y.cm(HERO);
    let standing = cm.stand_half_height_m + RADIUS;
    let want = roof + DVec3::Y * (2.0 * standing * cam.tuning.pivot_height_ratio);
    let pivot_off = (cam.pivot.to_dvec3() - want).length();
    println!(
        "=== the boarding camera ===\n  the claim held {held} of {ground_steps} ground steps; {after_driving_held} steps after the wheel; the drive pivot is {pivot_off:.4} m from the roof-derived one"
    );
    assert!(
        ground_steps > 60,
        "the choreography took {ground_steps} ground steps"
    );
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

/// **The boarding camera never pops** (VEH3d audit, priority f') — the
/// camera's own per-step translation over the WHOLE board (press to a second
/// at the wheel) and the WHOLE exit (press to a second standing), against the
/// CHAR1c director's blend law: no single step may carry CHAR1c's own measured
/// worst share of the move (15.9 %, [`CHAR1C_WORST_STEP_FRAC`]; its arm's
/// bound is 20 %). The multiple of the walking control's worst step is
/// REPORTED: the change-over into the drive block is a blend of several
/// metres in [`d3::camera::BOARDING_CAMERA_BLEND_S`], and it is the director's
/// blend that moves the camera there, not a target.
///
/// The implementer's claim target was framed off the BODY's feet and yaw, so
/// it rode the seat warp: 0.42 m in one step through `EnteringIK`, a camera
/// doing 25 m/s beside a body doing 2. The claim is anchored on the CAR now
/// (the take node in the chassis frame, the flank's own facing), so the only
/// thing that moves the camera is the director's blend.
///
/// **The mutation** (run in the audit): the claim framed off the body's feet
/// again — the per-step bound reds.
#[test]
fn the_boarding_camera_never_pops() {
    let mut y = Yard::new("sedan");
    let mut cam = inf_ecs::camera::LocomotionCamera::default();
    for _ in 0..20 {
        d3::step_camera_with_requests(&mut y.world, &mut y.bridge, &mut cam, HERO, DT);
    }
    // The walking control: the camera's worst step while the hero walks.
    let mut walking = 0.0f64;
    let mut prev = cam.pose.position.to_dvec3();
    y.stick(HERO, 0.0, 1.0);
    for _ in 0..60 {
        y.step(1);
        d3::step_camera_with_requests(&mut y.world, &mut y.bridge, &mut cam, HERO, DT);
        let p = cam.pose.position.to_dvec3();
        walking = walking.max((p - prev).length());
        prev = p;
    }
    y.stick(HERO, 0.0, 0.0);
    for _ in 0..60 {
        y.step(1);
        d3::step_camera_with_requests(&mut y.world, &mut y.bridge, &mut cam, HERO, DT);
    }
    // One leg: press E, then step until `done`, then a second more.
    let leg = |y: &mut Yard,
               cam: &mut inf_ecs::camera::LocomotionCamera,
               done: &dyn Fn(&Yard) -> bool| {
        y.press(HERO);
        let from = cam.pose.position.to_dvec3();
        let mut prev = from;
        let (mut worst, mut worst_phase, mut path) = (0.0f64, BoardPhase::Idle, 0.0f64);
        let mut per_phase: std::collections::BTreeMap<u8, f64> = std::collections::BTreeMap::new();
        let mut after = 0;
        for _ in 0..1200 {
            y.step(1);
            d3::step_camera_with_requests(&mut y.world, &mut y.bridge, cam, HERO, DT);
            let p = cam.pose.position.to_dvec3();
            let d = (p - prev).length();
            path += d;
            let ph = y.cm(HERO).runtime.boarding.phase.as_u8();
            let e = per_phase.entry(ph).or_insert(0.0);
            *e = e.max(d);
            if d > worst {
                worst = d;
                worst_phase = y.cm(HERO).runtime.boarding.phase;
            }
            prev = p;
            if done(y) {
                after += 1;
                if after > 60 {
                    break;
                }
            }
        }
        let phases: Vec<String> = per_phase
            .iter()
            .map(|(p, d)| {
                let name = [
                    BoardPhase::Idle,
                    BoardPhase::Locked,
                    BoardPhase::Unlocking,
                    BoardPhase::OpeningDoor,
                    BoardPhase::EnteringIK,
                    BoardPhase::Seated,
                    BoardPhase::Driving,
                    BoardPhase::Exiting,
                    BoardPhase::ClosingDoor,
                    BoardPhase::Jacked,
                ]
                .into_iter()
                .find(|x| x.as_u8() == *p)
                .map(|x| x.name())
                .unwrap_or("?");
                format!("{name} {d:.4}")
            })
            .collect();
        println!("    worst step per phase: {}", phases.join(", "));
        // **The claim's own target is still** while the body does its work at
        // the car: no step in the door, the seat warp, the seat or the shut may
        // move the camera further than a walk does.
        for ph in [
            BoardPhase::OpeningDoor,
            BoardPhase::EnteringIK,
            BoardPhase::Seated,
            BoardPhase::ClosingDoor,
        ] {
            let d = per_phase.get(&ph.as_u8()).copied().unwrap_or(0.0);
            assert!(
                d <= walking,
                "in `{}` the camera moved {d:.4} m in one step, more than a walk's {walking:.4} m — its target is riding the body",
                ph.name()
            );
        }
        (worst, worst_phase, (prev - from).length(), path)
    };
    let board = leg(&mut y, &mut cam, &|y: &Yard| {
        y.cm(HERO).runtime.boarding.phase == BoardPhase::Driving
    });
    let exit = leg(&mut y, &mut cam, &|y: &Yard| {
        let cm = y.cm(HERO);
        cm.runtime.boarding.phase == BoardPhase::Idle && !cm.runtime.seat.is_seated()
    });
    println!(
        "=== the boarding camera, per step ===\n  walking control: worst {walking:.4} m a step"
    );
    for (name, (worst, phase, total, path)) in [("board", board), ("exit", exit)] {
        println!(
            "  {name}: worst {worst:.4} m a step (in `{}`), {:.1} % of the {total:.3} m move ({path:.3} m of path), {:.2}x the walking step",
            phase.name(),
            worst / total.max(1e-9) * 100.0,
            worst / walking.max(1e-9)
        );
    }
    for (name, (worst, phase, total, _)) in [("board", board), ("exit", exit)] {
        assert!(
            total > 0.5,
            "{name}: the camera barely moved ({total:.3} m) — nothing was measured"
        );
        assert!(
            worst < 0.2 * total,
            "{name}: one step (in `{}`) took {:.1} % of the camera's move — a pop, not a blend",
            phase.name(),
            worst / total * 100.0
        );
        assert!(
            worst < CHAR1C_WORST_STEP_FRAC * total,
            "{name}: one step (in `{}`) took {:.1} % of the move — over CHAR1c's own measured worst",
            phase.name(),
            worst / total * 100.0
        );
    }
}

/// CHAR1c's own measured worst step of a vehicle entry, as a share of the move
/// (15.9 %): the brief's "a step over that is a pop".
const CHAR1C_WORST_STEP_FRAC: f64 = 0.159;

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
            61..=120 if jack && step.is_multiple_of(4) => (true, 0.0, false),
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
    let at = DVec3::new(
        0.0,
        inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15,
        0.0,
    );

    type Row = (Vec<u8>, [u64; 3], u8, [u64; 3], u8);
    let hero_row = |world: &EcsWorld| -> Row {
        let e = world.entity_of(HERO).expect("the hero");
        let t = world
            .world()
            .get::<Transform>(e)
            .expect("placed")
            .translation;
        let cm = world.world().get::<CharacterMovement>(e).expect("a mover");
        let (vt, vp) = world
            .entity_of(VICTIM)
            .map(|v| {
                let t = world
                    .world()
                    .get::<Transform>(v)
                    .expect("placed")
                    .translation;
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
            doc.world_mut()
                .world_mut()
                .entity_mut(h)
                .insert(hero_bits());
            if jack {
                let v = doc.create_with_guid(VICTIM, SpawnKind::Empty, "Driver", None);
                doc.world_mut()
                    .world_mut()
                    .entity_mut(v)
                    .insert(victim_bits());
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
                shipped
                    .iter()
                    .any(|r| r.0.len() >= 2 * board::BOARDING_TRACE_BYTES),
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

/// The saloon, the hero beside it with the mannequin rig and a one-state
/// machine, in the SHIPPED player's own sim (`RuntimeSim`) — the fixture the
/// audit's shipped-host arms share.
fn rigged_runtime_sim() -> inf_player::runtime_sim::RuntimeSim {
    use inf_player::runtime_sim::RuntimeSim;
    const IDLE: inf_anim::ClipRef = [0xd3; 16];
    let def = catalogue_def("sedan");
    let mut world = EcsWorld::new();
    ground(&mut world);
    car(
        &mut world,
        CHASSIS,
        DVec3::new(
            0.0,
            inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15,
            0.0,
        ),
        0.0,
        &def,
    );
    stand(&mut world, HERO, "Hero", HERO_AT, 0.0, true);
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
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
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

/// **THE SHIPPED HOST MEASURES THE POSED JOINTS AGAINST THE SOCKETS** (VEH3d
/// audit, priority a') — `RuntimeSim::boarding_residuals`, the number
/// `hero.csv`'s boarding columns and the HUD row carry, on the shipped
/// player's own sim with the mannequin rig, over a board / full lock both ways
/// / throttle / exit course driven through the INPUT door.
///
/// The implementer's columns carried the IK solver's `reach_error` — the chain
/// end against the target the boarding module handed it — so they read 0.0 mm
/// on every row by construction. This arm reads the evaluated pose's hand and
/// foot JOINTS (the pose the GPU skins) against sockets recomputed from the
/// live chassis, the door's own body and the rack.
///
/// * the OUTER handle, the INNER handle, the rim through a full lock, the
///   pedals under throttle: each at weight 1, each <= 2 cm, each with an
///   engagement count;
/// * **the mutation** (run in the audit): the reach solve's weight forced to 0
///   in `pose::apply_hand_ik` — the old columns still read 0.0; this arm reds
///   with the hand hundreds of millimetres off.
#[test]
fn the_shipped_host_measures_the_posed_joints_against_the_sockets() {
    use inf_ecs::movement::actions::{HANDBRAKE, INTERACT, MOVE_X, MOVE_Y};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    let mut sim = rigged_runtime_sim();
    let phase = |sim: &RuntimeSim| {
        let e = sim.world().entity_of(HERO).unwrap();
        let cm = sim.world().world().get::<CharacterMovement>(e).unwrap();
        (cm.runtime.boarding.phase, cm.runtime.seat.is_seated())
    };
    let mut outer: Vec<f64> = Vec::new();
    let mut inner: Vec<f64> = Vec::new();
    let mut inner_reached: Vec<f64> = Vec::new();
    let mut inner_held: Vec<(BoardPhase, f64)> = Vec::new();
    let mut rim: Vec<f64> = Vec::new();
    let mut pedal: Vec<f64> = Vec::new();
    let mut rim_seen = 0.0f64;
    let mut driving_at: Option<u32> = None;
    let mut exited = false;
    for i in 0..2400u32 {
        let (p, seated) = phase(&sim);
        let mut input = RuntimeInput::default();
        if i == 60 {
            input = input.press(INTERACT);
        }
        if let Some(d) = driving_at {
            let k = i - d;
            match k {
                30..=150 => input = input.axis_at(MOVE_X, 1.0),
                151..=270 => input = input.axis_at(MOVE_X, -1.0),
                300..=360 => input = input.axis_at(MOVE_Y, 1.0),
                361..=560 => input = input.press(HANDBRAKE),
                600 => input = input.press(INTERACT),
                _ => {}
            }
        } else if p == BoardPhase::Driving || (p == BoardPhase::Idle && seated) {
            driving_at = Some(i);
        }
        sim.step_once(input);
        let (p, seated) = phase(&sim);
        if driving_at.is_some() && p == BoardPhase::Idle && !seated {
            exited = true;
            break;
        }
        let Some(r) = sim.boarding_residuals(HERO) else {
            continue;
        };
        let holding = r.sockets.hand_weight >= 0.999;
        let on_handle = r.sockets.handle_weight >= 0.999;
        match r.sockets.phase {
            BoardPhase::OpeningDoor if on_handle => outer.extend(r.handle_m),
            BoardPhase::Seated | BoardPhase::Exiting => {
                if let Some(h) = r.handle_m {
                    if on_handle {
                        inner.push(h);
                        let e = sim.world().entity_of(HERO).expect("the hero");
                        let deg = sim
                            .world()
                            .world()
                            .get::<CharacterMovement>(e)
                            .expect("a mover")
                            .runtime
                            .boarding
                            .door_deg;
                        inner_held.push((r.sockets.phase, deg));
                    }
                    inner_reached.push(h);
                }
            }
            BoardPhase::Driving => {
                if holding {
                    rim.extend(r.grips_m);
                }
                pedal.extend(r.feet_m);
                rim_seen = rim_seen
                    .max(d3::boarding::vehicle_steer(sim.world(), sim.bridge3d(), CHASSIS).abs());
            }
            _ => {}
        }
    }
    let worst = |v: &[f64]| v.iter().copied().fold(0.0f64, f64::max) * 1000.0;
    let best = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min) * 1000.0;
    println!("=== the shipped host: POSED joints against the LIVE sockets ===");
    println!(
        "  outer handle at weight 1: {} rows, worst {:.2} mm",
        outer.len(),
        worst(&outer)
    );
    println!(
        "  inner handle at weight 1: {} rows, worst {:.2} mm (every inner row: {}, nearest {:.2} mm)",
        inner.len(),
        worst(&inner),
        inner_reached.len(),
        best(&inner_reached)
    );
    for p in [BoardPhase::Seated, BoardPhase::Exiting] {
        let d: Vec<f64> = inner_held
            .iter()
            .filter(|x| x.0 == p)
            .map(|x| x.1)
            .collect();
        println!(
            "    held the inner handle in `{}`: {} rows, the door {:.1} .. {:.1} deg",
            p.name(),
            d.len(),
            d.iter().copied().fold(f64::INFINITY, f64::min),
            d.iter().copied().fold(0.0f64, f64::max)
        );
    }
    println!(
        "  rim at weight 1: {} rows, worst {:.2} mm, the rim reached {rim_seen:.1} deg",
        rim.len(),
        worst(&rim)
    );
    println!(
        "  pedals while driving: {} rows, worst {:.2} mm",
        pedal.len(),
        worst(&pedal)
    );
    assert!(exited, "the course never got the hero back out of the car");
    assert!(
        outer.len() >= 5,
        "only {} rows held the outer handle",
        outer.len()
    );
    assert!(
        worst(&outer) <= 20.0,
        "the posed hand was {:.2} mm off the outer handle at weight 1",
        worst(&outer)
    );
    assert!(
        rim.len() >= 60 && rim_seen > 400.0,
        "the rim course was not driven: {} rows, {rim_seen:.1} deg",
        rim.len()
    );
    assert!(
        worst(&rim) <= 20.0,
        "the posed hands were {:.2} mm off the rim grips at weight 1",
        worst(&rim)
    );
    assert!(pedal.len() >= 60, "only {} pedal rows", pedal.len());
    assert!(
        worst(&pedal) <= 20.0,
        "the posed feet were {:.2} mm off the pedals",
        worst(&pedal)
    );
    assert!(
        inner.len() >= 3,
        "only {} rows held the INNER handle at weight 1",
        inner.len()
    );
    for p in [BoardPhase::Seated, BoardPhase::Exiting] {
        assert!(
            inner_held.iter().any(|x| x.0 == p),
            "no hand held the INNER handle at weight 1 in `{}` — the door moved with no hand on it",
            p.name()
        );
    }
    assert!(
        worst(&inner) <= 20.0,
        "the posed hand was {:.2} mm off the INNER handle at weight 1",
        worst(&inner)
    );
}

/// **THE PREVIEW DOORS HOLD A BEAT AND SEE THROUGH THE CAR, AND TOUCH NO
/// STEP** (VEH3d audit, priorities a' and b') — `pie_drive::BoardHold`
/// (`INF_PIE_BOARD_HOLD`) and `pie_drive::Cutaway` (`INF_PIE_CUTAWAY`), driven
/// the way `window.rs` drives them: one `tick` per display frame, and no fixed
/// step while a hold runs.
///
/// * a hold fires on `take`, `pull`, `lock`, `throttle` and `push`, once each,
///   and its close-up joint is the beat's own joint: the hand residual at
///   every hand beat <= 2 cm, read by `boarding_residuals` at the hold;
/// * the car the subject sits in is drawn translucent while seated and every
///   `Material` is put back exactly on the way out;
/// * a twin sim stepped through the same inputs with no doors folds the
///   byte-identical state at the end — the doors change which WALL frames run
///   steps, never a step;
/// * `window.rs` gates `run_frame` on the hold (a source pin: the window
///   cannot be driven headless).
#[test]
fn the_preview_doors_hold_a_beat_and_see_through_the_car_and_touch_no_step() {
    use inf_ecs::movement::actions::{HANDBRAKE, INTERACT, MOVE_X, MOVE_Y};
    use inf_player::pie_drive::{BoardHold, Cutaway};
    use inf_player::runtime_sim::RuntimeInput;
    const SRC: &str = include_str!("../src/window.rs");
    assert!(
        SRC.contains("!(self.pie.is_some() && self.board_hold.holding())"),
        "`window.rs` no longer holds its fixed steps on a boarding beat"
    );
    let input = |i: u32| -> RuntimeInput {
        let mut r = RuntimeInput::default();
        match i {
            60 => r = r.press(INTERACT),
            400..=520 => r = r.axis_at(MOVE_X, 1.0),
            521..=600 => r = r.axis_at(MOVE_Y, 1.0),
            601..=800 => r = r.press(HANDBRAKE),
            840 => r = r.press(INTERACT),
            _ => {}
        }
        r
    };
    const STEPS: u32 = 1100;
    let mut doors = rigged_runtime_sim();
    let mut twin = rigged_runtime_sim();
    let mut hold = BoardHold::new(0.25, 1.2);
    let mut cut = Cutaway::with_alpha(0.25);
    let original: std::collections::BTreeMap<Uuid, inf_ecs::components::Material> = {
        let w = twin.world();
        let root = w.entity_of(CHASSIS).expect("the car");
        w.subtree(root)
            .into_iter()
            .filter_map(|e| {
                Some((
                    w.world().get::<inf_ecs::components::Guid>(e)?.0,
                    *w.world().get::<inf_ecs::components::Material>(e)?,
                ))
            })
            .collect()
    };
    let mut notes: Vec<String> = Vec::new();
    let mut held_frames = 0u32;
    let mut translucent_max = 0usize;
    let mut i = 0u32;
    let mut frames = 0u32;
    while i < STEPS {
        frames += 1;
        assert!(frames < 10 * STEPS, "the holds never let go");
        if let Some(n) = hold.tick(&doors, 1.0 / 60.0) {
            notes.push(n);
        }
        if let Some(n) = cut.tick(&mut doors) {
            notes.push(n);
        }
        let w = doors.world();
        let root = w.entity_of(CHASSIS).expect("the car");
        let translucent = w
            .subtree(root)
            .into_iter()
            .filter_map(|e| w.world().get::<inf_ecs::components::Material>(e))
            .filter(|m| {
                m.blend == inf_ecs::components::BlendMode::Translucent && m.base_color.a == 0.25
            })
            .count();
        translucent_max = translucent_max.max(translucent);
        if hold.holding() {
            held_frames += 1;
            continue;
        }
        doors.step_once(input(i));
        twin.step_once(input(i));
        i += 1;
    }
    println!("=== the preview doors ===");
    for n in &notes {
        println!("  {n}");
    }
    println!(
        "  {held_frames} display frames held; at most {translucent_max} drawn parts translucent"
    );
    for beat in inf_player::pie_drive::BOARD_HOLD_BEATS {
        let n = notes
            .iter()
            .filter(|s| s.contains(&format!("held `{beat}`")))
            .count();
        assert_eq!(n, 1, "the `{beat}` beat was held {n} times");
    }
    for n in notes.iter().filter(|s| s.contains("held `")) {
        let mm: f64 = n
            .split("(residual ")
            .nth(1)
            .and_then(|r| r.split(' ').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(f64::INFINITY);
        assert!(
            mm <= 20.0,
            "a hold was taken with the joint {mm} mm off its socket: {n}"
        );
    }
    assert!(held_frames >= 5 * 15, "only {held_frames} frames were held");
    assert!(
        translucent_max >= 1,
        "no part of the seated car was ever drawn translucent"
    );
    let w = doors.world();
    for (g, m) in &original {
        let now = w
            .entity_of(*g)
            .and_then(|e| w.world().get::<inf_ecs::components::Material>(e))
            .copied();
        assert_eq!(
            now,
            Some(*m),
            "part {g}'s material was not put back after the cutaway"
        );
    }
    assert_eq!(doors.steps(), twin.steps());
    assert!(
        doors.state_bytes() == twin.state_bytes(),
        "the preview doors changed the simulation"
    );
}

/// **A `Near` car that is driving has somebody in it, and the body is SEEN in
/// the seat** (VEH3d audit, priority d' — the implementer's carried 4, "Near-tier
/// traffic carries nobody", was a dropped deliverable).
///
/// A town at half past eight on the SHIPPED player's own sim, with the
/// mannequin as the level's body: every `Near` car that is driving seats a
/// drawn rider (`crowd::spawn_rider`) in its driver's seat, and one in the
/// passenger's where `traffic::carries_passenger` draws it. Read on the POSED
/// joints: the rider's pelvis joint on the seat's cushion, its head joint under
/// the car's roof, in the car's own frame.
///
/// **The mutations** (run in the audit): the pelvis pin deleted from the pose
/// step — the bench posture leaves the pelvis a chair's height off the
/// cushion; the riders never seated (`seat_riders` answering 0) — no occupied
/// `Near` car.
#[test]
fn a_near_car_that_is_driving_draws_its_riders_in_their_seats() {
    use inf_ecs::components::{PcgVolume, ResidentSlot, SlotRole, StreamingSource, TimeOfDay};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    const PITCH: f64 = 100.0;
    const STREET: f64 = 20.0;
    const IDLE: inf_anim::ClipRef = [0xd3; 16];
    let mut world = EcsWorld::new();
    let half = (PITCH - STREET) * 0.5;
    for row in 0..3i32 {
        for col in 0..3i32 {
            let c = glam::DVec2::new(f64::from(col) * PITCH, f64::from(row) * PITCH);
            let guid = Uuid::from_u64_pair(0x51, (row as u64) << 32 | col as u64);
            let e = world.spawn_with_guid(guid, "block", None);
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
            world
                .world_mut()
                .entity_mut(e)
                .insert((Transform::from_translation(DVec3::new(c.x, 0.0, c.y)), v));
        }
    }
    let g = world.spawn_with_guid(GROUND, "Ground", None);
    world.world_mut().entity_mut(g).insert((
        Transform::from_translation(DVec3::new(100.0, -0.5, 100.0)),
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(400.0, 0.5, 400.0),
            ..Default::default()
        },
    ));
    // The band's anchor, and the level's one body (the archetype every
    // resident, driver and rider wears).
    let h = world.spawn_with_guid(HERO, "Anchor", None);
    world.world_mut().entity_mut(h).insert((
        Transform::from_translation(DVec3::new(50.0, 0.0, 50.0)),
        StreamingSource { radius_m: 512.0 },
    ));
    let b = world.spawn_with_guid(RIDER, "Body", None);
    world.world_mut().entity_mut(b).insert((
        Transform::from_translation(DVec3::new(-500.0, 0.0, -500.0)),
        inf_ecs::components::AnimStateMachine {
            sm: Some(SM_GUID),
            ..Default::default()
        },
        inf_ecs::components::SkeletalMesh {
            mesh: None,
            skeleton: Some(SKEL_GUID),
        },
    ));
    let sky = world.spawn_with_guid(WALL, "Sky", None);
    world.world_mut().entity_mut(sky).insert(TimeOfDay {
        seconds: 8.5 * 3600.0,
        rate: 0.0,
        ..Default::default()
    });
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
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
    let mut cars: std::collections::BTreeSet<Uuid> = std::collections::BTreeSet::new();
    let mut passengers = 0usize;
    let mut measured = 0usize;
    let mut worst_pelvis = 0.0f64;
    let mut worst_head_under_roof = f64::INFINITY;
    let mut most = 0usize;
    for _ in 0..600 {
        sim.step_once(RuntimeInput::default());
        most = most.max(sim.traffic_stats().near_riders);
        let Some(riders) = sim
            .world()
            .world()
            .get_resource::<inf_ecs::traffic::SeatedRidersRes>()
            .cloned()
        else {
            continue;
        };
        for (rider, (chassis, seat)) in riders.riders {
            let w = sim.world();
            let (Some(ce), Some(_re)) = (w.entity_of(chassis), w.entity_of(rider)) else {
                continue;
            };
            let (Some(t), Some(col)) = (
                w.world().get::<Transform>(ce).copied(),
                w.world().get::<Collider3D>(ce).copied(),
            ) else {
                continue;
            };
            let half = inf_ecs::vehicle::chassis_half_extents(&col);
            let rot = glam::DQuat::from_rotation_y(t.rotation.y.to_radians());
            let sockets = board::sockets_of(half, col.offset, &[]);
            let cushion =
                t.translation.to_dvec3() + rot * sockets.seat(SeatIndex::from_u8(seat)).to_dvec3();
            let roof = t.translation.y + col.offset.y + half.y;
            let (Some(pelvis), Some(head)) = (
                sim.posed_joint(
                    rider,
                    inf_anim::BoneRoleKind::Pelvis,
                    inf_anim::BoneSide::Center,
                ),
                sim.posed_joint(
                    rider,
                    inf_anim::BoneRoleKind::Head,
                    inf_anim::BoneSide::Center,
                ),
            ) else {
                continue;
            };
            measured += 1;
            cars.insert(chassis);
            passengers += usize::from(seat == SeatIndex::Passenger.as_u8());
            worst_pelvis = worst_pelvis.max((pelvis - cushion).length());
            worst_head_under_roof = worst_head_under_roof.min(roof - head.y);
        }
    }
    println!(
        "=== Near riders ===\n  {} occupied Near cars drew a posed body ({measured} rider-steps, {passengers} of them passengers); at most {most} riders at once\n  the pelvis joint at worst {:.2} mm off its cushion; the head at least {:.3} m under the roof",
        cars.len(),
        worst_pelvis * 1000.0,
        worst_head_under_roof
    );
    assert!(
        cars.len() >= 3,
        "only {} Near cars drew a body in a seat",
        cars.len()
    );
    assert!(passengers > 0, "no Near car drew its passenger");
    assert!(
        worst_pelvis <= 0.03,
        "a rider's pelvis joint was {:.2} mm off its cushion",
        worst_pelvis * 1000.0
    );
    assert!(
        worst_head_under_roof > 0.0,
        "a rider's head is {:.3} m through its car's roof",
        -worst_head_under_roof
    );
}

/// **What the `Near` riders cost** (VEH3d audit, priority d': "budget it") —
/// `step_pose_evaluation` over 32 drawn riders (the mannequin, the `Near`
/// LOD: the machine, the posture, the pin, no hand or foot pass) against the
/// same world with none. Min of five warmed rounds; asserted in a RELEASE
/// build off CI only (`NEAR_RIDERS_BUDGET_MS`), reported everywhere.
#[test]
fn thirty_two_near_riders_cost_what_they_cost() {
    const IDLE: inf_anim::ClipRef = [0xd3; 16];
    const RIDERS: u32 = 32;
    const NEAR_RIDERS_BUDGET_MS: f64 = 0.5;
    let skeleton = inf_anim::build_template(
        inf_anim::BodyPlan::Biped,
        &inf_anim::BodyParams {
            height_m: 1.8,
            ..Default::default()
        },
    )
    .expect("the mannequin builds");
    let machine = inf_anim::StateMachine {
        states: vec![inf_anim::SmState::clip("idle", IDLE)],
        entry: 0,
        ..Default::default()
    };
    let clips: std::collections::BTreeMap<inf_anim::ClipRef, inf_anim::AnimClip> =
        [(IDLE, inf_anim::AnimClip::new("idle", Vec::new()))]
            .into_iter()
            .collect();
    let archetype = inf_ecs::crowd::CrowdArchetype::humanoid(None, Some(SKEL_GUID), Some(SM_GUID));
    let build = |n: u32| -> EcsWorld {
        let mut world = EcsWorld::new();
        let mut res = inf_ecs::traffic::SeatedRidersRes::default();
        for i in 0..n {
            let g = Uuid::from_u128(0x7D_0000 + u128::from(i));
            inf_ecs::crowd::spawn_rider(
                &mut world,
                g,
                &archetype,
                DVec3::new(f64::from(i) * 4.0, 0.5, 0.0),
                0.0,
            );
            res.riders.insert(g, (Uuid::from_u128(1), 0));
        }
        if n > 0 {
            world.world_mut().insert_resource(res);
        }
        world.propagate();
        world
    };
    let time = |world: &mut EcsWorld| -> f64 {
        let machines = |g: Uuid| (g == SM_GUID).then_some(&machine);
        let skels = |g: Uuid| (g == SKEL_GUID).then_some(&skeleton);
        let clip = |c: inf_anim::ClipRef| clips.get(&c);
        let vars = |_: Uuid| std::collections::BTreeMap::new();
        for _ in 0..10 {
            inf_ecs::pose::step_pose_evaluation(world, DT, &machines, &skels, &clip, &vars);
        }
        (0..5)
            .map(|_| {
                let t0 = std::time::Instant::now();
                inf_ecs::pose::step_pose_evaluation(world, DT, &machines, &skels, &clip, &vars);
                t0.elapsed().as_secs_f64() * 1000.0
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut empty = build(0);
    let mut full = build(RIDERS);
    let control = time(&mut empty);
    let riders = time(&mut full);
    let posed = full
        .world()
        .get_resource::<inf_ecs::pose::PoseStoreRes>()
        .map(|r| r.0.len())
        .unwrap_or(0);
    println!(
        "=== {RIDERS} Near riders ===\n  the pose step {riders:.4} ms against {control:.4} ms with none: {:.4} ms, {:.2} us a rider ({posed} posed)",
        riders - control,
        (riders - control) * 1000.0 / f64::from(RIDERS)
    );
    assert_eq!(posed, RIDERS as usize, "not every rider was posed");
    if clock_may_assert() {
        assert!(
            riders - control <= NEAR_RIDERS_BUDGET_MS,
            "{RIDERS} Near riders cost {:.4} ms of pose step, over the {NEAR_RIDERS_BUDGET_MS} ms budget",
            riders - control
        );
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
        DVec3::new(
            0.0,
            inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15,
            0.0,
        ),
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
        sim.world()
            .world()
            .get::<CharacterMovement>(e)
            .unwrap()
            .runtime
            .boarding
            .phase
    };
    assert_eq!(phase(&sim), BoardPhase::Idle);
    // The press lands on a frame too short to run a step, and is let go on the
    // next frame, which runs one.
    let before = sim.steps();
    let ran = sim.run_frame(
        0.001,
        RuntimeInput::default().press(inf_ecs::movement::actions::INTERACT),
    );
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

/// **EVERY host that owns a frame loop sees a press exactly once, whatever its
/// frame rate** (VEH3d audit, priority e'). Wave VEH3d fixed
/// `RuntimeSim::run_frame` (the shipped window, both PIE windows, web,
/// android — they all reach it through `PlayerApp::frame`) and armed only the
/// zero-step shape on that one host. The editor's Simulate runs its OWN
/// accumulator (`SimSession::tick`) and still dropped a press on a zero-step
/// frame and gave a RELEASE to both steps of a two-step frame; and the weapon
/// wheel, an edge wearing an axis, was lost on a zero-step frame and switched
/// twice on a two-step one, on both hosts.
///
/// Both hosts, three shapes each, through the hosts' own frame doors:
/// * **a press on a zero-step frame** boards the car;
/// * **a release on a two-step frame** is ONE crouch click (two were a crouch
///   and an un-crouch);
/// * **a wheel notch** on a zero-step frame equips the next weapon, and one on
///   a two-step frame equips the next weapon and not the one after it.
///
/// **The mutations** (run in the audit): the zero-step carry deleted from
/// either host; the `i > 0` clear deleted from either host; the wheel carry
/// deleted from either host.
#[test]
fn every_host_sees_a_press_once_whatever_its_frame_rate() {
    use inf_ecs::item::{Inventory, ItemDef, ItemDefs};
    use inf_ecs::movement::actions::{CROUCH, INTERACT, WEAPON_SWITCH};
    use inf_editor_core::scene::SceneDoc;
    use inf_editor_core::simulate::{SimInput, SimSession};
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    use std::collections::BTreeMap;

    fn defs() -> ItemDefs {
        let mut d = ItemDefs::default();
        for id in ["pistol_a", "pistol_b"] {
            d.insert(ItemDef {
                id: id.into(),
                label: id.into(),
                stack_max: 1,
                mass_kg: 1.0,
                mesh: None,
                weapon: Some(inf_ecs::weapon::WeaponDef {
                    magazine: 5,
                    reserve: 10,
                    ..Default::default()
                }),
            });
        }
        d
    }
    /// The saloon, the hero beside it with two pistols in the bag.
    fn build(world: &mut EcsWorld) {
        let def = catalogue_def("sedan");
        ground(world);
        car(
            world,
            CHASSIS,
            DVec3::new(
                0.0,
                inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15,
                0.0,
            ),
            0.0,
            &def,
        );
        stand(world, HERO, "Hero", HERO_AT, 0.0, true);
        let d = defs();
        let mut inv = Inventory::default();
        inv.add(&d, "pistol_a", 1);
        inv.add(&d, "pistol_b", 1);
        let e = world.entity_of(HERO).expect("the hero");
        world.world_mut().entity_mut(e).insert(inv);
        world.world_mut().insert_resource(d);
        world.propagate();
    }
    /// One frame: `(frame seconds, keys held, the wheel)`.
    type Frame = (f64, Vec<&'static str>, f32);
    const F: f64 = 1.0 / 60.0;
    /// What the course ends in: the boarding phase, the mode, the equipped slot.
    type End = (BoardPhase, MovementMode, Option<usize>);
    let read = |w: &EcsWorld| -> End {
        let e = w.entity_of(HERO).expect("the hero");
        let cm = w.world().get::<CharacterMovement>(e).expect("a mover");
        let inv = w.world().get::<Inventory>(e).expect("a bag");
        (cm.runtime.boarding.phase, cm.mode, inv.equipped)
    };
    let shipped = |frames: &[Frame]| -> End {
        let mut world = EcsWorld::new();
        build(&mut world);
        let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
        for _ in 0..60 {
            sim.run_frame(F, RuntimeInput::default());
        }
        for (dt, keys, wheel) in frames {
            let mut i = RuntimeInput::with_down(keys.iter().copied());
            if *wheel != 0.0 {
                i = i.axis_at(WEAPON_SWITCH, *wheel);
            }
            sim.run_frame(*dt, i);
        }
        read(sim.world())
    };
    let preview = |frames: &[Frame]| -> End {
        let mut doc = SceneDoc::new();
        build(doc.world_mut());
        let mut session =
            SimSession::enter(&mut doc, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
        for _ in 0..60 {
            session.tick(&mut doc, F, SimInput::default());
        }
        for (dt, keys, wheel) in frames {
            let mut axes: BTreeMap<String, f32> = BTreeMap::new();
            if *wheel != 0.0 {
                axes.insert(WEAPON_SWITCH.to_string(), *wheel);
            }
            session.tick(
                &mut doc,
                *dt,
                SimInput::with_down(keys.iter().copied()).with_axes(axes),
            );
        }
        let end = read(doc.world());
        session.exit(&mut doc);
        end
    };
    let idle = |n: usize| -> Vec<Frame> { (0..n).map(|_| (F, Vec::new(), 0.0)).collect() };
    // (1) E on a frame too short to run a step, let go on the next.
    let mut press: Vec<Frame> = vec![(0.001, vec![INTERACT], 0.0)];
    press.extend(idle(3));
    // (2) C held for one step, let go on a frame that runs two.
    let mut release: Vec<Frame> = vec![(F, vec![CROUCH], 0.0), (2.5 * F, Vec::new(), 0.0)];
    release.extend(idle(40));
    // (3a) a notch on a zero-step frame; (3b) a notch on a two-step frame.
    let mut notch0: Vec<Frame> = vec![(0.001, Vec::new(), 1.0)];
    notch0.extend(idle(3));
    let mut notch2: Vec<Frame> = vec![(2.5 * F, Vec::new(), 1.0)];
    notch2.extend(idle(3));
    for (host, run) in [
        (
            "shipped (RuntimeSim::run_frame)",
            &shipped as &dyn Fn(&[Frame]) -> End,
        ),
        ("preview (SimSession::tick)", &preview),
    ] {
        let (p1, _, _) = run(&press);
        let (_, m2, _) = run(&release);
        let (_, _, e3a) = run(&notch0);
        let (_, _, e3b) = run(&notch2);
        println!(
            "=== {host} ===\n  a zero-step press of E: the hero is `{}`\n  a release on a two-step frame: the hero is {m2:?}\n  a wheel notch on a zero-step frame: slot {e3a:?}; on a two-step frame: slot {e3b:?}",
            p1.name()
        );
        assert_ne!(
            p1,
            BoardPhase::Idle,
            "{host}: a press on a zero-step frame was dropped"
        );
        assert_eq!(
            m2,
            MovementMode::Crouch,
            "{host}: a tap of crouch released on a two-step frame did not end crouched"
        );
        assert_eq!(
            e3a,
            Some(1),
            "{host}: a wheel notch on a zero-step frame was dropped"
        );
        assert_eq!(
            e3b,
            Some(1),
            "{host}: a wheel notch on a two-step frame switched more than once"
        );
    }
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
        stand(
            &mut y.world,
            *d,
            "Driver",
            DVec3::new(0.0, 0.0, 60.0 + i as f64),
            0.0,
            false,
        );
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
    assert_eq!(
        report.hands as usize, CARS,
        "not every driver's hands were placed"
    );
    assert_eq!(
        report.feet as usize, CARS,
        "not every driver's feet were placed"
    );
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
    assert_eq!(
        keys.len(),
        3,
        "the file names more than what changed: {keys:?}"
    );
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

/// **A changed camera table reaches the RUNNING rig on both hosts** (VEH3d
/// audit, priority g'). The implementer shipped the write half
/// (`write_camera_beside`, Live Tuning's "Save camera to level") and it wrote a
/// file only the `--level` dev boot and the editor's Simulate ever read: a PIE
/// session built its world from a payload that carries no camera, and a
/// `--pack` boot never looked — the VEH3b tuning-door lesson, a door that wrote
/// a thing nothing read. And a rig created mid-session (the weapon's ADS blend
/// writes one key through `set_camera_rig_value`) started from the DEFAULTS,
/// so the level's table was thrown away the first time the hero aimed.
///
/// Which file each host reads now:
/// * **PIE**: the editor names its open level in `INF_PIE_LEVEL_PATH` on the
///   spawned player (`PieSession::spawn_scene_for_level`), and
///   `sim_from_payload` reads the `camera.toml` beside it;
/// * **shipped** (`--pack`): the `camera.toml` beside the pack, which the cook
///   copies there (`cook_blocking::a_cook_carries_the_level_camera_table_beside_the_pack`);
/// * **`--level`** and **Simulate**: the `camera.toml` beside the level, as before.
///
/// The arm writes a table through the write half with the walk boom at 6.5 m
/// (the default is 3), reads it back through BOTH hosts' boot doors, installs
/// each on the shipped sim, and reads the camera's own boom after it settles
/// — then aims (the ADS door) and reads it again.
///
/// **The mutations** (run in the audit): `set_camera_rig_value` back on
/// `CameraRig::default()` — the boom snaps to 3 m on the aim; the PIE reader
/// answering `None` — the PIE boom is 3 m.
#[test]
fn a_changed_camera_table_reaches_the_running_rig_on_both_hosts() {
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
    const BOOM: f64 = 6.5;
    const SRC: &str = include_str!("../../../editor/studio/src-tauri/src/commands/pie.rs");
    assert_eq!(
        inf_player::PIE_LEVEL_ENV,
        inf_editor_core::pie::PIE_LEVEL_ENV
    );
    assert!(
        SRC.contains("PieSession::spawn_scene_for_level(&bin, &payload, level.as_deref())"),
        "the editor's Play no longer names its level to the player"
    );
    let dir = std::env::temp_dir().join(format!("veh3d-camrig-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a temp dir");
    let level = dir.join("Level.inf_lvl");
    let mut t = inf_ecs::camera::CameraTuning::default();
    assert!(t.set("walk.arm_length_m", BOOM));
    inf_ecs::camera::write_camera_beside(&level, &t).expect("the write half");
    let pie = inf_player::pie_camera_table_from(Some(&level)).expect("the PIE reader");
    let shipped = inf_player::camera_table_for(&inf_player::args::WorldChoice::Pack(dir.clone()))
        .expect("the pack reader");
    let boom_of = |table: Option<inf_ecs::camera::CameraTuning>| -> (f64, f64) {
        let def = catalogue_def("sedan");
        let mut world = EcsWorld::new();
        ground(&mut world);
        car(
            &mut world,
            CHASSIS,
            DVec3::new(
                0.0,
                inf_ecs::vehicle::resting_origin_y(&def, 0.0) + 0.15,
                0.0,
            ),
            0.0,
            &def,
        );
        stand(&mut world, HERO, "Hero", HERO_AT, 0.0, true);
        world.propagate();
        let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), 60.0);
        if let Some(t) = table {
            sim.camera_mut().tuning = t;
        }
        for _ in 0..240 {
            sim.step_once(RuntimeInput::default());
        }
        let before = sim.camera().arm_m;
        // The ADS door: one key of a rig the hero did not have.
        assert!(inf_ecs::camera::set_camera_rig_value(
            sim.world_mut(),
            HERO,
            "aim_blend_speed",
            12.0
        ));
        for _ in 0..240 {
            sim.step_once(RuntimeInput::default());
        }
        (before, sim.camera().arm_m)
    };
    let control = boom_of(None);
    let on_pie = boom_of(Some(pie));
    let on_pack = boom_of(Some(shipped));
    std::fs::remove_dir_all(&dir).ok();
    println!(
        "=== the level's camera table, in the running rig ===\n  the boom (settled, then after an ADS rig was created): defaults {:.3} / {:.3} m; PIE {:.3} / {:.3} m; shipped pack {:.3} / {:.3} m (the table says {BOOM} m)",
        control.0, control.1, on_pie.0, on_pie.1, on_pack.0, on_pack.1
    );
    assert!(
        (control.0 - BOOM).abs() > 1.0,
        "the default boom is already {BOOM} m — the arm cannot tell the table from the defaults"
    );
    for (host, (a, b)) in [("PIE", on_pie), ("shipped", on_pack)] {
        assert!(
            (a - BOOM).abs() < 0.05,
            "{host}: the running boom is {a:.3} m, not the table's {BOOM}"
        );
        assert!(
            (b - BOOM).abs() < 0.05,
            "{host}: after the ADS door made a rig the boom is {b:.3} m — the rig threw the level's table away"
        );
    }
}

/// **The shipped host draws the boarding row**, and the row reads what the
/// machine is doing.
#[test]
fn the_shipped_host_draws_the_boarding_row() {
    const SRC: &str = include_str!("../src/window.rs");
    assert!(
        SRC.contains("inf_ecs::boarding::boarding_readout"),
        "`window.rs` no longer calls `boarding_readout` — the boarding is simulated and nothing draws it"
    );
    assert!(
        SRC.contains("sim.boarding_residuals(guid)"),
        "`window.rs`'s boarding row no longer measures the POSED joints against the sockets"
    );
    let mut b = inf_ecs::boarding::BoardingState::default();
    assert_eq!(inf_ecs::boarding::boarding_readout(&b, None, None), None);
    b.enter(BoardPhase::OpeningDoor, 0.0);
    b.hand_weight = 1.0;
    b.door = Uuid::from_u128(9);
    b.door_deg = 32.4;
    let row = inf_ecs::boarding::boarding_readout(&b, Some(0.0123), None).expect("a row");
    println!("the boarding row reads: {row}");
    assert_eq!(
        row,
        "BOARDING opening  SEAT driver  HAND 0.012 m  DOOR 32 deg"
    );
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
    let at = BOARDING
        .find("pub struct BoardingState {")
        .expect("the machine");
    let head = &BOARDING[at.saturating_sub(300)..at];
    assert!(
        !head.contains("Serialize"),
        "`BoardingState` grew a `Serialize`"
    );
    let at = BOARDING
        .find("pub struct VehicleSockets {")
        .expect("the sockets");
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
        .and_then(|e| {
            sim.world()
                .world()
                .get::<inf_ecs::components::SkeletalMesh>(e)
        })
        .and_then(|m| m.skeleton)?;
    let rig = sim.skeleton_of(skel)?;
    let j = rig.role_index().first(role, side)?;
    let posed = inf_ecs::pose::evaluated_pose(sim.world(), who)?;
    let to_world = inf_ecs::pose::model_to_world_of(sim.world(), who)?;
    let g = inf_anim::pose::global_transforms(&rig.skeleton, &posed.pose);
    let p = g.get(j as usize)?.to_scale_rotation_translation().2;
    Some(to_world.transform_point3(DVec3::new(f64::from(p.x), f64::from(p.y), f64::from(p.z))))
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
                .and_then(|e| {
                    sim.world()
                        .world()
                        .get::<inf_ecs::components::VehicleClass>(e)
                        .copied()
                })
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
        let rays = [car.offset.y + 0.2, car.offset.y - car.half.y + 0.12]
            .iter()
            .all(|y| {
                let from = car.world(Vec3d::new(
                    car.offset.x,
                    *y,
                    car.offset.z + car.half.z + 0.3,
                ));
                sim.bridge3d_mut()
                    .world_mut()
                    .cast_ray(from, ahead, 20.0)
                    .is_none()
            });
        let from = car.world(Vec3d::new(
            car.offset.x,
            car.offset.y + 0.1,
            car.offset.z + car.half.z + 0.5,
        ));
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
    let car =
        d3::boarding::car_frame(sim.world(), sim.bridge3d(), chassis, true).expect("its frame");
    let lift = {
        let e = sim.world().entity_of(hero).unwrap();
        let cm = sim.world().world().get::<CharacterMovement>(e).unwrap();
        let r = sim
            .world()
            .world()
            .get::<Collider3D>(e)
            .map(|c| c.radius)
            .unwrap_or(0.3);
        cm.stand_half_height_m + r
    };
    let beside = car.world(Vec3d::new(car.half.x + 1.6, 0.0, -1.2));
    let ground = sim.terrain_height_at(beside.x, beside.z);
    set_hero(
        &mut sim,
        hero,
        DVec3::new(beside.x, ground + lift + 0.05, beside.z),
    );
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }
    let mode_at_press = {
        let e = sim.world().entity_of(hero).unwrap();
        sim.world()
            .world()
            .get::<CharacterMovement>(e)
            .unwrap()
            .mode
    };
    sim.step_once(RuntimeInput::default().press(inf_ecs::movement::actions::INTERACT));
    let mut takes: Vec<f64> = Vec::new();
    let mut inner: Vec<f64> = Vec::new();
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
            if let Some(h) =
                d3::boarding::handle_world(sim.world(), sim.bridge3d(), &car, b.door, false)
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
        if b.phase == BoardPhase::Seated && b.handle_weight >= 0.999 {
            if let Some(h) = sim.boarding_residuals(hero).and_then(|r| r.handle_m) {
                inner.push(h);
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
    assert_eq!(
        phases.last(),
        Some(&BoardPhase::Driving),
        "the hero never reached the wheel"
    );
    assert!(
        !takes.is_empty(),
        "the hand was never measured on the handle"
    );
    let worst_take = takes.iter().copied().fold(0.0f64, f64::max);
    // Seated: the pelvis inside, the hands on the rim, the feet on the pedals.
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }
    let car = d3::boarding::car_frame(sim.world(), sim.bridge3d(), chassis, true).unwrap();
    let roof = car
        .world(Vec3d::new(
            car.offset.x,
            car.offset.y + car.half.y,
            car.offset.z,
        ))
        .y;
    let pelvis = sim_joint(
        &sim,
        hero,
        inf_anim::BoneRoleKind::Pelvis,
        inf_anim::BoneSide::Center,
    )
    .expect("a pelvis");
    // The POSED hands and feet against the live sockets (VEH3d audit): this
    // arm used to measure the hands against the hand REQUEST's own target and
    // did not measure the feet at all (carried 7).
    let seated = sim
        .boarding_residuals(hero)
        .expect("the seated hero has sockets");
    let rim_err = seated.grips_m.unwrap_or(f64::INFINITY);
    let feet_err = seated.feet_m.unwrap_or(f64::INFINITY);
    println!(
        "  seated: the pelvis {:+.3} m from the roof; the hands {:.2} mm off the rim; the feet {:.2} mm off the pedals",
        pelvis.y - roof,
        rim_err * 1000.0,
        feet_err * 1000.0
    );
    println!(
        "  the inner handle at weight 1 while boarding: {} rows, worst {:.2} mm",
        inner.len(),
        inner.iter().copied().fold(0.0f64, f64::max) * 1000.0
    );
    // Drive, then bail.
    for _ in 0..240 {
        sim.step_once(RuntimeInput::default().axis_at(inf_ecs::movement::actions::MOVE_Y, 1.0));
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
        let mode = sim
            .world()
            .world()
            .get::<CharacterMovement>(e)
            .unwrap()
            .mode;
        if modes.last() != Some(&mode) {
            modes.push(mode);
        }
        if let Some(s) = inf_ecs::anim_bridge::anim_state(sim.world(), hero) {
            if states.last() != Some(&s.name) {
                states.push(s.name.clone());
            }
        }
    }
    println!("  bailed out at {speed:.2} m/s: modes {modes:?}; machine states {states:?}");
    assert!(
        worst_take <= 0.02,
        "the island hero's hand ended {:.2} mm from the handle at weight 1",
        worst_take * 1000.0
    );
    assert!(
        pelvis.y < roof - 0.3,
        "the island hero is not inside the car"
    );
    assert!(
        rim_err <= 0.02,
        "the island hero's hands are {:.2} mm off the rim",
        rim_err * 1000.0
    );
    assert!(
        feet_err <= 0.02,
        "the island hero's feet are {:.2} mm off the pedals",
        feet_err * 1000.0
    );
    assert!(
        !inner.is_empty(),
        "no hand held the island car's INNER handle at weight 1"
    );
    assert!(
        inner.iter().all(|h| *h <= 0.02),
        "the island hero's hand was {:.2} mm off the INNER handle at weight 1",
        inner.iter().copied().fold(0.0f64, f64::max) * 1000.0
    );
    assert!(
        speed > 2.0 * board::EXIT_ROLL_MPS,
        "the car only reached {speed:.2} m/s"
    );
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
    // **THE ENGAGEMENT HALF** (VEH3d audit, priority i'): the census with no
    // boarding in it "would pass with no boarding at all" — the implementer's
    // own words. So the hero boards during the window: every 6 000 steps it is
    // put beside the nearest STANDING car — an empty one on even rounds, an
    // OCCUPIED one on odd rounds (the carjack) — and presses E, drives nowhere,
    // and presses E again to get out. The arm counts boardings that reached the
    // wheel and seats held, and a census that saw neither is not an arm.
    const ROUND: u32 = 6_000;
    let hero = {
        for _ in 0..240 {
            sim.step_once(RuntimeInput::default());
        }
        inf_ecs::movement::camera_subject(sim.world()).expect("the island's hero")
    };
    let (mut worst, mut samples, mut seats_seen, mut passengers) = (0u32, 0u32, 0usize, 0usize);
    let (mut boarded, mut jacked, mut attempts) = (0u32, 0u32, 0u32);
    let mut at_wheel_since: Option<u32> = None;
    let mut last_phase = BoardPhase::Idle;
    for i in 0..STEPS {
        let mut input = RuntimeInput::default();
        if i % ROUND == 60 {
            // Beside the nearest standing car of the round's kind.
            let want_occupied = (i / ROUND) % 2 == 1;
            let here = d3::boarding::capsule_centre(sim.world(), hero).expect("placed");
            let occupied = d3::carjack::occupied_chassis(sim.world());
            let mut cars: Vec<(f64, Uuid)> = sim
                .bridge3d()
                .vehicle_guids()
                .into_iter()
                .filter(|g| occupied.contains(g) == want_occupied)
                .filter_map(|g| {
                    let (seat, _, v) = d3::vehicle::seat_pose(sim.bridge3d(), g)?;
                    (v.length() < 0.2).then(|| ((seat - here).length(), g))
                })
                .collect();
            cars.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            if let Some(car) = cars
                .first()
                .and_then(|(_, g)| d3::boarding::car_frame(sim.world(), sim.bridge3d(), *g, false))
            {
                let beside = car.world(Vec3d::new(car.half.x + 1.6, 0.0, -1.2));
                let ground = sim.terrain_height_at(beside.x, beside.z);
                let lift = {
                    let e = sim.world().entity_of(hero).unwrap();
                    let cm = sim.world().world().get::<CharacterMovement>(e).unwrap();
                    cm.stand_half_height_m + RADIUS
                };
                set_hero(
                    &mut sim,
                    hero,
                    DVec3::new(beside.x, ground + lift + 0.05, beside.z),
                );
                attempts += 1;
            }
        }
        // Press E through the round's first seconds (a carjack's victim resists
        // a few presses), and again to get out after two seconds at the wheel.
        let phase = {
            let e = sim.world().entity_of(hero).unwrap();
            let cm = sim.world().world().get::<CharacterMovement>(e).unwrap();
            if cm.runtime.seat.is_seated()
                && !cm.runtime.seat.entering
                && cm.runtime.boarding.phase == BoardPhase::Driving
            {
                BoardPhase::Driving
            } else {
                cm.runtime.boarding.phase
            }
        };
        let r = i % ROUND;
        if (90..=600).contains(&r)
            && r % 8 == 0
            && phase == BoardPhase::Idle
            && at_wheel_since.is_none()
        {
            input = input.press(inf_ecs::movement::actions::INTERACT);
        }
        if phase == BoardPhase::Driving && last_phase != BoardPhase::Driving {
            boarded += 1;
            at_wheel_since = Some(i);
            let e = sim.world().entity_of(hero).unwrap();
            if sim
                .world()
                .world()
                .get::<CharacterMovement>(e)
                .unwrap()
                .runtime
                .boarding
                .carjack
            {
                jacked += 1;
            }
        }
        if let Some(t) = at_wheel_since {
            if i == t + 120 {
                input = input.press(inf_ecs::movement::actions::INTERACT);
            }
            if i > t + 120 && phase == BoardPhase::Idle {
                at_wheel_since = None;
            }
        }
        last_phase = phase;
        sim.step_once(input);
        if i % 30 == 0 {
            let census = d3::boarding::seat_census(sim.world());
            worst = worst.max(census.values().copied().max().unwrap_or(0));
            seats_seen = seats_seen.max(census.len());
            passengers = passengers.max(census.keys().filter(|(_, s)| *s == 1).count());
            samples += 1;
        }
    }
    println!(
        "=== the island census ===\n  {STEPS} steps ({:.1} min), {samples} samples: at most {seats_seen} seats held at once, {passengers} of them passengers; the busiest seat ever held {worst}\n  the hero was put beside a standing car {attempts} times and reached the wheel {boarded} times ({jacked} of them carjacks)",
        f64::from(STEPS) / 3600.0
    );
    assert!(
        seats_seen >= 5,
        "only {seats_seen} seats were ever held on the island — the census is about nothing"
    );
    assert!(
        boarded >= 2,
        "the hero reached the wheel {boarded} times in {attempts} attempts — the census measured no boarding"
    );
    assert!(worst <= 1, "a seat on the island held {worst} bodies");
}
