//! **The traffic step, against a world** (wave VEH2b) — the streets a level's
//! own blocks imply, the cars on them, and the ladder that decides which of them
//! is a rig.
//!
//! The unit arms for the map and the controller are in `inf_ecs::traffic`, where
//! they need no rapier. This file exists for the three claims that are only
//! true of a *world*:
//!
//! * a car inside [`inf_ecs::traffic::TRAFFIC_FULL_M`] is a real
//!   `RaycastVehicle` the vehicle phase drives, and one outside it is
//!   **structurally invisible** to that phase — not "skipped", *absent*;
//! * a commuter actually drives: it leaves its space, covers metres, and does
//!   so because a driver's stick was written by
//!   [`inf_ecs::traffic::drive_intent`] and turned into controls by the same
//!   `VehicleControls::from_intent` a player's stick goes through;
//! * the promotion rule of clause 4 is one door: the tier decides everything,
//!   and there is no second test anywhere for "is this car enterable".

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterMovement, Collider3D, ColliderShape3DKind, MovementMode, PcgVolume,
    ResidentSlot, RigidBody3D, SlotRole, StreamingSource, Terrain, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::traffic;
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;

const DT: f64 = 1.0 / 60.0;

/// A 3x3 grid of 80 m blocks on a 100 m pitch — two 20 m streets each way, the
/// shape `inf_editor_core::settlement` plans for a city.
const PITCH: f64 = 100.0;
const STREET: f64 = 20.0;

const HERO: Uuid = Uuid::from_u128(0x7000_0001);
const GROUND: Uuid = Uuid::from_u128(0x7000_0002);
const SKY: Uuid = Uuid::from_u128(0x7000_0003);

fn blocks(world: &mut EcsWorld, cols: i32, rows: i32) {
    let half = (PITCH - STREET) * 0.5;
    for row in 0..rows {
        for col in 0..cols {
            let c = DVec2::new(f64::from(col) * PITCH, f64::from(row) * PITCH);
            let guid = Uuid::from_u64_pair(0x51, (row as u64) << 32 | col as u64);
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
                face: glam::DVec3::ZERO,
            }];
            world.world_mut().entity_mut(e).insert(v);
        }
    }
}

/// A big static floor, so a `Full` car's four rays land on something.
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
            half_extents: Vec3d::new(400.0, 0.5, 400.0),
            ..Default::default()
        },
    ));
    let _ = Terrain::default();
}

/// The band's anchor. Where this stands is what every tier in the level is.
fn hero(world: &mut EcsWorld, at: DVec3) {
    let e = match world.entity_of(HERO) {
        Some(e) => e,
        None => {
            let e = world.spawn_with_guid(HERO, "Hero", None);
            world
                .world_mut()
                .entity_mut(e)
                .insert(StreamingSource { radius_m: 512.0 });
            e
        }
    };
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::from_dvec3(at);
    } else {
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(at));
    }
    world.propagate();
}

struct Town {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

impl Town {
    fn new(at: DVec3) -> Self {
        let mut world = EcsWorld::new();
        blocks(&mut world, 3, 3);
        ground(&mut world);
        hero(&mut world, at);
        Self {
            world,
            bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
        }
    }

    fn step(&mut self, n: usize) -> traffic::TrafficStats {
        let mut last = traffic::TrafficStats::default();
        for _ in 0..n {
            last = inf_physics::d3::traffic::step_traffic(&mut self.world, &mut self.bridge, DT);
            self.bridge
                .sync_from_world_sim(&self.world, &Default::default(), &Default::default());
            inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
        last
    }

    fn stand(&mut self, at: DVec3) {
        hero(&mut self.world, at);
    }

    /// Wind the level clock to `hour` — the door the commute is read through.
    fn set_hour(&mut self, hour: f64) {
        use inf_ecs::components::TimeOfDay;
        let e = match self.world.entity_of(SKY) {
            Some(e) => e,
            None => self.world.spawn_with_guid(SKY, "Sky", None),
        };
        self.world.world_mut().entity_mut(e).insert(TimeOfDay {
            seconds: hour * 3600.0,
            rate: 0.0,
            ..Default::default()
        });
    }
}

/// **The whole ladder, in one world.** A settled town derives its streets once,
/// parks cars on them, and gives exactly the ones inside 64 m a rig.
#[test]
fn a_towns_kerbs_fill_with_cars_and_only_the_near_ones_are_rigs() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    let stats = town.step(40);

    // The map, derived once.
    let res = traffic::carriageway_of(&town.world).expect("a carriageway");
    assert_eq!(res.derivations, 1, "the derivation is not a cache");
    assert_eq!(res.streets.len(), 4, "{:?}", res.streets);
    assert!(res.lanes.len() >= 8, "{}", res.lanes.len());

    // The population: cars on the kerbs, a fraction of them with a day.
    assert!(stats.cars > 20, "only {} cars", stats.cars);
    assert!(stats.commuters > 0, "no commuter has a day");
    assert!(
        stats.commuters < stats.cars,
        "every car commutes, which is not a street"
    );

    // The ladder: some Full, some Near, and NONE anywhere else — a car has
    // three rungs, and `Far` is unreachable by construction.
    let [full, near, far, dormant] = stats.per_tier;
    assert!(full > 0, "nothing is near the hero: {:?}", stats.per_tier);
    assert!(near > 0, "nothing is beyond it: {:?}", stats.per_tier);
    assert_eq!(far, 0, "a car reached the Far rung, which has no meaning");
    assert_eq!(full + near + far + dormant, stats.cars);

    // …and the ladder is a fact about the WORLD, not about the report.
    let mut rigs = 0;
    let mut bodies = 0;
    for (guid, rec) in inf_physics::d3::traffic::records(&town.world) {
        let Some(e) = town.world.entity_of(guid) else {
            assert_eq!(rec.tier, inf_ecs::crowd::CrowdTier::Dormant);
            continue;
        };
        let has_rig = inf_ecs::vehicle::rig_of(&town.world, guid).is_some();
        let kind = town
            .world
            .world()
            .get::<RigidBody3D>(e)
            .map(|b| b.kind)
            .expect("every built car has a body");
        match rec.tier {
            inf_ecs::crowd::CrowdTier::Full => {
                assert!(has_rig, "a Full car has no wheels");
                assert_eq!(kind, BodyKind3D::Dynamic);
                assert!(town.bridge.vehicle_of(guid).is_some(), "not in the bridge");
                rigs += 1;
            }
            _ => {
                // THE WATCHED-CAR SENTENCE, as a fact: no wheels, so no rig, so
                // `step_vehicles` cannot reach it even by accident.
                assert!(!has_rig, "a Near car has wheels");
                assert_eq!(kind, BodyKind3D::Kinematic);
                assert!(town.bridge.vehicle_of(guid).is_none(), "in the bridge");
                bodies += 1;
            }
        }
    }
    assert_eq!(rigs, full);
    assert_eq!(bodies, near);
    println!(
        "traffic: {} cars, {} commuters, tiers {:?}, {} rigs / {} bodies",
        stats.cars, stats.commuters, stats.per_tier, rigs, bodies
    );
}

/// **The promotion rule is the tier, and there is no second one.**
///
/// Clause 4's door: a parked car near the hero must be enterable. The reach is
/// `ENTER_REACH_M`, 3 m; the Full radius is 64. So the claim is not "we also
/// check whether you can enter it" — it is that **every car within reach of the
/// hero's feet has a seat the one interaction door can find**, because the only
/// thing that could stop it is a missing rig and the tier has already built one.
#[test]
fn every_car_the_hero_can_reach_is_one_the_interact_door_can_enter() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.step(40);
    // Stand on top of the nearest parked car's own space.
    let near = inf_physics::d3::traffic::cars_near(&town.world, DVec3::new(50.0, 0.0, 50.0), 60.0);
    assert!(!near.is_empty(), "no car within 60 m of the crossroads");
    let (target, _) = near[0];
    let at = inf_physics::d3::traffic::records(&town.world)[&target].last;
    town.stand(at + DVec3::new(2.0, 0.0, 0.0));
    town.step(4);

    let feet = at + DVec3::new(2.0, 0.0, 0.0);
    let seat = inf_physics::d3::vehicle::try_enter(&town.bridge, feet, &Default::default());
    assert_eq!(
        seat,
        Some(target),
        "the nearest car is not the seat the one door answers"
    );
    // …and the reason is the tier and nothing else.
    assert_eq!(
        inf_physics::d3::traffic::records(&town.world)[&target].tier,
        inf_ecs::crowd::CrowdTier::Full
    );
}

/// **A commuter drives**, and it drives because a driver held a stick.
#[test]
fn a_commuter_leaves_its_space_at_eight_oclock_with_somebody_at_the_wheel() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    // Midnight: every commuter is parked and only the night shift is out.
    town.set_hour(0.0);
    let night = town.step(60);

    // Half past eight: the commuters are out too.
    town.set_hour(8.5);
    let rush = town.step(90);
    assert!(rush.driving > 0, "nobody drives at half past eight");
    assert!(
        rush.driving > night.driving * 3,
        "the rush ({}) is not visibly busier than midnight ({})",
        rush.driving,
        night.driving
    );
    assert!(
        rush.drivers > 0,
        "{} cars are driving and none of them has a driver",
        rush.driving
    );

    // One of them has a person in its seat, in `Driving` mode, with this car
    // named on the seat — the same `SeatState` the hero's own enter writes.
    let mut seated = 0;
    let mut moved = 0.0f64;
    for (guid, rec) in inf_physics::d3::traffic::records(&town.world) {
        if rec.tier != inf_ecs::crowd::CrowdTier::Full || !rec.commutes() {
            continue;
        }
        let driver = traffic::driver_guid(guid);
        let Some(de) = town.world.entity_of(driver) else {
            continue;
        };
        let cm = town
            .world
            .world()
            .get::<CharacterMovement>(de)
            .expect("a driver has a movement model");
        if cm.mode == MovementMode::Driving && cm.runtime.seat.vehicle == guid {
            seated += 1;
            // …and the stick is not zero: something is asking the car to go.
            let ask = cm.runtime.intent_move.y.abs() + cm.runtime.intent_move.x.abs();
            moved = moved.max(ask);
        }
    }
    assert!(seated > 0, "no commuter has a person in it");
    assert!(moved > 0.0, "every driver is holding the stick at zero");
    println!(
        "rush: {} driving, {} drivers, {} seated, worst stick {:.3}",
        rush.driving, rush.drivers, seated, moved
    );
}

/// **A car the player touches stops being traffic's**, once and for ever.
#[test]
fn a_car_somebody_sits_in_is_let_go_of_and_never_taken_back() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.step(40);
    let near = inf_physics::d3::traffic::cars_near(&town.world, DVec3::new(50.0, 0.0, 50.0), 60.0);
    let (target, _) = near[0];
    assert!(!traffic::is_taken(&town.world, target));

    // A hero character sits in it, through the same `SeatState` the enter door
    // writes.
    let e = town.world.entity_of(HERO).expect("the hero");
    town.world
        .world_mut()
        .entity_mut(e)
        .insert(CharacterMovement {
            mode: MovementMode::Driving,
            ..Default::default()
        });
    if let Some(mut cm) = town.world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: target,
            entering: false,
            time_s: 0.0,
            start: Vec3d::ZERO,
            start_yaw_deg: 0.0,
        };
    }
    let stats = town.step(2);
    assert!(traffic::is_taken(&town.world, target), "{stats:?}");
    assert!(stats.taken > 0);

    // …and it stays let go of when the hero walks away, and its rig stays.
    if let Some(mut cm) = town.world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.mode = MovementMode::Grounded;
        cm.runtime.seat = inf_ecs::components::SeatState::default();
    }
    town.stand(DVec3::new(900.0, 0.0, 900.0));
    town.step(20);
    assert!(traffic::is_taken(&town.world, target));
    assert!(
        town.world.entity_of(target).is_some(),
        "a stolen car left far away was despawned out from under the player"
    );
}

/// **`audit:` VEH2b — a person standing in a parking space is not the ground.**
///
/// `settle_on_the_ground` casts `AllSolid`, which is every solid body and not
/// only the level's geometry, and a `Full` crowd agent carries a KINEMATIC
/// capsule. `ground_y` is latched on purpose, so one ray that landed on a
/// pedestrian's head is a car parked in mid-air for the rest of the session — a
/// `Near` car is kinematic and `place` writes that number onto its transform
/// every step.
///
/// Measured: without the exclusion set this arm reads **1.500 m** of ground
/// under a slot whose ground is at zero.
#[test]
fn a_car_does_not_settle_onto_the_pedestrian_standing_in_its_space() {
    let hero_at = DVec3::new(50.0, 0.0, 50.0);
    // The slot, chosen the way the derivation chooses it — the occupancy draw
    // and the day draw, so this is a car that really is parked there.
    let streets = {
        let mut w = EcsWorld::new();
        blocks(&mut w, 3, 3);
        w.propagate();
        inf_ecs::traffic::streets_of(&w)
    };
    let slot = inf_ecs::traffic::kerb_slots(&streets)
        .into_iter()
        .map(|(p, _)| p)
        .filter(|p| {
            let g = inf_ecs::traffic::parked_car_guid(*p);
            inf_ecs::crowd::agent_unit(g, 0, inf_ecs::traffic::SALT_PARK)
                < inf_ecs::traffic::KERB_OCCUPANCY
                && inf_ecs::traffic::day_of(g) == inf_ecs::traffic::TrafficDay::Parked
        })
        .filter(|p| (*p - hero_at).length() < 40.0)
        .min_by(|a, b| (*a - hero_at).length().total_cmp(&(*b - hero_at).length()))
        .expect("a parked car's space near the crossroads");

    let mut town = Town::new(hero_at);
    town.set_hour(12.0);
    // Somebody is standing in it, built through the crowd's own door so the
    // capsule is the one a resident wears. **Mirrored into rapier before the
    // first traffic step**, because the step runs before the physics sync and a
    // body spawned this step is invisible to this step's rays — which is the
    // real world's ordering for a pedestrian who has been crossing for a while.
    let arch = inf_ecs::society::level_archetype(&town.world);
    inf_ecs::crowd::spawn_body(
        &mut town.world,
        Uuid::from_u128(0x9E_D0),
        &arch,
        DVec3::new(slot.x, 0.9, slot.z),
    );
    town.world.propagate();
    town.bridge
        .sync_from_world_sim(&town.world, &Default::default(), &Default::default());

    town.step(40);
    let guid = inf_ecs::traffic::parked_car_guid(slot);
    let rec = inf_physics::d3::traffic::records(&town.world)
        .remove(&guid)
        .expect("the car whose space the pedestrian is standing in");
    let ground = rec.ground_y.expect("it settled on something");
    println!("settle: slot {slot:?} -> ground_y {ground:.3} m");
    assert!(
        ground.abs() < 0.05,
        "the car settled at {ground:.3} m — it took the pedestrian's head for the road"
    );
}

/// **A level with no blocks has no traffic and costs one look.**
#[test]
fn a_world_with_no_streets_grows_no_traffic() {
    let mut world = EcsWorld::new();
    ground(&mut world);
    hero(&mut world, DVec3::ZERO);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let stats = inf_physics::d3::traffic::step_traffic(&mut world, &mut bridge, DT);
    assert_eq!(stats.cars, 0);
    assert!(traffic::traffic_of(&world).is_none_or(|p| p.records.is_empty()));
    assert!(inf_ecs::traffic::traffic_state_bytes(&world).is_empty());
}

/// **The trace sees the ladder.** Two worlds that tiered the same car
/// differently produce different bytes — which is what makes the PIE compare
/// able to see a traffic divergence at all.
#[test]
fn the_trace_moves_when_a_car_changes_tier() {
    let mut a = Town::new(DVec3::new(50.0, 0.0, 50.0));
    a.step(30);
    let near = inf_ecs::traffic::traffic_state_bytes(&a.world);
    assert!(!near.is_empty());
    a.stand(DVec3::new(900.0, 0.0, 900.0));
    a.step(30);
    let far = inf_ecs::traffic::traffic_state_bytes(&a.world);
    assert_eq!(
        near.len(),
        far.len(),
        "the record set changed, not the tier"
    );
    assert_ne!(near, far, "the tier decision is invisible to the trace");
    // …and every car is Dormant now, which is the thing the bytes moved about.
    let stats = a.step(1);
    assert_eq!(stats.per_tier[0], 0);
    assert_eq!(stats.per_tier[1], 0);
    assert!(stats.per_tier[3] > 0);
}

/// **`audit:` VEH2b — WHAT THE PLAYER SEES AT 64 m.**
///
/// The module doc promises the demotion *"moves nothing"* — `rephase_delta_on`
/// hands the clock the metre the BODY reached — and nothing measured it. It is
/// worth measuring because the two tiers are not two views of one thing: a
/// `Full` car is a dynamic chassis on four rays, somewhere near its lane; a
/// `Near` car is a kinematic body written to `place(clock)`, which is *exactly*
/// on the lane centreline at the latched `ground_y`. Whatever lateral error the
/// steering was carrying is spent in one step, and this is how big that step is.
///
/// The falsifier is the number, not the sentence: **without the rephase the
/// hand-off is the whole distance between where the body got to and where the
/// clock ran on to**, which on a car that has spent ten seconds behind a queue
/// is tens of metres. Measured here at well under a metre.
#[test]
fn a_car_leaving_the_steered_tier_lands_where_its_body_already_was() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.set_hour(8.5);
    town.step(240);
    // A car that is being STEERED — the only kind the hand-off is about.
    let (target, _) = inf_physics::d3::traffic::records(&town.world)
        .into_iter()
        .find(|(g, r)| {
            r.tier == inf_ecs::crowd::CrowdTier::Full
                && r.commutes()
                && town
                    .world
                    .entity_of(traffic::driver_guid(*g))
                    .is_some_and(|e| {
                        town.world
                            .world()
                            .get::<CharacterMovement>(e)
                            .is_some_and(|cm| cm.runtime.seat.vehicle == *g)
                    })
        })
        .expect("a steered commuter");
    let at = |t: &Town| {
        t.world
            .entity_of(target)
            .and_then(|e| t.world.world().get::<Transform>(e))
            .map(|x| x.translation.to_dvec3())
            .expect("the chassis")
    };
    // Stand between the two rungs — 100 m is outside `TRAFFIC_FULL_M` and well
    // inside `TRAFFIC_NEAR_M`, so the car drops to `Near` rather than vanishing.
    town.stand(at(&town) + DVec3::new(100.0, 0.0, 0.0));
    // The band is read off `GlobalTransform`, which the propagate at the END of
    // a step publishes — so the anchor's move lands a step later, and `before`
    // has to be re-read each step or it is stale by a step of driving.
    let mut before = at(&town);
    let mut stats = traffic::TrafficStats::default();
    let mut demoted = false;
    for _ in 0..20 {
        before = at(&town);
        stats = town.step(1);
        if inf_physics::d3::traffic::records(&town.world)[&target].tier
            == inf_ecs::crowd::CrowdTier::Near
        {
            demoted = true;
            break;
        }
    }
    assert!(demoted, "the car never left the steered rung: {stats:?}");
    assert!(
        stats.rephased > 0,
        "nothing handed its clock back: {stats:?}"
    );
    assert!(
        inf_ecs::vehicle::rig_of(&town.world, target).is_none(),
        "it kept its wheels across the boundary"
    );
    let after = at(&town);
    let jump = ((after.x - before.x).powi(2) + (after.z - before.z).powi(2)).sqrt();
    println!(
        "hand-off: {jump:.3} m across the 64 m boundary, dy {:.3}",
        after.y - before.y
    );
    // **The bound is the rephase's own geometry, not a round number** (wave
    // ROAD1b). Handing the clock back projects the body onto its lane, so the
    // most it can move is the half-lane it is projected across plus the metre
    // the body drives while the step it is demoted on finishes. The failure
    // mode this arm exists for is the car landing back where it was *promoted*,
    // which is tens of metres.
    //
    // It read `jump < 1.0` until wave ROAD1b, which is UNDER the half-lane and
    // was therefore never a bound — it was a threshold that held for the car
    // this fixture happened to draw. Measured, by sweeping an unrelated
    // constant (`kerb_park_offset_m`, which moves the parked lattice and so
    // re-draws which commuter is picked): the jump reads 0.144 m at 5.0,
    // 1.385 m at 5.2, 0.000 m at 6.0 and 1.041 m at 6.8. A quantity that swings
    // tenfold on a 20 cm change somewhere else is a sample, and 1.0 m was
    // fitting it.
    let bound = inf_ecs::traffic::DEFAULT_LANE_WIDTH_M * 0.5
        + inf_ecs::traffic::street_speed_mps() * DT * 2.0;
    assert!(
        jump < bound,
        "a car crossing the steered boundary jumped {jump:.2} m, past the \
         {bound:.2} m its own rephase can move it (half a lane plus two steps \
         of driving) — the clock was not handed the metre the body reached"
    );
}

/// Two identical towns run byte for byte — the determinism the whole ladder
/// rests on.
#[test]
fn two_identical_towns_produce_the_same_traffic() {
    let mut a = Town::new(DVec3::new(50.0, 0.0, 50.0));
    let mut b = Town::new(DVec3::new(50.0, 0.0, 50.0));
    a.set_hour(8.5);
    b.set_hour(8.5);
    for _ in 0..60 {
        a.step(1);
        b.step(1);
        assert_eq!(
            inf_ecs::traffic::traffic_state_bytes(&a.world),
            inf_ecs::traffic::traffic_state_bytes(&b.world)
        );
    }
}

// ── the carjack (clause 5) ──────────────────────────────────────────────────

/// **THE CARJACK.** A commuter is driving, the hero walks up to the driver's
/// door, presses the one interact key, and ends up behind the wheel with the
/// driver staggering away down the street.
///
/// Every step of it goes through a door that already existed: the prompt is
/// `inf_ecs::interact::resolve`, the eject is `finish_driving`, the seat entry
/// is the P29.7 warp, and the victim's walk is an ordinary `CrowdRecord`.
#[test]
fn the_hero_pulls_a_commuter_out_of_a_moving_car_and_drives_off_in_it() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.set_hour(8.5);
    town.step(90);

    // Find a Full-tier car with somebody at the wheel.
    let mut target = None;
    for (guid, rec) in inf_physics::d3::traffic::records(&town.world) {
        if rec.tier != inf_ecs::crowd::CrowdTier::Full {
            continue;
        }
        if let Some(v) = inf_physics::d3::carjack::occupant_of(&town.world, guid) {
            target = Some((guid, v));
            break;
        }
    }
    let (chassis, victim) = target.expect("no Full-tier car has a driver in it");
    assert!(inf_physics::d3::carjack::is_ejectable(&town.world, victim));

    // The wrong side of the car offers nothing at all — the door-side rule.
    let seat = inf_physics::d3::vehicle::seat_pose(&town.bridge, chassis).expect("a seat");
    let wrong = seat.0 - (seat.1 * DVec3::X) * 2.0;
    assert!(
        !inf_physics::d3::carjack::at_the_door(&town.bridge, chassis, wrong),
        "the passenger side counts as the driver door"
    );
    // …and THIS car is not on offer from there. (Another car may be — the set
    // is every carjackable car in reach of the point, and a street has cars on
    // both sides of it.)
    assert!(
        !inf_physics::d3::carjack::carjackable(&town.world, &town.bridge, wrong).contains(&chassis),
        "a candidate from the wrong side of the car"
    );

    // The driver's side offers the car, with the verb and the label a player
    // reads — through the SAME resolution site the enter prompt uses.
    let feet = seat.0 + (seat.1 * DVec3::X) * 2.0;
    assert!(inf_physics::d3::carjack::at_the_door(
        &town.bridge,
        chassis,
        feet
    ));
    let hit = inf_physics::d3::interact::resolve(
        &town.world,
        &town.bridge,
        feet,
        0.0,
        &Default::default(),
    )
    .expect("the one door answers something");
    assert_eq!(hit.guid, chassis);
    assert_eq!(hit.verb, inf_ecs::interact::InteractVerb::Carjack);
    assert_eq!(
        inf_ecs::interact::prompt_text(hit.verb, &hit.label, "E"),
        "[E] Pull out driver"
    );

    // The press. The resist draw is a function of the step, so a real player
    // pressing twice is what the arm does — and BOTH outcomes are asserted, so
    // neither branch is dead.
    let overlays = inf_ecs::movement::overlay_registry(
        &town.world,
        &inf_ecs::movement::movement_targets(&town.world),
    );
    // **The resist branch is armed against the DRAW, not against the tick this
    // fixture happened to land on.** Whether a given press resists is
    // `agent_unit(victim, step, SALT_RESIST) < RESIST_CHANCE`, and which side of
    // it the first press falls on is a fact about one seed. The claim worth
    // making is that the branch is LIVE and roughly the documented share.
    let draws: u64 = 200;
    let refusals = (0..draws)
        .filter(|t| {
            inf_ecs::crowd::agent_unit(victim, *t, inf_physics::d3::carjack::SALT_RESIST)
                < inf_physics::d3::carjack::RESIST_CHANCE
        })
        .count();
    assert!(
        refusals as u64 > draws / 8 && (refusals as u64) < draws / 2,
        "{refusals} of {draws} draws resist against a documented {}",
        inf_physics::d3::carjack::RESIST_CHANCE
    );

    let mut resisted = 0;
    let mut ejected = None;
    for _ in 0..12 {
        match inf_physics::d3::carjack::try_carjack(
            &mut town.world,
            &mut town.bridge,
            chassis,
            HERO,
            DT,
            &overlays,
        ) {
            Some(inf_physics::d3::carjack::Carjack::Resisted { .. }) => {
                resisted += 1;
                town.step(1);
            }
            Some(inf_physics::d3::carjack::Carjack::Ejected { victim, .. }) => {
                ejected = Some(victim);
                break;
            }
            None => panic!("the door refused a car with a driver in it"),
        }
    }
    let out = ejected.expect("twelve presses and the driver never let go");
    assert_eq!(out, victim);
    println!("  (the driver held on {resisted} time(s) before letting go)");

    // THE WORLD, not the report. The seat is free…
    assert!(inf_physics::d3::carjack::occupant_of(&town.world, chassis).is_none());
    // …the victim is out of the car, staggering, with its collider back…
    let ve = town
        .world
        .entity_of(victim)
        .expect("the victim still exists");
    let cm = town
        .world
        .world()
        .get::<CharacterMovement>(ve)
        .expect("with a movement model");
    assert_eq!(cm.mode, MovementMode::FallControlled);
    assert!(!cm.runtime.seat.is_seated());
    // …standing at the driver's door rather than inside the car…
    let at = town
        .world
        .world()
        .get::<Transform>(ve)
        .expect("a transform")
        .translation
        .to_dvec3();
    let d = at - seat.0;
    assert!(
        (d.x * d.x + d.z * d.z).sqrt() > 1.0,
        "the victim is still inside the car"
    );
    // …and the traffic has let the car go.
    assert!(traffic::is_taken(&town.world, chassis));

    // The victim now has somewhere to be, as an ordinary crowd agent.
    let route_end = {
        let p = town
            .world
            .world()
            .get_resource::<inf_ecs::crowd::CrowdPopulationRes>()
            .expect("the crowd adopted the victim");
        p.records
            .get(&victim)
            .expect("the victim is a crowd agent now")
            .route
            .destination()
    };
    let flee = route_end - at;
    assert!(
        ((flee.x * flee.x + flee.z * flee.z).sqrt() - inf_physics::d3::carjack::FLEE_M).abs() < 1.0,
        "the flee route is not {} m long",
        inf_physics::d3::carjack::FLEE_M
    );

    // …and the seat the hero now walks up to is a free one, answered by the
    // ordinary enter door.
    let seat_now = inf_physics::d3::vehicle::try_enter(&town.bridge, feet, &Default::default());
    assert_eq!(seat_now, Some(chassis));

    // The victim walks: some steps later it is further from the car than it
    // started, and it is doing that as a crowd agent rather than a statue.
    let before = at;
    town.step(120);
    let after = town
        .world
        .entity_of(victim)
        .and_then(|e| town.world.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .unwrap_or(before);
    let walked = ((after.x - before.x).powi(2) + (after.z - before.z).powi(2)).sqrt();
    assert!(walked > 1.0, "the victim has not moved: {walked} m");
    println!("carjack: {resisted} resists, victim walked {walked:.2} m");
}

/// **You cannot pull yourself out of your own car**, and a car nobody is in is
/// not a carjack at all — it is an ordinary theft, through the ordinary door.
#[test]
fn an_empty_car_is_not_a_carjack_and_neither_is_your_own() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.set_hour(0.0);
    town.step(40);
    // **A car with NOBODY IN IT**, chosen by asking rather than by assuming the
    // hour. A night shift runs through midnight, so "everything is parked at
    // midnight" is not true of this engine and an arm that rested on it was
    // resting on the draw.
    let target =
        inf_physics::d3::traffic::cars_near(&town.world, DVec3::new(50.0, 0.0, 50.0), 60.0)
            .into_iter()
            .map(|(g, _)| g)
            .find(|g| {
                inf_physics::d3::carjack::occupant_of(&town.world, *g).is_none()
                    && inf_physics::d3::vehicle::seat_pose(&town.bridge, *g).is_some()
            })
            .expect("a parked car near the crossroads");
    let seat = inf_physics::d3::vehicle::seat_pose(&town.bridge, target).expect("a seat");
    let feet = seat.0 + (seat.1 * DVec3::X) * 2.0;
    assert!(
        !inf_physics::d3::carjack::carjackable(&town.world, &town.bridge, feet).contains(&target),
        "an empty car is on offer as a carjack"
    );
    // …and the one door offers the ordinary Enter instead.
    let hit = inf_physics::d3::interact::resolve(
        &town.world,
        &town.bridge,
        feet,
        0.0,
        &Default::default(),
    )
    .expect("a free seat");
    assert_eq!(hit.verb, inf_ecs::interact::InteractVerb::Enter);
    assert_eq!(hit.guid, target);

    // A PLAYER-CONTROLLED occupant is never ejectable, which is what makes
    // "your own car" answer no without anybody passing an actor in.
    let e = town.world.entity_of(HERO).expect("the hero");
    town.world
        .world_mut()
        .entity_mut(e)
        .insert(CharacterMovement {
            player_controlled: true,
            mode: MovementMode::Driving,
            ..Default::default()
        });
    if let Some(mut cm) = town.world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: target,
            entering: false,
            time_s: 0.0,
            start: Vec3d::ZERO,
            start_yaw_deg: 0.0,
        };
    }
    assert_eq!(
        inf_physics::d3::carjack::occupant_of(&town.world, target),
        Some(HERO)
    );
    assert!(!inf_physics::d3::carjack::is_ejectable(&town.world, HERO));
    assert!(
        !inf_physics::d3::carjack::carjackable(&town.world, &town.bridge, feet).contains(&target),
        "your own car is on offer as a carjack"
    );
}

// ── the living street (clauses 3 and 6) ─────────────────────────────────────

/// **A street at night is sparse, not empty** — `frames/steal-car/0028`.
///
/// The commute alone cannot do this: its legs run at eight and at six, so
/// between them the town is a car park. The circuit can, because it is a loop at
/// a STATED speed rather than a fraction of a window, and a night shift is a
/// circuit whose hours run through midnight.
#[test]
fn the_street_is_busy_at_eight_and_sparse_at_three_and_never_empty() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    // Let every route plan before the clock is read, so the census is about the
    // hour and not about how far the batch queue got.
    town.set_hour(8.5);
    town.step(240);
    let rush = town.step(30);
    assert_eq!(rush.pending, 0, "the plan queue never drained");

    town.set_hour(3.0);
    let night = town.step(30);
    town.set_hour(8.5);
    let morning = town.step(30);
    town.set_hour(13.0);
    let noon = town.step(30);

    assert!(
        night.driving > 0,
        "nothing at all is on the road at three in the morning"
    );
    assert!(
        morning.driving > night.driving,
        "rush hour ({}) is no busier than three a.m. ({})",
        morning.driving,
        night.driving
    );
    assert!(
        noon.driving > 0 && noon.driving < morning.driving,
        "midday ({}) is not between the night ({}) and the rush ({})",
        noon.driving,
        night.driving,
        morning.driving
    );
    // …and the night-shift cars are the only ones out, so the ones that are not
    // working are NOT parked back in their spaces pretending to be.
    let hour_now = 3.0;
    town.set_hour(hour_now);
    town.step(2);
    for (_, rec) in inf_physics::d3::traffic::records(&town.world) {
        if let Some(c) = rec.circuit.as_ref() {
            if !c.running(hour_now) {
                assert_eq!(
                    rec.tier,
                    inf_ecs::crowd::CrowdTier::Dormant,
                    "a car that is not working is standing in the street"
                );
            }
        }
    }
    println!(
        "street: 03:00 {} driving, 08:30 {}, 13:00 {}",
        night.driving, morning.driving, noon.driving
    );
}

/// **A circuit car keeps its speed round the seam.**
///
/// A loop's start and end are one point, so a controller that read the path as
/// finite would brake to a halt there — and it would do it every lap, for ever.
#[test]
fn a_loop_does_not_brake_at_its_own_seam() {
    // The seam is in the MIDDLE OF AN EDGE, not on a corner: a loop that
    // closed on a right angle would slow there legitimately (the bend rule) and
    // the arm could not tell that from the defect it is about.
    let ring = inf_ecs::traffic::LanePath::new([
        DVec3::new(100.0, 0.0, 0.0),
        DVec3::new(200.0, 0.0, 0.0),
        DVec3::new(200.0, 0.0, 200.0),
        DVec3::new(0.0, 0.0, 200.0),
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(100.0, 0.0, 0.0),
    ]);
    let len = ring.length_m();
    let at_seam = |loops: bool| {
        traffic::drive_intent(&traffic::DriveView {
            at: DVec3::new(92.0, 0.0, 0.0),
            forward: DVec3::X,
            forward_mps: 8.0,
            path: &ring,
            s_m: len - 8.0,
            speed_limit_mps: 8.4,
            gap_m: None,
            lateral_bias_m: 0.0,
            loops,
        })
    };
    // A finite path stops at its end…
    assert!(at_seam(false).target_mps < 8.0, "{:?}", at_seam(false));
    // …and a loop does not.
    let looped = at_seam(true);
    assert_eq!(looped.target_mps, 8.4, "{looped:?}");
    assert!(!looped.handbrake);
    // …and its aim point wraps past the seam rather than clamping to it, so the
    // wheel is not held hard over for the last eight metres of every lap.
    assert!(looped.move_input.x.abs() < 1.0, "{looped:?}");
}

/// **The pavement, the parked row and the carriageway do not overlap.**
///
/// Two systems that never met until this wave: `inf_ecs::society` lays a
/// pedestrian ring `PAVEMENT_M` outside every block, and this wave lays lanes
/// down the middle of the gap and parked cars between the two. Nothing had ever
/// checked that a resident's own walking surface is clear of the road, because
/// until now there was nothing on the road.
///
/// This is clause 6's real content: the peds were ALREADY on the pavement (the
/// society has never routed one down the road centre), and what was missing was
/// an arm saying the traffic this wave put in the street does not drive over
/// them.
#[test]
fn the_pavement_the_parked_row_and_the_lanes_are_three_separate_places() {
    use inf_ecs::society::PAVEMENT_M;
    let streets = {
        let mut w = EcsWorld::new();
        blocks(&mut w, 3, 3);
        w.propagate();
        inf_ecs::traffic::streets_of(&w)
    };
    assert!(!streets.is_empty());
    let lanes = inf_ecs::traffic::carriageway(&streets);
    let slots = inf_ecs::traffic::kerb_slots(&streets);
    assert!(!slots.is_empty());

    // The pavement's own line, measured the way the society lays it: the block
    // edge plus `PAVEMENT_M`, which on a 20 m street is 8 m from the centre.
    let pavement_from_centre = STREET * 0.5 - PAVEMENT_M;
    assert_eq!(pavement_from_centre, 8.0);

    // A lane is inside 3.5 m of its own centreline…
    for lane in lanes.lanes() {
        for p in lane.path.points() {
            let off = perp_to_nearest(&streets, *p);
            assert!(
                off < inf_ecs::traffic::DEFAULT_LANE_WIDTH_M,
                "a lane at {off} m is out of the carriageway"
            );
        }
    }
    // …a parked car is between the carriageway and the pavement, on both
    // counts and with its own body allowed for…
    for (p, _) in &slots {
        let off = perp_to_nearest(&streets, *p);
        assert!(
            off > inf_ecs::traffic::DEFAULT_LANE_WIDTH_M / 2.0 + 1.0,
            "a parked car at {off} m is in a lane"
        );
        assert!(
            off + 1.0 <= pavement_from_centre,
            "a parked car at {off} m is on the pavement"
        );
    }
    // …and the pavement is outside both.
    assert!(pavement_from_centre > inf_ecs::traffic::KERB_PARK_OFFSET_M);
}

/// The shortest distance from `p` to any street centreline, metres.
fn perp_to_nearest(streets: &[inf_ecs::traffic::Street], p: DVec3) -> f64 {
    let mut best = f64::INFINITY;
    for s in streets {
        let (ax, az) = (s.a.x, s.a.y);
        let (dx, dz) = (s.b.x - s.a.x, s.b.y - s.a.y);
        let len2 = dx * dx + dz * dz;
        let t = if len2 > 0.0 {
            (((p.x - ax) * dx + (p.z - az) * dz) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (qx, qz) = (ax + dx * t, az + dz * t);
        let d = ((p.x - qx).powi(2) + (p.z - qz).powi(2)).sqrt();
        best = best.min(d);
    }
    best
}

/// **Traffic yields to whatever is standing in its lane** — which is why a town
/// does not run over its own residents, and why a hero standing in the road
/// stops the street.
#[test]
fn a_body_in_the_lane_is_a_gap_the_car_slows_for() {
    let lane =
        inf_ecs::traffic::LanePath::new([DVec3::new(0.0, 0.0, 0.0), DVec3::new(300.0, 0.0, 0.0)]);
    let clear = traffic::drive_intent(&traffic::DriveView {
        at: DVec3::new(0.0, 0.0, 0.0),
        forward: DVec3::X,
        forward_mps: 8.0,
        path: &lane,
        s_m: 0.0,
        speed_limit_mps: 8.4,
        gap_m: None,
        lateral_bias_m: 0.0,
        loops: false,
    });
    assert_eq!(clear.target_mps, 8.4);
    // A pedestrian twelve metres up the lane.
    let blocked = traffic::drive_intent(&traffic::DriveView {
        gap_m: Some(12.0),
        ..clear_view(&lane)
    });
    assert!(blocked.target_mps < clear.target_mps, "{blocked:?}");
    assert!(blocked.move_input.y < 0.0, "the car is not braking");
    // …and standing right in front of it stops it dead rather than nudging it.
    let touching = traffic::drive_intent(&traffic::DriveView {
        gap_m: Some(inf_ecs::traffic::STANDING_GAP_M),
        forward_mps: 0.2,
        ..clear_view(&lane)
    });
    assert_eq!(touching.target_mps, 0.0);
    assert!(touching.handbrake);
}

fn clear_view(lane: &inf_ecs::traffic::LanePath) -> traffic::DriveView<'_> {
    traffic::DriveView {
        at: DVec3::new(0.0, 0.0, 0.0),
        forward: DVec3::X,
        forward_mps: 8.0,
        path: lane,
        s_m: 0.0,
        speed_limit_mps: 8.4,
        gap_m: None,
        lateral_bias_m: 0.0,
        loops: false,
    }
}

/// **A re-derivation is not a fresh start.** A block paging in across town
/// changes the level's own stamp; the cars that did not move keep everything
/// they had — including the fact that the player stole one.
#[test]
fn a_block_arriving_does_not_un_steal_the_car_the_player_is_in() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.set_hour(8.5);
    town.step(120);
    let before = inf_physics::d3::traffic::records(&town.world).len();
    assert!(before > 20);

    // Steal one.
    let near = inf_physics::d3::traffic::cars_near(&town.world, DVec3::new(50.0, 0.0, 50.0), 60.0);
    let (target, _) = near[0];
    let e = town.world.entity_of(HERO).expect("the hero");
    town.world
        .world_mut()
        .entity_mut(e)
        .insert(CharacterMovement {
            player_controlled: true,
            mode: MovementMode::Driving,
            ..Default::default()
        });
    if let Some(mut cm) = town.world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: target,
            entering: false,
            time_s: 0.0,
            start: Vec3d::ZERO,
            start_yaw_deg: 0.0,
        };
    }
    town.step(2);
    assert!(traffic::is_taken(&town.world, target));
    let phases: Vec<f64> = inf_physics::d3::traffic::records(&town.world)
        .values()
        .map(|r| r.rephase_m)
        .collect();

    // A fourth column of blocks arrives — a real streaming event, and a
    // different block stamp.
    blocks(&mut town.world, 4, 3);
    town.world.propagate();
    let stamp_before = traffic::carriageway_of(&town.world).unwrap().stamp;
    town.step(4);
    let res = traffic::carriageway_of(&town.world).expect("a carriageway");
    assert_ne!(res.stamp, stamp_before, "the new block changed nothing");
    assert_eq!(res.derivations, 2, "it re-derived more than once");

    // The town grew…
    let after = inf_physics::d3::traffic::records(&town.world);
    assert!(after.len() > before, "{} then {}", before, after.len());
    // …the stolen car is still stolen…
    assert!(
        traffic::is_taken(&town.world, target),
        "the re-derivation un-stole the player's car"
    );
    // …and the cars that were already there kept their phase rather than
    // snapping back to their clock.
    let now: Vec<f64> = after.values().map(|r| r.rephase_m).collect();
    assert!(
        now.iter().filter(|p| **p != 0.0).count() >= phases.iter().filter(|p| **p != 0.0).count(),
        "phases were reset by the re-derivation"
    );
}

/// **`audit:` VEH2b — the kerb's guids are the SPACE's, and paging OUT is the
/// direction the carry-forward was written for.**
///
/// `a_block_arriving_does_not_un_steal_the_car_the_player_is_in` armed a block
/// APPEARING. The claim `derive_parked` actually rests on is the other
/// direction: a settlement whose blocks page out has no slots at all, so every
/// derived guid disappears from the derivation and the only thing keeping the
/// player's stolen car is the `taken` branch. Paging back in must then re-mint
/// **the same guids** — which is what makes `parked_car_guid` a function of the
/// space rather than of the derivation — and must leave the stolen car's own
/// entity untouched rather than rebuilding it.
#[test]
fn a_town_that_pages_out_and_back_keeps_its_guids_and_the_car_the_player_stole() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.set_hour(12.0);
    town.step(60);
    let before: std::collections::BTreeSet<Uuid> = inf_physics::d3::traffic::records(&town.world)
        .into_keys()
        .collect();
    assert!(before.len() > 20, "{} cars", before.len());

    // The player takes one, through the seat state the enter door writes.
    let (target, _) =
        inf_physics::d3::traffic::cars_near(&town.world, DVec3::new(50.0, 0.0, 50.0), 60.0)[0];
    let hero = town.world.entity_of(HERO).expect("the hero");
    town.world
        .world_mut()
        .entity_mut(hero)
        .insert(CharacterMovement {
            player_controlled: true,
            mode: MovementMode::Driving,
            ..Default::default()
        });
    if let Some(mut cm) = town.world.world_mut().get_mut::<CharacterMovement>(hero) {
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: target,
            entering: false,
            time_s: 0.0,
            start: Vec3d::ZERO,
            start_yaw_deg: 0.0,
        };
    }
    town.step(4);
    assert!(traffic::is_taken(&town.world, target));
    let entity_before = town.world.entity_of(target).expect("a chassis");
    let where_it_was = inf_physics::d3::traffic::records(&town.world)[&target].last;

    // ── PAGE OUT. Every block goes; the level's own stamp moves; the
    //    derivation names no slot at all.
    for row in 0..3i32 {
        for col in 0..3i32 {
            let g = Uuid::from_u64_pair(0x51, (row as u64) << 32 | col as u64);
            if let Some(e) = town.world.entity_of(g) {
                town.world.despawn(e);
            }
        }
    }
    town.world.propagate();
    town.step(6);
    assert!(
        inf_ecs::traffic::streets_of(&town.world).is_empty(),
        "the blocks are gone and the streets are not"
    );
    let while_out = inf_physics::d3::traffic::records(&town.world);
    assert_eq!(
        while_out.len(),
        1,
        "a town with no blocks kept {} cars — only the stolen one is the traffic's to keep",
        while_out.len()
    );
    assert!(
        while_out.contains_key(&target),
        "the stolen car was forgotten"
    );
    assert!(traffic::is_taken(&town.world, target));
    assert_eq!(
        town.world.entity_of(target),
        Some(entity_before),
        "the stolen car's rig was taken down and rebuilt while it paged out"
    );

    // ── PAGE BACK IN, the same blocks in the same places.
    blocks(&mut town.world, 3, 3);
    town.world.propagate();
    town.step(6);
    let after: std::collections::BTreeSet<Uuid> = inf_physics::d3::traffic::records(&town.world)
        .into_keys()
        .collect();
    assert_eq!(
        after, before,
        "the town came back with a different set of cars — a guid is not a function of the space"
    );
    assert!(
        traffic::is_taken(&town.world, target),
        "paging the town back in un-stole the player's car"
    );
    assert_eq!(
        town.world.entity_of(target),
        Some(entity_before),
        "the stolen car was re-derived under the player"
    );
    let now = inf_physics::d3::traffic::records(&town.world)[&target].last;
    assert!(
        (now - where_it_was).length() < 1.0,
        "the stolen car moved {:.2} m across a page cycle",
        (now - where_it_was).length()
    );
}

/// **`audit:` VEH2b — the instrument reads the controller the engine runs.**
///
/// `d3::traffic::probe_intent`'s doc says it exists so *"an arm that rebuilt the
/// view itself would be measuring a controller the engine does not run"*. Until
/// this arm it had no caller at all, which made that a claim about a gate that
/// did not exist. Here it is, held against the stick the step actually wrote
/// onto the driver.
///
/// The comparison is taken **immediately after a bare `step_traffic`**, before
/// the solver has moved anything: `view_of` reads the live chassis pose, so a
/// probe taken a whole frame later is a different view of a car that has since
/// travelled, and the two numbers would differ by a step of motion rather than
/// by a defect. Measured: 0.0006 of steer over one frame.
#[test]
fn the_intent_probe_answers_the_same_stick_the_step_wrote() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.set_hour(8.5);
    town.step(120);
    inf_physics::d3::traffic::step_traffic(&mut town.world, &mut town.bridge, DT);
    let mut checked = 0;
    for (guid, rec) in inf_physics::d3::traffic::records(&town.world) {
        if rec.tier != inf_ecs::crowd::CrowdTier::Full {
            continue;
        }
        let Some(de) = town.world.entity_of(traffic::driver_guid(guid)) else {
            continue;
        };
        let Some(cm) = town.world.world().get::<CharacterMovement>(de) else {
            continue;
        };
        let probed = inf_physics::d3::traffic::probe_intent(&town.world, &town.bridge, guid, DT)
            .expect("a driving car's driver is asking for something");
        assert_eq!(
            probed.move_input, cm.runtime.intent_move,
            "the probe and the step disagree about what {guid} is asking for"
        );
        assert!(probed.target_mps.is_finite());
        checked += 1;
    }
    assert!(checked > 0, "no Full-tier car had a driver to probe");
    println!("probe: {checked} driver(s) agree with the step");
}

/// **A stolen car drives.** The falsifier for the island gate's own modest
/// number: on the CI fixture the hero's stolen car tops out under a metre a
/// second, and this arm is what says that is the *town* — two streets, sixteen
/// cars and three hundred and twenty-nine residents crossing them — rather than
/// a car this wave cannot make go.
///
/// Here the street is empty of everything but the traffic itself, and the same
/// press through the same door produces a car that accelerates.
#[test]
fn a_stolen_car_answers_the_throttle_on_an_empty_street() {
    let mut town = Town::new(DVec3::new(50.0, 0.0, 50.0));
    town.set_hour(8.5);
    town.step(60);
    // **ONE car, and nothing else on the street.** A derived town parks a car
    // every fourteen metres, so a stolen one meets the next bumper after nine —
    // which is a true fact about a kerb and the wrong thing for this arm to
    // measure. `set_traffic` installs a population BY HAND, which also stops the
    // derivation (`hand_installed`), so what is left is one rig on an empty
    // road. A `Near` car is KINEMATIC and has no rig, and a car with a driver
    // already in it is two people in one seat; both are asked about rather than
    // assumed.
    let (target, one) = inf_physics::d3::traffic::records(&town.world)
        .into_iter()
        .find(|(g, r)| {
            r.tier == inf_ecs::crowd::CrowdTier::Full
                && inf_physics::d3::carjack::occupant_of(&town.world, *g).is_none()
        })
        .expect("an empty rig near the crossroads");
    let keep: std::collections::BTreeMap<Uuid, inf_ecs::traffic::TrafficRecord> =
        [(target, one)].into_iter().collect();
    inf_ecs::traffic::set_traffic(&mut town.world, keep);
    town.step(30);
    assert_eq!(inf_physics::d3::traffic::records(&town.world).len(), 1);
    assert!(
        inf_ecs::vehicle::rig_of(&town.world, target).is_some(),
        "the one car left is not a rig"
    );

    // Sit in it, the way the seat step does.
    let e = town.world.entity_of(HERO).expect("the hero");
    town.world
        .world_mut()
        .entity_mut(e)
        .insert(CharacterMovement {
            player_controlled: true,
            mode: MovementMode::Driving,
            ..Default::default()
        });
    let seat = inf_physics::d3::vehicle::seat_pose(&town.bridge, target).expect("a seat");
    if let Some(mut cm) = town.world.world_mut().get_mut::<CharacterMovement>(e) {
        cm.runtime.seat = inf_ecs::components::SeatState {
            vehicle: target,
            entering: false,
            time_s: 1.0,
            start: Vec3d::from_dvec3(seat.0),
            start_yaw_deg: 0.0,
        };
        cm.runtime.intent_move = inf_ecs::math::Vec2d::new(0.0, 1.0);
    }
    inf_physics::d3::vehicle::park_collider(&mut town.bridge, HERO, true);
    town.step(2);
    assert!(
        traffic::is_taken(&town.world, target),
        "the traffic kept the car"
    );

    // Hold the throttle. The intent has to be re-asserted each step because
    // `write_driver_back` writes the whole model back and the gate is not
    // running `apply_intent`.
    let from = town
        .world
        .entity_of(target)
        .and_then(|c| town.world.world().get::<Transform>(c))
        .map(|t| t.translation.to_dvec3())
        .expect("a chassis");
    let mut top = 0.0f64;
    for _ in 0..600 {
        if let Some(mut cm) = town.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.intent_move = inf_ecs::math::Vec2d::new(0.0, 1.0);
        }
        town.step(1);
        if let Some(body) = town.bridge.body_of(target) {
            if let (Some(v), Some(r)) = (
                town.bridge.world().body_linvel(body),
                town.bridge.world().body_rotation(body),
            ) {
                top = top.max(v.dot(r * DVec3::Z));
            }
        }
    }
    let to = town
        .world
        .entity_of(target)
        .and_then(|c| town.world.world().get::<Transform>(c))
        .map(|t| t.translation.to_dvec3())
        .expect("a chassis");
    let went = (to - from).length();
    println!("stolen: {went:.1} m, top {top:.2} m/s");
    assert!(
        top > 4.0,
        "the stolen car topped {top:.2} m/s on an empty street"
    );
    assert!(
        went > 25.0,
        "the stolen car covered {went:.1} m in ten seconds"
    );
}
