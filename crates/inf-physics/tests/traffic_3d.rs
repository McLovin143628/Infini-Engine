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
    // Midnight: everything is parked and nothing has a driver.
    town.set_hour(0.0);
    let night = town.step(60);
    assert_eq!(night.driving, 0, "somebody is driving at midnight");
    assert_eq!(night.drivers, 0);

    // Half past eight: the commuters are out.
    town.set_hour(8.5);
    let rush = town.step(90);
    assert!(rush.driving > 0, "nobody drives at half past eight");
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
    let mut resisted = 0;
    let mut ejected = None;
    for _ in 0..12 {
        match inf_physics::d3::carjack::try_carjack(
            &mut town.world,
            &mut town.bridge,
            chassis,
            HERO,
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
    assert!(
        resisted > 0,
        "the resist draw never fired in twelve attempts, so its branch is dead"
    );

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
    // Midnight: every car is parked, so nothing at all is carjackable, anywhere.
    let near = inf_physics::d3::traffic::cars_near(&town.world, DVec3::new(50.0, 0.0, 50.0), 60.0);
    let (target, _) = near[0];
    let seat = inf_physics::d3::vehicle::seat_pose(&town.bridge, target).expect("a seat");
    let feet = seat.0 + (seat.1 * DVec3::X) * 2.0;
    assert!(inf_physics::d3::carjack::carjackable(&town.world, &town.bridge, feet).is_empty());
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
    assert!(inf_physics::d3::carjack::carjackable(&town.world, &town.bridge, feet).is_empty());
}
