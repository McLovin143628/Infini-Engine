//! **WAVE VEH3f.2b -- THE DCC CAR SHELLS + THE CARRIED VEHICLE CLOSURES**:
//! the gate.
//!
//! Every arm below reads the WORLD or the committed bytes, never a table's
//! claim about them. The header of each arm says what it READS, and whether
//! VEH3f's panels-on-boxes (or the pre-wave machine) would pass it.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::boarding::BoardPhase;
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    RigidBody3D, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::VehicleDef;
use inf_ecs::EcsWorld;

const RADIUS: f64 = 0.3;
const GROUND: Uuid = Uuid::from_u128(0x5E3F_2B00);
const CHASSIS: Uuid = Uuid::from_u128(0x5E3F_2B01);
const HERO: Uuid = Uuid::from_u128(0x5E3F_2B03);
const SKEL_GUID: Uuid = Uuid::from_u128(0x5E3F_2B07);
const SM_GUID: Uuid = Uuid::from_u128(0x5E3F_2B08);

// ── the fixture (veh3d_gate's, for its reasons) ─────────────────────────────

fn catalogue_def(id: &str) -> VehicleDef {
    *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .or_else(|| inf_ecs::roster::roster().get(id))
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
fn stand(world: &mut EcsWorld, guid: Uuid, name: &str, at: DVec3, yaw_deg: f64) {
    let cm = CharacterMovement {
        player_controlled: true,
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

/// The SHIPPED host (the player's own `RuntimeSim`) with the mannequin rig,
/// one row's car at the origin and the hero beside its driver's flank.
fn rigged_sim(row: &str, hero_at: DVec3) -> inf_player::runtime_sim::RuntimeSim {
    use inf_player::runtime_sim::RuntimeSim;
    const IDLE: inf_anim::ClipRef = [0xd3; 16];
    let def = catalogue_def(row);
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
    stand(&mut world, HERO, "Hero", hero_at, 0.0);
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

/// Where a row's hero is put down: a metre out from the driver's (`+X`)
/// flank, a little behind the driver's cushion.
fn hero_at(def: &VehicleDef) -> DVec3 {
    let parts: &[inf_ecs::vehicle::BodyPart] = match def.art {
        Some(k) if !k.parts().is_empty() => k.parts(),
        _ => def.body.parts(),
    };
    let z = parts
        .iter()
        .filter(|p| p.kind == inf_ecs::vehicle::BodyPartKind::Seat && p.centre.x > 0.0)
        .map(|p| p.centre.z * def.half_extents.z)
        .next()
        .unwrap_or(0.0);
    DVec3::new(def.half_extents.x + 1.1, 0.0, z - 0.6)
}

fn boarding(sim: &inf_player::runtime_sim::RuntimeSim) -> inf_ecs::boarding::BoardingState {
    let e = sim.world().entity_of(HERO).expect("the hero");
    sim.world()
        .world()
        .get::<CharacterMovement>(e)
        .expect("a mover")
        .runtime
        .boarding
}

/// **THE INNER HANDLE: the latch waits for the reach, and the hand holds on
/// to the shut** -- READS the shipped host's boarding machine (the latch's
/// moment, `BoardingState::mark_s`, and the door's hinge angle read back off
/// its joint) and the posed hand (`RuntimeSim::boarding_residuals`: the
/// hand's weight on the pull and the POSED hand joint's distance to it), per
/// `Seated` step, on the five shells and four more rows.
///
/// The claims: (1) the door's motor is told to shut only after the hand's
/// reach has finished (`HAND_REACH_S`) and held (`HANDLE_HOLD_S`) -- the
/// VEH3f audit's "latches on the motor before the reach ramps in"; (2) the
/// door is still open at that moment; (3) once the hand is on the pull at
/// weight 1 it stays on it (<= 2 cm) until the door is shut -- it never lets
/// go early. Pre-wave machine: the motor latched on the step `Seated` began
/// (t = 0, the reach not started) -- FAILS (1). Printed, not asserted: on
/// which rows the pull comes into a seated arm's reach before the shut
/// (the pull 0.25 m ahead of the H-point swings out of reach at the boarding
/// angle; a torso lean is the primitive the pose system lacks -- CARRIED).
#[test]
fn the_inner_latch_waits_for_the_reach_and_the_hand_holds_to_the_shut() {
    use inf_ecs::movement::actions::INTERACT;
    use inf_player::runtime_sim::RuntimeInput;
    println!("=== the inner handle on the shipped host ===");
    let mut bad = Vec::new();
    let mut held_rows = 0usize;
    for row in [
        "sedan", "sports", "suv", "truck", "cruiser", "van", "ambulance", "brute_bus",
        "vapid_contender",
    ] {
        let def = catalogue_def(row);
        let mut sim = rigged_sim(row, hero_at(&def));
        let (mut latch, mut latch_deg) = (None::<f64>, 0.0f64);
        let (mut on, mut off_early, mut worst) = (0usize, 0usize, 0.0f64);
        let mut was_on = false;
        let mut first_on = None::<f64>;
        for i in 0..900u32 {
            let input = if i == 60 {
                RuntimeInput::default().press(INTERACT)
            } else {
                RuntimeInput::default()
            };
            sim.step_once(input);
            let b = boarding(&sim);
            if b.phase == BoardPhase::Seated {
                if latch.is_none() && b.mark_s >= 0.0 {
                    latch = Some(b.mark_s);
                    latch_deg = b.door_deg;
                }
                let r = sim.boarding_residuals(HERO);
                let w = r.as_ref().map(|r| r.sockets.handle_weight).unwrap_or(0.0);
                let shut = b.door_deg <= inf_ecs::boarding::DOOR_SHUT_DEG;
                if w >= 0.999 {
                    on += 1;
                    was_on = true;
                    first_on.get_or_insert(b.time_s);
                    let d = r.and_then(|r| r.handle_m).unwrap_or(f64::INFINITY);
                    worst = worst.max(d);
                } else if was_on && !shut {
                    off_early += 1;
                }
            }
            if b.phase == BoardPhase::Driving {
                break;
            }
        }
        let latch_s = latch.unwrap_or(f64::NAN);
        println!(
            "  {row:<16} latch at {latch_s:.3} s (door {latch_deg:.1} deg); hand on the pull {on} rows from {}, {:.2} mm worst; let go early {off_early}",
            first_on.map(|t| format!("{t:.3} s")).unwrap_or_else(|| "never".into()),
            worst * 1000.0
        );
        let reach = inf_ecs::boarding::HAND_REACH_S + inf_ecs::boarding::HANDLE_HOLD_S;
        if !(latch_s >= reach - 1e-9) {
            bad.push(format!("{row}: the door latched at {latch_s:.3} s, before the reach ({reach:.2} s)"));
        }
        if latch_deg < inf_ecs::boarding::DOOR_BOARD_DEG {
            bad.push(format!("{row}: the door was already swinging ({latch_deg:.1} deg) at the latch"));
        }
        if on > 0 {
            held_rows += 1;
            if worst > 0.02 {
                bad.push(format!("{row}: the posed hand was {:.1} mm off the pull", worst * 1000.0));
            }
            if off_early > 0 {
                bad.push(format!("{row}: the hand let go of the pull {off_early} step(s) before the shut"));
            }
        }
    }
    println!("  rows whose hand held the pull before the shut: {held_rows}");
    assert!(held_rows >= 3, "only {held_rows} row(s) ever held the pull");
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
