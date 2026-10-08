//! **The traffic OBEYS the signals** (wave PAR1b, clause 2) — against a world
//! with real rigs, real drivers and the real step.
//!
//! The unit half (the phase table, the stop-line gap) is in `inf_ecs::traffic`;
//! this file holds the claims that are only true of a world: a steered car
//! approaching a red comes to rest SHORT of the stop line (not the junction
//! centre), holds there, and goes on green; and cross traffic does not meet in
//! the junction box more often with the signals than without them.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, PcgVolume, ResidentSlot, RigidBody3D, SlotRole,
    StreamingSource, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::traffic;
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;

const DT: f64 = 1.0 / 60.0;
const PITCH: f64 = 100.0;
const STREET: f64 = 20.0;
const HERO: Uuid = Uuid::from_u128(0x7100_0001);
const GROUND: Uuid = Uuid::from_u128(0x7100_0002);
const SKY: Uuid = Uuid::from_u128(0x7100_0003);

/// A grid of 80 m blocks on a 100 m pitch with several residents each — the
/// city shape `inf_editor_core::settlement` plans, busy enough for a rush.
fn town(cols: i32, rows: i32, hero_at: DVec3, hour: f64) -> (EcsWorld, PhysicsBridge3D) {
    let mut world = EcsWorld::new();
    let half = (PITCH - STREET) * 0.5;
    for row in 0..rows {
        for col in 0..cols {
            let c = DVec2::new(f64::from(col) * PITCH, f64::from(row) * PITCH);
            let guid = Uuid::from_u64_pair(0x52, (row as u64) << 32 | col as u64);
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
            v.residents = (0..6)
                .map(|k| ResidentSlot {
                    role: if k % 2 == 0 {
                        SlotRole::Home
                    } else {
                        SlotRole::Work
                    },
                    at: DVec3::new(c.x + f64::from(k) * 3.0, 0.0, c.y),
                    room: k,
                    building: 0,
                    floor: 0,
                    index: k,
                    node: 0,
                    posture: inf_ecs::components::SlotPosture::Stand,
                    shift: inf_ecs::components::SlotShift::Day,
                    face: DVec3::ZERO,
                })
                .collect();
            world.world_mut().entity_mut(e).insert(v);
        }
    }
    let g = world.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(100.0, -0.5, 100.0);
    world.world_mut().entity_mut(g).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(600.0, 0.5, 600.0),
            ..Default::default()
        },
    ));
    let h = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(h).insert((
        StreamingSource { radius_m: 512.0 },
        Transform::from_translation(hero_at),
    ));
    let s = world.spawn_with_guid(SKY, "Sky", None);
    world
        .world_mut()
        .entity_mut(s)
        .insert(inf_ecs::components::TimeOfDay {
            seconds: hour * 3600.0,
            rate: 0.0,
            ..Default::default()
        });
    world.propagate();
    (world, PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)))
}

fn step(world: &mut EcsWorld, bridge: &mut PhysicsBridge3D, signals_off: bool) {
    inf_physics::d3::traffic::step_traffic(world, bridge, DT);
    if signals_off {
        if let Some(mut r) = world.world_mut().get_resource_mut::<traffic::TrafficRes>() {
            r.junctions.clear();
        }
    }
    bridge.sync_from_world_sim(world, &Default::default(), &Default::default());
    inf_physics::d3::step_character_movement(world, bridge, DT);
    inf_physics::d3::step_vehicles(world, bridge, DT);
    bridge.step(DT);
    bridge.write_back_into(world);
    world.propagate();
}

/// What one run measured.
#[derive(Debug, Default)]
struct Run {
    /// Distinct `(car, junction)` waits: a steered car at rest within 4 m of
    /// its stop line, the line ahead of its nose, its approach red or amber.
    waits: std::collections::BTreeSet<(Uuid, i64, i64)>,
    /// Of those, how many later drove on through the junction.
    went_on: usize,
    /// Steps on which two steered cars' footprints overlapped inside a
    /// junction box (the crossing street's kerb on each side).
    box_contacts: usize,
    /// The worst nose-past-the-line, metres, of a car at rest on a red.
    worst_overrun_m: f64,
    /// Steered cars observed driving, summed over steps.
    driving_steps: usize,
}

fn run(signals_off: bool, steps: usize) -> Run {
    let (mut world, mut bridge) = town(4, 4, DVec3::new(150.0, 0.0, 150.0), 8.4);
    let mut out = Run::default();
    let mut waiting: std::collections::BTreeMap<Uuid, (i64, i64)> = Default::default();
    for _ in 0..steps {
        step(&mut world, &mut bridge, signals_off);
        let Some(res) = traffic::carriageway_of(&world) else {
            continue;
        };
        let junctions = traffic::signal_junctions(&res.streets);
        let t_s = traffic::signal_clock_of(&world);
        let recs = inf_physics::d3::traffic::records(&world);
        // Footprints of every steered car this step.
        let mut boxes: Vec<(Uuid, DVec3, DVec3, f64, f64)> = Vec::new();
        for (guid, rec) in &recs {
            if rec.tier != inf_ecs::crowd::CrowdTier::Full {
                continue;
            }
            let Some(e) = world.entity_of(*guid) else {
                continue;
            };
            let Some(t) = world.world().get::<Transform>(e) else {
                continue;
            };
            let at = t.translation.to_dvec3();
            let yaw = t.rotation.y.to_radians();
            let fwd = DVec3::new(inf_math::psin64(yaw), 0.0, inf_math::pcos64(yaw));
            boxes.push((
                *guid,
                at,
                fwd,
                rec.def.half_extents.x.abs(),
                rec.def.half_extents.z.abs(),
            ));
            let Some(v) = bridge
                .body_of(*guid)
                .and_then(|b| bridge.world().body_linvel(b))
            else {
                continue;
            };
            let speed = DVec3::new(v.x, 0.0, v.z).length();
            if speed > 0.1 {
                out.driving_steps += 1;
            }
            // A car at rest just short of a junction it is approaching.
            for j in &junctions {
                let along_x = fwd.x.abs() >= fwd.z.abs();
                let (d_along, d_lat) = if along_x {
                    (
                        (j.centre.x - at.x) * fwd.x.signum(),
                        (j.centre.y - at.z).abs(),
                    )
                } else {
                    (
                        (j.centre.y - at.z) * fwd.z.signum(),
                        (j.centre.x - at.x).abs(),
                    )
                };
                if d_lat > traffic::SIGNAL_LANE_REACH_M {
                    continue;
                }
                let nose_to_line = d_along - rec.def.half_extents.z.abs() - traffic::STOP_LINE_M;
                let key = ((j.centre.x * 10.0) as i64, (j.centre.y * 10.0) as i64);
                if speed < traffic::STOPPED_MPS && (-1.0..=4.0).contains(&nose_to_line) {
                    let aspect = traffic::signal_aspect(j, along_x, t_s);
                    if aspect != traffic::Aspect::Green {
                        out.waits.insert((*guid, key.0, key.1));
                        waiting.insert(*guid, key);
                        out.worst_overrun_m = out.worst_overrun_m.max(-nose_to_line);
                    }
                }
                // A car that was waiting here and is now inside the box.
                if waiting.get(guid) == Some(&key) && d_along < traffic::STOP_LINE_M - 2.0 {
                    waiting.remove(guid);
                    out.went_on += 1;
                }
            }
        }
        // Junction-box contacts between steered cars.
        for j in &junctions {
            let hx = inf_ecs::traffic::street_kerb_offset_m(j.gap_z);
            let hz = inf_ecs::traffic::street_kerb_offset_m(j.gap_x);
            let inside: Vec<&(Uuid, DVec3, DVec3, f64, f64)> = boxes
                .iter()
                .filter(|b| (b.1.x - j.centre.x).abs() < hx && (b.1.z - j.centre.y).abs() < hz)
                .collect();
            let mut touched = false;
            for a in 0..inside.len() {
                for b in a + 1..inside.len() {
                    let (p, q) = (inside[a], inside[b]);
                    // Bounding circles of the two footprints, less a hand's
                    // clearance: a real overlap of two car bodies.
                    let rp = (p.3 * p.3 + p.4 * p.4).sqrt();
                    let rq = (q.3 * q.3 + q.4 * q.4).sqrt();
                    let d = DVec3::new(p.1.x - q.1.x, 0.0, p.1.z - q.1.z).length();
                    if d < 0.75 * (rp + rq) {
                        touched = true;
                    }
                }
            }
            out.box_contacts += usize::from(touched);
        }
    }
    out
}

/// **A SIGNAL CARS OBEY** (wave PAR1b clause 2). Over 2 400 steps (40 s, most
/// of a cycle) of the morning rush on a 4 x 4 city grid whose nine interior
/// crossings are signalised:
///
/// * steered cars come to REST at red stop lines — counted as distinct
///   `(car, junction)` waits, the nose between 1 m past and 4 m short of the
///   line (not the junction centre, which is `STOP_LINE_M` further on), and at
///   least one of them goes on into the box afterwards (it held for the red
///   and left on the green);
/// * cross traffic meets in the junction box no more often with the signals
///   than with them switched off (the same run with `TrafficRes::junctions`
///   emptied every step — the mutation the arm is built around).
///
/// Mutation (measured): the stop-line gap removed from `view_of` -> waits 0,
/// red.
#[test]
fn a_traffic_car_waits_at_a_red_and_goes_on_green_and_the_box_stays_clear() {
    let on = run(false, 2400);
    let off = run(true, 2400);
    println!(
        "PAR1b SIGNALS: on: {} waits at red, {} went on, worst overrun {:.2} m, {} box-contact steps, {} driving car-steps | off: {} waits, {} box-contact steps, {} driving car-steps",
        on.waits.len(),
        on.went_on,
        on.worst_overrun_m,
        on.box_contacts,
        on.driving_steps,
        off.waits.len(),
        off.box_contacts,
        off.driving_steps
    );
    assert!(on.driving_steps > 0, "nothing drove");
    assert!(!on.waits.is_empty(), "no steered car ever waited at a red");
    assert!(on.went_on > 0, "a car waited at a red and never went on");
    assert!(
        on.worst_overrun_m <= 1.0,
        "a car stopped {:.2} m past its line",
        on.worst_overrun_m
    );
    assert!(
        on.box_contacts <= off.box_contacts,
        "junction-box contacts rose with the signals: {} vs {}",
        on.box_contacts,
        off.box_contacts
    );
}
