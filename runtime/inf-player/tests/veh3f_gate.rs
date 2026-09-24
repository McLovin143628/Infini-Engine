//! **WAVE VEH3f — THE VEHICLE ROSTER.** The gate.
//!
//! (The arm table is written at the end of the wave, once every arm exists.)

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::roster::{self, RosterClass};
use inf_ecs::vehicle::{VehicleControls, VehicleDef};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::RuntimeSim;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;
const CAR: Uuid = Uuid::from_u128(0x5E3F_0001);
const GROUND: Uuid = Uuid::from_u128(0x5E3F_0002);
const HALF: f64 = 3_000.0;

// ── the shipped host, a slab and one car ────────────────────────────────────

fn slab(world: &mut EcsWorld, guid: Uuid, centre: DVec3, half: DVec3, friction: f64) {
    let e = world.spawn_with_guid(guid, "Slab", None);
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
            friction,
            ..Default::default()
        },
    ));
}

fn spawn(world: &mut EcsWorld, guid: Uuid, def: &VehicleDef, at: DVec3, yaw_deg: f64) {
    inf_ecs::vehicle::spawn_rig(
        world,
        guid,
        def,
        &inf_ecs::vehicle::RigSpawn {
            name: "Car".into(),
            at,
            yaw_deg,
            paint: inf_ecs::math::Color::new(0.3, 0.3, 0.7, 1.0),
            clip: None,
            engine_voice: false,
            livery: None,
        },
    );
}

/// A shipped `RuntimeSim` holding one `def` at rest on a 6 km slab, nose `+Z`,
/// at the far `-Z` end so a sprint has room.
fn flat_sim(def: &VehicleDef) -> RuntimeSim {
    let mut world = EcsWorld::new();
    slab(
        &mut world,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(HALF, 0.5, HALF),
        0.9,
    );
    let at = DVec3::new(
        0.0,
        inf_ecs::vehicle::resting_origin_y(def, 0.0),
        -HALF + 60.0,
    );
    spawn(&mut world, CAR, def, at, 0.0);
    world.propagate();
    RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ)
}

fn drive(sim: &mut RuntimeSim, guid: Uuid, c: VehicleControls) {
    if let Some(v) = sim.bridge3d_mut().vehicle_mut(guid) {
        v.control(VehicleControls {
            occupied: true,
            ..c
        });
    }
    sim.step_once(Default::default());
}

fn velocity(sim: &RuntimeSim, guid: Uuid) -> DVec3 {
    let b = sim.bridge3d();
    b.body_of(guid)
        .and_then(|body| b.world().body_linvel(body))
        .unwrap_or(DVec3::ZERO)
}

fn yaw_rate(sim: &RuntimeSim, guid: Uuid) -> f64 {
    let b = sim.bridge3d();
    b.body_of(guid)
        .and_then(|body| b.world().body_angvel(body))
        .map(|w| w.y)
        .unwrap_or(0.0)
}

fn position(sim: &RuntimeSim, guid: Uuid) -> DVec3 {
    let b = sim.bridge3d();
    b.body_of(guid)
        .and_then(|body| b.world().body_translation(body))
        .unwrap_or(DVec3::ZERO)
}

/// The mean share of its travel the car's struts are compressed, standing.
fn static_fraction(sim: &RuntimeSim, guid: Uuid, travel: f64) -> f64 {
    let Some(v) = sim.bridge3d().vehicle_of(guid) else {
        return f64::NAN;
    };
    let rest = v.suspension_rest_m();
    let ws = v.wheels();
    let grounded: Vec<f64> = ws
        .iter()
        .filter(|w| w.contact.is_some())
        .map(|w| rest - w.length_m)
        .collect();
    if grounded.is_empty() || !(travel > 0.0) {
        return f64::NAN;
    }
    grounded.iter().sum::<f64>() / grounded.len() as f64 / travel
}

/// **One row's feel, MEASURED on the shipped host.**
#[derive(Clone, Copy, Debug)]
struct Feel {
    /// Share of travel used standing still.
    static_frac: f64,
    /// The speed the sprint is timed to, m/s.
    sprint_to: f64,
    /// Seconds to reach it (infinity if never).
    sprint_s: f64,
    /// Metres to stop from it.
    brake_m: f64,
    /// Mean deceleration over that stop, g.
    brake_g: f64,
    /// Steady lateral acceleration on full lock at the skidpad speed, g.
    lat_g: f64,
    /// The highest speed seen over the sprint, m/s.
    top: f64,
}

/// The speed a class is timed to: 100 km/h, or 80 % of its own limiter for a
/// row that cannot reach 100 (the VEH2a feel table's rule).
fn sprint_target(def: &VehicleDef) -> f64 {
    (100.0 / 3.6f64).min(0.8 * def.class.max_speed_mps)
}

fn measure(def: &VehicleDef, sprint_cap_s: f64) -> Feel {
    let mut sim = flat_sim(def);
    for _ in 0..120 {
        drive(&mut sim, CAR, VehicleControls::default());
    }
    let static_frac = static_fraction(&sim, CAR, def.class.travel_m);
    let target = sprint_target(def);
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    let (mut sprint_s, mut top) = (f64::INFINITY, 0.0f64);
    let cap = (sprint_cap_s * HZ) as usize;
    for i in 0..cap {
        drive(&mut sim, CAR, full);
        let v = velocity(&sim, CAR);
        let s = DVec3::new(v.x, 0.0, v.z).length();
        top = top.max(s);
        if s >= target {
            sprint_s = (i + 1) as f64 * DT;
            break;
        }
    }
    // The stop, from where the sprint got to.
    let from = position(&sim, CAR);
    let v0 = {
        let v = velocity(&sim, CAR);
        DVec3::new(v.x, 0.0, v.z).length()
    };
    let mut brake_m = f64::INFINITY;
    let mut brake_s = 0.0;
    for i in 0..3_600 {
        drive(
            &mut sim,
            CAR,
            VehicleControls {
                brake: 1.0,
                ..Default::default()
            },
        );
        let v = velocity(&sim, CAR);
        if DVec3::new(v.x, 0.0, v.z).length() < 0.5 {
            brake_m = (position(&sim, CAR) - from).length();
            brake_s = (i + 1) as f64 * DT;
            break;
        }
    }
    let brake_g = if brake_s > 0.0 {
        v0 / brake_s / 9.81
    } else {
        0.0
    };
    // **The RAMP STEER** (the standard handling test): a fresh car brought to
    // 60 % of its sprint target (capped at 15 m/s), then the wheel wound from
    // straight to full lock over four seconds with the throttle HOLDING that
    // speed. The lateral acceleration is `v x yaw rate`, smoothed over a quarter
    // second, and the answer is its PEAK -- the most the tyres gave before the
    // car ploughed or spun. A skidpad at full lock measures the scrub of a
    // front axle at forty degrees, which is a deceleration, not a grip.
    let mut sim = flat_sim(def);
    for _ in 0..120 {
        drive(&mut sim, CAR, VehicleControls::default());
    }
    let pad = (0.6 * target).min(15.0);
    for _ in 0..(40.0 * HZ) as usize {
        drive(&mut sim, CAR, full);
        let v = velocity(&sim, CAR);
        if DVec3::new(v.x, 0.0, v.z).length() >= pad {
            break;
        }
    }
    let ramp = (4.0 * HZ) as usize;
    let mut window: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    let mut peak = 0.0f64;
    for i in 0..ramp {
        let v = velocity(&sim, CAR);
        let s = DVec3::new(v.x, 0.0, v.z).length();
        let throttle = ((pad - s) / 2.0).clamp(0.0, 1.0);
        let brake = ((s - pad) / 4.0).clamp(0.0, 1.0);
        drive(
            &mut sim,
            CAR,
            VehicleControls {
                throttle,
                brake,
                steer: (i + 1) as f64 / ramp as f64,
                ..Default::default()
            },
        );
        let v = velocity(&sim, CAR);
        let s = DVec3::new(v.x, 0.0, v.z).length();
        window.push_back(s * yaw_rate(&sim, CAR).abs());
        if window.len() > (HZ / 4.0) as usize {
            window.pop_front();
        }
        peak = peak.max(window.iter().sum::<f64>() / window.len() as f64);
    }
    let lat_g = peak / 9.81;
    Feel {
        static_frac,
        sprint_to: target,
        sprint_s,
        brake_m,
        brake_g,
        lat_g,
        top,
    }
}

/// **THE ROSTER FEEL SWEEP** -- every wheeled roster row, driven on the shipped
/// host, printed. (Being built: this prints; the per-class bands assert below.)
#[test]
fn the_roster_feel_sweep_prints_every_wheeled_row() {
    let mut rows = 0usize;
    for (id, def) in roster::roster().0.iter() {
        let class = def.roster_class.expect("a roster row has a class");
        if !class.road() || class == RosterClass::Trailer {
            continue;
        }
        let f = measure(def, 60.0);
        rows += 1;
        println!(
            "FEEL {:<12} {:<28} {:>6.0}kg static {:>4.2} to {:>3.0}kmh {:>6.2}s brake {:>6.1}m {:>4.2}g lat {:>4.2}g top {:>5.1}",
            class.name(),
            id,
            def.chassis_mass_kg(),
            f.static_frac,
            f.sprint_to * 3.6,
            f.sprint_s,
            f.brake_m,
            f.brake_g,
            f.lat_g,
            f.top
        );
    }
    // …and the ELEVEN island rows that predate the roster, on the same host.
    for (id, def) in inf_editor_core::vehicle::island_vehicles().0.iter() {
        if !def.body.wheeled() {
            continue;
        }
        let f = measure(def, 60.0);
        println!(
            "FEEL {:<12} {:<28} {:>6.0}kg static {:>4.2} to {:>3.0}kmh {:>6.2}s brake {:>6.1}m {:>4.2}g lat {:>4.2}g top {:>5.1}",
            "island",
            id,
            def.chassis_mass_kg(),
            f.static_frac,
            f.sprint_to * 3.6,
            f.sprint_s,
            f.brake_m,
            f.brake_g,
            f.lat_g,
            f.top
        );
    }
    assert!(rows > 100);
}

#[test]
#[ignore]
fn zz_debug_feel() {
    for id in ["hvy_dozer", "hvy_cutter"] {
        for wi in [10.0, 80.0, 300.0] {
            let mut def = *roster::roster().get(id).unwrap();
            def.class.set("wheel_inertia_kgm2", wi);
            println!("wi {wi}");
            let mut sim = flat_sim(&def);
            for _ in 0..120 {
                drive(&mut sim, CAR, VehicleControls::default());
            }
            for i in 0..300 {
                let steer = 0.0;
                drive(
                    &mut sim,
                    CAR,
                    VehicleControls {
                        throttle: 0.6,
                        steer,
                        ..Default::default()
                    },
                );
                if i % 30 == 0 {
                    let v = velocity(&sim, CAR);
                    let w = sim.bridge3d().vehicle_of(CAR).unwrap().wheels().to_vec();
                    let d = sim
                        .bridge3d()
                        .vehicle_of(CAR)
                        .unwrap()
                        .drivetrain()
                        .unwrap();
                    println!("{id} {i} v {:.2} yaw {:.3} omega {:.1} load {:.0} gear {} rpm {:.0} lock {:.2} slip {:.1} tc {:.2}", DVec3::new(v.x,0.0,v.z).length(), yaw_rate(&sim, CAR), w[2].omega_rad_s, w[0].load_n, d.gear, d.rpm, d.clutch_lock, d.clutch_slip_rad_s, w[2].tc_cut);
                }
            }
        }
    }
}

#[test]
#[ignore]
fn zz_sports_variants() {
    let base = *inf_editor_core::vehicle::island_vehicles()
        .get("sports")
        .unwrap();
    let variants: Vec<(&str, Vec<(&str, f64)>)> = vec![
        ("base", vec![]),
        ("tc.12", vec![("traction_control_slip", 0.12)]),
        ("tc.10", vec![("traction_control_slip", 0.10)]),
        ("g1 3.8", vec![("gear_1_ratio", 3.8)]),
        (
            "g1 3.4 g2 2.5",
            vec![("gear_1_ratio", 3.4), ("gear_2_ratio", 2.5)],
        ),
        (
            "lsd",
            vec![
                ("diff_lock_rear", 0.0),
                ("lsd_preload_rear_nm", 60.0),
                ("lsd_power_ramp_rear", 0.6),
                ("lsd_coast_ramp_rear", 0.35),
            ],
        ),
        ("split.2", vec![("front_torque_split", 0.2)]),
        ("split.3", vec![("front_torque_split", 0.3)]),
        (
            "lsd+g1 3.8+tc.12",
            vec![
                ("diff_lock_rear", 0.0),
                ("lsd_preload_rear_nm", 60.0),
                ("lsd_power_ramp_rear", 0.6),
                ("lsd_coast_ramp_rear", 0.35),
                ("gear_1_ratio", 3.8),
                ("traction_control_slip", 0.12),
            ],
        ),
        ("clutch", vec![("clutch_engage_s", 0.12)]),
        ("shift .08", vec![("shift_time_s", 0.08)]),
    ];
    for (name, keys) in variants {
        let mut d = base;
        for (k, v) in &keys {
            assert!(d.set(k, *v));
        }
        let f = measure(&d, 30.0);
        println!(
            "SPORTS {name:<20} 0-100 {:.2} s brake {:.1} m lat {:.2} g static {:.2}",
            f.sprint_s, f.brake_m, f.lat_g, f.static_frac
        );
    }
}
