//! **WAVE VEH3g — AIR + SEA CLASSES.** The gate.
//!
//! Fixed-wing flight at last (`inf_ecs::aero`), the roster's helicopters with a
//! winch and a police air lane, the marine rows with planing hulls and a sail
//! (`inf_ecs::marine`), the rotor / propeller / turbine / hull voices, and the
//! Harbour City airfield. Every arm reads the TRACE -- an altitude over time, a
//! lift coefficient in the force log, a draught off the hull's world position, a
//! cable's tension, a sail's force against the wind VECTOR -- never a class name
//! or the value the code asked for. Mutations are recorded in the wave's report
//! (`campaign-briefs/veh3g-implementer-report.md`), each run by hand and reverted.
//!
//! # What every arm reads, and whether VEH2c's craft would pass it
//!
//! "VEH2c" = the tree before this wave: `RotorVehicle` + `HullVehicle`, no wing,
//! no winch, no sail, no planing, no air lane, no strip.
//!
//! | arm | reads | VEH2c? |
//! |---|---|---|
//! | `the_dodo_takes_off_climbs_and_lands_on_a_strip` | the chassis's altitude trace, its wheels' contacts, the vertical speed at the first contact after flight, the roll-out's end | **fails** -- a wheeled chassis never left the ground |
//! | `the_keyboard_pilot_rotates_with_space` | the Dodo's height over the strip flown with the handbrake on the elevator's key, as the shipped map binds them | **fails** -- no lift-off |
//! | `a_wing_with_no_area_never_leaves_the_ground` | the same Dodo at full power with `wing_area_m2` 0: its peak altitude | passes (the control) |
//! | `every_aeroplane_lifts_off_inside_the_islands_runway` | each of the five rows' ground roll to the LAST wheel contact before 35 ft, against the recipe's own runway length; the distance to 35 ft printed | **fails** -- no lift-off |
//! | `the_dodo_floats_and_flies_off_the_water` | the floatplane's hull bottom against the surface at rest (wheels touching: none), its settle, the water run, its height a minute later | **fails** -- it floats (VEH2c's buoyancy) and never leaves the water |
//! | `the_stall_collapses_the_lift_and_drops_the_nose` | the force log's alpha and CL -- the highest CL 3..8 deg past the stall against the peak -- the pitch and the altitude after full back stick | **fails** -- no CL at all |
//! | `five_helicopters_lift_off_hover_and_translate` | each row's climb, its vertical speed held, its ground track | passes (the rows flew on the rotorcraft family) |
//! | `the_cargobob_winch_lifts_a_car_and_lets_it_go` | the car's altitude against the lifter's, the rope's limit impulse as tension, the fall after release | **fails** -- no winch |
//! | `the_police_helicopter_flies_its_air_lane_and_never_the_road` | the unit's run `path` every step, its track's height and straightness, its arrival | **fails** -- no air unit |
//! | `a_planing_hull_rises_and_a_displacement_hull_does_not` | each motor hull's draught off its world position against its forward speed | **fails** -- the draught never moves |
//! | `the_sail_drives_across_the_wind_and_not_into_it` | the yacht's forward speed at five headings (head to wind, 20, 45, beam, run), and again with the wind VECTOR turned | **fails** -- no sail |
//! | `the_tug_pushes_the_superyacht` | the yacht's velocity with the tug's bow on its transom | passes (a screw pushes) |
//! | `the_big_hulls_float_where_archimedes_puts_them` | the three ships' rest draught against `rho_b / rho_f x 2h` | passes (the Box branch is exact) |
//! | `the_titan_ramp_swings_down_on_its_hinge` | the ramp part's rear edge in the chassis frame, closed and open | **fails** -- no ramp kind |
//! | `pie_equals_shipping_for_a_plane_a_helicopter_and_a_boat` | every craft chassis pose, bit for bit, in `RuntimeSim` and `SimSession` | n/a -- a determinism arm |
//! | `the_craft_voices_reach_the_command_stream_on_both_hosts` | the Plays of the five craft clips, the rotor's pitch against its blade-pass, `dropped == 0`, the two streams equal | **fails** -- craft sang VEH1a's one loop |
//! | `eighteen_air_and_sea_craft_cost_what_they_print` | the vehicle phase's clock, 18 craft against 18 of the roster's sedans | n/a -- COST, release-only |
//! | `no_std_trig_on_the_air_and_sea_paths` | `aero.rs` and `marine.rs` source against the libm ban list | passes (the files did not exist) |
//! | `this_wave_moved_no_schema` | scene / payload versions, the tunable count | passes |
//! | `the_shipped_host_draws_the_craft_row_and_logs_the_columns` | `window.rs` / `pie_drive.rs` source and the Ring-0 row | n/a |
//! | `the_real_islands_runway_is_flat_and_the_dodo_leaves_it` | LOCAL: the built island's terrain under the paving and a take-off off the committed level's paving | skips on CI |

use std::collections::BTreeMap;

use glam::{DQuat, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::roster;
use inf_ecs::vehicle::{VehicleControls, VehicleDef};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::RuntimeSim;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;
const CRAFT: Uuid = Uuid::from_u128(0x5E3E_0001);
const GROUND: Uuid = Uuid::from_u128(0x5E3E_0002);
const LOAD: Uuid = Uuid::from_u128(0x5E3E_0020);
const PUSHER: Uuid = Uuid::from_u128(0x5E3E_0021);
const SEA: Uuid = Uuid::from_u128(0x5E3E_0010);
const SKY: Uuid = Uuid::from_u128(0x5E3E_0011);
/// The flight lab's slab: 8 km long so a circuit has somewhere to land.
const STRIP_HALF_L: f64 = 4_000.0;
const STRIP_HALF_W: f64 = 400.0;

// ── the labs ─────────────────────────────────────────────────────────────────

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

fn slab(world: &mut EcsWorld, guid: Uuid, centre: DVec3, half: DVec3, friction: f64) {
    let e = world.spawn_with_guid(guid, "Slab", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(slab_bits(centre, half, friction));
}

fn spawn_spec(name: &str, at: DVec3, yaw_deg: f64, voice: bool) -> inf_ecs::vehicle::RigSpawn {
    inf_ecs::vehicle::RigSpawn {
        name: name.into(),
        at,
        yaw_deg,
        paint: inf_ecs::math::Color::new(0.8, 0.8, 0.85, 1.0),
        clip: None,
        engine_voice: voice,
        livery: None,
    }
}

fn row(id: &str) -> VehicleDef {
    *roster::roster()
        .get(id)
        .unwrap_or_else(|| panic!("no roster row {id}"))
}

/// A shipped host holding `def` at rest at the lab strip's `-Z` end, nose `+Z`.
fn strip_sim(def: &VehicleDef) -> RuntimeSim {
    let mut world = EcsWorld::new();
    slab(
        &mut world,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(STRIP_HALF_W, 0.5, STRIP_HALF_L),
        0.9,
    );
    let at = DVec3::new(
        0.0,
        inf_ecs::vehicle::resting_origin_y(def, 0.0),
        -STRIP_HALF_L + 80.0,
    );
    inf_ecs::vehicle::spawn_rig(&mut world, CRAFT, def, &spawn_spec("Craft", at, 0.0, false));
    world.propagate();
    RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ)
}

fn sea_bits() -> (Transform, inf_ecs::components::WaterBody) {
    (
        Transform::IDENTITY,
        inf_ecs::components::WaterBody {
            wave_amplitude_m: 0.0,
            ..inf_ecs::components::WaterBody::lake(0.0, inf_ecs::math::Vec2d::new(8_000.0, 8_000.0))
        },
    )
}

fn sky_bits(
    wind: (f32, f32),
) -> (
    inf_ecs::components::TimeOfDay,
    inf_ecs::components::SkyAtmosphere,
) {
    (
        inf_ecs::components::TimeOfDay::default(),
        inf_ecs::components::SkyAtmosphere {
            weather_enabled: false,
            cloud_wind_x: wind.0,
            cloud_wind_z: wind.1,
            ..Default::default()
        },
    )
}

fn sea_world(wind: (f32, f32)) -> EcsWorld {
    let mut world = EcsWorld::new();
    let sea = world.spawn_with_guid(SEA, "Sea", None);
    world.world_mut().entity_mut(sea).insert(sea_bits());
    let sky = world.spawn_with_guid(SKY, "Sky", None);
    world.world_mut().entity_mut(sky).insert(sky_bits(wind));
    world
}

/// A flat 8 km lake at y = 0, a sky whose wind is `wind`, and `def` floating at
/// rest at the origin heading `yaw_deg`.
fn sea_sim(def: &VehicleDef, wind: (f32, f32), yaw_deg: f64) -> RuntimeSim {
    let mut world = sea_world(wind);
    let at = DVec3::new(0.0, inf_ecs::vehicle::floating_origin_y(def, 0.0), 0.0);
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        CRAFT,
        def,
        &spawn_spec("Boat", at, yaw_deg, false),
    );
    world.propagate();
    RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ)
}

fn body_state(sim: &RuntimeSim, guid: Uuid) -> (DVec3, DQuat, DVec3) {
    let b = sim.bridge3d();
    let body = b.body_of(guid).expect("a body");
    let w = b.world();
    (
        w.body_translation(body).unwrap_or(DVec3::ZERO),
        w.body_rotation(body).unwrap_or(DQuat::IDENTITY),
        w.body_linvel(body).unwrap_or(DVec3::ZERO),
    )
}

fn grounded(sim: &RuntimeSim, guid: Uuid) -> usize {
    sim.bridge3d()
        .vehicle_of(guid)
        .map(|v| v.wheels().iter().filter(|w| w.contact.is_some()).count())
        .unwrap_or(0)
}

fn flight(sim: &RuntimeSim, guid: Uuid) -> inf_ecs::aero::FlightState {
    sim.bridge3d()
        .vehicle_of(guid)
        .and_then(|v| v.flight())
        .unwrap_or_default()
}

fn command(sim: &mut RuntimeSim, guid: Uuid, c: VehicleControls) {
    if let Some(v) = sim.bridge3d_mut().vehicle_mut(guid) {
        v.control(VehicleControls {
            occupied: true,
            ..c
        });
    }
    sim.step_once(Default::default());
}

/// Pitch attitude, degrees, nose-up positive.
fn pitch_deg(rot: DQuat) -> f64 {
    let f = rot * DVec3::Z;
    inf_math::patan2_64(f.y, (f.x * f.x + f.z * f.z).sqrt()).to_degrees()
}

/// The hull's draught off its WORLD position: the water (y = 0) minus the
/// chassis box's bottom.
fn draught(sim: &RuntimeSim, def: &VehicleDef) -> f64 {
    let (p, _, _) = body_state(sim, CRAFT);
    -(p.y - def.half_extents.y)
}

fn forward_speed(sim: &RuntimeSim, guid: Uuid) -> f64 {
    let (_, r, v) = body_state(sim, guid);
    (r * DVec3::Z).dot(v)
}

/// Write a trace as CSV into `VEH3G_TRACE_DIR` when it is set (the report's
/// plots are drawn from these); a no-op otherwise.
fn write_csv(name: &str, header: &str, rows: &[String]) {
    let Some(dir) = std::env::var_os("VEH3G_TRACE_DIR") else {
        return;
    };
    let mut out = String::from(header);
    out.push('\n');
    for r in rows {
        out.push_str(r);
        out.push('\n');
    }
    let _ = std::fs::create_dir_all(&dir);
    let _ = std::fs::write(std::path::Path::new(&dir).join(format!("{name}.csv")), out);
}

// ── the pilot ────────────────────────────────────────────────────────────────

/// One row of a flight trace.
#[derive(Clone, Copy, Debug)]
struct Sample {
    t: f64,
    z: f64,
    alt: f64,
    vs: f64,
    ias: f64,
    alpha: f64,
    cl: f64,
    pitch: f64,
    thrust: f64,
    lift: f64,
    grounded: usize,
    stalled: bool,
}

fn sample(sim: &RuntimeSim, guid: Uuid, t: f64, rest_y: f64) -> Sample {
    let (p, r, v) = body_state(sim, guid);
    let f = flight(sim, guid);
    Sample {
        t,
        z: p.z,
        alt: p.y - rest_y,
        vs: v.y,
        ias: f.airspeed_mps,
        alpha: f.alpha_deg,
        cl: f.cl,
        pitch: pitch_deg(r),
        thrust: f.thrust_n,
        lift: f.lift_n,
        grounded: grounded(sim, guid),
        stalled: f.stalled,
    }
}

fn trace_rows(trace: &[Sample]) -> Vec<String> {
    trace
        .iter()
        .map(|s| {
            format!(
                "{:.4},{:.3},{:.3},{:.3},{:.3},{:.3},{:.4},{:.3},{:.1},{:.1},{},{}",
                s.t,
                s.z,
                s.alt,
                s.vs,
                s.ias,
                s.alpha,
                s.cl,
                s.pitch,
                s.thrust,
                s.lift,
                s.grounded,
                u8::from(s.stalled)
            )
        })
        .collect()
}

const TRACE_HEADER: &str = "t,z,alt,vs,ias,alpha,cl,pitch,thrust,lift,grounded,stalled";

/// The pilot's elevator: a pitch-attitude hold.
fn pitch_hold(target_deg: f64, pitch: f64) -> f64 {
    (0.25 + (target_deg - pitch) * 0.12).clamp(-1.0, 1.0)
}

/// **A circuit**: full power, rotate at `vr`, climb at `climb_pitch` to
/// `cruise_alt`, level off for `level_s`, then a stabilised descent, a flare
/// and a landing, braking to a stop.
struct Circuit {
    vr: f64,
    climb_pitch: f64,
    cruise_alt: f64,
    level_s: f64,
}

/// **The screen height**, metres: 35 ft, the obstacle the certified TAKE-OFF
/// DISTANCE is measured to (the ground roll is only its first part).
const SCREEN_HEIGHT_M: f64 = 10.7;

/// **The attitude that flies at `vr`**, degrees of pitch: the angle of attack
/// whose lift coefficient carries the weight at `vr` (CL max / (vr/Vs)^2 on the
/// wing's own lift slope), less the wing's incidence -- what a pilot rotates
/// to. A fixed 8 degrees under-rotated the heavies by five to seven degrees and
/// left them skimming for kilometres (measured: the Luxor reached 35 ft
/// 2 849 m from brake release at 8 degrees, 1 827 m at its rotate attitude).
fn rotate_attitude_deg(def: &VehicleDef, vr_over_vs: f64) -> f64 {
    let wing = inf_ecs::aero::Wing::of(&def.class.to_tuning()).expect("a wing");
    (wing.cl_max() / (vr_over_vs * vr_over_vs) / wing.lift_slope()).to_degrees()
        - inf_ecs::aero::WING_INCIDENCE_DEG
}

#[derive(Debug, Default)]
struct CircuitReport {
    /// The ground roll to the LAST wheel contact before the screen height --
    /// a hop at rotation that comes back down does not end the roll.
    liftoff_roll_m: Option<f64>,
    /// Brake release to 35 ft ([`SCREEN_HEIGHT_M`]), metres.
    screen_m: Option<f64>,
    liftoff_ias: f64,
    peak_alt: f64,
    climb_rate: f64,
    touchdown_vs: Option<f64>,
    rollout_m: Option<f64>,
    trace: Vec<Sample>,
}

fn fly_circuit(mut sim: RuntimeSim, plan: &Circuit, max_s: f64) -> CircuitReport {
    for _ in 0..90 {
        sim.step_once(Default::default());
    }
    let (p0, _, _) = body_state(&sim, CRAFT);
    let (rest_y, start_z) = (p0.y, p0.z);
    let mut rep = CircuitReport::default();
    let mut phase = 0u8; // 0 roll, 1 climb, 2 level, 3 descend, 4 roll-out
    let mut level_t = 0.0;
    let mut touch_z = 0.0;
    let mut prev_vs = 0.0;
    for i in 0..(max_s * HZ) as usize {
        let s = sample(&sim, CRAFT, i as f64 * DT, rest_y);
        let mut c = VehicleControls::default();
        match phase {
            0 => {
                c.throttle = 1.0;
                if s.ias >= plan.vr {
                    c.vertical = pitch_hold(plan.climb_pitch, s.pitch);
                }
                if s.grounded == 0 {
                    rep.liftoff_roll_m = Some(s.z - start_z);
                    rep.liftoff_ias = s.ias;
                    phase = 1;
                }
            }
            1 => {
                c.throttle = 1.0;
                c.vertical = pitch_hold(plan.climb_pitch, s.pitch);
                if s.grounded > 0 && rep.screen_m.is_none() {
                    // It came back down: the roll was not over.
                    rep.liftoff_roll_m = Some(s.z - start_z);
                    rep.liftoff_ias = s.ias;
                }
                if s.alt >= SCREEN_HEIGHT_M && rep.screen_m.is_none() {
                    rep.screen_m = Some(s.z - start_z);
                }
                if s.alt >= plan.cruise_alt {
                    phase = 2;
                }
            }
            2 => {
                c.throttle = 0.7;
                let want = ((plan.cruise_alt - s.alt) * 0.4 - s.vs * 1.5).clamp(-4.0, 6.0);
                c.vertical = pitch_hold(want, s.pitch);
                level_t += DT;
                if level_t >= plan.level_s {
                    phase = 3;
                }
            }
            3 => {
                let sink = if s.alt > 8.0 { -3.0 } else { -0.8 };
                c.throttle = if s.ias > plan.vr * 1.05 { 0.25 } else { 0.55 };
                let want = ((sink - s.vs) * 2.0).clamp(-6.0, 10.0);
                c.vertical = pitch_hold(want, s.pitch);
                if s.grounded > 0 {
                    rep.touchdown_vs = Some(prev_vs);
                    touch_z = s.z;
                    phase = 4;
                }
            }
            _ => {
                c.brake = 1.0;
                if s.ias < 0.3 && rep.rollout_m.is_none() {
                    rep.rollout_m = Some(s.z - touch_z);
                }
            }
        }
        rep.peak_alt = rep.peak_alt.max(s.alt);
        if phase == 1 && s.alt > 5.0 {
            rep.climb_rate = rep.climb_rate.max(s.vs);
        }
        prev_vs = s.vs;
        rep.trace.push(s);
        command(&mut sim, CRAFT, c);
        if rep.rollout_m.is_some() {
            break;
        }
    }
    rep
}

fn stall_speed(def: &VehicleDef) -> f64 {
    let wing = inf_ecs::aero::Wing::of(&def.class.to_tuning()).expect("a wing");
    wing.stall_speed_mps(def.chassis_mass_kg() * 9.81)
}

/// The vertical speed a light aeroplane's gear is rated to arrive at, m/s -- a
/// firm landing; over it is a heavy one.
const TOUCHDOWN_CEILING_MPS: f64 = 2.5;

// ── 1. THE FIXED WING ────────────────────────────────────────────────────────

/// **The Dodo takes off from a strip, climbs, flies a circuit and lands** --
/// read off its own altitude trace.
///
/// The roll is measured to the first step no wheel has a contact; the climb is
/// the vertical speed on the climb-out; the landing is the vertical speed at the
/// first contact after flight, under [`TOUCHDOWN_CEILING_MPS`], and a roll-out
/// that stops under braking.
///
/// **Mutation → red**: `wing_area_m2` 0 on the row (it never lifts off -- the
/// next arm IS that mutation, kept as a control).
#[test]
fn the_dodo_takes_off_climbs_and_lands_on_a_strip() {
    let def = row("mammoth_dodo");
    let vs = stall_speed(&def);
    let rep = fly_circuit(
        strip_sim(&def),
        &Circuit {
            vr: vs * 1.1,
            climb_pitch: rotate_attitude_deg(&def, 1.1),
            cruise_alt: 40.0,
            level_s: 10.0,
        },
        200.0,
    );
    write_csv("dodo_circuit", TRACE_HEADER, &trace_rows(&rep.trace));
    println!(
        "THE DODO: stall {vs:.1} m/s; lift-off after {:.1} m at {:.1} m/s; climb {:.2} m/s; \
         peak {:.1} m; touchdown at {:.2} m/s down; roll-out {:.1} m; {} samples",
        rep.liftoff_roll_m.unwrap_or(f64::NAN),
        rep.liftoff_ias,
        rep.climb_rate,
        rep.peak_alt,
        -rep.touchdown_vs.unwrap_or(f64::NAN),
        rep.rollout_m.unwrap_or(f64::NAN),
        rep.trace.len()
    );
    let roll = rep.liftoff_roll_m.expect("the Dodo never lifted off");
    assert!(
        roll > 50.0 && roll < 600.0,
        "a {roll:.0} m roll is not a light aeroplane's"
    );
    assert!(rep.peak_alt >= 39.0, "it climbed to {:.1} m", rep.peak_alt);
    assert!(rep.climb_rate > 1.5, "a climb of {:.2} m/s", rep.climb_rate);
    let td = rep.touchdown_vs.expect("it never came back down");
    assert!(
        -td <= TOUCHDOWN_CEILING_MPS,
        "it arrived at {:.2} m/s down, over the {TOUCHDOWN_CEILING_MPS} m/s gear rating",
        -td
    );
    let out = rep.rollout_m.expect("it never stopped");
    assert!(out > 0.0 && out < 800.0, "a {out:.0} m roll-out");
    assert!(
        rep.trace.len() > 3_000,
        "the circuit is {} samples",
        rep.trace.len()
    );
}

/// **THE KEYBOARD PILOT ROTATES WITH SPACE** -- the shipped input map puts
/// `move_up` (the elevator) and `handbrake` on the same key, so a player who
/// pulls back to rotate commands the handbrake with it. The Dodo is flown the
/// way the keyboard flies it: full throttle, then from Vr the stick on its
/// stop WITH the handbrake, and the arm reads the height over the runway.
///
/// **Mutation → red**: the wing's "the elevator is not the parking brake" line
/// removed (the gear locks at rotation).
#[test]
fn the_keyboard_pilot_rotates_with_space() {
    let def = row("mammoth_dodo");
    let vr = stall_speed(&def) * 1.1;
    let mut sim = strip_sim(&def);
    for _ in 0..90 {
        sim.step_once(Default::default());
    }
    let (p0, _, _) = body_state(&sim, CRAFT);
    let mut screen: Option<f64> = None;
    let mut rotating = false;
    for _ in 0..(60.0 * HZ) as usize {
        let (p, _, _) = body_state(&sim, CRAFT);
        if screen.is_none() && p.y - p0.y >= SCREEN_HEIGHT_M {
            screen = Some(p.z - p0.z);
        }
        rotating |= flight(&sim, CRAFT).airspeed_mps >= vr;
        let c = VehicleControls {
            throttle: 1.0,
            vertical: if rotating { 1.0 } else { 0.0 },
            handbrake: rotating,
            ..Default::default()
        };
        command(&mut sim, CRAFT, c);
        if screen.is_some() {
            break;
        }
    }
    println!(
        "THE KEYBOARD PILOT: Space held from Vr ({vr:.1} m/s); 35 ft after {} m",
        screen.map_or("never".to_string(), |d| format!("{d:.0}"))
    );
    let d = screen.expect("the keyboard pilot never left the ground");
    assert!(d < 1_700.0, "35 ft only after {d:.0} m");
}

/// **The recogniser IS the wing area**: the same Dodo with `wing_area_m2` 0 is a
/// car with a propeller it cannot use -- full power and full back stick for a
/// minute and it stays on the ground.
#[test]
fn a_wing_with_no_area_never_leaves_the_ground() {
    let mut def = row("mammoth_dodo");
    def.class.wing_area_m2 = 0.0;
    let mut sim = strip_sim(&def);
    for _ in 0..90 {
        sim.step_once(Default::default());
    }
    let (p0, _, _) = body_state(&sim, CRAFT);
    let mut peak = 0.0f64;
    for _ in 0..3_600 {
        command(
            &mut sim,
            CRAFT,
            VehicleControls {
                throttle: 1.0,
                vertical: 1.0,
                ..Default::default()
            },
        );
        peak = peak.max(body_state(&sim, CRAFT).0.y - p0.y);
    }
    let travelled = body_state(&sim, CRAFT).0.z - p0.z;
    let flying = sim
        .bridge3d()
        .vehicle_of(CRAFT)
        .and_then(|v| v.flight())
        .is_some();
    println!(
        "NO WING: 60 s at full power moved it {travelled:.1} m and lifted it {peak:.3} m; \
         a flight state: {flying}"
    );
    assert!(peak < 0.5, "a wingless Dodo rose {peak:.2} m");
    assert!(!flying, "a class with no wing area has a flight state");
}

/// **Every aeroplane lifts off inside the island's runway** -- the five rows'
/// rolls on the flat lab, against the committed recipe's own runway length.
///
/// The ROLL is to the last wheel contact before 35 ft (the first cut ended it
/// at the first step with no contact, which a rotation hop satisfies: the
/// heavies' "roll" read 20-80 m short). The distance to 35 ft -- what a
/// certified take-off distance is -- is printed beside it and NOT asserted
/// against the runway: the Luxor (1 827 m, against the Global 7500's published
/// 1 768 m) and the Jetliner reach it past the far threshold. Carried.
///
/// **Mutation → red**: the Titan's `max_engine_force_n` back to VEH3f's
/// 160 kN (measured in the mutation table of the VEH3g report).
#[test]
fn every_aeroplane_lifts_off_inside_the_islands_runway() {
    let recipe = inf_island::IslandRecipe::load(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../samples/island/island.toml"
    )))
    .expect("the island recipe loads");
    let runway = recipe
        .airstrips
        .iter()
        .find(|a| !a.apron)
        .expect("the island has a runway");
    println!(
        "THE TAKE-OFF TABLE (flat lab, sea-level air, full power, rotate at 1.1 Vs to the \
         attitude that flies there) against the {:.0} m {}:",
        runway.length_m, runway.name
    );
    println!(
        "  row                        mass t   Vs m/s   rotate deg   roll m   to 35 ft m   \
         lift-off m/s   climb m/s"
    );
    let mut longest = 0.0f64;
    let mut rows = Vec::new();
    for id in [
        "western_duster",
        "mammoth_dodo",
        "buckingham_luxor_deluxe",
        "titan_cargo",
        "flyus_jetliner",
    ] {
        let def = row(id);
        let vs = stall_speed(&def);
        let rotate = rotate_attitude_deg(&def, 1.1);
        let rep = fly_circuit(
            strip_sim(&def),
            &Circuit {
                vr: vs * 1.1,
                climb_pitch: rotate,
                cruise_alt: 30.0,
                level_s: 0.0,
            },
            90.0,
        );
        write_csv(
            &format!("takeoff_{id}"),
            TRACE_HEADER,
            &trace_rows(&rep.trace),
        );
        let roll = rep
            .liftoff_roll_m
            .unwrap_or_else(|| panic!("{id} never lifted off"));
        let screen = rep
            .screen_m
            .unwrap_or_else(|| panic!("{id} never reached 35 ft"));
        println!(
            "  {id:26} {:6.1}  {vs:6.1}  {rotate:10.1}  {roll:7.0}  {screen:10.0}  {:12.1}  {:9.2}",
            def.chassis_mass_kg() / 1000.0,
            rep.liftoff_ias,
            rep.climb_rate
        );
        rows.push(format!(
            "{id},{:.0},{vs:.2},{rotate:.2},{roll:.1},{screen:.1},{:.2},{:.2}",
            def.chassis_mass_kg(),
            rep.liftoff_ias,
            rep.climb_rate
        ));
        assert!(
            rep.climb_rate > 1.0,
            "{id} climbed at {:.2} m/s",
            rep.climb_rate
        );
        longest = longest.max(roll);
    }
    write_csv(
        "takeoff_table",
        "row,mass_kg,vs_mps,rotate_deg,roll_m,screen_m,liftoff_mps,climb_mps",
        &rows,
    );
    assert!(
        longest < runway.length_m,
        "the longest roll ({longest:.0} m) is longer than the runway ({:.0} m)",
        runway.length_m
    );
}

/// **THE DODO IS A FLOATPLANE: it floats, runs on the water and flies off it**
/// -- the handover at the waterline, read off its world position.
///
/// On the water no wheel has anything to touch (the lake is not a collider; the
/// suspension rays find nothing), so what carries the weight at rest is
/// P20.2's Archimedes on the row's `buoyancy_density_kg_m3`; the wing model
/// runs the whole time and takes the weight over as the airspeed builds. The arm
/// reads: the hull settled with its bottom BELOW the surface and no wheel in
/// contact (afloat, not on its gear), the water run to the last step the hull's
/// bottom was wet, and the altitude over the surface a minute later.
///
/// **Mutation → red**: the row's `buoyancy_density_kg_m3` 0 (it sinks: no
/// Archimedes, no floats).
#[test]
fn the_dodo_floats_and_flies_off_the_water() {
    let def = row("mammoth_dodo");
    let vs = stall_speed(&def);
    let rotate = rotate_attitude_deg(&def, 1.1);
    let mut sim = sea_sim(&def, (0.0, 0.0), 0.0);
    for _ in 0..(3.0 * HZ) as usize {
        command(&mut sim, CRAFT, VehicleControls::default());
    }
    let (p0, _, _) = body_state(&sim, CRAFT);
    let mut settle_dy = 0.0f64;
    for _ in 0..(2.0 * HZ) as usize {
        command(&mut sim, CRAFT, VehicleControls::default());
        settle_dy = settle_dy.max((body_state(&sim, CRAFT).0.y - p0.y).abs());
    }
    let rest_draught = draught(&sim, &def);
    let wheels_at_rest = grounded(&sim, CRAFT);
    let start_z = body_state(&sim, CRAFT).0.z;
    let mut rows = Vec::new();
    let mut last_wet_z = None;
    let mut peak_alt = f64::NEG_INFINITY;
    let mut lowest_bottom = f64::INFINITY;
    for i in 0..(60.0 * HZ) as usize {
        let (p, r, v) = body_state(&sim, CRAFT);
        let f = flight(&sim, CRAFT);
        let bottom = p.y - def.half_extents.y;
        lowest_bottom = lowest_bottom.min(bottom);
        if bottom < 0.0 {
            last_wet_z = Some(p.z - start_z);
        }
        peak_alt = peak_alt.max(bottom);
        let mut c = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        if f.airspeed_mps >= vs * 1.1 {
            c.vertical = pitch_hold(rotate, pitch_deg(r));
        }
        rows.push(format!(
            "{:.4},{:.3},{:.4},{:.3},{:.3},{:.3},{:.4}",
            i as f64 * DT,
            p.z - start_z,
            bottom,
            v.y,
            f.airspeed_mps,
            f.alpha_deg,
            f.cl
        ));
        command(&mut sim, CRAFT, c);
    }
    write_csv(
        "dodo_water",
        "t,dist_m,hull_bottom_m,vs,ias,alpha,cl",
        &rows,
    );
    println!(
        "THE FLOATPLANE: afloat with its hull bottom {rest_draught:.3} m under the surface, {} wheel(s) \
         in contact, settled to {:.1} mm; water run {:.0} m; the hull bottom {:.1} m over the surface at \
         its highest in 60 s",
        wheels_at_rest,
        settle_dy * 1000.0,
        last_wet_z.unwrap_or(f64::NAN),
        peak_alt
    );
    assert!(
        rest_draught > 0.0 && wheels_at_rest == 0,
        "the Dodo is not afloat (draught {rest_draught:.3} m, {wheels_at_rest} wheels touching)"
    );
    assert!(
        settle_dy < 0.05,
        "the Dodo is still moving {:.1} mm on the water",
        settle_dy * 1000.0
    );
    let run = last_wet_z.expect("the hull was never wet");
    assert!(run > 50.0, "a {run:.0} m water run is not a take-off");
    assert!(
        peak_alt > SCREEN_HEIGHT_M,
        "it never left the water: {peak_alt:.1} m at best (lowest {lowest_bottom:.2} m)"
    );
}

/// **THE STALL**: level at 300 m and 40 m/s, power off, full back stick. The
/// force log's angle of attack passes `stall_deg`, the lift coefficient
/// COLLAPSES (it falls below 80 % of its pre-stall peak once alpha is past the
/// stall), the nose drops ten degrees from its high point and the aeroplane
/// loses height.
///
/// **Mutation → red**: `POST_STALL_CL_FRAC` 1.0 with `STALL_BREAK_DEG` 90 (no
/// collapse).
#[test]
fn the_stall_collapses_the_lift_and_drops_the_nose() {
    let def = row("mammoth_dodo");
    let stall = def.class.stall_deg;
    let mut sim = strip_sim(&def);
    let at = DVec3::new(0.0, 300.0, -STRIP_HALF_L + 200.0);
    assert!(sim.place_vehicle(CRAFT, at, DQuat::IDENTITY));
    {
        let b = sim.bridge3d_mut();
        let body = b.body_of(CRAFT).expect("a body");
        b.world_mut()
            .set_body_linvel(body, DVec3::new(0.0, 0.0, 40.0));
    }
    let mut trace = Vec::new();
    let mut cl_peak = 0.0f64;
    let mut first_stall: Option<usize> = None;
    let mut cl_after = f64::INFINITY;
    let (mut band_worst, mut band_n) = (f64::MIN, 0usize);
    let mut pitch_peak = f64::MIN;
    let mut pitch_after = f64::INFINITY;
    for i in 0..(60 * 20) {
        // Two seconds of trimmed flight, then the stick on the stop.
        let back = i >= 120;
        command(
            &mut sim,
            CRAFT,
            VehicleControls {
                throttle: 0.0,
                vertical: if back { 1.0 } else { 0.0 },
                ..Default::default()
            },
        );
        let s = sample(&sim, CRAFT, i as f64 * DT, 0.0);
        if back {
            if first_stall.is_none() {
                cl_peak = cl_peak.max(s.cl);
                pitch_peak = pitch_peak.max(s.pitch);
            }
            if s.alpha > stall {
                first_stall.get_or_insert(i);
            }
            // The collapse is read in the BREAK BAND only -- 3 to 8 degrees
            // past the stall, where a wing that does not collapse still holds
            // ~90 % of its peak. Read over every later step (the first cut),
            // the dive brought CL down by itself; read over every step past
            // the stall angle (the second), the deep-alpha fade toward 90 deg
            // did (32 deg is reached, where a straight line to zero at 90 is
            // already 78 %). Both stayed green with `POST_STALL_CL_FRAC` 1.0 +
            // `STALL_BREAK_DEG` 90; this one does not. The WORST (highest) CL
            // in the band is the one held under the ceiling.
            if first_stall.is_some() && s.alpha > stall + 3.0 && s.alpha <= stall + 8.0 {
                band_worst = band_worst.max(s.cl);
                band_n += 1;
            }
            if first_stall.is_some() && s.alpha > stall {
                cl_after = cl_after.min(s.cl);
            }
            if first_stall.is_some() {
                pitch_after = pitch_after.min(s.pitch);
            }
        }
        trace.push(s);
    }
    write_csv("dodo_stall", TRACE_HEADER, &trace_rows(&trace));
    let at_stall = first_stall.expect("the wing never passed its stall angle");
    let lowest = trace
        .iter()
        .skip(at_stall)
        .map(|s| s.alt)
        .fold(f64::MAX, f64::min);
    let alt_loss = trace[at_stall].alt - lowest;
    println!(
        "THE STALL (stall_deg {stall}): alpha passed it {:.2} s after the stick came back, at \
         {:.1} m/s; CL {cl_peak:.3} -> {cl_after:.3} while past it ({:.0} %); pitch {pitch_peak:.1} -> \
         {pitch_after:.1} deg; {alt_loss:.1} m lost; in the break band (stall +3..+8 deg, {band_n} \
         steps) CL at most {band_worst:.3} ({:.0} % of the peak)",
        (at_stall - 120) as f64 * DT,
        trace[at_stall].ias,
        cl_after / cl_peak * 100.0,
        band_worst / cl_peak * 100.0
    );
    let stalled_steps = trace.iter().filter(|s| s.stalled).count();
    assert!(stalled_steps > 10, "only {stalled_steps} stalled steps");
    assert!(band_n >= 5, "only {band_n} steps in the break band");
    assert!(
        band_worst < 0.8 * cl_peak,
        "the lift did not collapse: {band_worst:.3} in the break band against a peak of {cl_peak:.3}"
    );
    assert!(
        pitch_peak - pitch_after > 10.0,
        "the nose did not drop: {pitch_peak:.1} -> {pitch_after:.1}"
    );
    assert!(alt_loss > 10.0, "a stall that lost {alt_loss:.1} m");
}

// ── 2. THE HELICOPTERS ───────────────────────────────────────────────────────

/// A collective pilot: hold a vertical speed.
fn collective(want_vs: f64, vs: f64) -> f64 {
    ((want_vs - vs) * 0.6).clamp(-1.0, 1.0)
}

/// **Every helicopter row lifts off, hovers and translates** -- off its own
/// trace: 20 m of climb, the vertical speed held at zero (under 0.3 m/s at the
/// end of the hover), then 10 s of forward stick covering 100 m.
#[test]
fn five_helicopters_lift_off_hover_and_translate() {
    println!("  row                       climb m   hover |vs|   forward m   at m/s");
    for id in [
        "nagasaki_buzzard",
        "buckingham_maverick",
        "buckingham_volatus",
        "buckingham_swift_deluxe",
        "cargobob",
    ] {
        let def = row(id);
        let mut sim = strip_sim(&def);
        for _ in 0..90 {
            sim.step_once(Default::default());
        }
        let (p0, _, _) = body_state(&sim, CRAFT);
        let mut rows = Vec::new();
        let (mut hover_vs, mut climbed) = (f64::NAN, 0.0);
        for i in 0..(60 * 25) {
            let t = i as f64 * DT;
            let (p, _, v) = body_state(&sim, CRAFT);
            let c = if p.y - p0.y < 20.0 && t < 10.0 {
                VehicleControls {
                    vertical: collective(4.0, v.y),
                    ..Default::default()
                }
            } else if t < 15.0 {
                VehicleControls {
                    vertical: collective(0.0, v.y),
                    ..Default::default()
                }
            } else {
                VehicleControls {
                    throttle: 1.0,
                    vertical: collective(0.0, v.y),
                    ..Default::default()
                }
            };
            if i == 60 * 15 - 1 {
                hover_vs = v.y.abs();
                climbed = p.y - p0.y;
            }
            command(&mut sim, CRAFT, c);
            rows.push(format!(
                "{t:.3},{:.3},{:.3},{:.3}",
                p.y - p0.y,
                v.y,
                p.z - p0.z
            ));
        }
        write_csv(&format!("heli_{id}"), "t,alt,vs,z", &rows);
        let (p, _, v) = body_state(&sim, CRAFT);
        println!(
            "  {id:24} {climbed:8.1}   {hover_vs:9.3}   {:9.1}   {:6.1}",
            p.z - p0.z,
            v.length()
        );
        assert!(climbed >= 19.0, "{id} climbed {climbed:.1} m");
        assert!(hover_vs < 0.3, "{id} hovered at {hover_vs:.2} m/s vertical");
        assert!(p.z - p0.z > 100.0, "{id} translated {:.1} m", p.z - p0.z);
    }
}

/// **The Cargobob's winch lifts a car, and lets it go** -- the car's altitude
/// follows the lifter's, the rope's tension at the hover is the car's weight
/// (the limit impulse over the step, all four substeps), and once released the
/// car FALLS.
///
/// **Mutation → red**: `winch` not written (the car stays on the ground, no
/// tension); `rope_impulse_ns` without the substep factor (a quarter of the
/// weight -- measured 4 660 N against 18 639 N).
#[test]
fn the_cargobob_winch_lifts_a_car_and_lets_it_go() {
    let heli = row("cargobob");
    let car = row("albany_washington");
    let rope = 8.0;
    let mut world = EcsWorld::new();
    slab(
        &mut world,
        GROUND,
        DVec3::new(0.0, -0.5, 0.0),
        DVec3::new(STRIP_HALF_W, 0.5, STRIP_HALF_L),
        0.9,
    );
    let heli_at = DVec3::new(
        0.0,
        inf_ecs::vehicle::resting_origin_y(&heli, 0.0) + 12.0,
        0.0,
    );
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        CRAFT,
        &heli,
        &spawn_spec("Cargobob", heli_at, 0.0, false),
    );
    let car_at = DVec3::new(0.0, inf_ecs::vehicle::resting_origin_y(&car, 0.0), 0.0);
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        LOAD,
        &car,
        &spawn_spec("Car", car_at, 0.0, false),
    );
    assert!(inf_ecs::vehicle::winch(
        &mut world, CRAFT, &heli, LOAD, &car, rope
    ));
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
    let weight = car.chassis_mass_kg() * 9.81;
    let mut rows = Vec::new();
    let (mut car_top, mut hover_tension) = (0.0f64, None);
    let mut released_at: Option<f64> = None;
    let mut car_after = f64::MAX;
    for i in 0..(60 * 30) {
        let t = i as f64 * DT;
        if i == 60 * 22 {
            assert!(inf_ecs::vehicle::release_winch(sim.world_mut(), LOAD));
            released_at = Some(body_state(&sim, LOAD).0.y);
        }
        let (hp, _, hv) = body_state(&sim, CRAFT);
        let want_vs = ((40.0 - hp.y) * 0.5).clamp(-3.0, 4.0);
        command(
            &mut sim,
            CRAFT,
            VehicleControls {
                vertical: collective(want_vs, hv.y),
                ..Default::default()
            },
        );
        let (cp, _, _) = body_state(&sim, LOAD);
        let tension = sim.winch_tension_n(CRAFT);
        rows.push(format!(
            "{t:.3},{:.3},{:.3},{:.1}",
            hp.y,
            cp.y,
            tension.unwrap_or(0.0)
        ));
        if released_at.is_none() {
            car_top = car_top.max(cp.y);
            if i == 60 * 20 {
                hover_tension = tension;
            }
        } else {
            car_after = car_after.min(cp.y);
        }
    }
    write_csv("winch", "t,lifter_y,car_y,tension_n", &rows);
    let tension = hover_tension.expect("no cable on the hook at the hover");
    let released = released_at.expect("released");
    println!(
        "THE WINCH: the car rose to {car_top:.1} m (from {:.2}); tension at the hover {tension:.0} N \
         against its {weight:.0} N weight ({:+.2} %); released at {released:.1} m, fell to \
         {car_after:.1} m",
        car_at.y,
        (tension / weight - 1.0) * 100.0
    );
    assert!(
        car_top - car_at.y > 20.0,
        "the car rose {:.1} m",
        car_top - car_at.y
    );
    assert!(
        (tension / weight - 1.0).abs() < 0.05,
        "the cable held {tension:.0} N under a {weight:.0} N car"
    );
    assert!(
        sim.winch_tension_n(CRAFT).is_none(),
        "the cable is still on the hook after the release"
    );
    assert!(
        released - car_after > 15.0,
        "the released car fell {:.1} m",
        released - car_after
    );
}

// ── the air lane lab: a 3x3 town, a cruiser and a police helicopter ─────────

mod air_lane_lab {
    use super::*;
    use inf_ecs::components::{PcgVolume, ResidentSlot, SlotRole, StreamingSource};
    use inf_ecs::dispatch::UnitKind;
    use inf_ecs::math::{Color, Vec2d};
    use inf_ecs::vehicle::{BodyPart, Livery, PartPaint, RigSpawn};
    use inf_physics::d3::PhysicsBridge3D;

    pub const PITCH: f64 = 100.0;
    pub const STREET: f64 = 20.0;
    pub const HERO: Uuid = Uuid::from_u128(0x5E3E_1001);
    pub const TOWN_GROUND: Uuid = Uuid::from_u128(0x5E3E_1002);
    pub const CRUISER: Uuid = Uuid::from_u128(0x5E3E_1004);
    pub const HELI: Uuid = Uuid::from_u128(0x5E3E_1008);
    pub const WITNESS: Uuid = Uuid::from_u128(0x5E3E_1005);

    static BAR: BodyPart = BodyPart {
        name: "light_bar",
        centre: Vec3d::new(0.0, 1.02, 0.0),
        half: Vec3d::new(0.5, 0.06, 0.18),
        primitive: inf_ecs::components::Primitive::Cube,
        kind: inf_ecs::vehicle::BodyPartKind::Panel,
    };
    static BLUE: PartPaint = PartPaint {
        base_color: Color::new(0.9, 0.9, 0.95, 1.0),
        emissive: Color::new(0.15, 0.35, 1.0, 1.0),
        emissive_intensity: 3.0,
    };
    pub static POLICE: Livery = Livery {
        name: "police",
        parts: &[],
        extra: &[(BAR, BLUE)],
        service: Some(UnitKind::Police),
    };

    pub struct Town {
        pub world: EcsWorld,
        pub bridge: PhysicsBridge3D,
    }

    fn blocks(world: &mut EcsWorld) {
        let half = (PITCH - STREET) * 0.5;
        for row in 0..3 {
            for col in 0..3 {
                let c = DVec3::new(f64::from(col) * PITCH, 0.0, f64::from(row) * PITCH);
                let guid = Uuid::from_u64_pair(0x5E3E_0E52, ((row as u64) << 32) | col as u64);
                let e = world.spawn_with_guid(guid, "block", None);
                world
                    .world_mut()
                    .entity_mut(e)
                    .insert(Transform::from_translation(c));
                let mut v = PcgVolume {
                    extent: Vec2d::new(half, half),
                    ..Default::default()
                };
                v.residents = vec![ResidentSlot {
                    role: SlotRole::Home,
                    at: c,
                    room: 0,
                    building: 0,
                    floor: 0,
                    index: 0,
                    node: 0,
                    posture: inf_ecs::components::SlotPosture::Stand,
                    shift: inf_ecs::components::SlotShift::Day,
                    face: DVec3::ZERO,
                }];
                world.world_mut().entity_mut(e).insert(v);
            }
        }
    }

    impl Town {
        /// The town, a cruiser and the police helicopter on the station apron.
        pub fn new(heli_row: &VehicleDef) -> Self {
            let mut world = EcsWorld::new();
            blocks(&mut world);
            slab(
                &mut world,
                TOWN_GROUND,
                DVec3::new(100.0, -0.5, 100.0),
                DVec3::new(600.0, 0.5, 600.0),
                0.9,
            );
            let mut cruiser = VehicleDef::default();
            inf_ecs::traffic::size_the_suspension(&mut cruiser);
            let sag = cruiser.class.travel_m * inf_ecs::traffic::STATIC_SAG_FRAC;
            let rest_y = -cruiser.wheel_drop_m + cruiser.wheel_radius_m - sag;
            for (guid, def, at) in [
                (CRUISER, &cruiser, DVec3::new(-46.0, rest_y, 12.0)),
                (
                    HELI,
                    heli_row,
                    DVec3::new(
                        -46.0,
                        inf_ecs::vehicle::resting_origin_y(heli_row, 0.0),
                        40.0,
                    ),
                ),
            ] {
                inf_ecs::vehicle::spawn_rig_at(
                    &mut world,
                    guid,
                    def,
                    &RigSpawn {
                        name: "Unit".to_string(),
                        at,
                        yaw_deg: 0.0,
                        paint: Color::new(0.35, 0.36, 0.38, 1.0),
                        clip: None,
                        engine_voice: false,
                        livery: Some(&POLICE),
                    },
                    true,
                );
            }
            let e = world.spawn_with_guid(HERO, "Hero", None);
            world.world_mut().entity_mut(e).insert((
                StreamingSource { radius_m: 1024.0 },
                Transform::from_translation(DVec3::new(150.0, 0.0, 250.0)),
            ));
            world.mark_dirty();
            world.propagate();
            let mut town = Self {
                world,
                bridge: PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0)),
            };
            town.bridge
                .sync_from_world_sim(&town.world, &Default::default(), &Default::default());
            town
        }

        pub fn step(&mut self) {
            inf_physics::d3::traffic::step_traffic(&mut self.world, &mut self.bridge, DT);
            let _ = inf_physics::d3::dispatch::step_dispatch(&mut self.world, &mut self.bridge, DT);
            self.bridge
                .sync_from_world_sim(&self.world, &Default::default(), &Default::default());
            inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }

        pub fn body_at(&self, guid: Uuid) -> DVec3 {
            let b = &self.bridge;
            b.body_of(guid)
                .and_then(|body| b.world().body_translation(body))
                .unwrap_or(DVec3::ZERO)
        }
    }
}

/// **THE POLICE HELICOPTER FLIES ITS AIR LANE AND NEVER THE ROAD** -- VEH2c's
/// refusal closed by a routing tier. A killing on file (`Response::MultiUnit`,
/// two stars) brings both the cruiser and the helicopter; the helicopter's run
/// holds NO route at any step, its track once clear of its pad stays over 40 m
/// up and within 15 m of the straight line from the pad to its hover point, and
/// it arrives (`OnScene`) -- while the cruiser, the control, drives a route.
///
/// **Mutation → red**: `assign` handing an air unit `drive_path` (its run holds
/// a path); `run_units` without the `fly_unit` branch (it never climbs and never
/// arrives).
#[test]
fn the_police_helicopter_flies_its_air_lane_and_never_the_road() {
    use air_lane_lab::*;
    use inf_ecs::dispatch::UnitState;
    use inf_ecs::witness::{ActKind, WitnessedAct};
    let heli = row(inf_editor_core::island::POLICE_HELI_ROW);
    let mut town = Town::new(&heli);
    for _ in 0..30 {
        town.step();
    }
    let pad = town.body_at(HELI);
    assert!(inf_ecs::dispatch::is_air_unit(&town.world, HELI));
    let scene = DVec3::new(250.0, 0.0, 250.0);
    inf_ecs::witness::record_act(
        &mut town.world,
        WitnessedAct {
            kind: ActKind::Killed,
            actor: HERO,
            at: scene,
            step: 100,
            observers: vec![WITNESS],
            actor_look: 0,
            actor_vehicle: None,
            heard_by: 0,
        },
    );
    let hover = inf_ecs::dispatch::air_hover_point(pad, scene);
    let line = DVec3::new(hover.x - pad.x, 0.0, hover.z - pad.z).normalize();
    let (mut steps_out, mut path_steps, mut on_scene) = (0usize, 0usize, None);
    let (mut worst_off, mut lowest) = (0.0f64, f64::MAX);
    let mut cruiser_routed = false;
    let mut rows = Vec::new();
    for i in 0..(60 * 60) {
        town.step();
        let res = inf_ecs::dispatch::dispatch_of(&town.world).expect("a dispatcher");
        let run = res.runs.get(&HELI).cloned().unwrap_or_default();
        cruiser_routed |= res.runs.get(&CRUISER).is_some_and(|r| r.path.is_some());
        if run.state == UnitState::InStation {
            continue;
        }
        steps_out += 1;
        if run.path.is_some() {
            path_steps += 1;
        }
        let p = town.body_at(HELI);
        rows.push(format!(
            "{:.3},{:.2},{:.2},{:.2},{:?}",
            i as f64 * DT,
            p.x,
            p.y,
            p.z,
            run.state
        ));
        if run.state == UnitState::OnScene && on_scene.is_none() {
            on_scene = Some(i as f64 * DT);
        }
        let flat = DVec3::new(p.x - pad.x, 0.0, p.z - pad.z);
        if run.state == UnitState::EnRoute && flat.length() > 60.0 {
            let off = (flat - line * flat.dot(line)).length();
            worst_off = worst_off.max(off);
            lowest = lowest.min(p.y);
        }
    }
    write_csv("air_lane", "t,x,y,z,state", &rows);
    println!(
        "THE AIR LANE: {steps_out} steps out, {path_steps} of them holding a road route; lowest \
         {lowest:.1} m over the lane, worst {worst_off:.1} m off the straight line; on scene after \
         {on_scene:?} s; the cruiser drove a route: {cruiser_routed}"
    );
    assert!(
        steps_out > 600,
        "the helicopter was out for {steps_out} steps"
    );
    assert_eq!(
        path_steps, 0,
        "the air unit held a road route on {path_steps} steps"
    );
    assert!(lowest > 40.0, "the lane came down to {lowest:.1} m");
    assert!(
        worst_off < 15.0,
        "the track wandered {worst_off:.1} m off the line"
    );
    assert!(on_scene.is_some(), "the helicopter never arrived");
    assert!(
        cruiser_routed,
        "the control: the cruiser never drove a route"
    );
}

// ── 3. THE MARINE ROWS ───────────────────────────────────────────────────────

/// **A planing hull rises; a displacement hull does not** -- the draught off the
/// hull's world position every ten seconds of a minute at full power, for the
/// seven motor hulls. A hull with a `planing_speed_mps` ends above it at under
/// 70 % of its rest draught; one without ends at its rest draught to the
/// centimetre.
///
/// **Mutation → red**: the jetski's `planing_speed_mps` 0 (its draught stays at
/// rest); `PLANING_LIFT_FRAC` 0 (every hull stays).
#[test]
fn a_planing_hull_rises_and_a_displacement_hull_does_not() {
    println!("THE DRAUGHT TABLE (flat water, full power; draught off the hull's world position):");
    println!("  row                   planes at   rest m    v m/s:draught m every 10 s");
    let mut csv = Vec::new();
    let mut planed = 0usize;
    for id in [
        "nagasaki_seashark",
        "shitzu_jetmax",
        "pegassi_speeder",
        "nagasaki_dinghy",
        "buckingham_tug",
        "galaxy_super_yacht",
        "oceanic_odyssey",
    ] {
        let def = row(id);
        let mut sim = sea_sim(&def, (0.0, 0.0), 0.0);
        for _ in 0..240 {
            sim.step_once(Default::default());
        }
        let rest = draught(&sim, &def);
        let mut line = format!(
            "  {id:22} {:8.1}   {rest:6.3}  ",
            def.class.planing_speed_mps
        );
        let (mut v_end, mut d_end) = (0.0, 0.0);
        for i in 0..(60 * 60) {
            command(
                &mut sim,
                CRAFT,
                VehicleControls {
                    throttle: 1.0,
                    ..Default::default()
                },
            );
            if i % 600 == 599 {
                v_end = forward_speed(&sim, CRAFT);
                d_end = draught(&sim, &def);
                line.push_str(&format!(" {v_end:5.1}:{d_end:.3}"));
                csv.push(format!(
                    "{id},{:.0},{v_end:.3},{d_end:.4},{rest:.4}",
                    (i + 1) as f64 * DT
                ));
            }
        }
        println!("{line}");
        if def.class.planing_speed_mps > 0.0 {
            assert!(
                v_end > def.class.planing_speed_mps,
                "{id} ended at {v_end:.1} m/s, under its planing speed"
            );
            assert!(
                d_end < 0.7 * rest,
                "{id} planes at {d_end:.3} m against {rest:.3} m at rest"
            );
            planed += 1;
        } else {
            assert!(
                (d_end - rest).abs() < 0.01,
                "{id} is a displacement hull and its draught moved {rest:.3} -> {d_end:.3}"
            );
        }
    }
    write_csv("draught_table", "row,t_s,v_mps,draught_m,rest_m", &csv);
    assert_eq!(planed, 4, "four planing hulls, {planed} planed");
}

/// **The big hulls float where Archimedes puts them** -- the AABB-draught bound
/// MEASURED before closing it: a ship's chassis is a `Box`, whose buoyancy
/// branch states its own half-extent exactly, so its rest draught is `rho_b /
/// rho_f x 2h` (`Buoyancy::default().fluid_density_kg_m3`) to the millimetre.
/// The bound is on the convex-hull and trimesh branches, which no hull uses.
#[test]
fn the_big_hulls_float_where_archimedes_puts_them() {
    let fluid = inf_ecs::components::Buoyancy::default().fluid_density_kg_m3;
    for id in ["buckingham_tug", "galaxy_super_yacht", "oceanic_odyssey"] {
        let def = row(id);
        let mut sim = sea_sim(&def, (0.0, 0.0), 0.0);
        for _ in 0..600 {
            sim.step_once(Default::default());
        }
        let d = draught(&sim, &def);
        let want = def.buoyancy_density_kg_m3 / fluid * 2.0 * def.half_extents.y;
        println!(
            "  {id:20} draught {d:.4} m, Archimedes {want:.4} m ({:+.2} mm)",
            (d - want) * 1000.0
        );
        assert!((d - want).abs() < 0.005, "{id}: {d:.4} against {want:.4}");
    }
}

/// **THE SAIL** reads the P17 wind VECTOR: 8 m/s from the north, the yacht at
/// three headings with its engine off -- head to wind it goes nowhere (the sail
/// luffs), on a beam reach it sails fastest, dead downwind it runs -- and then
/// the WIND is turned a quarter round with the yacht still on the beam heading,
/// which puts it head to wind, and it goes nowhere.
///
/// PINCHING, 20 deg off the wind, it makes no way either, and close-hauled at
/// 45 deg it makes way. Head to wind alone reads no sail: lift is square to
/// the apparent wind and drag along it, so NO sail drives straight into the
/// wind, filled or not.
///
/// **Mutation → red**: `ChassisState::wind` fed zero by the door (nothing
/// sails). **Mutation → GREEN, said**: the no-go fill removed (a full sail
/// at 20 deg still makes no way: the polar's drag rising off the close reach
/// and the hull's own drag out-pull lift's sin-20 share). The fill SHAPES the
/// close-hauled band; the no-go edge is the force geometry's, not the fill's.
#[test]
fn the_sail_drives_across_the_wind_and_not_into_it() {
    let def = row("dundreary_marquis");
    let run = |wind: (f32, f32), yaw: f64, name: &str| -> (f64, f64, f64) {
        let mut sim = sea_sim(&def, wind, yaw);
        for _ in 0..240 {
            sim.step_once(Default::default());
        }
        let mut heel = 0.0f64;
        let mut rows = Vec::new();
        for i in 0..(60 * 60) {
            command(&mut sim, CRAFT, VehicleControls::default());
            let (_, r, _) = body_state(&sim, CRAFT);
            let right = r * DVec3::X;
            let h = inf_math::patan2_64(right.y.abs(), (r * DVec3::Y).y).to_degrees();
            heel = heel.max(h);
            if i % 30 == 0 {
                let m = sim
                    .bridge3d()
                    .vehicle_of(CRAFT)
                    .and_then(|v| v.marine())
                    .unwrap_or_default();
                rows.push(format!(
                    "{:.2},{:.3},{:.2},{:.1},{:.1},{:.1}",
                    i as f64 * DT,
                    forward_speed(&sim, CRAFT),
                    h,
                    m.apparent_wind_deg,
                    m.sail_force.x,
                    m.sail_force.z
                ));
            }
        }
        write_csv(
            &format!("sail_{name}"),
            "t,v_mps,heel_deg,apparent_deg,fx,fz",
            &rows,
        );
        let m = sim
            .bridge3d()
            .vehicle_of(CRAFT)
            .and_then(|v| v.marine())
            .unwrap_or_default();
        (forward_speed(&sim, CRAFT), heel, m.apparent_wind_deg)
    };
    // The wind blows TOWARD +Z (it comes from the north, -Z).
    let north = (0.0f32, 8.0f32);
    let into = run(north, 180.0, "into");
    let beam = run(north, 90.0, "beam");
    let runs = run(north, 0.0, "run");
    let pinch = run(north, 160.0, "pinch_20");
    let close = run(north, 135.0, "close_hauled_45");
    // The same beam heading with the wind turned a quarter: from the east,
    // blowing toward -X, head to wind for a boat pointing +X.
    let turned = run((-8.0, 0.0), 90.0, "beam_wind_turned");
    println!("THE SAIL (8 m/s wind, engine off, 60 s):");
    for (name, (v, heel, app)) in [
        ("head to wind", into),
        ("pinching, 20 off the wind", pinch),
        ("close-hauled, 45 off", close),
        ("beam reach", beam),
        ("dead run", runs),
        ("beam heading, wind turned 90", turned),
    ] {
        println!("  {name:30} {v:6.2} m/s  heel {heel:4.1} deg  apparent {app:5.1} deg");
    }
    assert!(into.0 < 0.2, "head to wind it made {:.2} m/s", into.0);
    assert!(
        pinch.0 < 0.2,
        "20 deg off the wind it made {:.2} m/s -- inside the no-go zone",
        pinch.0
    );
    // Makes WAY -- the same 0.2 m/s the no-way rows are held under. Slow
    // (0.49 m/s measured, the apparent wind 35 deg, five past the no-go edge
    // and a third of the way up the fill ramp): carried as the close-hauled
    // performance, not asserted as a speed.
    assert!(close.0 > 0.2, "close-hauled it made {:.2} m/s", close.0);
    assert!(beam.0 > 3.0, "on the beam it made {:.2} m/s", beam.0);
    assert!(runs.0 > 1.5, "running it made {:.2} m/s", runs.0);
    assert!(beam.0 > runs.0, "a reach is the fast point of sail");
    assert!(
        beam.1 > 0.5,
        "the sail did not heel the boat ({:.2} deg)",
        beam.1
    );
    assert!(
        turned.0 < 0.2,
        "the turned wind still drove it at {:.2} m/s",
        turned.0
    );
}

/// **The tug pushes the superyacht** -- its bow on the yacht's transom, full
/// power: the yacht, whose own engine is off, is moved.
#[test]
fn the_tug_pushes_the_superyacht() {
    let tug = row("buckingham_tug");
    let yacht = row("galaxy_super_yacht");
    let mut world = sea_world((0.0, 0.0));
    let yacht_at = DVec3::new(0.0, inf_ecs::vehicle::floating_origin_y(&yacht, 0.0), 0.0);
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        CRAFT,
        &yacht,
        &spawn_spec("Yacht", yacht_at, 0.0, false),
    );
    let tug_at = DVec3::new(
        0.0,
        inf_ecs::vehicle::floating_origin_y(&tug, 0.0),
        -yacht.half_extents.z - tug.half_extents.z - 0.5,
    );
    inf_ecs::vehicle::spawn_rig(
        &mut world,
        PUSHER,
        &tug,
        &spawn_spec("Tug", tug_at, 0.0, false),
    );
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
    for _ in 0..240 {
        sim.step_once(Default::default());
    }
    let z0 = body_state(&sim, CRAFT).0.z;
    let mut rows = Vec::new();
    for i in 0..(60 * 90) {
        if let Some(v) = sim.bridge3d_mut().vehicle_mut(PUSHER) {
            v.control(VehicleControls {
                throttle: 1.0,
                occupied: true,
                ..Default::default()
            });
        }
        sim.step_once(Default::default());
        if i % 60 == 0 {
            rows.push(format!(
                "{:.1},{:.3},{:.3}",
                i as f64 * DT,
                body_state(&sim, CRAFT).2.z,
                body_state(&sim, PUSHER).2.z
            ));
        }
    }
    write_csv("tug_push", "t,yacht_vz,tug_vz", &rows);
    let (p, _, v) = body_state(&sim, CRAFT);
    println!(
        "THE TUG: the {:.0} t superyacht moved {:.1} m and is at {:.2} m/s",
        yacht.chassis_mass_kg() / 1000.0,
        p.z - z0,
        v.z
    );
    assert!(v.z > 0.5, "the pushed yacht is at {:.2} m/s", v.z);
    assert!(p.z - z0 > 20.0, "the pushed yacht moved {:.1} m", p.z - z0);
}

// ── 4. THE RAMP ──────────────────────────────────────────────────────────────

/// **The Titan's ramp swings down on its hinge** -- the ramp part's rear edge,
/// in the chassis frame, closed and three seconds after it was told to open: it
/// has DROPPED by more than a metre, and the ramp stands at its open angle.
///
/// **Mutation → red**: `ramp_hinge`'s sign flipped to the boot lid's (the rear
/// edge rises).
#[test]
fn the_titan_ramp_swings_down_on_its_hinge() {
    let def = row("titan_cargo");
    let mut sim = strip_sim(&def);
    for _ in 0..120 {
        sim.step_once(Default::default());
    }
    let ramp = inf_ecs::vehicle::body_part_guid(CRAFT, "ramp");
    // The ramp's rear edge IN THE CHASSIS FRAME, off the world: its drawn unit
    // box's `-Z` face centre through its GlobalTransform, taken back through
    // the chassis's. A latched part is a child of the chassis and a live one a
    // root body (VEH3c), and the GlobalTransform is the one number both have.
    let edge = |sim: &RuntimeSim| -> DVec3 {
        let w = sim.world();
        let gt = |g: Uuid| {
            w.world()
                .get::<inf_ecs::components::GlobalTransform>(w.entity_of(g).expect("an entity"))
                .expect("a global transform")
                .0
        };
        let world_edge = gt(ramp).transform_point3(DVec3::new(0.0, 0.0, -0.5));
        gt(CRAFT).inverse().transform_point3(world_edge)
    };
    let closed = edge(&sim);
    let opened = sim.open_vehicle_doors(CRAFT, true);
    for _ in 0..180 {
        sim.step_once(Default::default());
    }
    let open = edge(&sim);
    // The swing, off the same two points: the edge's drop over the ramp's
    // own length.
    let len = (closed - hinge_of(&def)).length();
    let angle = inf_math::patan2_64(closed.y - open.y, len).to_degrees();
    println!(
        "THE RAMP: {opened} hinged part(s) opened; the ramp at {angle:.1} deg; its rear edge \
         {closed:?} -> {open:?} ({:+.2} m)",
        open.y - closed.y
    );
    assert!(opened >= 1, "no hinged part opened");
    assert!(
        closed.y - open.y > 1.0,
        "the rear edge moved {:+.2} m",
        open.y - closed.y
    );
    assert!(
        (angle - inf_ecs::vehicle::RAMP_OPEN_DEG).abs() < 8.0,
        "the ramp swung {angle:.1} deg"
    );
}

/// The ramp's hinge in the Titan's chassis frame, metres -- off the family
/// table, for the arm's lever.
fn hinge_of(def: &VehicleDef) -> DVec3 {
    let part = def
        .body
        .parts()
        .iter()
        .find(|p| p.name == "ramp")
        .expect("a ramp part");
    let h = part.kind.hinge().expect("a hinge");
    DVec3::new(
        h.at.x * def.half_extents.x,
        h.at.y * def.half_extents.y,
        h.at.z * def.half_extents.z,
    )
}

// ── 5. DETERMINISM ───────────────────────────────────────────────────────────

/// Every vehicle chassis's pose bits, sorted by guid.
fn pose_bits(world: &EcsWorld) -> Vec<(Uuid, [u64; 6])> {
    let mut out = Vec::new();
    for e in world.world().iter_entities() {
        if e.get::<inf_ecs::components::VehicleClass>().is_none() {
            continue;
        }
        let (Some(g), Some(t)) = (e.get::<inf_ecs::components::Guid>(), e.get::<Transform>())
        else {
            continue;
        };
        out.push((
            g.0,
            [
                t.translation.x.to_bits(),
                t.translation.y.to_bits(),
                t.translation.z.to_bits(),
                t.rotation.x.to_bits(),
                t.rotation.y.to_bits(),
                t.rotation.z.to_bits(),
            ],
        ));
    }
    out.sort();
    out
}

/// The shared lab of the two two-host arms: a strip to the north, a lake to the
/// south, and a sky with a wind; authored in a world or through the editor's
/// own doors.
struct Lab {
    strip: (DVec3, DVec3),
    sea_centre: DVec3,
    wind: (f32, f32),
}

impl Lab {
    fn world(&self, craft: &[(Uuid, VehicleDef, DVec3, f64)], voice: bool) -> EcsWorld {
        let mut world = EcsWorld::new();
        slab(&mut world, GROUND, self.strip.0, self.strip.1, 0.9);
        let sea = world.spawn_with_guid(SEA, "Sea", None);
        let mut sb = sea_bits();
        sb.0.translation = Vec3d::from_dvec3(self.sea_centre);
        world.world_mut().entity_mut(sea).insert(sb);
        let sky = world.spawn_with_guid(SKY, "Sky", None);
        world
            .world_mut()
            .entity_mut(sky)
            .insert(sky_bits(self.wind));
        for (g, def, at, yaw) in craft {
            inf_ecs::vehicle::spawn_rig(
                &mut world,
                *g,
                def,
                &spawn_spec("Craft", *at, *yaw, voice),
            );
        }
        world
    }

    fn doc(
        &self,
        craft: &[(Uuid, VehicleDef, DVec3, f64)],
        voice: bool,
    ) -> inf_editor_core::scene::SceneDoc {
        use inf_editor_core::ipc::SpawnKind;
        let mut doc = inf_editor_core::scene::SceneDoc::new();
        let e = doc.create_with_guid(GROUND, SpawnKind::Empty, "Slab", None);
        doc.world_mut().world_mut().entity_mut(e).insert(slab_bits(
            self.strip.0,
            self.strip.1,
            0.9,
        ));
        let e = doc.create_with_guid(SEA, SpawnKind::Empty, "Sea", None);
        let mut sb = sea_bits();
        sb.0.translation = Vec3d::from_dvec3(self.sea_centre);
        doc.world_mut().world_mut().entity_mut(e).insert(sb);
        let e = doc.create_with_guid(SKY, SpawnKind::Empty, "Sky", None);
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(sky_bits(self.wind));
        for (g, def, at, yaw) in craft {
            inf_editor_core::vehicle::spawn_vehicle(
                &mut doc,
                *g,
                def,
                inf_editor_core::vehicle::VehicleSpawn {
                    name: "Craft",
                    at: *at,
                    yaw_deg: *yaw,
                    paint: inf_ecs::math::Color::new(0.8, 0.8, 0.85, 1.0),
                    clip: None,
                    livery: None,
                    engine_voice: voice,
                },
            );
        }
        doc
    }
}

/// **PIE == SHIPPING for a plane, a helicopter with a car on its winch, a
/// planing jetski and a yacht under sail** -- one lab holding all of them,
/// authored through each host's own doors and driven by the same controls:
/// every chassis pose, bit for bit, every step.
#[test]
fn pie_equals_shipping_for_a_plane_a_helicopter_and_a_boat() {
    use inf_editor_core::simulate::{SimInput, SimSession};
    const PLANE: Uuid = Uuid::from_u128(0x5E3E_0101);
    const HELI: Uuid = Uuid::from_u128(0x5E3E_0102);
    const CAR: Uuid = Uuid::from_u128(0x5E3E_0103);
    const SKI: Uuid = Uuid::from_u128(0x5E3E_0104);
    const YACHT: Uuid = Uuid::from_u128(0x5E3E_0105);
    let (dodo, bob, car, ski, yacht) = (
        row("mammoth_dodo"),
        row("cargobob"),
        row("albany_washington"),
        row("nagasaki_seashark"),
        row("dundreary_marquis"),
    );
    let lab = Lab {
        strip: (
            DVec3::new(0.0, -0.5, -2_300.0),
            DVec3::new(300.0, 0.5, 1_700.0),
        ),
        sea_centre: DVec3::new(0.0, 0.0, 3_400.0),
        // Blowing toward -Z: a headwind for the aeroplane's take-off roll and a
        // beam wind for the yacht heading +X.
        wind: (0.0, -8.0),
    };
    let craft: [(Uuid, VehicleDef, DVec3, f64); 5] = [
        (
            PLANE,
            dodo,
            DVec3::new(
                0.0,
                inf_ecs::vehicle::resting_origin_y(&dodo, 0.0),
                -3_900.0,
            ),
            0.0,
        ),
        (
            HELI,
            bob,
            DVec3::new(
                100.0,
                inf_ecs::vehicle::resting_origin_y(&bob, 0.0) + 12.0,
                -1_000.0,
            ),
            0.0,
        ),
        (
            CAR,
            car,
            DVec3::new(
                100.0,
                inf_ecs::vehicle::resting_origin_y(&car, 0.0),
                -1_000.0,
            ),
            0.0,
        ),
        (
            SKI,
            ski,
            DVec3::new(
                -100.0,
                inf_ecs::vehicle::floating_origin_y(&ski, 0.0),
                200.0,
            ),
            0.0,
        ),
        (
            YACHT,
            yacht,
            DVec3::new(
                100.0,
                inf_ecs::vehicle::floating_origin_y(&yacht, 0.0),
                200.0,
            ),
            90.0,
        ),
    ];
    let mut world = lab.world(&craft, false);
    assert!(inf_ecs::vehicle::winch(
        &mut world, HELI, &bob, CAR, &car, 8.0
    ));
    world.propagate();
    let mut shipped = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
    let mut doc = lab.doc(&craft, false);
    let joint = inf_ecs::vehicle::winch_joint(HELI, &bob, &car, 8.0).unwrap();
    let e = doc.entity_of(CAR).unwrap();
    doc.world_mut().world_mut().entity_mut(e).insert(joint);
    doc.world_mut().propagate();
    let mut preview = SimSession::enter(&mut doc, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);

    let mut moved: BTreeMap<Uuid, f64> = BTreeMap::new();
    let mut plane_peak = 0.0f64;
    for i in 0..1_800u32 {
        let plane_pitch = pitch_deg(body_state(&shipped, PLANE).1);
        let plane_ias = flight(&shipped, PLANE).airspeed_mps;
        let (hp, _, hv) = body_state(&shipped, HELI);
        let controls = [
            (
                PLANE,
                VehicleControls {
                    throttle: 1.0,
                    vertical: if plane_ias > stall_speed(&dodo) * 1.1 {
                        pitch_hold(8.0, plane_pitch)
                    } else {
                        0.0
                    },
                    occupied: true,
                    ..Default::default()
                },
            ),
            (
                HELI,
                VehicleControls {
                    vertical: collective(((30.0 - hp.y) * 0.5).clamp(-3.0, 4.0), hv.y),
                    occupied: true,
                    ..Default::default()
                },
            ),
            (
                SKI,
                VehicleControls {
                    throttle: 1.0,
                    occupied: true,
                    ..Default::default()
                },
            ),
        ];
        for (g, c) in controls {
            if let Some(v) = shipped.bridge3d_mut().vehicle_mut(g) {
                v.control(c);
            }
            if let Some(v) = preview.bridge3d_mut().vehicle_mut(g) {
                v.control(c);
            }
        }
        shipped.step_once(Default::default());
        preview.step_once(&mut doc, SimInput::default());
        let (a, b) = (pose_bits(shipped.world()), pose_bits(doc.world()));
        assert_eq!(a, b, "step {i}: the hosts' craft poses differ");
        for (g, _, at, _) in &craft {
            let d = (body_state(&shipped, *g).0 - *at).length();
            let m = moved.entry(*g).or_default();
            *m = m.max(d);
        }
        plane_peak = plane_peak.max(body_state(&shipped, PLANE).0.y);
    }
    println!(
        "PIE == SHIPPING: 1 800 steps, 5 craft bit-identical; the Dodo reached {plane_peak:.1} m, \
         the winched car moved {:.1} m, the jetski {:.1} m, the yacht {:.1} m",
        moved[&CAR], moved[&SKI], moved[&YACHT]
    );
    let rest = craft[0].2.y;
    assert!(
        plane_peak - rest > 5.0,
        "the plane never flew ({:.1} m over its rest)",
        plane_peak - rest
    );
    assert!(moved[&CAR] > 15.0, "the winched car never rose");
    assert!(moved[&SKI] > 200.0, "the jetski never ran");
    assert!(moved[&YACHT] > 30.0, "the yacht never sailed");
}

// ── 6. THE VOICES ────────────────────────────────────────────────────────────

/// **The craft voices reach the command stream on both hosts** -- a helicopter,
/// a propeller aeroplane, a jet and a jetski, each with an engine emitter and a
/// pilot: the rotor, propeller, turbine, hull-slap and hull-spray clips are all
/// `Play`ed; the rotor's pitch is its blade-pass over the clip's reference;
/// nothing is dropped on either host; and the two streams are the same commands.
///
/// **Mutation → red**: `RotorVehicle::voice` removed (no rotor Play); the
/// planner's `craft_voices` not pushed (no craft clip at all).
#[test]
fn the_craft_voices_reach_the_command_stream_on_both_hosts() {
    use inf_audio::AudioCommand;
    use inf_ecs::vehicle_audio as va;
    use inf_editor_core::simulate::{SimInput, SimSession};
    const HELI: Uuid = Uuid::from_u128(0x5E3E_0201);
    const PROP: Uuid = Uuid::from_u128(0x5E3E_0202);
    const JET: Uuid = Uuid::from_u128(0x5E3E_0203);
    const SKI: Uuid = Uuid::from_u128(0x5E3E_0204);
    let (buzz, dodo, luxor, ski) = (
        row("nagasaki_buzzard"),
        row("mammoth_dodo"),
        row("buckingham_luxor_deluxe"),
        row("nagasaki_seashark"),
    );
    let lab = Lab {
        strip: (
            DVec3::new(0.0, -0.5, -2_300.0),
            DVec3::new(400.0, 0.5, 1_700.0),
        ),
        sea_centre: DVec3::new(0.0, 0.0, 4_100.0),
        wind: (0.0, 0.0),
    };
    let craft: [(Uuid, VehicleDef, DVec3, f64); 4] = [
        (
            HELI,
            buzz,
            DVec3::new(
                -100.0,
                inf_ecs::vehicle::resting_origin_y(&buzz, 0.0),
                -1_500.0,
            ),
            0.0,
        ),
        (
            PROP,
            dodo,
            DVec3::new(
                0.0,
                inf_ecs::vehicle::resting_origin_y(&dodo, 0.0),
                -3_800.0,
            ),
            0.0,
        ),
        (
            JET,
            luxor,
            DVec3::new(
                150.0,
                inf_ecs::vehicle::resting_origin_y(&luxor, 0.0),
                -3_800.0,
            ),
            0.0,
        ),
        (
            SKI,
            ski,
            DVec3::new(0.0, inf_ecs::vehicle::floating_origin_y(&ski, 0.0), 1_000.0),
            0.0,
        ),
    ];
    let mut world = lab.world(&craft, true);
    world.propagate();
    let mut shipped = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
    let mut doc = lab.doc(&craft, true);
    doc.world_mut().propagate();
    let mut preview = SimSession::enter(&mut doc, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
    for i in 0..1_500u32 {
        for (g, _, _, _) in &craft {
            let c = VehicleControls {
                throttle: 1.0,
                vertical: if *g == HELI { 0.4 } else { 0.0 },
                occupied: true,
                ..Default::default()
            };
            if let Some(v) = shipped.bridge3d_mut().vehicle_mut(*g) {
                v.control(c);
            }
            if let Some(v) = preview.bridge3d_mut().vehicle_mut(*g) {
                v.control(c);
            }
        }
        shipped.step_once(Default::default());
        preview.step_once(&mut doc, SimInput::default());
        assert_eq!(
            shipped.dropped_audio_commands(),
            0,
            "the shipped log evicted at step {i}"
        );
        assert_eq!(
            preview.dropped_audio_commands(),
            0,
            "the preview log evicted at step {i}"
        );
    }
    let (a, b) = (shipped.audio_command_log(), preview.audio_command_log());
    assert_eq!(
        a.len(),
        b.len(),
        "the two hosts queued different stream lengths"
    );
    assert!(a == b, "the two hosts queued different commands");
    let played: BTreeMap<Uuid, usize> = a.iter().fold(BTreeMap::new(), |mut m, c| {
        if let AudioCommand::Play(p) = c {
            *m.entry(p.clip).or_default() += 1;
        }
        m
    });
    let rotor_key = va::voice_key(va::entity_key(HELI), va::VoiceLayer::Rotor);
    let rotor_pitch = a
        .iter()
        .rev()
        .find_map(|c| match c {
            AudioCommand::SetPitch { source, pitch } if *source == rotor_key => Some(*pitch),
            AudioCommand::Play(p) if p.source == rotor_key => Some(p.pitch),
            _ => None,
        })
        .unwrap_or(f64::NAN);
    let ratio = shipped
        .vehicles()
        .iter()
        .find(|o| o.chassis == HELI)
        .and_then(|o| o.voice)
        .map(|v| v.craft.rotor_hz / va::ROTOR_REF_HZ)
        .unwrap_or(f64::NAN);
    println!(
        "THE CRAFT VOICES: {} commands on each host, identical; Plays: rotor {}, prop {}, jet {}, \
         slap {}, spray {}; the rotor's last pitch {rotor_pitch:.4} against its blade-pass ratio \
         {ratio:.4}",
        a.len(),
        played.get(&va::rotor_clip()).copied().unwrap_or(0),
        played.get(&va::prop_clip()).copied().unwrap_or(0),
        played.get(&va::jet_clip()).copied().unwrap_or(0),
        played.get(&va::hull_slap_clip()).copied().unwrap_or(0),
        played.get(&va::hull_spray_clip()).copied().unwrap_or(0)
    );
    for (what, clip) in [
        ("rotor", va::rotor_clip()),
        ("propeller", va::prop_clip()),
        ("turbine", va::jet_clip()),
        ("hull slap", va::hull_slap_clip()),
        ("hull spray", va::hull_spray_clip()),
    ] {
        assert!(
            played.contains_key(&clip),
            "the {what} voice was never played"
        );
    }
    assert!(
        (rotor_pitch - ratio).abs() < 0.02,
        "the rotor sings at {rotor_pitch:.4} and its blade-pass says {ratio:.4}"
    );
    assert!(ratio > 0.9, "the rotor never spooled ({ratio:.3})");
}

// ── 7. THE COST ──────────────────────────────────────────────────────────────

/// **Eighteen air and sea craft cost what they print** -- the vehicle phase with
/// the five aeroplanes (on a strip), the five helicopters (hovering) and the
/// eight hulls (on a lake) all under power, against 18 of the roster's sedans on
/// the same strip: the minimum of five 40-step windows each, printed; asserted
/// against `VEHICLE_STEP_BUDGET_MS` on a release build off CI only.
#[test]
fn eighteen_air_and_sea_craft_cost_what_they_print() {
    fn window(ids: &[&str]) -> f64 {
        let mut world = sea_world((0.0, 6.0));
        slab(
            &mut world,
            GROUND,
            DVec3::new(0.0, -0.5, -3_000.0),
            DVec3::new(1_200.0, 0.5, 2_000.0),
            0.9,
        );
        for (k, id) in ids.iter().enumerate() {
            let def = row(id);
            let g = Uuid::from_u128(0x5E3E_3000 + k as u128);
            let x = -1_000.0 + 110.0 * k as f64;
            let mounts = def.body.mounts();
            let hull = !mounts.is_empty()
                && mounts
                    .iter()
                    .all(|m| m.kind == inf_ecs::vehicle::PartKind::Thruster);
            let rotor = mounts
                .iter()
                .any(|m| m.kind == inf_ecs::vehicle::PartKind::Rotor);
            let at = if hull {
                DVec3::new(
                    x,
                    inf_ecs::vehicle::floating_origin_y(&def, 0.0),
                    600.0 + 500.0 * (k % 2) as f64,
                )
            } else {
                DVec3::new(
                    x,
                    inf_ecs::vehicle::resting_origin_y(&def, 0.0) + if rotor { 20.0 } else { 0.0 },
                    -4_500.0,
                )
            };
            inf_ecs::vehicle::spawn_rig(&mut world, g, &def, &spawn_spec("Craft", at, 0.0, false));
        }
        world.propagate();
        let mut bridge = inf_physics::d3::PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        let drive = |bridge: &mut inf_physics::d3::PhysicsBridge3D| {
            for k in 0..ids.len() {
                let g = Uuid::from_u128(0x5E3E_3000 + k as u128);
                if let Some(v) = bridge.vehicle_mut(g) {
                    v.control(VehicleControls {
                        throttle: 1.0,
                        occupied: true,
                        ..Default::default()
                    });
                }
            }
        };
        for _ in 0..90 {
            drive(&mut bridge);
            inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
            bridge.step(DT);
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let mut clock = std::time::Duration::ZERO;
            for _ in 0..40 {
                drive(&mut bridge);
                let t = std::time::Instant::now();
                inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
                clock += t.elapsed();
                bridge.step(DT);
            }
            best = best.min(clock.as_secs_f64() * 1000.0 / 40.0);
        }
        best
    }
    const CRAFT_ROWS: [&str; 18] = [
        "western_duster",
        "mammoth_dodo",
        "buckingham_luxor_deluxe",
        "titan_cargo",
        "flyus_jetliner",
        "nagasaki_buzzard",
        "buckingham_maverick",
        "buckingham_volatus",
        "buckingham_swift_deluxe",
        "cargobob",
        "nagasaki_seashark",
        "shitzu_jetmax",
        "pegassi_speeder",
        "nagasaki_dinghy",
        "buckingham_tug",
        "dundreary_marquis",
        "galaxy_super_yacht",
        "oceanic_odyssey",
    ];
    let control = window(&["albany_washington"; 18]);
    let measured = window(&CRAFT_ROWS);
    println!(
        "THE AIR AND SEA STEP: 18 craft {measured:.4} ms against 18 sedans {control:.4} ms \
         ({:.2}x); {:.2} us a craft; the budget is {} ms at {} cars",
        measured / control,
        measured * 1000.0 / 18.0,
        inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        inf_player::budget::VEHICLE_BUDGET_CARS
    );
    assert!(measured.is_finite() && measured > 0.0 && control > 0.0);
    if cfg!(debug_assertions) {
        eprintln!("dev build: the craft step is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the craft step is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        measured <= inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        "18 craft cost {measured:.4} ms against {} ms {}",
        inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

// ── 8. HYGIENE ───────────────────────────────────────────────────────────────

/// **No std trig on the air and sea paths** -- the P14 law, read as source:
/// every angle a wing or a sail computes goes through `inf_math::portable`.
///
/// **Mutation → red**: `.atan2(` in `aero.rs`.
#[test]
fn no_std_trig_on_the_air_and_sea_paths() {
    for (name, src) in [
        (
            "aero.rs",
            include_str!("../../../crates/inf-ecs/src/aero.rs"),
        ),
        (
            "marine.rs",
            include_str!("../../../crates/inf-ecs/src/marine.rs"),
        ),
    ] {
        let code: String = src
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in inf_math::libm_ban::ALL {
            assert!(!code.contains(banned), "{name} reaches for `{banned}`");
        }
        assert!(
            code.contains("patan2_64"),
            "{name} names no portable angle at all"
        );
    }
}

/// **This wave moved no schema.**
#[test]
fn this_wave_moved_no_schema() {
    assert_eq!(inf_scene::SCHEMA_VERSION, 28);
    assert_eq!(inf_runtime::pie::SCENE_PAYLOAD_VERSION, 13);
    assert_eq!(
        inf_ecs::vehicle::VehicleTuning::names().len(),
        100,
        "the tunable count moved"
    );
}

/// **The shipped host draws the craft row and logs the columns** -- the HUD
/// calls the Ring-0 row, the hero log writes the eight craft columns, and the
/// row reads a wing, a hull and a cable.
#[test]
fn the_shipped_host_draws_the_craft_row_and_logs_the_columns() {
    let window = include_str!("../src/window.rs");
    assert!(window.contains(
        "inf_ecs::vehicle::craft_instruments(flight, marine, sim.winch_tension_n(vehicle))"
    ));
    let drive = include_str!("../src/pie_drive.rs");
    for col in [
        "c_alt",
        "c_ias",
        "c_alpha",
        "c_cl",
        "c_spool",
        "c_winch",
        "c_draught",
        "c_wind",
    ] {
        assert!(drive.contains(col), "pie_drive.rs does not log `{col}`");
    }
    let row = inf_ecs::vehicle::craft_instruments(
        Some(inf_ecs::aero::FlightState {
            airspeed_mps: 45.2,
            alpha_deg: 5.2,
            cl: 0.81,
            spool: 1.0,
            stalled: true,
            ..Default::default()
        }),
        Some(inf_ecs::marine::MarineState {
            draught_m: 0.061,
            planing_share: 0.8,
            ..Default::default()
        }),
        Some(18_624.0),
    );
    println!("THE CRAFT ROW: {row}");
    assert!(row.starts_with("FLIGHT  IAS 45.2 m/s  AOA 5.2  CL 0.81  THR 100%  STALL"));
    assert!(row.contains("HULL  DRAUGHT 0.061 m  PLANE 80%"));
    assert!(row.ends_with("WINCH 18624 N"));
    assert_eq!(inf_ecs::vehicle::craft_instruments(None, None, None), "");
}

// ── 9. THE REAL ISLAND (local only) ──────────────────────────────────────────

/// **The real island's runway is flat, and the Dodo leaves it** -- LOCAL ONLY
/// (the built terrain is a build artifact; CI skips and says so): the built
/// terrain sampled every 5 m along three lines under the paving sits at the
/// recipe's stated height to 5 cm, and a Dodo at the western threshold of the
/// committed LEVEL's own paving, at full power, lifts off before the far end.
#[test]
fn the_real_islands_runway_is_flat_and_the_dodo_leaves_it() {
    let content = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../island-build/project/Content");
    let terrain_path = content.join("VancouverIsland.inf_terrain");
    if !terrain_path.exists() {
        println!(
            "SKIP: no built island at {} (a build artifact; CI has none)",
            terrain_path.display()
        );
        return;
    }
    let recipe = inf_island::IslandRecipe::load(std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../samples/island/island.toml"
    )))
    .expect("the island recipe loads");
    let runway = recipe
        .airstrips
        .iter()
        .find(|a| !a.apron)
        .expect("a runway")
        .clone();
    let asset = inf_terrain::asset::read_terrain_asset(&terrain_path).expect("the terrain reads");
    let r = asset.reader();
    let res = r.tile_resolution();
    let mps = r.meters_per_sample();
    let span = (res - 1) as f64 * mps;
    let mut cache: BTreeMap<(i32, i32), Option<inf_terrain::TerrainTile>> = BTreeMap::new();
    let mut h = |x: f64, z: f64| -> Option<f64> {
        let (tx, tz) = ((x / span).floor() as i32, (z / span).floor() as i32);
        let t = cache
            .entry((tx, tz))
            .or_insert_with(|| {
                r.tile_bytes(inf_terrain::TileKey::lod0((tx, tz)))
                    .and_then(|b| inf_terrain::asset::decode_tile(&b).ok())
            })
            .as_ref()?;
        let u = ((x - t.origin.x) / mps)
            .round()
            .clamp(0.0, (res - 1) as f64) as u32;
        let v = ((z - t.origin.z) / mps)
            .round()
            .clamp(0.0, (res - 1) as f64) as u32;
        Some(t.world_height(res, u, v))
    };
    let along = runway.along();
    let across = glam::DVec2::new(-along.y, along.x);
    let centre = glam::DVec2::new(runway.x, runway.z);
    let (mut lo, mut hi, mut n) = (f64::MAX, f64::MIN, 0usize);
    let mut max_slope = 0.0f64;
    let mut prev: Option<f64> = None;
    let mut s = -runway.length_m * 0.5;
    while s <= runway.length_m * 0.5 {
        for w in [-0.45, 0.0, 0.45] {
            let p = centre + along * s + across * (w * runway.width_m);
            if let Some(y) = h(p.x, p.y) {
                lo = lo.min(y);
                hi = hi.max(y);
                n += 1;
                if w == 0.0 {
                    if let Some(q) = prev {
                        max_slope = max_slope.max((y - q).abs() / 5.0);
                    }
                    prev = Some(y);
                }
            }
        }
        s += 5.0;
    }
    println!(
        "THE REAL RUNWAY: {n} samples under the paving at {lo:.3}..{hi:.3} m against the stated \
         {} m; steepest 5 m slope along the centreline {:.4} %",
        runway.elevation_m,
        max_slope * 100.0
    );
    assert!(n > 900, "only {n} terrain samples under the runway");
    assert!(
        (lo - runway.elevation_m).abs() < 0.05 && (hi - runway.elevation_m).abs() < 0.05,
        "the ground under the runway is {lo:.3}..{hi:.3}, not {}",
        runway.elevation_m
    );

    // The committed level's own paving, and the Dodo on it.
    let lvl = std::path::Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../samples/island/VancouverIsland.inf_lvl"
    ));
    let doc = match inf_editor_core::scene::serialize::load(lvl) {
        Ok(d) => d,
        Err(e) => panic!("the committed level did not open: {e}"),
    };
    let mut world = EcsWorld::new();
    let mut paved = 0usize;
    for e in doc.world().world().iter_entities() {
        let (Some(name), Some(t), Some(c)) = (
            e.get::<inf_ecs::components::Name>(),
            e.get::<Transform>(),
            e.get::<Collider3D>(),
        ) else {
            continue;
        };
        if !name.0.starts_with(runway.name.as_str()) {
            continue;
        }
        let g = Uuid::from_u128(0x5E3E_4000 + paved as u128);
        let ent = world.spawn_with_guid(g, "Paving", None);
        world.world_mut().entity_mut(ent).insert((
            *t,
            Visibility::default(),
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..Default::default()
            },
            *c,
        ));
        paved += 1;
    }
    assert!(
        paved >= 7,
        "the committed level paves the runway in {paved} segments"
    );
    let def = row("mammoth_dodo");
    let top = runway.elevation_m + inf_editor_core::island::AIRSTRIP_LIP_M;
    let start = centre - along * (runway.length_m * 0.5 - 40.0);
    let at = DVec3::new(
        start.x,
        inf_ecs::vehicle::resting_origin_y(&def, top),
        start.y,
    );
    let yaw = inf_math::patan2_64(along.x, along.y).to_degrees();
    inf_ecs::vehicle::spawn_rig(&mut world, CRAFT, &def, &spawn_spec("Dodo", at, yaw, false));
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), glam::DVec2::new(0.0, -9.81), HZ);
    for _ in 0..90 {
        sim.step_once(Default::default());
    }
    let vr = stall_speed(&def) * 1.1;
    let mut lifted: Option<f64> = None;
    for _ in 0..(60 * 60) {
        let (p, r, _) = body_state(&sim, CRAFT);
        let f = flight(&sim, CRAFT);

        command(
            &mut sim,
            CRAFT,
            VehicleControls {
                throttle: 1.0,
                vertical: if f.airspeed_mps >= vr {
                    pitch_hold(8.0, pitch_deg(r))
                } else {
                    0.0
                },
                ..Default::default()
            },
        );
        if grounded(&sim, CRAFT) == 0 && p.y > at.y + 2.0 {
            lifted = Some((glam::DVec2::new(p.x, p.z) - start).dot(along));
            break;
        }
    }
    let roll = lifted.expect("the Dodo never left the real runway");
    println!(
        "THE REAL RUNWAY: {paved} paved segments; the Dodo was 2 m up {roll:.0} m from its \
         start, {:.0} m before the far threshold",
        runway.length_m - 40.0 - roll
    );
    assert!(
        roll < runway.length_m - 40.0,
        "it ran off the end ({roll:.0} m)"
    );
}
