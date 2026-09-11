//! **WAVE VEH3b — THE DRIVETRAIN.** The gate.
//!
//! Every claim of the wave, in one file, each arm **mutation-verified** and each
//! carrying an **engagement count** — so "the arm ran" and "something happened"
//! are two different facts and the second one is asserted.
//!
//! # What every arm in this file reads
//!
//! The WORLD, never a tuning table and never a report's opinion of itself. An
//! rpm is `DrivetrainState::rpm` after the solve published it; a launch is a
//! distance the rapier body actually travelled; a per-axle load is
//! `RaycastVehicle::axle_load_n` after the suspension pass wrote it; a trace
//! claim is real bytes out of the fold both hosts call.
//!
//! # Would a car with the OLD RIGID DRIVELINE pass?
//!
//! The question the audit brief asks of every arm, answered in the table. The
//! honest way to ask it is `flywheel_inertia_kgm2 = 0`, which is not a hostile
//! edit but the field's own documented sentinel: it restores the pre-VEH3b model
//! exactly, to the printed digit -- **3.98 s and 31.1 m on the sports row, and
//! VEH2a's own pair on the other four** (`audit:` VEH3b; the wave's prose said
//! 30.8 m, which is a number out of its own defect narrative and not this
//! measurement, and the true one is the stronger claim).
//! `vehicle_grade::the_flywheel_sentinel_restores_the_pre_veh3b_feel_table` is
//! where that is an arm rather than a sentence. Seven of
//! the arms below use it as their CONTROL rather than as a mutation, which is
//! the same fact read the useful way round.
//!
//! | arm | the mutation that reds it | engagement counted | rigid driveline? |
//! |---|---|---|---|
//! | `a_launch_flares_the_crank` | `flywheel_inertia_kgm2` → 0 | slipping steps, and the rigid CONTROL | **fails** |
//! | `a_downshift_flares_the_crank` | `flywheel_inertia_kgm2` → 0 | downshifts seen, steps of flare | **fails** |
//! | `the_limiter_cuts_and_restores` | `FUEL_CUT_HYSTERESIS_RPM` → a band nothing escapes, or the cut branch deleted | cut edges, steps at the limiter | **fails** |
//! | `the_three_differentials_launch_differently` | `lsd_transfer_nm` → 0 | contact steps on each of three runs | passes — this is a diff claim, not a crank one |
//! | `the_lsd_transfer_is_a_preload_and_a_ramp` | the same, one level down — the anti-vacuity arm beside it | six properties of the function | passes, by design |
//! | `the_boost_spools_and_blows_off` | `turbo_spool_s` → instant | steps with boost, and the NA control | **fails** |
//! | `the_axle_loads_are_the_formulas` | `cog_height_m` → 0 | braked steps, launched steps | passes — weight transfer predates this wave |
//! | `the_nose_dives_and_the_tail_squats` | `cog_height_m` → 0 | steps of each | passes |
//! | `every_drivetrain_tunable_moves_the_world` | any of the thirteen ignored | thirteen fields, each with its own delta | **fails on twelve of thirteen** |
//! | `the_differential_word_is_resolved_before_the_numbers` | the branch moved after the numeric loop | rows parsed | n/a — a codec claim |
//! | `the_clutch_is_never_weaker_than_its_engine` | the floor dropped from `clutch_capacity_nm` | rows checked, and a real launch | **fails** |
//! | `a_quiet_level_folds_no_drivetrain_bytes` | `DrivetrainState::is_quiet` → `false` | parked cars, driven steps | passes — the section would be empty either way, which is the point |
//! | `the_shipped_host_draws_the_drivetrain_row` | the call deleted from `inf_player::window` | three source fragments | n/a — a source pin |
//! | `the_hud_row_says_what_the_drivetrain_knows` | the row's own `format!` | four rows built from real state |  n/a |
//! | `the_vehicle_phase_costs_what_it_prints` | any per-wheel derivation put back | cars, control + measured | n/a |
//! | `two_runs_of_one_drive_fold_the_same_bytes` | a clock or an RNG anywhere in `crank_step` | 213 distinct states of 240, plus a source ban | passes |
//! | `a_class_edited_after_creation_reaches_the_car_only_through_the_tuner` | the live-tuner half deleted from `pie_drive.rs` | two 360-step drives, 0.000 boost against 0.812 | passes |
//!
//! # PIE == shipping, the mirrors, and the wire
//!
//! Not duplicated here, for `veh3a_gate`'s reason verbatim.
//! `island_gate::pie_equals_shipping_on_an_island_drive` drives the cooked
//! island on both hosts and compares them step for step; it is green at this
//! HEAD **with the crank, the clutch, the limiter, the differentials and the
//! turbo in the model and folded into the trace**, which is the re-bless this
//! wave owes and its cause. The mirror fences are
//! `inf-editor-core::fixed_step_mirror` and `projector_mirror` (whose `SECTIONS`
//! allowlist grew its seventeenth row in the same commit that folded it). The
//! thirteen fields' wire round-trip is
//! `veh3a_gate::every_v28_tunable_survives_the_wire`, which walks all hundred
//! through the editor's own codec — this wave adds no persisted field and
//! therefore adds no codec claim.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Terrain, Transform, Visibility,
};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{
    drivetrain_readout, drivetrain_state_bytes, lsd_transfer_nm, DrivetrainState, SurfaceClass,
    VehicleControls, VehicleDefs, VehicleTuning, DRIVETRAIN_TRACE_BYTES,
};
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;
use inf_terrain::TerrainData;

const DT: f64 = 1.0 / 60.0;
const CHASSIS: Uuid = Uuid::from_u128(0x5E3B_0001);
const GROUND: Uuid = Uuid::from_u128(0x5E3B_0002);
const PAD: Uuid = Uuid::from_u128(0x5E3B_0003);
const SLAB: Uuid = Uuid::from_u128(0x5E3B_0004);

/// The tyre's radius in the fixture rig, metres.
const WHEEL_RADIUS: f64 = 0.35;
/// Where a wheel centre sits below the chassis origin at full extension.
const WHEEL_Y: f64 = -0.6;
/// Half the width of the fixture's terrain, metres.
const HALF: f64 = 200.0;

// ── the fixture ─────────────────────────────────────────────────────────────

/// A flat terrain with a sealed pad over it, and a car on the pad.
///
/// The pad is a real `Collider3D` at a sealed friction, so `surface_under`
/// classifies it through the same band the island's kerbs and bridges go
/// through — the fixture says *this is a road* the way content says it.
struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
}

fn ground_world() -> EcsWorld {
    let mut world = EcsWorld::default();
    let mut data = TerrainData::new(64, 2.0);
    let n = (HALF / 64.0) as i32 + 1;
    for tz in -n..=n {
        for tx in -n..=n {
            data.author_tile((tx, tz), |_, _| 0.0);
        }
    }
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform {
            translation: Vec3d::new(0.0, -0.5, 0.0),
            ..Default::default()
        })
        .insert(Visibility::default())
        .insert(Terrain {
            meters_per_sample: 2.0,
            tile_resolution: 64,
            data,
            ..Default::default()
        })
        .insert(RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        })
        .insert(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(HALF, 0.5, HALF),
            ..Default::default()
        });
    world
}

/// Lay a slab of a named surface over part of the field.
fn slab(world: &mut EcsWorld, guid: Uuid, centre_x: f64, half_x: f64, mu: f64) {
    let e = world.spawn_with_guid(guid, "Slab", None);
    world
        .world_mut()
        .entity_mut(e)
        .insert(Transform {
            translation: Vec3d::new(centre_x, 0.02, 0.0),
            ..Default::default()
        })
        .insert(Visibility::default())
        .insert(RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        })
        .insert(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(half_x, 0.02, HALF),
            friction: mu,
            ..Default::default()
        });
}

fn car(world: &mut EcsWorld, at: DVec3) {
    car_from(world, at, &inf_ecs::vehicle::VehicleDef::default());
}

/// The same, from a CATALOGUE row's own definition (`audit:` VEH3b) — its body,
/// its mass, its travel and its spring, which is what clause 5 has to be
/// measured on if it is to be about a car this engine ships.
fn catalogue_def(id: &str) -> inf_ecs::vehicle::VehicleDef {
    *inf_editor_core::vehicle::island_vehicles()
        .get(id)
        .unwrap_or_else(|| panic!("the catalogue has no `{id}` row"))
}

fn car_from(world: &mut EcsWorld, at: DVec3, def: &inf_ecs::vehicle::VehicleDef) {
    let spawn = inf_ecs::vehicle::RigSpawn {
        name: "Runner".into(),
        at,
        yaw_deg: 0.0,
        paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
        clip: None,
        engine_voice: false,
        livery: None,
    };
    inf_ecs::vehicle::spawn_rig(world, CHASSIS, def, &spawn);
}

impl Rig {
    /// A car on a sealed pad.
    fn sealed() -> Self {
        Self::build(&[(PAD, 0.0, 60.0, 0.9)])
    }

    /// A CATALOGUE ROW on a sealed pad (`audit:` VEH3b) — its own body, mass,
    /// travel and spring, settled on its own suspension.
    fn sealed_row(id: &str) -> Self {
        let def = catalogue_def(id);
        let mut world = ground_world();
        slab(&mut world, PAD, 0.0, 60.0, 0.9);
        // A hand's lift over the pad, exactly as the island's own wizard parks
        // one (`CAR_LIFT_M`): the springs settle it on the first steps.
        let y = inf_ecs::vehicle::resting_origin_y(&def, 0.02) + 0.15;
        car_from(&mut world, DVec3::new(0.0, y, -80.0), &def);
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        Self { world, bridge }
    }

    /// A car with its LEFT wheels on mud and its right on sealed road — the
    /// one-wheel-on-grass launch the differential arms need.
    fn split() -> Self {
        Self::build(&[
            (PAD, 30.0, 30.0, 0.9),
            (SLAB, -30.0, 30.0, SurfaceClass::Mud.dry_mu()),
        ])
    }

    /// A car twenty metres up, with nothing under it for the whole of a short
    /// window (`audit:` VEH3b). Sixty steps of free fall is 4.9 m, so every
    /// wheel is in the air for every step of it -- which is the one place a
    /// wheel's speed can be changed without changing the chassis at all.
    fn airborne() -> Self {
        Self::at_height(&[(PAD, 0.0, 60.0, 0.9)], 20.0)
    }

    fn build(slabs: &[(Uuid, f64, f64, f64)]) -> Self {
        Self::at_height(slabs, -WHEEL_Y + WHEEL_RADIUS + 0.04)
    }

    fn at_height(slabs: &[(Uuid, f64, f64, f64)], y: f64) -> Self {
        let mut world = ground_world();
        for (guid, cx, hx, mu) in slabs {
            slab(&mut world, *guid, *cx, *hx, *mu);
        }
        car(&mut world, DVec3::new(0.0, y, -80.0));
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        Self { world, bridge }
    }

    fn tune(&mut self, pairs: &[(&str, f64)]) {
        for (name, value) in pairs {
            assert!(
                self.bridge
                    .vehicle_mut(CHASSIS)
                    .is_some_and(|v| v.tune(name, *value)),
                "the fixture set an unknown tunable `{name}`"
            );
        }
    }

    fn step(&mut self, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
    }

    fn drive(&mut self, c: VehicleControls, n: u32) {
        for _ in 0..n {
            if let Some(v) = self.bridge.vehicle_mut(CHASSIS) {
                v.control(c);
            }
            self.step(1);
        }
    }

    fn speed(&self) -> f64 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_linvel(b))
            .map(|v| DVec3::new(v.x, 0.0, v.z).length())
            .unwrap_or(0.0)
    }

    fn at(&self) -> DVec3 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_translation(b))
            .unwrap_or(DVec3::ZERO)
    }

    fn drivetrain(&self) -> DrivetrainState {
        self.bridge
            .vehicle_of(CHASSIS)
            .and_then(|v| v.drivetrain())
            .unwrap_or_default()
    }

    fn wheel(&self, i: usize) -> inf_ecs::vehicle::WheelState {
        self.bridge
            .vehicle_of(CHASSIS)
            .map(|v| v.wheels()[i])
            .unwrap_or_default()
    }

    fn wheels(&self) -> Vec<inf_ecs::vehicle::WheelState> {
        self.bridge
            .vehicle_of(CHASSIS)
            .map(|v| v.wheels().to_vec())
            .unwrap_or_default()
    }

    /// The load the suspension pass put on one axle this step, newtons.
    fn axle_load_n(&self, front: bool) -> f64 {
        self.bridge
            .vehicle_of(CHASSIS)
            .map(|v| {
                v.rig()
                    .wheels
                    .iter()
                    .zip(v.wheels())
                    .filter(|(m, _)| m.steered() == front)
                    .map(|(_, w)| w.load_n)
                    .sum()
            })
            .unwrap_or(0.0)
    }

    /// The mean suspension compression on one axle, metres.
    fn axle_compression_m(&self, front: bool) -> f64 {
        let rest = self
            .bridge
            .vehicle_of(CHASSIS)
            .map(|v| v.suspension_rest_m())
            .unwrap_or(0.0);
        let mut sum = 0.0;
        let mut n = 0usize;
        if let Some(v) = self.bridge.vehicle_of(CHASSIS) {
            for (m, w) in v.rig().wheels.iter().zip(v.wheels()) {
                if m.steered() == front {
                    sum += rest - w.length_m;
                    n += 1;
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            sum / n as f64
        }
    }

    fn mass_kg(&self) -> f64 {
        self.bridge
            .body_of(CHASSIS)
            .and_then(|b| self.bridge.world().body_mass(b))
            .unwrap_or(0.0)
    }

    /// The rig's own wheelbase, metres — derived from the mounts exactly as the
    /// model derives it.
    fn wheelbase_m(&self) -> f64 {
        self.bridge
            .vehicle_of(CHASSIS)
            .map(|v| {
                2.0 * v
                    .rig()
                    .wheels
                    .iter()
                    .map(|w| w.mount_local.z.abs())
                    .fold(0.0f64, f64::max)
            })
            .unwrap_or(0.0)
    }
}

/// A settled car: ninety steps on its springs with nobody touching it.
fn settled(mut rig: Rig) -> Rig {
    rig.step(90);
    rig
}

// ── 1. THE FLYWHEEL AND THE CLUTCH ──────────────────────────────────────────

/// **A LAUNCH FLARES THE CRANK, AND A RIGID DRIVELINE CANNOT** (wave VEH3b
/// clause 1).
///
/// The headline of the wave. A parked car's clutch is open; the throttle opens;
/// the clutch bites over `clutch_engage_s` and the crank — which now has
/// `flywheel_inertia_kgm2` of its own — rises ABOVE the speed the gearing
/// implies while the car is still crawling. That gap IS the flare, and a rigid
/// driveline cannot produce it by construction: `max(idle, wheels)` is a
/// function of the wheels and the wheels are barely turning.
///
/// **The mutation**: `flywheel_inertia_kgm2` → 0, which is the field's own
/// documented sentinel and restores the pre-VEH3b model exactly. It runs here as
/// the CONTROL rather than as a hostile edit, so the arm prints both numbers.
#[test]
fn a_launch_flares_the_crank() {
    let launch = |inertia: f64| -> (f64, f64, usize, usize) {
        let mut rig = settled(Rig::sealed());
        rig.tune(&[("flywheel_inertia_kgm2", inertia)]);
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        let (mut peak, mut flare, mut slipping) = (0.0f64, 0.0f64, 0usize);
        let mut contacts = 0usize;
        for _ in 0..60 {
            rig.drive(full, 1);
            let d = rig.drivetrain();
            peak = peak.max(d.rpm);
            // **THE FLARE IS THE CRANK'S EXCESS OVER THE GEARING**, not its
            // excess over idle. A rigid driveline revs too — it revs because the
            // car is moving — and the first cut of this arm compared the two
            // rises and found the RIGID one bigger, because a rigid car
            // accelerates harder off the line. What a flywheel buys is the gap
            // between the crank and the wheels, and that gap is exactly
            // `clutch_slip_rad_s`, which is zero for a rigid driveline by
            // construction.
            flare = flare.max(inf_ecs::vehicle::crank_rpm(d.clutch_slip_rad_s));
            if d.clutch_slip_rad_s > 0.0 {
                slipping += 1;
            }
            contacts += usize::from(rig.wheel(0).contact.is_some());
        }
        (flare, peak, slipping, contacts)
    };
    let (flare, peak, slipping, contacts) = launch(0.18);
    let (rigid_flare, rigid_peak, rigid_slipping, rigid_contacts) = launch(0.0);
    println!("VEH3b LAUNCH: the crank ran {flare:.0} rpm ahead of its own gearing, peaking at {peak:.0} rpm, slipping on {slipping} of 60 steps ({contacts} contact-steps)");
    println!("VEH3b LAUNCH (rigid control, flywheel 0): {rigid_flare:.0} rpm ahead, peak {rigid_peak:.0} rpm, {rigid_slipping} slipping steps");
    assert!(
        contacts > 50 && rigid_contacts > 50,
        "the car spent the launch in the air: {contacts} and {rigid_contacts} contact-steps of 60"
    );
    assert!(
        slipping > 5,
        "the clutch slipped on {slipping} of sixty steps of a standing start, so nothing was measured"
    );
    assert!(
        flare > 200.0,
        "the crank never got more than {flare:.0} rpm ahead of its gearing at a standing start"
    );
    assert_eq!(
        rigid_flare, 0.0,
        "a rigid driveline ran {rigid_flare:.0} rpm ahead of its own gearing, which it cannot do"
    );
    assert_eq!(
        rigid_slipping, 0,
        "a rigid driveline slipped its clutch on {rigid_slipping} steps, so the sentinel is not restoring the old model"
    );
}

/// **A DOWNSHIFT FLARES THE CRANK THROUGH THE CLUTCH** (wave VEH3b clause 1).
///
/// The other half of the same mechanism, and the one the research doc's *"rather
/// than applying direct force to the chassis"* is really about: on a downshift
/// the gearing implies a HIGHER crank speed than the crank has, so the box
/// drives the engine — through the clutch, bounded by what it takes to sync —
/// and the needle sweeps up over several steps instead of teleporting.
///
/// **What a rigid driveline does instead** is the control: the rpm is a pure
/// function of the wheels, so at the instant the gear changes it JUMPS, in one
/// step, with nothing in between. The arm measures the number of steps the
/// sweep takes.
///
/// **The mutation**: `flywheel_inertia_kgm2` → 0.
#[test]
fn a_downshift_flares_the_crank() {
    let run = |inertia: f64| -> (usize, f64, usize, usize) {
        let mut rig = settled(Rig::sealed());
        // **The governor is lifted for this arm**, and the reason is a fact
        // about the Ring-0 rig worth recording: its `max_speed_mps` is 25 and
        // its second gear reaches 5 300 rpm there, so the shipped default car
        // uses TWO of its six gears and can only ever make one downshift.
        // Nothing about a downshift is different in fifth.
        rig.tune(&[("flywheel_inertia_kgm2", inertia), ("max_speed_mps", 70.0)]);
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..1_800 {
            rig.drive(full, 1);
            if rig.speed() > 45.0 {
                break;
            }
        }
        // A PART brake rather than a panic stop: a car slowing gently
        // downshifts through the whole box, where a 0.9 g stop is over in two
        // gears and a trailing throttle alone barely slows this rig at all
        // (measured: 900 steps of coasting from 25 m/s and not one downshift).
        // Recorded and then read, because a sweep is a property of a window and
        // not of a step.
        let coast = VehicleControls {
            brake: 0.6,
            ..Default::default()
        };
        let mut trace: Vec<(f64, i32)> = Vec::new();
        let mut contacts = 0usize;
        for _ in 0..900 {
            rig.drive(coast, 1);
            contacts += usize::from(rig.wheel(0).contact.is_some());
            let d = rig.drivetrain();
            trace.push((d.rpm, d.gear));
            if rig.speed() < 2.0 {
                break;
            }
        }
        let (mut downshifts, mut worst_sweep, mut worst_steps) = (0usize, 0.0f64, 0usize);
        for i in 1..trace.len() {
            if trace[i].1 >= trace[i - 1].1 {
                continue;
            }
            downshifts += 1;
            let from = trace[i - 1].0;
            let mut top = from;
            let mut at = 0usize;
            for (k, row) in trace.iter().enumerate().skip(i).take(40) {
                if row.0 > top {
                    top = row.0;
                    at = k - i + 1;
                }
            }
            if top - from > worst_sweep {
                worst_sweep = top - from;
                worst_steps = at;
            }
        }
        let gmin = trace.iter().map(|t| t.1).min().unwrap_or(0);
        let gmax = trace.iter().map(|t| t.1).max().unwrap_or(0);
        println!(
            "VEH3b DOWNSHIFT: the box walked gears {gmin}..{gmax} over {} braked steps",
            trace.len()
        );
        (downshifts, worst_sweep, worst_steps, contacts)
    };
    let (downs, sweep, steps, contacts) = run(0.18);
    let (rigid_downs, rigid_sweep, rigid_steps, rigid_contacts) = run(0.0);
    println!("VEH3b DOWNSHIFT: {downs} downshifts on a part brake, the biggest sweep {sweep:.0} rpm over {steps} steps ({contacts} contact-steps)");
    println!("VEH3b DOWNSHIFT (rigid control): {rigid_downs} downshifts, biggest sweep {rigid_sweep:.0} rpm over {rigid_steps} steps ({rigid_contacts} contact-steps)");
    assert!(
        downs >= 2 && rigid_downs >= 2,
        "the box downshifted {downs} and {rigid_downs} times on the way down, so there is little to measure"
    );
    assert!(
        contacts > 100 && rigid_contacts > 100,
        "the car coasted in the air: {contacts} and {rigid_contacts} contact-steps"
    );
    assert!(
        sweep > 200.0,
        "the biggest downshift sweep was {sweep:.0} rpm, which is not a flare"
    );
    assert!(
        steps > 2,
        "the crank reached the new gear's speed in {steps} steps, which is a jump and not a sweep"
    );
    assert!(
        rigid_steps <= 1,
        "the RIGID control took {rigid_steps} steps to reach the new gear's speed — a rigid driveline has no in-between"
    );
}

/// **THE LIMITER CUTS AND RESTORES, AND THE PLATEAU IS GONE** (wave VEH3b
/// clause 2 — VEH2a's carried item).
///
/// `engine_torque_nm`'s own doc has carried this since VEH2a: *"It is a
/// **plateau**, not a fuel cut: past the redline this answers
/// `redline_torque_frac × peak_torque_nm` for ever rather than falling to zero
/// … Nothing in this model cuts fuel."* It does now, and the shape of the answer
/// is the point: a cut with [`inf_ecs::vehicle::FUEL_CUT_HYSTERESIS_RPM`] of
/// hysteresis BOUNCES, so the trace has edges in it.
///
/// **How the world is made to reach the limiter**: the box is told to hold first
/// gear (`shift_up_rpm` above the redline) and the driven wheels are put on mud
/// with the traction control off, so they spin up into the cut the way a car
/// leaving a wet junction does. On dry tarmac this rig cannot reach its own
/// limiter at any throttle a script can apply, which is the same arithmetic the
/// VEH3a audit recorded about the burnout frame.
///
/// **The mutation**: `fuel_cut_rpm` raised above the over-rev ceiling, which is
/// the plateau restored — the crank pins and the edges vanish.
#[test]
fn the_limiter_cuts_and_restores() {
    let run = |cut_rpm: f64| -> (usize, f64, f64, f64, usize, usize, usize) {
        let mut rig = settled(Rig::build(&[(SLAB, 0.0, 60.0, SurfaceClass::Mud.dry_mu())]));
        rig.tune(&[
            ("shift_up_rpm", 20_000.0),
            ("traction_control_slip", 0.0),
            ("stability_control", 0.0),
            ("fuel_cut_rpm", cut_rpm),
        ]);
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        let (mut edges, mut hi) = (0usize, 0.0f64);
        let (mut cutting, mut at_limit, mut contacts) = (false, 0usize, 0usize);
        // The SAWTOOTH, read between the limiter's own edges: how far the crank
        // falls between a cut and the restore that follows it, and how many
        // steps that takes. A plateau has no edges at all and therefore no saw.
        let (mut saw_amp, mut saw_period, mut saws) = (0.0f64, 0usize, 0usize);
        let (mut cut_at_rpm, mut cut_at_step) = (0.0f64, 0usize);
        for step in 0..600 {
            rig.drive(full, 1);
            let d = rig.drivetrain();
            contacts += usize::from(rig.wheel(0).contact.is_some());
            if d.rpm > 5_800.0 {
                at_limit += 1;
                hi = hi.max(d.rpm);
            }
            if d.fuel_cut != cutting {
                edges += 1;
                cutting = d.fuel_cut;
                if cutting {
                    cut_at_rpm = d.rpm;
                    cut_at_step = step;
                } else if cut_at_rpm > 0.0 {
                    saws += 1;
                    saw_amp += cut_at_rpm - d.rpm;
                    saw_period += step - cut_at_step;
                }
            }
        }
        let n = saws.max(1) as f64;
        (
            edges,
            hi,
            saw_amp / n,
            saw_period as f64 / n,
            at_limit,
            contacts,
            saws,
        )
    };
    // `0` is the sentinel: the limiter sits at the redline, 6 500 rpm.
    let (edges, hi, amp, period, at_limit, contacts, saws) = run(0.0);
    // Above the over-rev ceiling the cut can never fire, which is the plateau.
    let (flat_edges, flat_hi, _, _, flat_at_limit, _, _) = run(9_000.0);
    println!("VEH3b LIMITER: {edges} cut edges over ten seconds and {saws} complete saws, mean amplitude {amp:.0} rpm over a mean period of {period:.1} steps, peaking at {hi:.0} rpm on {at_limit} steps at the limiter ({contacts} contact-steps)");
    println!("VEH3b LIMITER (the plateau restored): {flat_edges} edges, peaking at {flat_hi:.0} rpm on {flat_at_limit} steps");
    assert!(
        at_limit > 30 && flat_at_limit > 30,
        "the crank never got near its limiter: {at_limit} and {flat_at_limit} steps above 5 800 rpm"
    );
    assert!(
        edges >= 4 && saws >= 2,
        "the limiter made {edges} cut edges and {saws} complete saws, so it is not bouncing — it is a switch that went once"
    );
    assert_eq!(
        flat_edges, 0,
        "the plateau control cut fuel {flat_edges} times, so the mutation is not restoring the plateau"
    );
    assert!(
        amp > 50.0,
        "the crank fell {amp:.0} rpm between a cut and its restore, which is a plateau with a flag on it"
    );
    assert!(
        (1.0..60.0).contains(&period),
        "the limiter's mean period is {period:.1} steps"
    );
    assert!(
        hi < flat_hi,
        "the limiter let the crank to {hi:.0} rpm and the plateau to {flat_hi:.0} — the cut is not holding it down"
    );
}

// ── 2. THE DIFFERENTIALS ────────────────────────────────────────────────────

/// **OPEN, LSD AND LOCKED LAUNCH DIFFERENTLY WITH ONE WHEEL ON MUD**
/// (wave VEH3b clause 3).
///
/// The research doc's own test of a differential: *"a limited-slip differential
/// dynamically balances output torque between left and right wheels based on
/// velocity delta"*, and what that BUYS is a car that drives away from a wet
/// manhole. The fixture puts the left-hand wheels on mud (µ 0.35) and the right
/// on sealed road (µ 1.00) and lets each of the three differentials launch.
///
/// **What it reads**: the speed difference across the driven axle after three
/// seconds, and the distance the car actually covered — both off the world, one
/// from `WheelState::omega_rad_s` and one from the rapier body.
///
/// **The mutation**: `lsd_preload_rear_nm` and both rear ramps → 0, which is the
/// open differential — the LSD row's numbers collapse onto the open row's.
#[test]
fn the_three_differentials_launch_differently() {
    let launch = |name: &str| -> (f64, f64, usize) {
        let mut rig = settled(Rig::split());
        // The scalar lock is the SPOOL and the six v28 fields refine it: `open`
        // is all seven at zero, `locked` is the lock at one, and `lsd` is the
        // clutch pack. Set here through the live door rather than through a
        // catalogue row, because the row's own resolution is its own arm below.
        match name {
            "open" => rig.tune(&[
                ("diff_lock_rear", 0.0),
                ("lsd_preload_rear_nm", 0.0),
                ("lsd_power_ramp_rear", 0.0),
            ]),
            "lsd" => rig.tune(&[
                ("diff_lock_rear", 0.0),
                ("lsd_preload_rear_nm", 60.0),
                ("lsd_power_ramp_rear", 0.6),
            ]),
            _ => rig.tune(&[
                ("diff_lock_rear", 1.0),
                ("lsd_preload_rear_nm", 0.0),
                ("lsd_power_ramp_rear", 0.0),
            ]),
        }
        // Rear drive and no traction control, so the differential is the only
        // thing deciding where the torque goes.
        rig.tune(&[("front_torque_split", 0.0), ("traction_control_slip", 0.0)]);
        let from = rig.at();
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        let mut contacts = 0usize;
        for _ in 0..180 {
            rig.drive(full, 1);
            contacts += usize::from(rig.wheel(2).contact.is_some());
        }
        let w = rig.wheels();
        let delta = (w[2].omega_rad_s - w[3].omega_rad_s).abs();
        ((rig.at() - from).length(), delta, contacts)
    };
    let (open_m, open_d, open_c) = launch("open");
    let (lsd_m, lsd_d, lsd_c) = launch("lsd");
    let (lock_m, lock_d, lock_c) = launch("locked");
    println!("VEH3b DIFF open:   {open_m:.2} m in 3 s, rear axle {open_d:.1} rad/s apart ({open_c} contact-steps)");
    println!("VEH3b DIFF lsd:    {lsd_m:.2} m in 3 s, rear axle {lsd_d:.1} rad/s apart ({lsd_c} contact-steps)");
    println!("VEH3b DIFF locked: {lock_m:.2} m in 3 s, rear axle {lock_d:.1} rad/s apart ({lock_c} contact-steps)");
    for (n, c) in [("open", open_c), ("lsd", lsd_c), ("locked", lock_c)] {
        assert!(
            c > 150,
            "the {n} run spent the launch in the air: {c} of 180 contact-steps"
        );
    }
    assert!(
        open_d > 1.0,
        "the OPEN differential left its axle {open_d:.2} rad/s apart on a split surface, so the muddy wheel is not spinning and this fixture has no split in it"
    );
    assert!(
        lsd_d < open_d * 0.9,
        "the LSD left the axle {lsd_d:.1} rad/s apart against the open one's {open_d:.1} — it is not biasing anything"
    );
    // **A SPOOL STARVES AND A CLUTCH PACK BRAKES**, and that is why the LSD
    // ends up TIGHTER than the lock rather than looser. The scalar lock hands
    // the axle to the slowest wheel and stops feeding the fastest — which is
    // what the model's own doc says it does and is not the same thing as tying
    // the two shafts together — so the spun-up wheel coasts down on tyre drag
    // alone. The clutch pack applies a NEGATIVE torque to it, which is the
    // mechanism a real limited-slip differential has and the reason the two
    // rows below are not interchangeable.
    assert!(
        lock_d < open_d * 0.5,
        "the LOCKED axle was {lock_d:.1} rad/s apart against the open one's {open_d:.1} — it is not starving the runaway"
    );
    assert!(
        lsd_d < open_d * 0.5,
        "the LSD left the axle {lsd_d:.1} rad/s apart against the open one's {open_d:.1}"
    );
    assert!(
        lsd_m > open_m * 1.02,
        "the LSD covered {lsd_m:.2} m against the open differential's {open_m:.2} — it bought nothing"
    );
    assert!(
        lock_m > open_m * 1.02,
        "the LOCKED axle covered {lock_m:.2} m against the open differential's {open_m:.2}"
    );
}

/// **`differential = "lsd"` IS RESOLVED BEFORE THE NUMBERS** (wave VEH3b clause
/// 3) — the `drivetrain` ruling applied to the second enum an author reaches
/// for.
///
/// A codec claim and it says so. What it pins is the ORDER: the word sets seven
/// fields, and an explicit numeric key in the same table wins over it whatever
/// order the TOML map happened to iterate in. That is the whole reason
/// `drivetrain` is resolved where it is, and a second spelling resolved after
/// the loop would be a table whose meaning depended on a hash.
///
/// **The mutation**: move the `differential` branch below the numeric loop — the
/// third row's explicit `lsd_power_ramp_rear` is then overwritten by the word.
#[test]
fn the_differential_word_is_resolved_before_the_numbers() {
    let mut defs = VehicleDefs::default();
    let rows = defs
        .merge_toml(concat!(
            "[o.vehicle]\ndifferential = \"open\"\n\n",
            "[l.vehicle]\ndifferential = \"locked\"\n\n",
            "[s.vehicle]\ndifferential = \"lsd\"\n\n",
            "[x.vehicle]\ndifferential = \"lsd\"\nlsd_power_ramp_rear = 0.11\ndiff_lock_rear = 0.42\n",
        ))
        .expect("the catalogue parses");
    assert_eq!(rows, 4, "the fixture declared four rows and {rows} parsed");
    let c = |id: &str| defs.0.get(id).expect("the row").class;
    assert_eq!(c("o").diff_lock_rear, 0.0);
    assert_eq!(c("o").lsd_power_ramp_rear, 0.0);
    assert_eq!(c("l").diff_lock_rear, 1.0);
    assert_eq!(c("s").diff_lock_rear, 0.0);
    assert_eq!(
        c("s").lsd_preload_front_nm,
        inf_ecs::vehicle::LSD_ROW_PRELOAD_NM
    );
    assert!(c("s").lsd_power_ramp_rear > c("s").lsd_coast_ramp_rear);
    assert_eq!(
        c("x").lsd_power_ramp_rear,
        0.11,
        "an explicit ramp beside the word lost to the word, so the word is resolved after the numbers"
    );
    assert_eq!(c("x").diff_lock_rear, 0.42);
    // …and an unknown word is a refusal BY NAME, not a silent open diff.
    let err = VehicleDefs::default()
        .merge_toml("[x.vehicle]\ndifferential = \"torsen\"\n")
        .expect_err("an unknown differential is refused");
    assert!(err.contains("torsen"), "{err}");
    assert!(VehicleDefs::default()
        .merge_toml("[x.vehicle]\ndifferential = 3\n")
        .is_err());
    println!("VEH3b DIFF ROW: four rows parsed; `lsd` sets preload {:.0} N.m and ramps {:.2}/{:.2}; an explicit key beside the word still wins", c("s").lsd_preload_rear_nm, c("s").lsd_power_ramp_rear, c("s").lsd_coast_ramp_rear);
}

/// **THE LSD'S OWN ARITHMETIC** — the anti-vacuity arm beside the launch above.
///
/// The launch arm measures a car; this measures the function, for the reason
/// `veh3a_gate::the_magic_formula_is_not_the_old_curve` exists: a world
/// measurement cannot separate "the LSD worked" from "the fixture's two surfaces
/// did it". Four properties, each of which a wrong implementation breaks:
/// the preload bites with no ramp at all, the ramp scales with the axle's
/// torque, the transfer is bounded, and it engages over the speed difference
/// rather than switching.
#[test]
fn the_lsd_transfer_is_a_preload_and_a_ramp() {
    let band = inf_ecs::vehicle::DIFF_SPEED_BAND;
    assert_eq!(
        lsd_transfer_nm(0.0, 0.0, 500.0, 10.0),
        0.0,
        "an open differential transferred torque"
    );
    let preload_only = lsd_transfer_nm(60.0, 0.0, 0.0, 10.0);
    assert_eq!(
        preload_only, 60.0,
        "a preload with no torque through the axle transferred {preload_only} N.m"
    );
    let ramped = lsd_transfer_nm(0.0, 0.5, 400.0, 10.0);
    assert_eq!(ramped, 200.0, "a 0.5 ramp on 400 N.m transferred {ramped}");
    let capped = lsd_transfer_nm(60.0, 1.0, 400.0, 10.0);
    assert_eq!(
        capped,
        60.0 + 400.0 * inf_ecs::vehicle::LSD_MAX_BIAS_FRAC,
        "a ramp of 1.0 was not bounded: {capped} N.m"
    );
    let half = lsd_transfer_nm(0.0, 0.5, 400.0, band * 0.5);
    assert!(
        (half - 100.0).abs() < 1e-9,
        "at half the engagement band the transfer was {half} and not half of 200"
    );
    assert_eq!(
        lsd_transfer_nm(60.0, 0.6, 400.0, 0.0),
        0.0,
        "an axle whose wheels turn together was biased anyway"
    );
    println!("VEH3b LSD: preload {preload_only:.0} N.m alone, ramp {ramped:.0} N.m on a 400 N.m axle, bounded at {capped:.0}, half-engaged at {half:.0}");
}

// ── 3. THE TURBO ────────────────────────────────────────────────────────────

/// **THE BOOST SPOOLS, LAGS AND BLOWS OFF** (wave VEH3b clause 4).
///
/// A first-order state with a dead time in front of it, measured on a launch:
/// the rise time to half of the boost the throttle is asking for, and the drop
/// when the throttle shuts. The class the fixture uses is turbocharged through
/// the live tuning door, because **every catalogue row shipped today is
/// naturally aspirated** — `turbo_boost_max` is `0.0` on all eleven, which is
/// the second control this arm prints.
///
/// **The mutation**: `turbo_spool_s` → a step, which makes the boost arrive with
/// the throttle and collapses the rise to one step.
#[test]
fn the_boost_spools_and_blows_off() {
    let run = |boost_max: f64, spool: f64| -> (f64, usize, usize, f64, usize) {
        let mut rig = settled(Rig::sealed());
        rig.tune(&[
            ("turbo_boost_max", boost_max),
            ("turbo_spool_s", spool),
            ("turbo_lag_s", 0.15),
        ]);
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        let (mut peak, mut rise_steps, mut spooled, mut dead) = (0.0f64, 0usize, 0usize, 0usize);
        for i in 0..240 {
            rig.drive(full, 1);
            let b = rig.drivetrain().boost;
            if b > 0.0 {
                spooled += 1;
            } else if i < 60 {
                dead += 1;
            }
            if b < 0.5 {
                rise_steps = i + 1;
            }
            peak = peak.max(b);
        }
        // …and the valve: the throttle shuts and the plenum dumps.
        let coast = VehicleControls::default();
        let before = rig.drivetrain().boost;
        rig.drive(coast, 6);
        let after = rig.drivetrain().boost;
        (peak, rise_steps, spooled, before - after, dead)
    };
    let (peak, rise, spooled, dumped, dead) = run(0.6, 0.7);
    let (fast_peak, fast_rise, _, _, fast_dead) = run(0.6, 1e-4);
    let (na_peak, _, na_spooled, _, _) = run(0.0, 0.7);
    println!("VEH3b TURBO: peak boost {peak:.2}, half of it reached after {rise} steps, {spooled} steps on boost, {dead} steps of dead time, {dumped:.2} dumped in six steps");
    println!("VEH3b TURBO (spool made instant): peak {fast_peak:.2}, half reached after {fast_rise} steps, {fast_dead} dead steps");
    println!(
        "VEH3b TURBO (naturally aspirated control): peak {na_peak:.2} over {na_spooled} steps"
    );
    assert!(
        spooled > 60,
        "the turbo was on boost for {spooled} of 240 steps, so nothing was measured"
    );
    assert!(
        peak > 0.5,
        "the boost never got past {peak:.2} of its own peak at full throttle"
    );
    // **THE DEAD TIME, MEASURED WHERE IT IS THE DEAD TIME** (`audit:` VEH3b).
    //
    // This used to be `dead`: the steps of the first second with no boost, and
    // an assertion that there were at least eight of them, commented *"and
    // `turbo_lag_s` is nine of them"*. It is not. Boost is also zero below
    // `TURBO_THRESHOLD_FRAC` of the redline, which a standing start spends about
    // a second under, so the count is the THRESHOLD and not the lag — measured,
    // by forcing `turbo_lag_s` to zero inside `turbo_step`: the arm stayed GREEN
    // and printed the same "60 steps of dead time".
    //
    // A dead time is the delay between asking and getting, so it is measured
    // with the revs already up: drive onto boost, LIFT (the valve dumps and the
    // dead time re-arms — the model's own documented behaviour), then open the
    // throttle again and count the steps until the shaft answers.
    let relag = |lag: f64| -> usize {
        let mut rig = settled(Rig::sealed());
        rig.tune(&[
            ("turbo_boost_max", 0.6),
            ("turbo_spool_s", 0.7),
            ("turbo_lag_s", lag),
        ]);
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        rig.drive(full, 240);
        assert!(
            rig.drivetrain().boost > 0.0,
            "the fixture never got on boost, so there is no lift to re-arm"
        );
        rig.drive(VehicleControls::default(), 10);
        assert_eq!(
            rig.drivetrain().boost,
            0.0,
            "the blow-off did not empty the plenum in ten steps"
        );
        for i in 0..120 {
            rig.drive(full, 1);
            if rig.drivetrain().boost > 0.0 {
                return i;
            }
        }
        120
    };
    let lagged = relag(0.15);
    let instant = relag(0.0);
    println!("VEH3b TURBO: with the revs already up, a 0.15 s dead time cost {lagged} steps between the throttle and the boost, and no dead time cost {instant}");
    assert!(
        dead >= 8,
        "the compressor made no boost for {dead} steps of a standing start — the THRESHOLD, not the lag; see `relag` below for the dead time itself"
    );
    assert!(
        lagged >= 8,
        "a 0.15 s dead time is nine steps and the boost arrived after {lagged}"
    );
    assert!(
        instant <= 1,
        "with `turbo_lag_s` at zero the boost still took {instant} steps to appear, so this pair is not measuring the dead time"
    );
    assert!(
        lagged > instant + 5,
        "the dead time cost {lagged} steps and its absence {instant} — the same turbo twice"
    );
    assert!(
        rise > fast_rise + 5,
        "the boost took {rise} steps to half with a 0.7 s spool and {fast_rise} with none, which is the same turbo twice"
    );
    assert!(
        dumped > 0.3,
        "the blow-off dumped {dumped:.2} of boost in a tenth of a second"
    );
    assert_eq!(
        na_peak, 0.0,
        "a naturally-aspirated class made {na_peak:.2} of boost"
    );
    assert_eq!(na_spooled, 0);
    // Every catalogue row is naturally aspirated today, and that is a fact about
    // the content rather than about the model: wave VEH3f's roster is where a
    // turbocharged row lands.
    let turbos = inf_editor_core::vehicle::island_vehicles()
        .0
        .values()
        .filter(|d| d.class.turbo_boost_max > 0.0)
        .count();
    println!("VEH3b TURBO: {turbos} of the eleven catalogue rows carry a turbocharger");
    assert_eq!(
        turbos, 0,
        "{turbos} catalogue rows are turbocharged, so this wave's `0` control is no longer the shipped case"
    );
}

// ── 4. WEIGHT TRANSFER ──────────────────────────────────────────────────────

/// **THE PER-AXLE LOADS ARE THE DOC'S FORMULA, WITHIN TEN PER CENT**
/// (wave VEH3b clause 5).
///
/// The research doc asks for `Fz_front = Fz_static − m·a_x·h / L`. This engine
/// does not compute that at all: it has no door to move rapier's centre of mass,
/// so it applies the horizontal tyre force at `contact − up·cog_height_m` and
/// lets the SOLVER produce the moment. The claim under test is that the trick
/// and the formula agree, which is a claim about rapier and can only be made by
/// measuring both.
///
/// # `h` is the CENTRE OF GRAVITY'S HEIGHT ABOVE THE ROAD, and it is derived
///
/// Not `cog_height_m`, which is an offset from the chassis collider's CENTRE
/// (its default −0.25 puts the mass a quarter-metre into the floor of a 1.24 m
/// body). The height the formula wants is measured from the world: the chassis
/// body's own `y`, minus the contact point's, plus `cog_height_m`. Getting this
/// wrong is an eleven-to-one error and the first cut of this arm made it.
///
/// # The springs are stiffened, and that is a FINDING rather than a fudge
///
/// The Ring-0 default rig **sits on its bump stops under a 0.9 g stop**: static
/// compression is 0.149 m of a 0.25 m travel, so 0.10 m of bump is all there is
/// and a hard stop uses it in a fifth of a second — measured, both axles pinned
/// at `travel_m` from step 40 of a 150-step stop, on the rigid driveline
/// identically, so it is not this wave's. A car on its bump stops measures its
/// bump stops, so the fixture is given a spring that keeps the suspension inside
/// its own travel and the arm measures the moment it is named for. The
/// bottoming-out is carried by name — and `audit:` VEH3b re-ruled what it is:
/// `the_shipped_spring_bottoms_out_and_the_formula_is_what_pays` measures this
/// same stop on the SHIPPED spring and finds the transfer **118.8 % from the
/// formula with the sign inverted**, because `suspension_force_n` has no bump
/// stop at all. The 3.7 % below is a true statement about a 90 000 N/m fixture.
///
/// **The mutation**: `cog_height_m` → the chassis centre's own height below the
/// road, i.e. an `h` of zero — the force is applied AT the contact patch, the
/// moment goes to zero and the measured transfer collapses while the formula
/// still predicts hundreds of newtons.
#[test]
fn the_axle_loads_are_the_formulas() {
    // A spring stiff enough to stay inside its travel under 1 g, and a damper
    // scaled with it so the pitch settles rather than rings.
    const STIFF: [(&str, f64); 2] = [
        ("stiffness_n_per_m", 90_000.0),
        ("damping_ns_per_m", 9_000.0),
    ];
    let mut rig = settled(Rig::sealed());
    rig.tune(&STIFF);
    rig.step(120);
    let mass = rig.mass_kg();
    let wheelbase = rig.wheelbase_m();
    let cog = VehicleTuning::default().cog_height_m;
    let contact_y = rig
        .wheel(0)
        .contact
        .map(|c| c.point.y)
        .expect("the car is standing on something");
    let h = rig.at().y - contact_y + cog;
    let static_front = rig.axle_load_n(true);
    let static_rear = rig.axle_load_n(false);
    println!("VEH3b LOAD: {mass:.0} kg on a {wheelbase:.2} m wheelbase; the chassis centre is {:.3} m over the road and `cog_height_m` is {cog:.2}, so h = {h:.3} m; at rest the axles carry {static_front:.0} N front and {static_rear:.0} N rear", rig.at().y - contact_y);
    assert!(
        (static_front + static_rear - mass * 9.81).abs() < mass * 0.5,
        "the car weighs {:.0} N and its springs carry {:.0}",
        mass * 9.81,
        static_front + static_rear
    );
    assert!(
        h > 0.2,
        "the centre of gravity is {h:.3} m over the road, which is not a car"
    );

    // ── under the brakes ──
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    for _ in 0..900 {
        rig.drive(full, 1);
        if rig.speed() > 22.0 {
            break;
        }
    }
    let brake = VehicleControls {
        brake: 1.0,
        ..Default::default()
    };
    // The first forty steps are the pitch TRANSIENT — the suspension's own
    // natural period is about 0.5 s on this spring — and a transient is not the
    // equilibrium the formula describes. Skipped, and the skip is asserted to
    // have left something behind.
    rig.drive(brake, 40);
    let (mut braked, mut braked_front, mut braked_rear, mut braked_accel) =
        (0usize, 0.0f64, 0.0f64, 0.0f64);
    let mut last = rig.speed();
    for _ in 0..90 {
        rig.drive(brake, 1);
        let v = rig.speed();
        let a = (v - last) / DT;
        last = v;
        if a < -4.0 && v > 3.0 {
            braked += 1;
            braked_front += rig.axle_load_n(true);
            braked_rear += rig.axle_load_n(false);
            braked_accel += a;
        }
    }
    assert!(
        braked > 20,
        "only {braked} steps of the stop pulled more than 0.4 g, so there is no transfer to measure"
    );
    let measured = braked_front / braked as f64;
    let measured_rear = braked_rear / braked as f64;
    let a_x = braked_accel / braked as f64;
    let predicted = static_front - mass * a_x * h / wheelbase;
    println!("VEH3b LOAD: braking at {a_x:.2} m/s2 over {braked} steps the axles carried {measured:.0} N front and {measured_rear:.0} N rear, against the formula's {predicted:.0} N front (static {static_front:.0}, transfer {:.0} N)", -mass * a_x * h / wheelbase);
    // **The TRANSFER is the claim, not the total** — and the difference is a
    // vacuity this arm was measured to have. The static load is two thirds of
    // the number, so a ten-per-cent band on the TOTAL passes a transfer that is
    // twenty-six per cent wrong: measured with the force offset deleted, the
    // front read 9 929 N against a formula's 9 110 — nine per cent, green — while
    // the transfer underneath it was 3 972 against 3 153. The doc's formula IS
    // the term `m a h / L`, so that is what is banded.
    let moved = measured - static_front;
    let want = -mass * a_x * h / wheelbase;
    let err = (moved - want).abs() / want.abs().max(1.0);
    assert!(
        err < 0.10,
        "the front axle took {moved:.0} N of transfer under {a_x:.2} m/s2 and the doc's formula says {want:.0} N — {:.1} % apart",
        err * 100.0
    );

    // ── and under the launch ──
    let mut rig = settled(Rig::sealed());
    rig.tune(&STIFF);
    rig.step(120);
    let (mut launched, mut launched_rear, mut launched_accel) = (0usize, 0.0f64, 0.0f64);
    rig.drive(full, 40);
    let mut last = rig.speed();
    for _ in 0..120 {
        rig.drive(full, 1);
        let v = rig.speed();
        let a = (v - last) / DT;
        last = v;
        if a > 2.0 {
            launched += 1;
            launched_rear += rig.axle_load_n(false);
            launched_accel += a;
        }
    }
    assert!(
        launched > 20,
        "only {launched} steps of the launch pulled more than 0.2 g"
    );
    let measured_rear = launched_rear / launched as f64;
    let a_launch = launched_accel / launched as f64;
    let predicted_rear = static_rear + mass * a_launch * h / wheelbase;
    println!("VEH3b LOAD: launching at {a_launch:.2} m/s2 over {launched} steps the rear axle carried {measured_rear:.0} N against the formula's {predicted_rear:.0} N (static {static_rear:.0}, transfer {:.0} N)", mass * a_launch * h / wheelbase);
    let moved = measured_rear - static_rear;
    let want = mass * a_launch * h / wheelbase;
    let err = (moved - want).abs() / want.abs().max(1.0);
    assert!(
        err < 0.10,
        "the rear axle took {moved:.0} N of transfer under {a_launch:.2} m/s2 and the doc's formula says {want:.0} N — {:.1} % apart",
        err * 100.0
    );
}

/// **THE NOSE DIVES AND THE TAIL SQUATS** (wave VEH3b clause 5) — the same
/// transfer, read where a player sees it rather than where the solver holds it.
///
/// A load is a number; a nose-dive is a body moving. The suspension's own
/// compression is the bridge between them, and it is what the frames this wave
/// captures actually show.
///
/// On the same stiffened spring as the arm above and for the same measured
/// reason: on the shipped spring both axles reach `travel_m` and stay there, so
/// the dive reads 101.1 mm with a centre of gravity and 101.1 mm without one —
/// the bump stop, twice.
///
/// **The mutation**: `cog_height_m` → the chassis centre's own height below the
/// road, which is an `h` of zero.
#[test]
fn the_nose_dives_and_the_tail_squats() {
    let run = |cog: f64| -> (f64, f64, usize) {
        let mut rig = settled(Rig::sealed());
        rig.tune(&[
            ("stiffness_n_per_m", 90_000.0),
            ("damping_ns_per_m", 9_000.0),
            ("cog_height_m", cog),
        ]);
        rig.step(120);
        let rest_front = rig.axle_compression_m(true);
        let rest_rear = rig.axle_compression_m(false);
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        let mut squat = 0.0f64;
        let mut engaged = 0usize;
        for _ in 0..120 {
            rig.drive(full, 1);
            squat = squat.max(rig.axle_compression_m(false) - rest_rear);
            engaged += 1;
        }
        for _ in 0..900 {
            rig.drive(full, 1);
            if rig.speed() > 22.0 {
                break;
            }
        }
        let brake = VehicleControls {
            brake: 1.0,
            ..Default::default()
        };
        let mut dive = 0.0f64;
        for _ in 0..90 {
            rig.drive(brake, 1);
            dive = dive.max(rig.axle_compression_m(true) - rest_front);
            engaged += 1;
        }
        (dive * 1_000.0, squat * 1_000.0, engaged)
    };
    let (dive, squat, engaged) = run(VehicleTuning::default().cog_height_m);
    // The chassis centre sits 0.95 m over the road on this rig, so putting the
    // centre of gravity THERE is an `h` of zero.
    let (flat_dive, flat_squat, flat_engaged) = run(-0.95);
    println!("VEH3b BODY: the nose dived {dive:.1} mm under the brakes and the tail squatted {squat:.1} mm on the launch, over {engaged} steps");
    println!("VEH3b BODY (centre of gravity at road level): {flat_dive:.1} mm and {flat_squat:.1} mm over {flat_engaged} steps");
    assert!(
        engaged > 200 && flat_engaged > 200,
        "the arm ran {engaged} and {flat_engaged} steps"
    );
    assert!(
        dive > 2.0,
        "the nose dived {dive:.2} mm under full braking, which nobody would see"
    );
    assert!(
        squat > 1.0,
        "the tail squatted {squat:.2} mm on a full-throttle launch"
    );
    assert!(
        dive > flat_dive * 2.0,
        "the nose dived {dive:.1} mm with a centre of gravity and {flat_dive:.1} mm with it at road level"
    );
    assert!(
        squat > flat_squat * 1.5,
        "the tail squatted {squat:.1} mm with a centre of gravity and {flat_squat:.1} mm with it at road level"
    );

    // ── AND ON THE CAR THAT SHIPS ── (`audit:` VEH3b). Everything above is the
    //    Ring-0 fixture on a spring stiffened for the arm; a player sees the
    //    SEDAN, on its own re-sprung rate, and a nose-dive is the one number in
    //    clause 5 that is a body moving rather than a load.
    let mut rig = settled(Rig::sealed_row("sedan"));
    rig.step(120);
    let rest_front = rig.axle_compression_m(true);
    let rest_rear = rig.axle_compression_m(false);
    let travel = catalogue_def("sedan").class.to_tuning().travel_m;
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    let mut squat = 0.0f64;
    for _ in 0..120 {
        rig.drive(full, 1);
        squat = squat.max(rig.axle_compression_m(false) - rest_rear);
    }
    for _ in 0..1_800 {
        rig.drive(full, 1);
        if rig.speed() > 28.0 {
            break;
        }
    }
    let brake = VehicleControls {
        brake: 1.0,
        ..Default::default()
    };
    let (mut dive, mut deepest) = (0.0f64, 0.0f64);
    for _ in 0..120 {
        rig.drive(brake, 1);
        let c = rig.axle_compression_m(true);
        deepest = deepest.max(c);
        dive = dive.max(c - rest_front);
    }
    println!(
        "VEH3b BODY (the shipped sedan): the nose dives {:.1} mm and the tail squats {:.1} mm; deepest front compression {:.1} mm of a {:.0} mm travel ({:.0} % -- the stop engages at 85)",
        dive * 1_000.0,
        squat * 1_000.0,
        deepest * 1_000.0,
        travel * 1_000.0,
        deepest / travel * 100.0
    );
    assert!(
        dive * 1_000.0 > 20.0,
        "the shipped sedan's nose dived {:.1} mm under full braking, which nobody would see",
        dive * 1_000.0
    );
    assert!(
        squat * 1_000.0 > 1.0,
        "the shipped sedan's tail squatted {:.2} mm on a launch",
        squat * 1_000.0
    );
    assert!(
        deepest < travel,
        "the shipped sedan reached {deepest:.3} m of its {travel:.2} m travel, so it is on its bump stop under the brakes"
    );
}

// ── 5. THE THIRTEEN FIELDS ──────────────────────────────────────────────────

/// **EVERY ONE OF THE THIRTEEN DRIVETRAIN TUNABLES MOVES THE WORLD**
/// (wave VEH3b, against VEH3a's carried item 5).
///
/// VEH3a spent the schema window and carried the honest consequence: *"31 of the
/// 38 have no consumer."* Thirteen of those thirty-one are this wave's, and this
/// is the arm that says so — not by reading `VehicleTuning::names()`, which
/// would pass for a field nothing reads, but by MUTATING each one and measuring
/// that the car ends somewhere else.
///
/// The wire round-trip is not repeated here:
/// `veh3a_gate::every_v28_tunable_survives_the_wire` walks all hundred through
/// the editor's own codec, and this wave adds no persisted field.
///
/// **One of the thirteen legitimately moves nothing on this fixture** and it is
/// named rather than hidden: `turbo_lag_s` has no effect on a class whose
/// `turbo_boost_max` is zero, which is every class in the tree. The arm turns the
/// turbo ON for the three turbo fields and says so.
#[test]
fn every_drivetrain_tunable_moves_the_world() {
    let run = |split: bool, pairs: &[(&str, f64)]| -> (DVec3, f64) {
        // **The LSD rows run on the SPLIT surface**, and they have to: a
        // differential only has something to do when its two wheels are turning
        // at different speeds, so on a uniform pad a preload of 400 N.m moved
        // the car by 0.19 micrometres — which reads exactly like a field nothing
        // consumes, and is the opposite. Measured before this fixture existed.
        let mut rig = settled(if split { Rig::split() } else { Rig::sealed() });
        rig.tune(pairs);
        // Two steps with the throttle shut, so a live-tuned `turbo_lag_s` gets
        // to re-arm: the dead time is a STATE seeded at construction and reset
        // by the blow-off, which is a real property of the model and is why an
        // author who retunes it mid-drive sees it from the next lift.
        rig.drive(VehicleControls::default(), 2);
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..240 {
            rig.drive(full, 1);
        }
        let brake = VehicleControls {
            brake: 1.0,
            ..Default::default()
        };
        for _ in 0..120 {
            rig.drive(brake, 1);
        }
        (rig.at(), rig.drivetrain().rpm)
    };
    let (base_at, base_rpm) = run(false, &[]);
    // The thirteen, each with a value a row might really author, and the three
    // turbo ones with the turbo switched on so the field has something to do.
    const TURBO: (&str, f64) = ("turbo_boost_max", 0.8);
    const NO_TC: (&str, f64) = ("traction_control_slip", 0.0);
    /// One case: the field's name, whether it needs the SPLIT surface, and the
    /// tuning that exercises it.
    type Case = (&'static str, bool, &'static [(&'static str, f64)]);
    let cases: [Case; 13] = [
        (
            "flywheel_inertia_kgm2",
            false,
            &[("flywheel_inertia_kgm2", 0.0)],
        ),
        ("clutch_torque_nm", false, &[("clutch_torque_nm", 3_000.0)]),
        ("clutch_engage_s", false, &[("clutch_engage_s", 2.5)]),
        ("fuel_cut_rpm", false, &[("fuel_cut_rpm", 3_000.0)]),
        ("turbo_boost_max", false, &[TURBO]),
        ("turbo_spool_s", false, &[TURBO, ("turbo_spool_s", 6.0)]),
        ("turbo_lag_s", false, &[TURBO, ("turbo_lag_s", 3.0)]),
        (
            "lsd_preload_rear_nm",
            true,
            &[
                ("lsd_preload_rear_nm", 400.0),
                ("front_torque_split", 0.0),
                NO_TC,
            ],
        ),
        (
            "lsd_preload_front_nm",
            true,
            &[
                ("lsd_preload_front_nm", 400.0),
                ("front_torque_split", 1.0),
                NO_TC,
            ],
        ),
        (
            "lsd_power_ramp_rear",
            true,
            &[
                ("lsd_power_ramp_rear", 1.0),
                ("front_torque_split", 0.0),
                NO_TC,
            ],
        ),
        (
            "lsd_power_ramp_front",
            true,
            &[
                ("lsd_power_ramp_front", 1.0),
                ("front_torque_split", 1.0),
                NO_TC,
            ],
        ),
        (
            "lsd_coast_ramp_rear",
            true,
            &[
                ("lsd_coast_ramp_rear", 1.0),
                ("front_torque_split", 0.0),
                NO_TC,
            ],
        ),
        (
            "lsd_coast_ramp_front",
            true,
            &[
                ("lsd_coast_ramp_front", 1.0),
                ("front_torque_split", 1.0),
                NO_TC,
            ],
        ),
    ];
    let mut moved = 0usize;
    for (name, split, pairs) in cases {
        // The CONTROL is the same fixture and the same drivetrain with THIS
        // field left alone, so a row cannot pass on the split or the surface it
        // set for itself.
        let control: Vec<(&str, f64)> = pairs.iter().filter(|(n, _)| *n != name).copied().collect();
        let control = if control.is_empty() && !split {
            (base_at, base_rpm)
        } else {
            run(split, &control)
        };
        let (at, rpm) = run(split, pairs);
        let dm = (at - control.0).length();
        let drpm = (rpm - control.1).abs();
        println!("VEH3b FIELD {name:>22}: {dm:8.3} m and {drpm:7.1} rpm away from its own control");
        assert!(
            dm > 1e-6 || drpm > 1e-6,
            "`{name}` moved the car by {dm} m and the crank by {drpm} rpm — it is on the wire and nothing reads it"
        );
        moved += 1;
    }
    assert_eq!(moved, 13, "the arm checked {moved} of the thirteen");
}

/// **A CLUTCH IS NEVER WEAKER THAN THE ENGINE IT IS BOLTED TO** (wave VEH3b).
///
/// The defect this floor exists for, kept as an arm because it cost a wave an
/// afternoon: the eleven catalogue rows author an engine and do not author a
/// clutch, so the sports row inherited the Ring-0 420 N.m against its own
/// 460 N.m and slipped for the whole of every gear — 0-100 km/h went from 3.98 s
/// to **8.85**.
///
/// **The mutation**: drop the `peak_torque_nm × CLUTCH_TORQUE_MARGIN` floor from
/// `clutch_capacity_nm` and the sports row is torque-limited again.
#[test]
fn the_clutch_is_never_weaker_than_its_engine() {
    let mut thin = 0usize;
    let mut rows = 0usize;
    for (id, def) in inf_editor_core::vehicle::island_vehicles().0.iter() {
        rows += 1;
        let cap = def.class.to_tuning().clutch_capacity_nm();
        assert!(
            cap >= def.class.peak_torque_nm,
            "`{id}` carries a {cap:.0} N.m clutch against a {:.0} N.m engine",
            def.class.peak_torque_nm
        );
        if def.class.clutch_torque_nm < def.class.peak_torque_nm {
            thin += 1;
        }
    }
    println!("VEH3b CLUTCH: {rows} catalogue rows, {thin} of them author a `clutch_torque_nm` below their own peak torque and are carried by the floor");
    assert!(rows >= 11, "{rows} rows in the catalogue");
    assert!(
        thin > 0,
        "no catalogue row needs the floor, so this arm cannot fail and the floor is not load-bearing"
    );
    // …and the floor is a floor rather than a replacement: a clutch authored
    // ABOVE the engine's own margin keeps its number.
    let mut t = VehicleTuning {
        peak_torque_nm: 100.0,
        clutch_torque_nm: 5_000.0,
        ..Default::default()
    };
    assert_eq!(t.clutch_capacity_nm(), 5_000.0);
    t.clutch_torque_nm = 10.0;
    assert_eq!(
        t.clutch_capacity_nm(),
        100.0 * inf_ecs::vehicle::CLUTCH_TORQUE_MARGIN
    );
}

// ── 6. THE TRACE ────────────────────────────────────────────────────────────

/// **A QUIET LEVEL FOLDS NO DRIVETRAIN BYTES, AND A DRIVEN ONE FOLDS ITS CARS**
/// (wave VEH3b).
///
/// The seventeenth trace section's empty case is the half that keeps every trace
/// committed before this wave byte-identical: a parked car sits at idle with its
/// clutch open and its turbo cold, which is `DrivetrainState::is_quiet`, so a
/// level nobody is driving folds nothing at all.
///
/// **The mutation**: `is_quiet` → `false` — the parked level starts folding
/// fifty-three bytes a car and every committed trace on a level with a car in it
/// moves.
#[test]
fn a_quiet_level_folds_no_drivetrain_bytes() {
    let mut rig = settled(Rig::sealed());
    let parked = drivetrain_state_bytes(&rig.world);
    let quiet_state = rig.drivetrain();
    println!("VEH3b TRACE: a parked car folds {} bytes; its drivetrain is {} rpm, gear {}, clutch {:.2}, boost {:.2}", parked.len(), quiet_state.rpm, quiet_state.gear, quiet_state.clutch_lock, quiet_state.boost);
    assert!(
        parked.is_empty(),
        "a level with one parked car folded {} drivetrain bytes",
        parked.len()
    );
    assert!(
        quiet_state.is_quiet(800.0),
        "the parked car's drivetrain is not quiet: {quiet_state:?}"
    );
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    rig.drive(full, 60);
    let driving = drivetrain_state_bytes(&rig.world);
    let running = rig.drivetrain();
    println!("VEH3b TRACE: after a second of throttle the same level folds {} bytes ({DRIVETRAIN_TRACE_BYTES} a car) at {:.0} rpm in gear {}", driving.len(), running.rpm, running.gear);
    assert_eq!(
        driving.len(),
        DRIVETRAIN_TRACE_BYTES,
        "one driven car folded {} bytes",
        driving.len()
    );
    assert!(
        !running.is_quiet(800.0),
        "a car at full throttle reports a quiet drivetrain"
    );
    // The guid is the first sixteen bytes, so a reader can tell whose crank it
    // is — and a fold that lost the key would be a fold two hosts could agree
    // about while disagreeing about which car was doing it.
    assert_eq!(&driving[..16], CHASSIS.as_bytes());
    // …and letting it roll to a stop makes it quiet again, which is what the
    // "empty" half has to mean if it is to hold on the island.
    rig.drive(VehicleControls::default(), 600);
    let stopped = drivetrain_state_bytes(&rig.world);
    println!(
        "VEH3b TRACE: ten seconds later, coasted to {:.2} m/s, the level folds {} bytes again",
        rig.speed(),
        stopped.len()
    );
    assert!(
        stopped.is_empty(),
        "a car that coasted to {:.2} m/s still folds {} bytes",
        rig.speed(),
        stopped.len()
    );
}

// ── 7. THE HUD ──────────────────────────────────────────────────────────────

/// **THE HUD ROW SAYS WHAT THE DRIVETRAIN KNOWS** (wave VEH3b clause 7).
#[test]
fn the_hud_row_says_what_the_drivetrain_knows() {
    let mut rig = settled(Rig::sealed());
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    // THREE steps, not twenty: `clutch_engage_s` is a quarter of a second, so
    // by step twenty the clutch has locked and the row correctly stops drawing a
    // column for it.
    rig.drive(full, 3);
    let d = rig.drivetrain();
    let row = drivetrain_readout(&d, false);
    println!("VEH3b HUD: {row}");
    assert!(
        row.contains("rpm"),
        "the drivetrain row does not name the revs: {row}"
    );
    assert!(
        !row.contains("boost"),
        "a naturally-aspirated car drew a boost column: {row}"
    );
    assert!(
        row.contains("clutch"),
        "the clutch is at {:.2} on a standing start and the row does not say so: {row}",
        d.clutch_lock
    );
    // A locked clutch draws no column, and a turbocharged car draws one.
    let locked = DrivetrainState {
        rpm: 3_210.0,
        gear: 3,
        clutch_lock: 1.0,
        clutch_slip_rad_s: 0.0,
        boost: 0.42,
        fuel_cut: false,
    };
    let row = drivetrain_readout(&locked, true);
    println!("VEH3b HUD: {row}");
    assert!(
        !row.contains("clutch"),
        "a locked clutch drew a column: {row}"
    );
    assert!(row.contains("boost 0.42"), "{row}");
    assert!(row.contains("3210 rpm"), "{row}");
    assert!(row.contains("   3"), "{row}");
    assert!(!row.contains("CUT"), "{row}");
    let cutting = DrivetrainState {
        fuel_cut: true,
        ..locked
    };
    let row = drivetrain_readout(&cutting, false);
    println!("VEH3b HUD: {row}");
    assert!(
        row.ends_with("CUT"),
        "the limiter is cutting and the row does not say so: {row}"
    );
}

/// **THE SHIPPED HOST DRAWS THE DRIVETRAIN ROW** — the source pin beside the
/// Ring-0 arm above, on `veh3a_gate::the_shipped_host_draws_the_tyre_row`'s
/// terms exactly.
///
/// `drivetrain_readout` is in Ring 0 precisely so a host function need not be
/// tested, which also means **nothing else can see its call site**. The VEH3a
/// audit measured that: deleting the `tyre_readout` call from the player left
/// the Ring-0 arm green. So the call site is pinned by source.
#[test]
fn the_shipped_host_draws_the_drivetrain_row() {
    const WINDOW: &str = include_str!("../src/window.rs");
    for fragment in [
        "inf_ecs::vehicle::drivetrain_readout",
        "let drivetrain = v.drivetrain();",
        "let turbocharged = v.turbocharged();",
    ] {
        assert!(
            WINDOW.contains(fragment),
            "`inf_player::window` no longer carries `{fragment}`, so the shipped host has stopped drawing the drivetrain row"
        );
    }
    // The player is the only host that draws a driving readout at all — the
    // editor draws neither `craft_readout` nor either row, which the VEH3a audit
    // established and is recorded here rather than re-discovered.
    println!("VEH3b HUD: the shipped player draws the drivetrain row; the editor draws no driving readout at all");
}

// ── 8. THE BUDGET ───────────────────────────────────────────────────────────

/// **THE VEHICLE PHASE COSTS WHAT IT PRINTS** — the budget arm, at the
/// population it names (wave VEH3b clause 6).
///
/// `VEHICLE_STEP_BUDGET_MS` is 0.5 **at 64 cars** and §8 forbids raising it.
/// VEH3a measured 0.3981 ms over a warm control; this wave adds a crank step, a
/// differential pass and a turbo step per vehicle, and a `BTreeMap` rebuild for
/// the trace.
///
/// A clock, so — like every budget arm in the house — it is REPORTED in every
/// build and ASSERTED only under `cargo test --release` off CI. VEH3a's arm
/// reddened three shared runners by asserting a dev build before it took this
/// conditioning.
#[test]
fn the_vehicle_phase_costs_what_it_prints() {
    const CARS: usize = 64;
    let build = |n: usize| -> Rig {
        let mut world = ground_world();
        slab(&mut world, PAD, 0.0, 80.0, 0.9);
        for i in 0..n {
            let guid = Uuid::from_u128(0x5E3B_1000 + i as u128);
            let def = inf_ecs::vehicle::VehicleDef::default();
            let spawn = inf_ecs::vehicle::RigSpawn {
                name: "Fleet".into(),
                at: DVec3::new(
                    (i % 8) as f64 * 8.0 - 32.0,
                    -WHEEL_Y + WHEEL_RADIUS + 0.04,
                    (i / 8) as f64 * 8.0 - 32.0,
                ),
                yaw_deg: 0.0,
                paint: inf_ecs::math::Color::new(0.2, 0.2, 0.6, 1.0),
                clip: None,
                engine_voice: false,
                livery: None,
            };
            inf_ecs::vehicle::spawn_rig(&mut world, guid, &def, &spawn);
        }
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        let mut rig = Rig { world, bridge };
        // WARMED: settled cars on their springs, not ones still falling.
        rig.step(90);
        rig
    };
    let time = |rig: &mut Rig| -> f64 {
        let mut best = f64::MAX;
        for _ in 0..5 {
            rig.bridge.sync_from_world(&rig.world);
            let t0 = std::time::Instant::now();
            for _ in 0..40 {
                inf_physics::d3::step_vehicles(&mut rig.world, &mut rig.bridge, DT);
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1_000.0 / 40.0);
        }
        best
    };
    let mut control = build(0);
    let mut measured = build(CARS);
    assert_eq!(
        measured.bridge.vehicle_count(),
        CARS,
        "the fixture built {CARS} cars and the bridge derived {}",
        measured.bridge.vehicle_count()
    );
    let c = time(&mut control);
    let m = time(&mut measured);
    let per_car = (m - c) * 1_000.0 / CARS as f64;
    println!("VEH3b BUDGET: {CARS} cars cost {m:.4} ms a step over a {c:.4} ms control — {per_car:.2} µs a car (budget {:.1} ms, VEH3a measured 0.3981)", inf_player::budget::VEHICLE_STEP_BUDGET_MS);
    assert!(
        m - c > 0.0,
        "sixty-four cars cost {} ms more than none, so this arm timed nothing",
        m - c
    );
    if cfg!(debug_assertions) {
        eprintln!("dev build: the vehicle phase is reported, not asserted");
        return;
    }
    if std::env::var_os("CI").is_some() {
        eprintln!("CI: the vehicle phase is reported, not asserted (shared runner)");
        return;
    }
    assert!(
        m <= inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        "the vehicle phase costs {m:.4} ms a step at {CARS} cars against a {} ms ceiling {}",
        inf_player::budget::VEHICLE_STEP_BUDGET_MS,
        inf_player::budget::RATCHET_NOTE
    );
}

// ── 9. DETERMINISM ──────────────────────────────────────────────────────────

/// **THE DRIVETRAIN IS A PURE FUNCTION OF THE STEP HISTORY** (wave VEH3b clause
/// 6) — no wall clock, no RNG, no platform trigonometry on the committed path.
///
/// Two runs of the same fixture with the same inputs produce the same bytes,
/// step for step. It is the cheap half of the PIE-versus-shipping claim and it
/// is here because it can fail for a reason that gate cannot see: **two hosts
/// that both read a clock agree with each other**.
#[test]
fn two_runs_of_one_drive_fold_the_same_bytes() {
    let run = || -> Vec<Vec<u8>> {
        let mut rig = settled(Rig::split());
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        let mut out = Vec::new();
        for i in 0..240 {
            let c = if i < 180 {
                full
            } else {
                VehicleControls {
                    brake: 1.0,
                    ..Default::default()
                }
            };
            rig.drive(c, 1);
            out.push(drivetrain_state_bytes(&rig.world));
        }
        out
    };
    let a = run();
    let b = run();
    let distinct: std::collections::BTreeSet<&Vec<u8>> = a.iter().collect();
    println!(
        "VEH3b DETERMINISM: {} steps, {} distinct drivetrain states, {} bytes at its widest",
        a.len(),
        distinct.len(),
        a.iter().map(|s| s.len()).max().unwrap_or(0)
    );
    assert!(
        distinct.len() > 100,
        "only {} of {} states differ — the drive is not moving the drivetrain",
        distinct.len(),
        a.len()
    );
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(x, y, "two runs of one drive diverged at step {i}");
    }
    // …and no `std` trigonometry reached the crank: the whole of `crank_step`
    // and `turbo_step` is arithmetic, and `lsd_transfer_nm` is arithmetic. The
    // portable-math law's subject in this crate is the tyre curve, which VEH3a
    // owns.
    //
    // **BOUNDED AT EACH FUNCTION'S OWN CLOSING BRACE** (`audit:` VEH3b), and
    // the reason is this wave's own scar one file over: the ban read a flat
    // 6 000-character window over a `crank_step` that measures **5 905**, which
    // is ninety-five characters of slack — one added comment line from silently
    // un-banning the tail of the function it names. `projector_mirror`'s twin
    // window was six thousand characters too, wave VEH3b's own entry call landed
    // at offset 6 001, and the fix that commit wrote in its own comment is the
    // one taken here: *"the honest fix is to bound the block at its own closing
    // brace rather than to move this number again."*
    //
    // And it scans all THREE functions the sentence above names. Two of them
    // were prose: `turbo_step` and `lsd_transfer_nm` were claimed arithmetic and
    // read by nothing.
    const VEHICLE: &str = include_str!("../../../crates/inf-ecs/src/vehicle.rs");
    let body_of = |decl: &str| -> &str {
        let start = VEHICLE
            .find(decl)
            .unwrap_or_else(|| panic!("`{decl}` is not in `vehicle.rs` at all"));
        let body = &VEHICLE[start..];
        // A top-level `fn` closes at column zero, so this is the function and
        // nothing after it — however long it grows.
        let end = body
            .find(
                "
}
",
            )
            .unwrap_or_else(|| panic!("`{decl}` never closes at column zero"));
        &body[..end]
    };
    let mut scanned = 0usize;
    for decl in [
        "pub fn crank_step(",
        "pub fn turbo_step(",
        "pub fn lsd_transfer_nm(",
    ] {
        let body = body_of(decl);
        // Non-vacuity: a window that found nothing would ban nothing.
        assert!(
            body.len() > 400,
            "`{decl}`'s body read back as {} characters, so this ban is scanning a window and not a function",
            body.len()
        );
        scanned += body.len();
        for banned in [
            ".sin()",
            ".cos()",
            ".tan()",
            ".atan2(",
            "rand",
            "SystemTime",
            "Instant",
        ] {
            assert!(
                !body.contains(banned),
                "`{decl}` reaches `{banned}`, which is not portable and not deterministic"
            );
        }
    }
    println!(
        "VEH3b DETERMINISM: {scanned} characters of crank, turbo and differential scanned for platform trigonometry, a clock and an RNG — each bounded at its own closing brace"
    );
}

/// **A CLASS EDITED AFTER THE CAR EXISTS REACHES IT ONLY THROUGH THE TUNER**
/// (wave VEH3b) — the seam `INF_PIE_TUNE_VEHICLE` was only half across.
///
/// `PhysicsBridge3D::reconcile_vehicles` installs an authored `VehicleClass`
/// **once, at creation**, and says why in its own doc: *"the component is the
/// STARTING point; the tuner owns it from there"* — re-installing every step
/// would silently undo every edit an author made during Simulate. That ruling
/// is right and this arm does not challenge it. What it pins is the
/// CONSEQUENCE, which VEH3a's preview tuning door did not carry: writing the
/// component on a car that already exists changes what the Details grid shows
/// and **nothing the car does**.
///
/// Measured by the demo loop, which is how it was found: a session that asked
/// for `turbo_boost_max=0.8` on twenty-three island chassis drove six hundred
/// and eleven rows at **0.000 boost**. `pie_drive.rs` now tunes the running
/// vehicles as well as the component, and the source pin below is there for
/// `the_shipped_host_draws_the_drivetrain_row`'s reason exactly: a host loop
/// cannot be reached from a test, so what can be pinned is that it is written.
#[test]
fn a_class_edited_after_creation_reaches_the_car_only_through_the_tuner() {
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    // (a) THE COMPONENT ALONE, on a car that already exists.
    let mut rig = settled(Rig::sealed());
    let e = rig.world.entity_of(CHASSIS).expect("the chassis entity");
    let mut class = rig
        .world
        .world()
        .get::<inf_ecs::components::VehicleClass>(e)
        .copied()
        .unwrap_or_default();
    assert!(class.set("turbo_boost_max", 0.8));
    rig.world.world_mut().entity_mut(e).insert(class);
    rig.world.mark_dirty();
    rig.world.propagate();
    rig.drive(full, 180);
    let component_only = (0..180).fold(0.0f64, |m, _| {
        rig.drive(full, 1);
        m.max(rig.drivetrain().boost)
    });

    // (b) THE SAME NUMBER THROUGH THE TUNER, which is what the bridge's own
    //     doc says owns it from creation onward.
    let mut rig = settled(Rig::sealed());
    rig.tune(&[("turbo_boost_max", 0.8)]);
    let tuned = (0..360).fold(0.0f64, |m, _| {
        rig.drive(full, 1);
        m.max(rig.drivetrain().boost)
    });
    println!("VEH3b SEAM: an edited component alone made {component_only:.3} of boost on a car that already existed; the same number through the live tuner made {tuned:.3}");
    assert_eq!(
        component_only, 0.0,
        "an edited component reached a running car's physics, so `reconcile_vehicles` has started re-installing and every live tune is now being undone"
    );
    assert!(
        tuned > 0.3,
        "the live tuner made {tuned:.3} of boost, so this arm's control is not a control"
    );

    // …and the shipped preview door crosses BOTH halves.
    const DRIVE: &str = include_str!("../src/pie_drive.rs");
    for fragment in [
        "if let Some(v) = sim.bridge3d_mut().vehicle_mut(*guid) {",
        "if v.tune(name, *value) {",
        "on the RUNNING vehicles",
    ] {
        assert!(
            DRIVE.contains(fragment),
            "`pie_drive.rs` no longer carries `{fragment}`, so `INF_PIE_TUNE_VEHICLE` is back to editing a component nothing reads"
        );
    }
}

/// **A WHEEL'S SPEED IS IN THE TRACE, AND IT ALWAYS WAS** — through the wheel's
/// own transform (`audit:` VEH3b, against the wave's carried item 4).
///
/// The wave carried this by name: *"what that leaves unfolded is the DRIVELINE's
/// speed, and with it the wheel speeds — which no trace section in this
/// repository has ever carried."* The first half is true and the conclusion is
/// **false**, and the difference matters because it is the whole of the audit
/// brief's worry: two hosts whose wheels turn at different speeds would, if it
/// were true, agree about every byte until a tyre touched something.
///
/// They do not. `WheelState::omega_rad_s` is integrated into
/// `WheelState::spin_deg` every step, and `inf_physics::d3::vehicle::step_one`
/// writes that straight onto the wheel entity's own `Transform` as
/// `rotation.x` — which `inf_ecs::sim::sim_snapshot` folds for every entity in
/// the world, as the FIRST section of `RuntimeSim::state_bytes` and since long
/// before this wave. A wheel-speed divergence is therefore a trace divergence in
/// the step it happens, not in the step the wheel next touches the ground.
///
/// # Measured where nothing else can move
///
/// A car in free fall. Its chassis is a ballistic body — the same arc whatever
/// the engine is doing — so the two runs below are **bit-identical in the
/// chassis** and differ only in what the wheels are doing. If the wheels were
/// not in the snapshot, the two traces would be equal, and they are not.
///
/// # The seam, stated
///
/// The write is skipped for a rig whose wheels have no ENTITY (`world
/// .entity_of(guid)`), which is `reconcile_vehicles`' own "a rig with no wheel
/// meshes simulates identically to one with them". Every vehicle the island
/// parks is spawned by `spawn_rig`, which gives each wheel an entity — asserted
/// here, so the day a rig arrives without one this arm is where it is read.
#[test]
fn a_wheels_speed_is_in_the_trace_through_its_own_transform() {
    let fall = |throttle: f64| -> (Vec<inf_ecs::sim::EntitySimState>, usize, usize) {
        let mut rig = Rig::airborne();
        // **Traction control off**, and the reason is worth the line: the aid
        // caps a wheel's drive torque at what its CONTACT PATCH can take, and an
        // airborne wheel's load is zero — so with the shipped aid on, sixty
        // steps of full throttle spin an airborne wheel by exactly nothing
        // (measured: 0 of 4 wheels folded a different transform). That is the
        // aid working, and it would have made this arm measure the aid.
        rig.tune(&[("traction_control_slip", 0.0)]);
        let c = VehicleControls {
            throttle,
            ..Default::default()
        };
        let mut airborne = 0usize;
        for _ in 0..60 {
            rig.drive(c, 1);
            airborne += usize::from(rig.wheels().iter().all(|w| w.contact.is_none()));
        }
        let wheels = rig.wheels().len();
        (inf_ecs::sim::sim_snapshot(&mut rig.world), airborne, wheels)
    };
    let (driven, driven_air, wheels) = fall(1.0);
    let (coasting, coast_air, _) = fall(0.0);
    assert_eq!(
        (driven_air, coast_air),
        (60, 60),
        "the car landed inside the window, so the chassis is no longer the control"
    );
    assert!(wheels >= 4, "the rig has {wheels} wheels");

    // The wheel entities, by the guids the rig itself names them with.
    let rig = Rig::airborne();
    let wheel_guids: Vec<Uuid> = rig
        .bridge
        .vehicle_of(CHASSIS)
        .expect("the bridge derived the rig")
        .rig()
        .wheels
        .iter()
        .map(|w| w.guid)
        .collect();
    assert_eq!(wheel_guids.len(), wheels);
    for g in &wheel_guids {
        assert!(
            rig.world.entity_of(*g).is_some(),
            "a wheel mount names a guid no entity carries, so nothing writes its \
             spin into the world and this arm's claim does not hold for this rig"
        );
    }

    let row = |snap: &[inf_ecs::sim::EntitySimState], g: Uuid| {
        snap.iter()
            .find(|s| s.guid == g)
            .cloned()
            .unwrap_or_else(|| panic!("`{g}` is not in the snapshot"))
    };
    // ── THE CHASSIS IS THE CONTROL ── a ballistic body, twice.
    let (a, b) = (row(&driven, CHASSIS), row(&coasting, CHASSIS));
    println!(
        "VEH3b TRACE: after sixty airborne steps the chassis is at {:?} driven and {:?} coasting",
        a.world_translation, b.world_translation
    );
    assert_eq!(
        a, b,
        "the chassis differs between a driven and a coasting free fall, so this \
         arm is measuring the body and not the wheels"
    );
    // ── AND THE WHEELS ARE NOT ──
    let mut spun = 0usize;
    for g in &wheel_guids {
        let (x, y) = (row(&driven, *g), row(&coasting, *g));
        if x != y {
            spun += 1;
        }
    }
    println!(
        "VEH3b TRACE: {spun} of {wheels} wheel entities fold different bytes, with \
         the chassis bit-identical — a wheel's speed reaches `state_bytes` through \
         its own `Transform::rotation`"
    );
    assert!(
        spun >= 2,
        "{spun} of {wheels} wheels folded a different transform under full \
         throttle than under none, so a wheel-speed divergence really would be \
         invisible to every trace comparison in this repository"
    );
    assert_ne!(
        driven, coasting,
        "two free falls differing only in wheel speed folded identical snapshots"
    );
}

/// **CLAUSE 5 ON THE SHIPPED ROWS** (`audit:` VEH3b — the clause, closed).
///
/// The wave measured weight transfer on a fixture stiffened to 90 000 N/m and
/// carried the reason: the Ring-0 rig sat on its stops under a 0.9 g stop, and a
/// car on its bump stops measures its bump stops. This audit measured what that
/// cost on the SHIPPED spring — the front axle took **-491 N** where the doc's
/// `m.a.h / L` predicts **+2 609**, 118.8 % apart with the sign INVERTED — and
/// the clause the brief actually wrote is *"weight transfer verified on the
/// shipped rows within 10 %"*.
///
/// Two things closed it and both are here:
///
/// 1. **a bump stop**, so a strut's rate keeps rising to the end of travel and
///    an axle never saturates flat ([`inf_ecs::vehicle::suspension_force_n`]);
/// 2. **the nine wheeled catalogue rows re-sprung** to stand at about a third of
///    their travel at rest, which is where a road car sits.
///
/// So this runs on two rows' OWN cars — the **sedan** and the **truck**, a
/// 1 185 kg saloon and a 2 341 kg pickup on different travels and different
/// rates — with nothing stiffened, and it bands the TRANSFER rather than the
/// total (that vacuity is the wave's own finding) at 10 % of the doc's formula
/// under a hard stop and a full launch.
///
/// # Which half closes which, measured
///
/// The two fixes are not interchangeable and the mutations say so:
///
/// * **the RE-SPRING is what this arm rests on.** Put the sedan back on its
///   21 000 N/m and the arm reds at the static-fraction guard before it ever
///   brakes — 55 % of travel. On their new rates neither row comes near the
///   stop: the sedan's deepest braked compression is **0.172 m of 0.25 (69 %)**
///   against an 85 % engagement, so the stop does not fire here at all;
/// * **the BUMP STOP is what closes the model**, and its own arm
///   (`the_bump_stop_is_a_rate_that_rises_and_the_rows_are_sprung_for_it`)
///   measures it where it fires — the UNSTIFFENED Ring-0 fixture, which reaches
///   0.239 m of its 0.25 under a 0.95 g stop. Take the stop out
///   (`BUMP_STOP_RATE_MULT` → 0) and THAT arm reds while this one stays green,
///   which is the honest division of labour: the re-spring keeps a road car off
///   its stops, and the stop is what makes the model right the day a kerb or a
///   landing puts it there anyway.
///
/// **The mutation for this arm**: the sedan's old `stiffness_n_per_m` — RED.
#[test]
fn the_axle_loads_are_the_formulas_on_the_shipped_rows() {
    /// How many steps of the stop are averaged after the transient is skipped.
    /// Longer is steadier: a damper still creeping is a force the quasi-static
    /// formula does not have, and averaging it out is what the window is for.
    const STOP_WINDOW: usize = 150;
    for id in ["sedan", "truck"] {
        let tuning = catalogue_def(id).class.to_tuning();
        let travel = tuning.travel_m;
        let mut rig = settled(Rig::sealed_row(id));
        rig.step(120);
        let mass = rig.mass_kg();
        let wheelbase = rig.wheelbase_m();
        let contact_y = rig
            .wheel(0)
            .contact
            .map(|c| c.point.y)
            .unwrap_or_else(|| panic!("the {id} is not standing on anything"));
        let h = rig.at().y - contact_y + tuning.cog_height_m;
        let static_front = rig.axle_load_n(true);
        let static_rear = rig.axle_load_n(false);
        let rest = rig.axle_compression_m(true);
        println!(
            "VEH3b CLAUSE5 {id}: {mass:.0} kg on a {wheelbase:.2} m wheelbase, h = {h:.3} m, standing on {rest:.3} m of {travel:.2} m ({:.0} % of its travel), axles {static_front:.0} N front / {static_rear:.0} N rear",
            rest / travel * 100.0
        );
        assert!(
            (0.25..0.50).contains(&(rest / travel)),
            "`{id}` stands on {:.0} % of its travel; the catalogue's own comment says a third",
            rest / travel * 100.0
        );
        assert!(
            h > 0.2,
            "`{id}`'s centre of gravity is {h:.3} m over the road"
        );

        // -- under the brakes --
        let full = VehicleControls {
            throttle: 1.0,
            ..Default::default()
        };
        // **As fast as the row itself will go, capped at 28 m/s**, and both
        // halves matter. These rows stop at up to one g, so the old 22 m/s entry
        // left only forty-eight steps above the 3 m/s floor once the transient
        // was skipped, which is a thin average of a quantity a damper is still
        // moving (the sedan read 9.7 % from 22 and 5.3 % from 28). And the cap
        // is the ROW's own governor rather than a literal, because the truck's
        // is 27 m/s: a fixed 28 means the accelerate loop never breaks, runs its
        // whole 1 800 steps, and drives the car seven hundred metres off the end
        // of the pad.
        let entry = (tuning.max_speed_mps * 0.9).min(28.0);
        for _ in 0..1_800 {
            rig.drive(full, 1);
            if rig.speed() > entry {
                break;
            }
        }
        let brake = VehicleControls {
            brake: 1.0,
            ..Default::default()
        };
        // The pitch TRANSIENT is skipped for the fixture arm's reason and with a
        // number of its own: these rows are sprung two to three times harder
        // than the fixture, so their suspension's own period is shorter and
        // their brakes pull over one g -- 60 steps of settling against the
        // fixture's 40, measured (at 40 the sedan read 12.2 % against the
        // formula and at 60 it reads what is printed below).
        rig.drive(brake, 60);
        let (mut braked, mut front, mut rear, mut accel) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let (mut pinned, mut deepest) = (0usize, 0.0f64);
        let mut last = rig.speed();
        for _ in 0..STOP_WINDOW {
            rig.drive(brake, 1);
            let v = rig.speed();
            let a = (v - last) / DT;
            last = v;
            let c = rig.axle_compression_m(true);
            deepest = deepest.max(c);
            if c >= travel - 1e-6 {
                pinned += 1;
            }
            if a < -4.0 && v > 3.0 {
                braked += 1;
                front += rig.axle_load_n(true);
                rear += rig.axle_load_n(false);
                accel += a;
            }
        }
        assert!(
            braked > 20,
            "`{id}`: only {braked} steps of the stop pulled more than 0.4 g"
        );
        let a_x = accel / braked as f64;
        // **THE TRANSFER IS HALF THE FRONT-TO-REAR DIFFERENCE**, which is what
        // the doc's formula says it is: `Fz_front = Fz_static - m.ax.h/L` and
        // `Fz_rear = Fz_static + m.ax.h/L`, so one axle gains exactly what the
        // other loses and the transfer is `(front - rear) / 2`.
        //
        // Reading the front's gain ALONE measures the transfer plus whatever
        // common-mode drift the chassis has, and it has some: on the sedan's
        // stop the front gains 2 074 N while the rear loses 2 358, because the
        // total normal force is 284 N (2.4 %) under the car's weight while the
        // body is still settling downward. The formula has no term for that and
        // is not wrong about the transfer; half the difference is the estimator
        // that does not carry it.
        let moved = (front - rear) / braked as f64 / 2.0;
        let want = -mass * a_x * h / wheelbase;
        let err = (moved - want).abs() / want.abs().max(1.0);
        println!(
            "VEH3b CLAUSE5 {id}: braking at {a_x:.2} m/s2 over {braked} steps the front axle took {moved:.0} N of transfer against the doc's {want:.0} N — {:.1} % apart; deepest compression {deepest:.3} m of {travel:.2}, pinned at travel on {pinned} of {STOP_WINDOW} steps (rear {:.0} N)",
            err * 100.0,
            rear / braked as f64
        );
        assert!(
            err < 0.10,
            "`{id}`: the front axle took {moved:.0} N of transfer under {a_x:.2} m/s2 and the doc's formula says {want:.0} N — {:.1} % apart",
            err * 100.0
        );

        // -- and under the launch --
        let mut rig = settled(Rig::sealed_row(id));
        rig.step(120);
        rig.drive(full, 40);
        let (mut launched, mut rear_sum, mut front_sum, mut la) = (0usize, 0.0f64, 0.0f64, 0.0f64);
        let mut last = rig.speed();
        for _ in 0..120 {
            rig.drive(full, 1);
            let v = rig.speed();
            let a = (v - last) / DT;
            last = v;
            if a > 1.0 {
                launched += 1;
                rear_sum += rig.axle_load_n(false);
                front_sum += rig.axle_load_n(true);
                la += a;
            }
        }
        assert!(
            launched > 20,
            "`{id}`: only {launched} steps of the launch pulled more than 0.1 g"
        );
        let a_l = la / launched as f64;
        let moved = (rear_sum - front_sum) / launched as f64 / 2.0;
        let want = mass * a_l * h / wheelbase;
        let err = (moved - want).abs() / want.abs().max(1.0);
        println!(
            "VEH3b CLAUSE5 {id}: launching at {a_l:.2} m/s2 over {launched} steps the rear axle took {moved:.0} N of transfer against the doc's {want:.0} N — {:.1} % apart",
            err * 100.0
        );
        assert!(
            err < 0.10,
            "`{id}`: the rear axle took {moved:.0} N of transfer under {a_l:.2} m/s2 and the doc's formula says {want:.0} N — {:.1} % apart",
            err * 100.0
        );
    }
}

/// **THE BUMP STOP IS A RATE THAT RISES, AND THE ROWS ARE SPRUNG FOR IT**
/// (`audit:` VEH3b).
///
/// The arm that used to stand here asserted the DEFECT, on P22's own precedent,
/// so that the day a stop landed it would red and the ledger would be rewritten.
/// The stop landed in the same audit. This is that arm inverted: it asserts the
/// fix, in the three places the fix exists.
///
/// 1. **The law.** `suspension_force_n` is the linear spring it always was below
///    `BUMP_STOP_ENGAGE_FRAC` of the travel, and above it gains a CUBIC whose
///    marginal rate at full travel is `3 x BUMP_STOP_RATE_MULT` times the main
///    one. A millimetre at the stop must cost far more than a millimetre in the
///    middle, which is the whole of what "a rate that rises" means.
/// 2. **The content.** All nine wheeled catalogue rows stand at about a third of
///    their travel at rest (they stood at 54-88 %), so the stop is a safety net
///    rather than the thing holding the car up.
/// 3. **The world.** The Ring-0 fixture, on its OWN unstiffened spring, agrees
///    with the doc's formula under a hard stop and never reaches `travel_m`.
///
/// **The mutation**: `BUMP_STOP_RATE_MULT` -> 0 (the stop removed) — the fixture
/// saturates again and the world half fails.
#[test]
fn the_bump_stop_is_a_rate_that_rises_and_the_rows_are_sprung_for_it() {
    // -- 1. THE LAW --
    let t = VehicleTuning::default();
    let engage = t.travel_m * inf_ecs::vehicle::BUMP_STOP_ENGAGE_FRAC;
    let below = inf_ecs::vehicle::suspension_force_n(&t, engage, 0.0);
    assert_eq!(
        below,
        t.stiffness_n_per_m * engage,
        "the stop is doing work before it engages"
    );
    let mid = inf_ecs::vehicle::suspension_force_n(&t, engage * 0.5 + 0.001, 0.0)
        - inf_ecs::vehicle::suspension_force_n(&t, engage * 0.5, 0.0);
    let last = inf_ecs::vehicle::suspension_force_n(&t, t.travel_m, 0.0)
        - inf_ecs::vehicle::suspension_force_n(&t, t.travel_m - 0.001, 0.0);
    println!(
        "VEH3b STOP: a millimetre at the stop costs {last:.0} N and one in the middle {mid:.0} N ({:.0}x); the strut carries {:.0} N at full travel against a bare spring's {:.0}",
        last / mid,
        inf_ecs::vehicle::suspension_force_n(&t, t.travel_m, 0.0),
        t.stiffness_n_per_m * t.travel_m
    );
    assert!(
        last > mid * 10.0,
        "the last millimetre of travel costs {last:.0} N and one in the middle {mid:.0} — that is not a rate that rises"
    );

    // -- 2. THE CONTENT --
    let catalogue = inf_editor_core::vehicle::island_vehicles();
    let (mut rows, mut worst) = (0usize, (String::new(), 0.0f64));
    for (id, def) in catalogue.0.iter() {
        if !def.body.wheeled() {
            continue;
        }
        rows += 1;
        let tuning = def.class.to_tuning();
        let e = def.half_extents;
        let kg = 8.0 * e.x * e.y * e.z * def.density_kg_m3;
        let used = kg * 9.81 / 4.0 / tuning.stiffness_n_per_m / tuning.travel_m;
        println!(
            "VEH3b STOP: {id:>10} stands on {:.0} % of its {:.2} m travel ({kg:.0} kg on {:.0} N/m corners)",
            used * 100.0,
            tuning.travel_m,
            tuning.stiffness_n_per_m
        );
        assert!(
            (0.25..0.50).contains(&used),
            "`{id}` stands on {:.0} % of its travel; the catalogue's own comment says about a third, and 54-88 % is what it used to be",
            used * 100.0
        );
        if used > worst.1 {
            worst = (id.clone(), used);
        }
    }
    assert!(rows >= 9, "only {rows} wheeled rows in the catalogue");
    println!(
        "VEH3b STOP: {rows} wheeled rows, the worst `{}` at {:.0} % of its travel standing still",
        worst.0,
        worst.1 * 100.0
    );

    // -- 3. THE WORLD -- the Ring-0 fixture on its own spring, unstiffened.
    let travel = t.travel_m;
    let mut rig = settled(Rig::sealed());
    rig.step(120);
    let mass = rig.mass_kg();
    let wheelbase = rig.wheelbase_m();
    let contact_y = rig.wheel(0).contact.map(|c| c.point.y).expect("on the pad");
    let h = rig.at().y - contact_y + t.cog_height_m;
    let static_front = rig.axle_load_n(true);
    let full = VehicleControls {
        throttle: 1.0,
        ..Default::default()
    };
    for _ in 0..900 {
        rig.drive(full, 1);
        if rig.speed() > 22.0 {
            break;
        }
    }
    let brake = VehicleControls {
        brake: 1.0,
        ..Default::default()
    };
    rig.drive(brake, 40);
    let (mut braked, mut front, mut accel) = (0usize, 0.0f64, 0.0f64);
    let (mut pinned, mut deepest) = (0usize, 0.0f64);
    let mut last = rig.speed();
    for _ in 0..90 {
        rig.drive(brake, 1);
        let v = rig.speed();
        let a = (v - last) / DT;
        last = v;
        let c = rig.axle_compression_m(true);
        deepest = deepest.max(c);
        if c >= travel - 1e-6 {
            pinned += 1;
        }
        if a < -4.0 && v > 3.0 {
            braked += 1;
            front += rig.axle_load_n(true);
            accel += a;
        }
    }
    assert!(braked > 20, "only {braked} braked steps over 0.4 g");
    let a_x = accel / braked as f64;
    let moved = front / braked as f64 - static_front;
    let want = -mass * a_x * h / wheelbase;
    let err = (moved - want).abs() / want.abs().max(1.0);
    println!(
        "VEH3b STOP: the UNSTIFFENED Ring-0 fixture, braking at {a_x:.2} m/s2, took {moved:.0} N of transfer against the doc's {want:.0} N — {:.1} % apart; deepest {deepest:.3} m of {travel:.2}, pinned at travel on {pinned} of 90 steps (it was 90 of 90 and -491 N before the stop)",
        err * 100.0
    );
    assert_eq!(
        pinned, 0,
        "the fixture reached `travel_m` on {pinned} of 90 braked steps, so the stop is not holding it out of the clamp"
    );
    assert!(
        err < 0.10,
        "on the shipped spring the transfer is {:.1} % from the doc's formula",
        err * 100.0
    );
}

/// **A CAR WITH NO WHEEL ON THE GROUND SAYS SO** (`audit:` VEH3b) — the column
/// that could not tell a road from a kerb.
///
/// Wave VEH3a's audit found an island car the hero could board and could not
/// move, and read this off `hero.csv`:
///
/// ```text
/// …,Driving,0.0000,…,asphalt,0.0000,0.0000,1.000
/// ```
///
/// and concluded *"`surface asphalt`, `mu 1.000`"* — i.e. the wheels have grip
/// and the throttle is not reaching them. Those three columns say **exactly the
/// same thing** about a chassis beached on a kerb with its wheels hanging, which
/// is the other half of that hypothesis and the likelier one: `step_one`
/// classifies a wheel whose raycast MISSED as `Asphalt` (the "no ground" default
/// it must answer something with), `mu_surface` is only written on a step with a
/// contact so it keeps whatever the last one was worth, and a slip ratio over a
/// stationary wheel on a stationary car is `0` either way.
///
/// So the census reads **contacts** now, and a car with none answers `air`.
/// Measured here on the two states a car can be in, on the same rig.
///
/// **The mutation**: drop the `contact.is_some()` filter from `surface_census`
/// and the airborne car reads `asphalt` again — verified RED.
#[test]
fn a_car_with_no_wheel_on_the_ground_says_air() {
    // ── ON THE ROAD ──
    let mut rig = settled(Rig::sealed());
    rig.drive(
        VehicleControls {
            throttle: 1.0,
            ..Default::default()
        },
        30,
    );
    let wheels = rig.wheels();
    let grounded = wheels.iter().filter(|w| w.contact.is_some()).count();
    let (on_road, road_mu) = inf_ecs::vehicle::surface_census(&wheels);
    let road_row = inf_ecs::vehicle::tyre_readout(&wheels);
    println!("VEH3b AIR: {grounded} of {} wheels in contact -> `{on_road}` at mu {road_mu:.3}; the row is `{road_row}`", wheels.len());
    assert_eq!(grounded, wheels.len(), "the car is not on the pad");
    assert_eq!(on_road, "asphalt");
    assert!(road_mu > 0.5, "the road is worth mu {road_mu:.3}");
    assert!(road_row.contains("asphalt"), "{road_row}");

    // ── AND IN THE AIR ── the same rig, twenty metres up, with the same
    //    `surface` field on every wheel (the bridge answers `Asphalt` for a
    //    raycast that found nothing) and no contact under any of them.
    let mut rig = Rig::airborne();
    rig.step(20);
    let wheels = rig.wheels();
    let grounded = wheels.iter().filter(|w| w.contact.is_some()).count();
    let claimed = wheels
        .iter()
        .filter(|w| w.surface == SurfaceClass::Asphalt)
        .count();
    let (in_air, air_mu) = inf_ecs::vehicle::surface_census(&wheels);
    let air_row = inf_ecs::vehicle::tyre_readout(&wheels);
    println!("VEH3b AIR: {grounded} of {} wheels in contact and {claimed} still CLAIM asphalt -> `{in_air}` at mu {air_mu:.3}; the row is `{air_row}`", wheels.len());
    assert_eq!(
        grounded, 0,
        "the car landed, so there is nothing to measure"
    );
    assert_eq!(
        claimed,
        wheels.len(),
        "no wheel claims asphalt any more, so the defect this arm records is somewhere else now"
    );
    assert_eq!(
        in_air,
        inf_ecs::vehicle::AIRBORNE_SURFACE,
        "a car with {grounded} wheels on the ground reported `{in_air}`"
    );
    assert_eq!(
        air_mu, 0.0,
        "an airborne car's contact is worth {air_mu:.3}"
    );
    // The surface sits between the temperatures and the slips, so the check is
    // positional: the row's LAST word is the ambient air temperature and always
    // has been, and a `contains("air")` would pass on that alone.
    assert!(
        air_row.contains("C  air  slip") && !air_row.contains("asphalt"),
        "the HUD row for a car in the air is `{air_row}`"
    );

    // …and the demo log's own column goes through the SAME door rather than a
    // second copy of the tie-break, which is what let the two disagree.
    const DRIVE: &str = include_str!("../src/pie_drive.rs");
    assert!(
        DRIVE.contains("inf_ecs::vehicle::surface_census(&tyres)"),
        "`pie_drive.rs` has stopped reading Ring 0's census, so `hero.csv`'s surface column can go back to calling a beached car a road"
    );
}
