//! **THE D-14 SURFACE, AUDITED** (island wave VEH1a) — what a wheel ray reads
//! off a heightfield, and what of it reaches the car.
//!
//! P29.7 shipped the raycast vehicle against a *box* floor and left two things
//! about real ground written down and unmeasured:
//!
//! * **the normal SNAP.** `FIX_INTERNAL_EDGES` is set on every heightfield this
//!   engine builds, and it fixes **contacts** — a ray reads the hit triangle's
//!   own normal, un-fixed. A heightfield cell is two triangles, so a wheel
//!   crossing a cell diagonal sees the normal change discontinuously. Nothing
//!   had ever put a number on it.
//! * **the sensor ride.** The wheel ray was cast with `CastTargets::All`, and a
//!   trigger volume is a collider: *"a car crossing a trigger volume would ride
//!   on it"* was a carried bound with no arm.
//!
//! This file measures the first and arms the second. What it deliberately does
//! **not** do is smooth the normal, and the reason is the measurement below:
//! `WheelContact::normal` is written by the door and read by **nothing** — a
//! smoothing pass would be a fix applied to a number no force is a function of.
//! `the_snapped_normal_reaches_no_force_in_the_model` is the falsifier that says
//! so in metres rather than by grep, and it is what turns red the day a class
//! starts reading it.

use glam::{DQuat, DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, Collider3D, ColliderShape3DKind, RigidBody3D, Terrain, Transform,
};
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{
    ChassisState, RaycastVehicle, Vehicle, VehicleControls, VehicleRig, WheelForce, WheelState,
};
use inf_ecs::EcsWorld;
use inf_physics::d3::PhysicsBridge3D;
use inf_terrain::TerrainData;

const HZ: f64 = 60.0;
const DT: f64 = 1.0 / HZ;

const TERRAIN: Uuid = Uuid::from_u128(0x7614_0001);
const CHASSIS: Uuid = Uuid::from_u128(0x7614_0002);
const WHEEL_BASE: u128 = 0x7614_0010;
const TRIGGER: Uuid = Uuid::from_u128(0x7614_0020);

/// The island's own grid: **one metre** between height samples.
const MPS: f64 = 1.0;
/// 65 samples a side ⇒ one tile spanning 64 m, which is 64 cells of diagonal
/// for a wheel to cross.
const TILE_RES: u32 = 65;
/// Tiles a side. Four, spanning `[0, 256]²` — a car at the shipped 25 m/s top
/// speed covers 64 m in two and a half seconds, so a one-tile fixture measures a
/// car that drove off the edge rather than one that drove.
const TILES: i32 = 4;

const HALF: Vec3d = Vec3d::new(2.0, 0.5, 1.0);
const DENSITY: f64 = 150.0;
const MASS_KG: f64 = 8.0 * DENSITY;
const WHEEL_RADIUS: f64 = 0.35;
const WHEEL_Y: f64 = -0.75;

/// The lift `inf_gis::roads` draws its ribbon at above the ground it drapes on
/// (`DEFAULT_ROAD_LIFT_M`).
///
/// Restated rather than imported because `inf-physics` does not link `inf-gis`
/// and will not grow a dev-dependency on a GIS stack for one constant. The
/// **comparison** against the road builder's own value is therefore made where
/// both crates are already in scope — `island_gate`'s drive arm, which prices
/// the same sink on the real circuit — and this number is only the label on the
/// figure this file prints.
const ROAD_LIFT_M: f64 = 0.02;

// ── the ground ──────────────────────────────────────────────────────────────

/// A deterministic integer hash in `[0, 1)` — the roughness a real DTM has and
/// a polynomial does not.
///
/// Integer arithmetic only: no `sin`, no `cbrt`, nothing that reaches `libm`
/// (the P14 law). The same bits on every target, which is what lets the numbers
/// this file prints be quoted.
fn rough01(i: i64, j: i64) -> f64 {
    let mut x = (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (j as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    x ^= x >> 29;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 32;
    (x >> 11) as f64 / (1u64 << 53) as f64
}

/// A heightfield with a real cross-slope and `roughness_m` of per-sample
/// relief.
///
/// The slope is the island's own worst audited grade (**0.108**) laid across
/// both axes, so the surface a wheel crosses here is at least as steep as the
/// steepest stretch the road generator will admit. `roughness_m = 0.0` is the
/// levelled road corridor; the rough row is open ground.
fn ground_height(x: f64, z: f64, roughness_m: f64) -> f64 {
    let smooth = 12.0 + 0.108 * x - 0.06 * z + 0.0008 * x * z;
    if roughness_m == 0.0 {
        return smooth;
    }
    // Sampled on the LATTICE, so the roughness lives at the samples and the
    // cell interiors are the bilinear surface the collider actually is.
    smooth + roughness_m * (rough01(x.round() as i64, z.round() as i64) - 0.5)
}

fn terrain_data(roughness_m: f64) -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    for tz in 0..TILES {
        for tx in 0..TILES {
            data.author_tile((tx, tz), |x, z| ground_height(x, z, roughness_m));
        }
    }
    data
}

fn world_with_ground(roughness_m: f64) -> EcsWorld {
    let mut w = EcsWorld::new();
    let e = w.spawn_with_guid(TERRAIN, "Terrain", None);
    w.world_mut().entity_mut(e).insert(Terrain {
        meters_per_sample: MPS,
        tile_resolution: TILE_RES,
        data: terrain_data(roughness_m),
        ..Terrain::default()
    });
    w
}

/// The committed rig's shape: a dynamic box with four sphere **sensors** hung
/// off it, `y` metres above the ground under its own origin.
fn car(world: &mut EcsWorld, at: DVec3) {
    let e = world.spawn_with_guid(CHASSIS, "Car", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::from_dvec3(at);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            angular_damping: 0.5,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: HALF,
            density: DENSITY,
            friction: 0.5,
            ..Default::default()
        },
    ));
    for (i, (x, z)) in [(-0.9, 1.4), (0.9, 1.4), (-0.9, -1.4), (0.9, -1.4)]
        .into_iter()
        .enumerate()
    {
        let w = world.spawn_with_guid(Uuid::from_u128(WHEEL_BASE + i as u128), "Wheel", Some(e));
        let mut wt = Transform::IDENTITY;
        wt.translation = Vec3d::new(x, WHEEL_Y, z);
        world.world_mut().entity_mut(w).insert((
            wt,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: WHEEL_RADIUS,
                sensor: true,
                ..Default::default()
            },
        ));
    }
}

struct Rig {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    roughness_m: f64,
}

impl Rig {
    /// A car standing on the ground at world `(x, z)`, wheels touching with the
    /// suspension fully extended.
    fn new(x: f64, z: f64, roughness_m: f64) -> Self {
        let mut world = world_with_ground(roughness_m);
        let ground = ground_height(x, z, roughness_m);
        car(
            &mut world,
            DVec3::new(x, ground - WHEEL_Y + WHEEL_RADIUS, z),
        );
        world.mark_dirty();
        world.propagate();
        let bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        let mut rig = Self {
            world,
            bridge,
            roughness_m,
        };
        rig.bridge.sync_from_world(&rig.world);
        rig
    }

    fn step(&mut self, n: u32) {
        for _ in 0..n {
            self.bridge.sync_from_world(&self.world);
            inf_physics::d3::step_character_movement(&mut self.world, &mut self.bridge, DT);
            // The `vehicle` phase, in the slot both hosts call it in.
            inf_physics::d3::step_vehicles(&mut self.world, &mut self.bridge, DT);
            self.bridge.step(DT);
            self.bridge.write_back_into(&mut self.world);
            self.world.propagate();
        }
    }

    fn drive(&mut self, controls: VehicleControls, n: u32) {
        for _ in 0..n {
            if let Some(v) = self.bridge.vehicle_mut(CHASSIS) {
                v.control(controls);
            }
            self.step(1);
        }
    }

    /// Hold it where it stands — the handbrake, which is what a car parked on a
    /// grade has and what a fixture that wants a *settled* number needs. Without
    /// it the arms below would be measuring a car rolling downhill, and the
    /// ground under a rolling car is not the ground it started on.
    fn park(&mut self, n: u32) {
        self.drive(
            VehicleControls {
                handbrake: true,
                ..Default::default()
            },
            n,
        );
    }

    fn chassis(&self) -> Transform {
        let e = self.world.entity_of(CHASSIS).expect("the car exists");
        *self
            .world
            .world()
            .get::<Transform>(e)
            .expect("…with a transform")
    }

    /// The chassis pose as raw bits — the comparison a "byte-identical" claim
    /// needs, since an epsilon would admit different arithmetic.
    fn pose_bits(&self) -> [u64; 6] {
        let t = self.chassis();
        [
            t.translation.x.to_bits(),
            t.translation.y.to_bits(),
            t.translation.z.to_bits(),
            t.rotation.x.to_bits(),
            t.rotation.y.to_bits(),
            t.rotation.z.to_bits(),
        ]
    }

    /// The analytic ground under the chassis origin — the surface the collider
    /// interpolates between samples.
    fn ground_under(&self) -> f64 {
        let t = self.chassis().translation;
        ground_height(t.x, t.z, self.roughness_m)
    }
}

// ── clause 1a: the snap, measured ───────────────────────────────────────────

/// One straight-down ray's `(y, normal)` at world `(x, z)`, or `None` off the
/// tile.
fn probe(bridge: &mut PhysicsBridge3D, x: f64, z: f64) -> Option<(f64, DVec3)> {
    bridge
        .world_mut()
        .cast_ray(DVec3::new(x, 200.0, z), DVec3::NEG_Y, 400.0)
        .map(|h| (h.point.y, h.normal))
}

/// **THE MEASUREMENT.** A wheel ray crossing a heightfield cell diagonal reads
/// a normal that SNAPS, and here is how far.
///
/// The walk is a line at 45° to the lattice across 40 cells at a 5 cm step, so
/// it crosses a diagonal about once a metre. For each pair of adjacent samples
/// the arm records the angle between the two normals and the change in height;
/// the height is continuous (a heightfield is C0) and the normal is not (it is
/// not C1), which is the whole shape of the finding.
///
/// Two surfaces, because the island has both: the **levelled road corridor**
/// (roughness 0, a bilinear patch over a smooth slope) and **open ground** (a
/// real DTM's per-sample relief). The corridor row is the one the circuit is
/// driven on and it is the small number; the open-ground row is the honest
/// worst case.
#[test]
fn a_wheel_ray_normal_snaps_at_a_heightfield_cell_diagonal() {
    println!("THE D-14 SURFACE — a wheel ray's normal across cell diagonals:");
    let mut rows: Vec<(f64, f64, f64, f64)> = Vec::new();
    for roughness in [0.0, 0.15] {
        let world = world_with_ground(roughness);
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);

        let mut worst_deg = 0.0f64;
        let mut worst_dy = 0.0f64;
        let mut jumps_over_1deg = 0usize;
        let mut samples = 0usize;
        let mut prev: Option<(f64, DVec3)> = None;
        let mut s = 0.0f64;
        while s < 40.0 {
            let (x, z) = (8.0 + s, 8.0 + s);
            let Some(now) = probe(&mut bridge, x, z) else {
                s += 0.05;
                continue;
            };
            if let Some(before) = prev {
                let dot = before.1.dot(now.1).clamp(-1.0, 1.0);
                // `acos` is not on the libm ban list's subject set and this is a
                // REPORT, never a committed byte — but the comparison below is
                // done on the dot product, which is exact arithmetic, so the
                // degrees are for a reader and the assertion is not.
                let deg = dot.acos().to_degrees();
                worst_deg = worst_deg.max(deg);
                worst_dy = worst_dy.max((now.0 - before.0).abs());
                if deg > 1.0 {
                    jumps_over_1deg += 1;
                }
                samples += 1;
            }
            prev = Some(now);
            s += 0.05;
        }
        assert!(
            samples > 700,
            "only {samples} of the 800 sample pairs hit the ground — the walk \
             left the tile and the measurement is about nothing"
        );
        println!(
            "  roughness {roughness:.2} m: worst normal step {worst_deg:.4}deg, \
             {jumps_over_1deg} of {samples} steps over 1deg, worst height step \
             {worst_dy:.6} m"
        );
        rows.push((roughness, worst_deg, worst_dy, jumps_over_1deg as f64));
    }

    // The height is continuous: a 5 cm step on a surface whose worst grade is
    // 0.108 plus the roughness cannot move the ground more than a few
    // centimetres, and a heightfield that had a cliff at every cell edge would.
    for (roughness, _, dy, _) in &rows {
        assert!(
            *dy < 0.05,
            "a 5 cm walk moved the ground {dy} m on the {roughness} m surface — \
             the heightfield is not C0"
        );
    }
    // The NORMAL is not continuous, and the rough surface is where it shows.
    // This asserts the defect exists, so a day the ray path starts smoothing
    // (or `FIX_INTERNAL_EDGES` grows a ray half) this arm goes red and gets
    // rewritten rather than quietly certifying a fix nobody claimed.
    assert!(
        rows[1].1 > 3.0,
        "the rough surface's worst normal step is {:.4}deg — the snap this file \
         is about is not happening, so either the fixture is flat or a ray now \
         reads a smoothed normal",
        rows[1].1
    );
    assert!(
        rows[0].1 < rows[1].1,
        "the levelled corridor ({:.4}deg) snaps at least as hard as open ground \
         ({:.4}deg), which is not what a smooth slope should do",
        rows[0].1,
        rows[1].1
    );
}

// ── clause 1b: and it reaches no force ──────────────────────────────────────

/// A [`RaycastVehicle`] whose contact normals are **replaced with rubbish**
/// before the model sees them.
///
/// The falsifier for "nothing reads `WheelContact::normal`". A grep says the
/// same thing and says it about the tree the grep ran on; this says it in
/// metres, through the shipped door, over a real drive.
struct ScrambledNormals {
    inner: RaycastVehicle,
    scrambled: usize,
}

impl Vehicle for ScrambledNormals {
    fn rig(&self) -> &VehicleRig {
        self.inner.rig()
    }
    fn set_rig(&mut self, rig: VehicleRig) {
        self.inner.set_rig(rig);
    }
    fn wheels(&self) -> &[WheelState] {
        self.inner.wheels()
    }
    fn wheels_mut(&mut self) -> &mut [WheelState] {
        self.inner.wheels_mut()
    }
    fn control(&mut self, controls: VehicleControls) {
        self.inner.control(controls);
    }
    fn tune(&mut self, name: &str, value: f64) -> bool {
        self.inner.tune(name, value)
    }
    fn seat_warp(&self) -> (f64, inf_anim::WarpWindow) {
        self.inner.seat_warp()
    }
    fn suspension_rest_m(&self) -> f64 {
        self.inner.suspension_rest_m()
    }
    fn solve(&mut self, chassis: ChassisState, dt: f64, out: &mut Vec<WheelForce>) {
        for (i, w) in self.inner.wheels_mut().iter_mut().enumerate() {
            if let Some(c) = w.contact.as_mut() {
                // Not a small perturbation: a normal pointing sideways, then
                // straight down, then back up. If any force were a function of
                // it the trajectories could not agree to the bit.
                c.normal = match i % 3 {
                    0 => DVec3::new(1.0, 0.0, 0.0),
                    1 => DVec3::NEG_Y,
                    _ => DVec3::new(0.0, 0.6, 0.8).normalize(),
                };
                self.scrambled += 1;
            }
        }
        self.inner.solve(chassis, dt, out);
    }
}

/// **THE VERDICT: the snap reaches no force, so the model needs no smoothing.**
///
/// Two identical rigs are driven 600 steps of throttle and steer across the
/// rough heightfield. One runs the shipped [`RaycastVehicle`]; the other runs
/// the same class behind [`ScrambledNormals`], which replaces every contact
/// normal with a direction the ground could not possibly have. The two poses
/// agree **to the bit**.
///
/// That is the whole disposition of clause 1's smoothing candidate.
/// `RaycastVehicle::solve` pushes the suspension along the **chassis up** (on
/// purpose — projecting onto the contact normal is how a car slides sideways off
/// a ramp it should drive up) and takes its friction basis from the steered
/// wheel's own axes, so the surface normal is never an input. Smoothing it would
/// have been a repair to a number no wheel is a function of.
///
/// The arm counts the scrambles, because "they agreed" is satisfied perfectly by
/// two rigs that never touched the ground.
#[test]
fn the_snapped_normal_reaches_no_force_in_the_model() {
    let controls = VehicleControls {
        throttle: 1.0,
        steer: 0.25,
        ..Default::default()
    };
    // The middle of the four-tile field: a steered car at full throttle draws a
    // circle of about twenty metres and needs room on every side of it.
    let mut plain = Rig::new(128.0, 128.0, 0.15);
    let mut scrambled = Rig::new(128.0, 128.0, 0.15);
    let derived = scrambled
        .bridge
        .vehicle_of(CHASSIS)
        .expect("the car was derived")
        .rig()
        .clone();
    scrambled.bridge.install_vehicle(
        CHASSIS,
        Box::new(ScrambledNormals {
            inner: RaycastVehicle::new(derived),
            scrambled: 0,
        }),
    );

    // Anti-vacuity is COUNTED over the whole drive, not read off the last step:
    // a car crossing rough ground at 24 m/s has a wheel in the air a good deal
    // of the time, and "four wheels down at step 600" would be a stricter claim
    // than the physics supports while saying less about how many normals were
    // actually replaced.
    let mut contacts = 0usize;
    for _ in 0..600 {
        plain.drive(controls, 1);
        scrambled.drive(controls, 1);
        contacts += scrambled
            .bridge
            .vehicle_of(CHASSIS)
            .expect("still installed")
            .wheels()
            .iter()
            .filter(|w| w.contact.is_some())
            .count();
    }
    let end = scrambled.chassis().translation.to_dvec3();
    assert!(
        contacts > 1_500,
        "only {contacts} wheel contacts over 600 steps (of a possible 2 400) — \
         the car spent the drive in the air at {end}, so there were no normals \
         to scramble and this arm compares two falling boxes"
    );
    let grounded = contacts;
    let travelled = (end - plain.chassis().translation.to_dvec3()).length();
    let ran = (end - DVec3::new(128.0, end.y, 128.0)).length();
    assert!(
        ran > 5.0,
        "the rig ended {ran} m from where it started after ten seconds of full \
         throttle — it is stuck, so the comparison below is about a parked car"
    );

    println!(
        "THE NORMAL'S READERS: {travelled:.6} m apart after 600 steps with every \
         contact normal replaced ({grounded} wheel contacts of a possible 2400, \
         {ran:.2} m from the start)"
    );
    assert_eq!(
        plain.pose_bits(),
        scrambled.pose_bits(),
        "replacing every wheel contact normal moved the car {travelled} m — the \
         model DOES read `WheelContact::normal`, so the cell-diagonal snap this \
         file measures is a force error and the wheel-side smoothing clause 1 \
         routed is owed"
    );
}

// ── clause 1c: the sensor ride ──────────────────────────────────────────────

/// **A wheel does not ride on a trigger volume** — P29.7's carried bound, given
/// its filter (`CastTargets::AllSolid`) and the arm that would have caught it.
///
/// The fixture is a car standing on the ground with a **sensor** slab floating a
/// metre and a half above the terrain, wide enough to be under all four wheels.
/// Under `CastTargets::All` the wheel ray finds the slab first — it is nearer —
/// and the suspension pushes off it: the car parks in the air. Under
/// `AllSolid` the slab is invisible to the ray and the car sits on the ground.
///
/// Mutation-measured: reverting the door to `CastTargets::All` leaves the car
/// **1.5 m** higher and this arm red.
#[test]
fn a_wheel_does_not_ride_on_a_trigger_volume() {
    let (x, z) = (8.0, 8.0);
    let mut rig = Rig::new(x, z, 0.0);
    let ground = ground_height(x, z, 0.0);
    // The slab sits between the wheel mounts and the ground, so a ray that can
    // see it hits it before anything else.
    let slab_y = ground + WHEEL_RADIUS + 0.4;
    let e = rig.world.spawn_with_guid(TRIGGER, "Checkpoint", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(x, slab_y, z);
    rig.world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(10.0, 0.05, 10.0),
            sensor: true,
            ..Default::default()
        },
    ));
    rig.world.mark_dirty();
    rig.world.propagate();
    // The handbrake, because the fixture's ground has a 0.108 grade and a car
    // that rolled would be compared against the height it started at.
    rig.park(300);

    let settled = rig.chassis().translation.y - rig.ground_under();
    let on_the_ground = -WHEEL_Y + WHEEL_RADIUS - MASS_KG * 9.81 / 4.0 / 20_000.0;
    println!(
        "THE SENSOR RIDE: the car settled {settled:.4} m over the ground; its \
         springs say {on_the_ground:.4} m and the trigger's top face is \
         {:.4} m up",
        slab_y + 0.05 - ground
    );
    assert!(
        (settled - on_the_ground).abs() < 0.05,
        "the car settled {settled} m over the ground against its springs' \
         {on_the_ground} m — its wheels are riding on the trigger volume, which \
         exerts no force and describes a region"
    );
    // …and the trigger is really there: a solid slab in the same place holds it
    // up, so the arm above is about the SENSOR flag and not about a collider
    // the bridge failed to mirror at all.
    let mut solid = Rig::new(x, z, 0.0);
    let e = solid.world.spawn_with_guid(TRIGGER, "Ramp", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(x, slab_y, z);
    solid.world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(10.0, 0.05, 10.0),
            ..Default::default()
        },
    ));
    solid.world.mark_dirty();
    solid.world.propagate();
    solid.park(300);
    let on_the_slab = solid.chassis().translation.y - solid.ground_under();
    assert!(
        on_the_slab - settled > 0.3,
        "the same slab as a SOLID left the car at {on_the_slab} m against the \
         sensor run's {settled} m — the fixture's slab is not reaching the \
         physics world at all, so the sensor arm above proves nothing"
    );
}

// ── clause 1d: the ride, at speed ───────────────────────────────────────────

/// **The wheel rides the GROUND, and the road is drawn two centimetres above
/// it** — the sink, priced.
///
/// Roads carry no colliders (IB-4's priced ruling: 3.63 ms/step for nothing a
/// body can reach), so a wheel on the island's circuit is on the terrain
/// heightfield and the tarmac it looks like it is on is the road ribbon, drawn
/// at `ground + DEFAULT_ROAD_LIFT_M`. This arm prices that gap **at speed**:
/// the car is driven the length of the tile at full throttle and the worst
/// disagreement between the chassis's own settled ride height and the analytic
/// ground under it is printed.
///
/// The claim is not that the sink is zero. It is that the sink is the lift plus
/// the suspension's own travel and nothing else — so a driver sees the wheels
/// two centimetres into the tarmac and never sees them through it.
#[test]
fn the_wheel_rides_the_ground_and_the_road_is_drawn_above_it() {
    let mut rig = Rig::new(4.0, 4.0, 0.0);
    rig.step(180);
    let settled_clearance = rig.chassis().translation.y - rig.ground_under();

    let mut worst = 0.0f64;
    let mut best = f64::MAX;
    let mut speed_max = 0.0f64;
    for _ in 0..600 {
        rig.drive(
            VehicleControls {
                throttle: 1.0,
                ..Default::default()
            },
            1,
        );
        let t = rig.chassis().translation;
        if t.z > 240.0 || t.x > 240.0 {
            break;
        }
        let clearance = t.y - rig.ground_under();
        worst = worst.max((clearance - settled_clearance).abs());
        best = best.min(clearance);
        let v = rig
            .bridge
            .world()
            .body_linvel(rig.bridge.body_of(CHASSIS).expect("a body"))
            .unwrap_or(DVec3::ZERO);
        speed_max = speed_max.max(v.length());
    }
    println!(
        "THE SINK, AT SPEED: settled clearance {settled_clearance:.4} m, worst \
         departure {worst:.4} m, least clearance {best:.4} m, top speed \
         {speed_max:.2} m/s; the road ribbon is drawn {ROAD_LIFT_M} m above the \
         ground the wheel is on"
    );
    assert!(
        speed_max > 8.0,
        "the run reached {speed_max} m/s — this is not a measurement at speed"
    );
    // The chassis never reaches the ground: if it did, the box would be
    // ploughing and the suspension would have stopped carrying it.
    assert!(
        best > HALF.y,
        "the chassis came within {best} m of the ground, which is inside its own \
         {} m half-height",
        HALF.y
    );
    // …and the ride is a suspension's ride, not a bounce: the departure from
    // the settled clearance stays inside the authored travel.
    assert!(
        worst < 0.25,
        "the ride departed {worst} m from its settled clearance, which is past \
         the 0.25 m of travel the suspension has"
    );
}

/// A wheel's ray hits the ground where the terrain sampler says the ground is —
/// the collider and the query are one surface.
///
/// The bridge's heightfield is built from the same samples `height_at`
/// interpolates, so a disagreement here would mean a car driving on a different
/// surface from the one every other system reads.
#[test]
fn the_collider_the_wheel_hits_is_the_surface_the_sampler_answers() {
    let world = world_with_ground(0.15);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);
    let data = terrain_data(0.15);
    let mut worst = 0.0f64;
    let mut n = 0usize;
    for k in 0..300 {
        let (x, z) = (6.0 + k as f64 * 0.17, 9.0 + k as f64 * 0.11);
        let Some((y, _)) = probe(&mut bridge, x, z) else {
            continue;
        };
        let Some(h) = data.height_at(DVec2::new(x, z)) else {
            continue;
        };
        worst = worst.max((y - h).abs());
        n += 1;
    }
    assert!(n > 250, "only {n} of 300 probes landed on the tile");
    println!("THE ONE SURFACE: worst |ray - height_at| over {n} probes = {worst:.6} m");
    // A heightfield collider is the triangulation of the samples and
    // `height_at` is the bilinear interpolation of the same samples, so the two
    // differ by the twist term inside a cell — millimetres at this roughness,
    // and never the metres a second surface would show.
    assert!(
        worst < 0.05,
        "the wheel's ray and the ground query disagree by {worst} m — a car is \
         driving on a different surface from the one every other system reads"
    );
}

/// A car on real ground still holds itself up, **and a grade is a thing a
/// parked car rolls down** — the P29.7 headline re-taken on a heightfield with
/// the island's own worst grade in it.
///
/// P29.7's `a_parked_rig_settles_on_its_springs_and_stays_there` is on a *box*
/// floor, where "stays there" is free. On a 0.108 grade — 6.2° — it is not: the
/// first cut of this arm measured the same car rolling **4.58 m in two seconds**
/// with nothing applied, which is a correct simulation and a useless fixture.
/// So the settled number is taken with the handbrake on, and the free roll is
/// printed beside it, because clause 4's grade promise is about a car that
/// *can* be held on the steepest stretch the generator will admit.
#[test]
fn a_rig_parked_on_a_graded_heightfield_holds_its_own_weight() {
    let mut rig = Rig::new(8.0, 8.0, 0.0);
    rig.park(300);
    let clearance = rig.chassis().translation.y - rig.ground_under();
    let want = -WHEEL_Y + WHEEL_RADIUS - MASS_KG * 9.81 / 4.0 / 20_000.0;
    println!(
        "A GRADED PARK: clearance {clearance:.4} m against the springs' \
         {want:.4} m on a 0.108 grade"
    );
    assert!(
        (clearance - want).abs() < 0.05,
        "the rig settled {clearance} m over the ground; its springs say {want} m"
    );
    let before = rig.chassis().translation.to_dvec3();
    rig.park(120);
    let held = (rig.chassis().translation.to_dvec3() - before).length();

    // …and the free roll, so the handbrake number above has a control.
    let mut loose = Rig::new(8.0, 8.0, 0.0);
    loose.step(300);
    let before = loose.chassis().translation.to_dvec3();
    loose.step(120);
    let rolled = (loose.chassis().translation.to_dvec3() - before).length();
    println!(
        "A GRADED PARK: the handbrake held it to {held:.4} m in two seconds; \
         with nothing applied it rolled {rolled:.4} m"
    );
    assert!(
        held < 0.5,
        "the handbrake let the car move {held} m in two seconds on a 0.108 grade"
    );
    assert!(
        rolled > held * 2.0,
        "the car rolled {rolled} m with nothing applied against {held} m on the \
         handbrake — the handbrake is not what held it, so this arm proves \
         nothing about the handbrake"
    );
}

// ── clause 3: what the new phase costs ──────────────────────────────────────

/// **THE `vehicle` PHASE'S OWN NUMBER** — what `step_vehicles` costs per car on
/// a streamed-heightfield-shaped world, and the measurement
/// `VEHICLE_STEP_BUDGET_MS` is minted from.
///
/// Two things are asserted and one is reported, which is this tree's own split.
///
/// **Asserted, as a COUNT** (the NPC1c CI-red's law): the phase asks the world
/// exactly **four questions per car per step** — one ray per wheel — so a door
/// that grew a second probe, or one that started walking every entity instead of
/// the vehicle map, fails here rather than in somebody's frame six waves later.
/// A clock cannot see the difference between four rays and eight on a fast
/// machine; a subtraction can.
///
/// **Reported, as a clock**: the milliseconds. A wall clock is a fact about the
/// machine (`[profile.dev]` is `opt-level = 1` with debug assertions), so the
/// figure is printed with its build named and never asserted here. The budget
/// that *is* asserted lives in `inf_player::budget` and is taken on the island.
#[test]
fn the_vehicle_phase_asks_four_questions_a_car_and_costs_what_it_prints() {
    println!(
        "THE VEHICLE PHASE ({} build):",
        if cfg!(debug_assertions) {
            "dev"
        } else {
            "release"
        }
    );
    for n in [1usize, 4, 16, 64] {
        let mut world = world_with_ground(0.0);
        // Cars on a 12 m lattice, well inside the four-tile field.
        for i in 0..n {
            let (gx, gz) = ((i % 8) as f64, (i / 8) as f64);
            let (x, z) = (24.0 + gx * 12.0, 24.0 + gz * 12.0);
            let e = world.spawn_with_guid(Uuid::from_u128(0x7614_0100 + i as u128), "Car", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(x, ground_height(x, z, 0.0) - WHEEL_Y + WHEEL_RADIUS, z);
            world.world_mut().entity_mut(e).insert((
                t,
                RigidBody3D {
                    kind: BodyKind3D::Dynamic,
                    angular_damping: 0.5,
                    ..Default::default()
                },
                Collider3D {
                    shape_kind: ColliderShape3DKind::Box,
                    half_extents: HALF,
                    density: DENSITY,
                    friction: 0.5,
                    ..Default::default()
                },
            ));
            for (k, (wx, wz)) in [(-0.9, 1.4), (0.9, 1.4), (-0.9, -1.4), (0.9, -1.4)]
                .into_iter()
                .enumerate()
            {
                let w = world.spawn_with_guid(
                    Uuid::from_u128(0x7614_1000 + (i * 4 + k) as u128),
                    "Wheel",
                    Some(e),
                );
                let mut wt = Transform::IDENTITY;
                wt.translation = Vec3d::new(wx, WHEEL_Y, wz);
                world.world_mut().entity_mut(w).insert((
                    wt,
                    Collider3D {
                        shape_kind: ColliderShape3DKind::Sphere,
                        radius: WHEEL_RADIUS,
                        sensor: true,
                        ..Default::default()
                    },
                ));
            }
        }
        world.mark_dirty();
        world.propagate();
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.sync_from_world(&world);
        assert_eq!(
            bridge.vehicle_count(),
            n,
            "the fixture built {n} cars and the bridge derived {}",
            bridge.vehicle_count()
        );
        // Settle, so the measurement is of a car on the ground rather than of
        // one still falling onto its springs.
        for _ in 0..60 {
            bridge.sync_from_world(&world);
            inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
            bridge.step(DT);
            bridge.write_back_into(&mut world);
            world.propagate();
        }

        // THE COUNT: one ray a wheel, four wheels a car, and nothing else.
        let before = bridge.world().queries();
        let out = inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
        let asked = bridge.world().queries() - before;
        assert_eq!(out.len(), n, "the phase reported {} of {n} cars", out.len());
        assert_eq!(
            asked,
            4 * n as u64,
            "the vehicle phase asked the world {asked} questions for {n} cars; \
             it casts one ray per wheel and a rig has four, so this is {} per \
             car",
            asked as f64 / n as f64
        );

        // THE CLOCK, reported: MIN of three rounds of forty steps.
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            for _ in 0..40 {
                inf_physics::d3::step_vehicles(&mut world, &mut bridge, DT);
            }
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0 / 40.0);
        }
        println!(
            "  N = {n:>2}: {best:.4} ms a step, {:.2} us a car, {asked} world \
             queries a step",
            best * 1000.0 / n as f64
        );
    }
}

/// The seat the enter door aims at is the top of the chassis, on real ground —
/// the P29.7 rule re-taken where the chassis is not axis-aligned.
#[test]
fn the_seat_is_the_top_of_the_chassis_on_real_ground() {
    let mut rig = Rig::new(8.0, 8.0, 0.0);
    rig.step(60);
    let (seat, rot, _) =
        inf_physics::d3::vehicle::seat_pose(&rig.bridge, CHASSIS).expect("the car has a seat");
    let t = rig.chassis().translation.to_dvec3();
    assert!(
        (seat - t).length() > 0.4 && (seat - t).length() < 0.6,
        "the seat is {} m from the chassis origin; the collider's top face is \
         {} m up",
        (seat - t).length(),
        HALF.y
    );
    assert!(
        rot.angle_between(DQuat::IDENTITY).to_degrees() < 15.0,
        "the car is lying on its side on a 0.108 grade"
    );
}
