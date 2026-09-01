//! **The one door that authors a car** (island wave VEH1a) — geometry, wheels,
//! a drawn body, and the class it is tuned with.
//!
//! # What was wrong, and it is the I8b finding wearing a bumper
//!
//! P29.7's committed car is a `Primitive::Cube` on a `Transform` with **scale
//! one**, over a `Collider3D` whose half-extents are `(2.0, 0.5, 1.0)`. Two
//! things follow, and neither had an arm:
//!
//! * it **drew as a one-metre cube** while its collider was eight cubic metres —
//!   the exact shape of wave I8b's finding (*"a building has never once been
//!   drawn at the dimensions it is built at"*), on a vehicle;
//! * and those half-extents describe a car **four metres wide and two metres
//!   long**, because `+Z` is forward in this engine. It stood across a 2.8 m
//!   wheelbase on a 1.8 m track. Nothing saw it, because nothing drew it.
//!
//! So a car is authored here, from one [`VehicleDef`](inf_ecs::vehicle::VehicleDef),
//! and the drawn thing is
//! the same size and shape as the solid thing by construction.
//!
//! # Why child entities and not a scatter batch
//!
//! Wave I8b's module meshes reach the screen through `ScatterMeshes`: a Ring-0
//! table of unit-space geometry, content-derived GUIDs, registered by hand in
//! both hosts. That is the right route for **static** content, and it is the
//! wrong one for a car, for a reason the batch type states itself:
//! `ScatterData::key` is a content hash over the *instance records*, and a
//! vehicle's instances carry its rotation. A moving car re-keys its batch every
//! frame — a fresh upload, sixty times a second, per car — where the whole
//! argument for the anchor being outside the key is that a batch's *content*
//! must not move.
//!
//! The parts are therefore ordinary scene entities drawing the built-in
//! primitives, which is the path an authored actor has always taken and which
//! costs the projectors nothing at all. What is borrowed from I8b is everything
//! else and it is the part that mattered: **a family of unit-space parts in
//! fractions of the hull** (`inf_ecs::vehicle::BodyPart`), so one table serves a
//! sedan and a truck at any size; **content-derived GUIDs**
//! (`inf_ecs::vehicle::body_part_guid`), so the same car authored twice produces
//! the same bytes; and **no committed mesh files**, so the island's content
//! folder does not grow.
//!
//! Axis-aligned boxes and one cylinder, because `inf-dcc` refuses a bevel on
//! edges that share an endpoint (the P23 finding) and a car body is nothing but
//! such edges.

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::{
    AudioSource, BodyKind3D, Collider3D, ColliderShape3DKind, Material, MeshRef, Primitive,
    RigidBody3D, Transform,
};
use inf_ecs::math::{Color, Vec3d};
use inf_ecs::vehicle::{VehicleDef, TYRE_ROLL_DEG, TYRE_WIDTH_FRAC};

use crate::ipc::SpawnKind;
use crate::scene::SceneDoc;

/// Insert a bundle onto `guid`'s entity, dirtying the doc.
///
/// The third copy of this macro (`samples.rs`, `island.rs`, here), and the third
/// time for the same reason `island.rs` records: `macro_rules!` is
/// module-scoped, and writing it once as a function needs a
/// `B: bevy_ecs::bundle::Bundle` bound — which would make this crate name
/// `bevy_ecs`, which is exactly what `inf-ecs` exists to prevent.
macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr $(,)?) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

/// The tyres' colour — dark, unlit-looking rubber, so a wheel reads against the
/// body whatever the body is painted.
pub const TYRE_COLOR: Color = Color::new(0.07, 0.07, 0.08, 1.0);

/// **Where one authored vehicle goes, and what it looks like** — the half of a
/// spawn that is not the class.
///
/// A struct and not five more parameters: [`spawn_vehicle`] was eight arguments
/// and the pairs in it (a pose, a dressing) are what a caller thinks in.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleSpawn<'a> {
    /// The entity name in the outliner.
    pub name: &'a str,
    /// The **chassis origin** in world space.
    pub at: DVec3,
    /// Heading about `+Y`, degrees. `0` faces `+Z`, which is forward.
    pub yaw_deg: f64,
    /// The body colour. The tyres are always [`TYRE_COLOR`].
    pub paint: Color,
    /// The engine loop's `.inf_audio` clip. `None` is a silent voice, and the
    /// *command stream* — which is what PIE == shipping compares — is the same
    /// either way.
    pub clip: Option<Uuid>,
}

/// **Author one vehicle**, returning its chassis `Guid`.
///
/// Every entity the call creates has a content-derived `Guid`, so authoring the
/// same car twice is byte-identical and re-authoring one in place replaces its
/// own parts rather than accumulating a second set.
///
/// What is built:
///
/// * the **chassis** — a `Dynamic` `RigidBody3D`, a box `Collider3D` at the
///   def's half-extents and density, and the def's
///   [`VehicleClass`](inf_ecs::components::VehicleClass) (scene v25, applied
///   once at creation by the bridge);
/// * the **body** — one child per [`BodyPart`] of the def's family, drawing its
///   own primitive at its own size;
/// * the **wheels** — four sphere **sensors** with no body of their own, which
///   is what `inf_ecs::vehicle::wheel_of` recognises, each with a **tyre** child
///   that draws the cylinder the door's own rotation write cannot lay down;
/// * an [`AudioSource`] on the chassis, looping and spatial, so the VEH1a engine
///   loop has an emitter to address. Its `clip` is the caller's: `None` is a
///   silent voice, and the *command stream* — which is what PIE == shipping
///   compares — is the same either way.
///
/// [`BodyPart`]: inf_ecs::vehicle::BodyPart
pub fn spawn_vehicle(
    doc: &mut SceneDoc,
    chassis: Uuid,
    def: &VehicleDef,
    spawn: VehicleSpawn<'_>,
) -> Uuid {
    let part_guid = |part: &str| inf_ecs::vehicle::body_part_guid(chassis, part);
    let h = def.half_extents;

    doc.create_with_guid(chassis, SpawnKind::Empty, spawn.name, None);
    insert!(
        doc,
        chassis,
        Transform {
            translation: Vec3d::from_dvec3(spawn.at),
            rotation: Vec3d::new(0.0, spawn.yaw_deg, 0.0),
            scale: Vec3d::ONE,
        }
    );
    insert!(
        doc,
        chassis,
        RigidBody3D {
            kind: BodyKind3D::Dynamic,
            // A car does not spin on its own axis for want of damping; the
            // suspension supplies the rest of the resistance. (P29.7's number.)
            angular_damping: 0.5,
            ..Default::default()
        }
    );
    insert!(
        doc,
        chassis,
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: h,
            density: def.density_kg_m3,
            friction: 0.5,
            ..Default::default()
        }
    );
    insert!(doc, chassis, def.class);
    insert!(
        doc,
        chassis,
        AudioSource {
            clip: spawn.clip,
            looping: true,
            spatial: true,
            // Not autoplay: the VEH1a engine loop emits the `Play` itself, on
            // the step the vehicle first appears in an outcome, and two paths
            // both starting one voice is two `Play`s for one source.
            autoplay: false,
            ..Default::default()
        }
    );

    // ── the body ──
    for part in def.body.parts() {
        let guid = part_guid(part.name);
        doc.create_with_guid(guid, SpawnKind::Empty, part.name, Some(chassis));
        insert!(
            doc,
            guid,
            Transform {
                translation: Vec3d::new(
                    part.centre.x * h.x,
                    part.centre.y * h.y,
                    part.centre.z * h.z,
                ),
                rotation: Vec3d::ZERO,
                // The built-in primitives span ±0.5, so a part's SCALE is its
                // full extent — twice its half-extent. This is the line the
                // committed car never had: without it every part draws as the
                // unit primitive whatever the collider says.
                scale: Vec3d::new(
                    2.0 * part.half.x * h.x,
                    2.0 * part.half.y * h.y,
                    2.0 * part.half.z * h.z,
                ),
            }
        );
        insert!(
            doc,
            guid,
            MeshRef {
                primitive: part.primitive,
                asset: None,
            }
        );
        insert!(
            doc,
            guid,
            Material {
                base_color: spawn.paint,
                metallic: 0.35,
                roughness: 0.42,
                ..Default::default()
            }
        );
    }

    // ── the wheels, and the tyres that draw them ──
    let r = def.wheel_radius_m;
    for (i, mount) in def.wheel_mounts().into_iter().enumerate() {
        let wheel = part_guid(&format!("wheel{i}"));
        doc.create_with_guid(wheel, SpawnKind::Empty, "Wheel", Some(chassis));
        insert!(doc, wheel, Transform::from_translation(mount.to_dvec3()));
        insert!(
            doc,
            wheel,
            Collider3D {
                shape_kind: ColliderShape3DKind::Sphere,
                radius: r,
                // A wheel is a SENSOR and has no body: that is the whole of
                // `inf_ecs::vehicle::wheel_of`, and the bridge consumes it
                // rather than mirroring it into rapier.
                sensor: true,
                ..Default::default()
            }
        );
        // The tyre is a child of the wheel because `step_vehicles` writes the
        // wheel's rotation every step as euler `(spin, steer, 0)` — there is no
        // roll slot left to lay a `+Y`-axis cylinder on its side with.
        let tyre = part_guid(&format!("tyre{i}"));
        doc.create_with_guid(tyre, SpawnKind::Empty, "Tyre", Some(wheel));
        insert!(
            doc,
            tyre,
            Transform {
                translation: Vec3d::ZERO,
                rotation: Vec3d::new(0.0, 0.0, TYRE_ROLL_DEG),
                scale: Vec3d::new(2.0 * r, 2.0 * r * TYRE_WIDTH_FRAC, 2.0 * r),
            }
        );
        insert!(
            doc,
            tyre,
            MeshRef {
                primitive: Primitive::Cylinder,
                asset: None,
            }
        );
        insert!(
            doc,
            tyre,
            Material {
                base_color: TYRE_COLOR,
                metallic: 0.0,
                roughness: 0.9,
                ..Default::default()
            }
        );
    }
    chassis
}

/// Where a vehicle's chassis origin goes if its wheels are to touch `ground`
/// with the suspension at full extension — the placement an author makes.
///
/// One function because three callers want it (the island's spawner, the
/// phase-29 course and any test that parks a car), and "wheel drop plus wheel
/// radius" written out three times is three chances to write it once with the
/// sign wrong.
pub fn resting_origin_y(def: &VehicleDef, ground_y: f64) -> f64 {
    ground_y - def.wheel_drop_m + def.wheel_radius_m
}

/// **The island's fleet, as authored content** (island wave VEH1a).
///
/// # Zero schema, on the `WeaponDef` precedent
///
/// `GAMEPLAY_ITEMS_TOML` is a `&str` in this crate rather than a file, because
/// `ItemDef`/`WeaponDef` are not serializable types and the catalogue is
/// *authoring input* — what ships is ordinary scene content. A vehicle catalogue
/// is the same shape one system over: `VehicleDefs::merge_toml` reads this at
/// **generation** time, and what reaches the committed `.inf_lvl` is a chassis,
/// four wheel sensors, a body and the `VehicleClass` the scene has carried since
/// v25. No loader ships, no asset kind is added, and no schema moves.
///
/// The two rows are the two silhouettes. The numbers are a road car and a light
/// pickup: the truck is longer, wider, heavier, geared shorter and softer on its
/// springs, and it turns less at speed because it is taller.
pub const ISLAND_VEHICLES_TOML: &str = "\
# The island's vehicle catalogue (island wave VEH1a).
#
# Every key is either a geometry name (`VehicleDef::geometry_names`) or a
# tuning name (`VehicleTuning::names`); an unknown one is refused BY NAME at
# parse time rather than defaulted.

[sedan]
label = \"Saloon\"

[sedan.vehicle]
body = \"sedan\"
half_width_m = 0.92
half_height_m = 0.62
half_length_m = 2.2
density_kg_m3 = 118.0
wheel_radius_m = 0.34
half_track_m = 0.84
half_wheelbase_m = 1.42
wheel_drop_m = -0.62
max_speed_mps = 34.0
max_engine_force_n = 9000.0
brake_force_n = 13000.0
stiffness_n_per_m = 21000.0
damping_ns_per_m = 3200.0
# A 2.0-litre petrol, geared to reach its limiter in top (island wave VEH2a).
peak_torque_nm = 245.0
peak_torque_rpm = 4000.0
redline_rpm = 6400.0
idle_rpm = 800.0
idle_torque_frac = 0.52
redline_torque_frac = 0.74
final_drive = 3.9
shift_up_rpm = 6000.0
shift_down_rpm = 2300.0

[truck]
label = \"Pickup\"

[truck.vehicle]
body = \"truck\"
half_width_m = 1.02
half_height_m = 0.82
half_length_m = 2.65
density_kg_m3 = 132.0
wheel_radius_m = 0.42
half_track_m = 0.92
half_wheelbase_m = 1.7
wheel_drop_m = -0.78
max_speed_mps = 27.0
max_engine_force_n = 14000.0
brake_force_n = 15000.0
stiffness_n_per_m = 26000.0
damping_ns_per_m = 3800.0
max_steer_deg = 32.0
min_steer_deg = 6.0
# A DIESEL, and the reason the row needs one at all (island wave VEH2a): the
# torque curve is what moves a car now, and a 2 341 kg pickup on the engine
# every default rig shares was still at IDLE at four metres a second — it
# climbed the audited grade at 0.55 m/s^2 and covered 17.9 m in ten seconds
# where the sedan covered 157.8. So: torque low and early (`bias` above 0.5),
# a short final drive, and a limiter it reaches at 27 m/s.
peak_torque_nm = 520.0
peak_torque_rpm = 2200.0
redline_rpm = 4200.0
idle_rpm = 700.0
idle_torque_frac = 0.72
redline_torque_frac = 0.55
torque_curve_bias = 0.75
final_drive = 4.3
shift_up_rpm = 3700.0
shift_down_rpm = 1400.0
wheel_inertia_kgm2 = 2.4
";

/// The island's catalogue, parsed.
///
/// Panics on a malformed table, which is right for a `const` this crate ships:
/// a catalogue that does not parse is a build error, not a runtime refusal.
pub fn island_vehicles() -> inf_ecs::vehicle::VehicleDefs {
    let mut defs = inf_ecs::vehicle::VehicleDefs::default();
    defs.merge_toml(ISLAND_VEHICLES_TOML)
        .expect("the island's committed vehicle catalogue parses");
    defs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The catalogue is two rows and both of them are cars.**
    #[test]
    fn the_island_catalogue_declares_a_sedan_and_a_truck() {
        let defs = island_vehicles();
        assert_eq!(defs.0.len(), 2);
        let sedan = defs.get("sedan").expect("the sedan row");
        let truck = defs.get("truck").expect("the truck row");
        assert_eq!(sedan.body, inf_ecs::vehicle::VehicleBody::Sedan);
        assert_eq!(truck.body, inf_ecs::vehicle::VehicleBody::Truck);
        for (id, def) in &defs.0 {
            let volume = 8.0 * def.half_extents.x * def.half_extents.y * def.half_extents.z;
            let kg = volume * def.density_kg_m3;
            println!(
                "{id}: {:.2} x {:.2} x {:.2} m, {kg:.0} kg, {:.0} km/h",
                def.half_extents.x * 2.0,
                def.half_extents.y * 2.0,
                def.half_extents.z * 2.0,
                def.class.max_speed_mps * 3.6
            );
            assert!(
                def.half_extents.z > def.half_extents.x,
                "{id} is wider than it is long"
            );
            assert!(
                (900.0..3600.0).contains(&kg),
                "{id} weighs {kg} kg, which is not a road vehicle"
            );
            // Its wheels are under it, not beside it.
            for m in def.wheel_mounts() {
                assert!(
                    m.z.abs() < def.half_extents.z,
                    "{id}: a wheel past the nose"
                );
                assert!(
                    m.x.abs() <= def.half_extents.x + def.wheel_radius_m,
                    "{id}: a wheel {} m out from a body {} m wide",
                    m.x.abs(),
                    def.half_extents.x * 2.0
                );
            }
        }
        // The truck is the heavier, slower, longer one — or the catalogue's two
        // rows are the same car with two names.
        assert!(truck.half_extents.z > sedan.half_extents.z);
        assert!(truck.class.max_speed_mps < sedan.class.max_speed_mps);
        assert!(truck.wheel_radius_m > sedan.wheel_radius_m);
    }

    /// **A spawned car is a rig the recogniser finds, with a body the right size
    /// on it.**
    ///
    /// The end-to-end arm for this door: the geometry the def describes reaches
    /// the world, `rig_of` derives four wheels from it, and every drawn part is
    /// inside the collider it is drawn on — which is the claim the committed
    /// car could not make.
    #[test]
    fn a_spawned_car_is_a_derivable_rig_wearing_its_own_collider() {
        let defs = island_vehicles();
        for (id, def) in &defs.0 {
            let mut doc = SceneDoc::new();
            let chassis = Uuid::from_u128(0x5EDA_0000 + id.len() as u128);
            spawn_vehicle(
                &mut doc,
                chassis,
                def,
                VehicleSpawn {
                    name: "Car",
                    at: DVec3::new(10.0, resting_origin_y(def, 0.0), -4.0),
                    yaw_deg: 35.0,
                    paint: Color::new(0.6, 0.1, 0.1, 1.0),
                    clip: None,
                },
            );
            doc.world_mut().propagate();

            let rig = inf_ecs::vehicle::rig_of(doc.world(), chassis)
                .unwrap_or_else(|| panic!("{id}: the spawned car is not a rig"));
            assert_eq!(rig.wheels.len(), 4, "{id}");
            assert_eq!(
                rig.seat_local,
                Vec3d::new(0.0, def.half_extents.y, 0.0),
                "{id}: the seat is the top face of the chassis collider"
            );
            assert_eq!(
                rig.wheels.iter().filter(|w| w.steered()).count(),
                2,
                "{id}: the front pair steers"
            );

            // Every drawn part is inside the hull it is drawn on, at its own
            // size — the defect this door exists to close.
            let world = doc.world();
            let mut drawn = 0usize;
            for part in def.body.parts() {
                let guid = inf_ecs::vehicle::body_part_guid(chassis, part.name);
                let e = world
                    .entity_of(guid)
                    .unwrap_or_else(|| panic!("{id}: no `{}` part", part.name));
                let t = world
                    .world()
                    .get::<Transform>(e)
                    .expect("a part has a transform");
                assert!(
                    world.world().get::<MeshRef>(e).is_some(),
                    "{id}/{}: a part that draws nothing",
                    part.name
                );
                for (axis, c, s, hull) in [
                    ("x", t.translation.x, t.scale.x, def.half_extents.x),
                    ("y", t.translation.y, t.scale.y, def.half_extents.y),
                    ("z", t.translation.z, t.scale.z, def.half_extents.z),
                ] {
                    assert!(
                        s > 0.0,
                        "{id}/{}: axis {axis} is drawn at scale {s}",
                        part.name
                    );
                    assert!(
                        c.abs() + s / 2.0 <= hull + 1e-9,
                        "{id}/{}: axis {axis} reaches {} past a hull half-extent \
                         of {hull}",
                        part.name,
                        c.abs() + s / 2.0
                    );
                }
                // …and it is not the unit primitive: the whole finding is that a
                // scale of one drew a one-metre cube over an eight-cubic-metre
                // collider.
                assert!(
                    t.scale != Vec3d::ONE,
                    "{id}/{}: drawn at scale one, which is the defect",
                    part.name
                );
                drawn += 1;
            }
            assert!(drawn >= 4, "{id}: only {drawn} drawn parts");

            // The tyres are cylinders laid on their sides, sized from the wheel.
            for i in 0..4 {
                let tyre = inf_ecs::vehicle::body_part_guid(chassis, &format!("tyre{i}"));
                let e = world.entity_of(tyre).expect("a tyre");
                let t = world.world().get::<Transform>(e).expect("its transform");
                assert_eq!(t.rotation.z, TYRE_ROLL_DEG);
                assert_eq!(t.scale.x, 2.0 * def.wheel_radius_m);
                assert_eq!(t.scale.z, 2.0 * def.wheel_radius_m);
                assert!(t.scale.y < t.scale.x, "{id}: a tyre wider than it is tall");
                assert_eq!(
                    world.world().get::<MeshRef>(e).map(|m| m.primitive),
                    Some(Primitive::Cylinder)
                );
            }

            // The class the catalogue authored is on the chassis, and it is the
            // catalogue's own numbers rather than the default.
            let e = world.entity_of(chassis).expect("the chassis");
            let class = world
                .world()
                .get::<inf_ecs::components::VehicleClass>(e)
                .copied()
                .unwrap_or_else(|| panic!("{id}: no VehicleClass"));
            assert_eq!(class.max_speed_mps, def.class.max_speed_mps);
            assert_ne!(
                class,
                inf_ecs::components::VehicleClass::default(),
                "{id}: the catalogue's tuning did not reach the entity"
            );
            // …and the engine loop has an emitter to address.
            assert!(
                world
                    .world()
                    .get::<AudioSource>(e)
                    .is_some_and(|a| a.looping),
                "{id}: no looping AudioSource, so the engine loop addresses nothing"
            );
        }
    }

    /// **THE PHASE-29 COURSE'S CAR IS STILL BUILT SIDEWAYS**, and this arm is
    /// the tripwire that says so.
    ///
    /// `PHASE29_CAR_HALF` is `(2.0, 0.5, 1.0)` and its own doc calls it
    /// "4 × 1 × 2 m", which it is — in the order **width, height, length**. `+Z`
    /// is forward in this engine, so the committed car is four metres across a
    /// 1.8 m track and two metres along a 2.8 m wheelbase: its wheels stick out
    /// of the ends and its body overhangs them by a metre on each side.
    ///
    /// It has never been visible because the entity draws `Primitive::Cube` at
    /// `Transform` scale **one** — a one-metre cube over an eight-cubic-metre
    /// collider, which is wave I8b's finding on a vehicle.
    ///
    /// **Not fixed here, on purpose.** The phase-29 locomotion course is a
    /// committed `.inf_lvl` with a twenty-arm gate over it, and re-proportioning
    /// its chassis changes the rig's inertia and every number that gate holds —
    /// a slice of its own, not a footnote in the wave that found it. The island's
    /// own cars come from the catalogue and are the right way round
    /// (`the_default_vehicle_is_longer_than_it_is_wide_and_covers_its_wheels`).
    ///
    /// So the defect is **asserted**: the day the course's car is fixed, this
    /// arm goes red and names the ledger entry that owed it.
    #[test]
    fn the_phase29_courses_car_is_still_wider_than_it_is_long() {
        let h = crate::samples::PHASE29_CAR_HALF;
        let mounts = crate::samples::phase29_wheel_mounts();
        let track = mounts.iter().map(|m| m.x.abs()).fold(0.0f64, f64::max);
        let base = mounts.iter().map(|m| m.z.abs()).fold(0.0f64, f64::max);
        println!(
            "PHASE 29's CAR: body {:.1} x {:.1} x {:.1} m over a {:.1} m track \
             and a {:.1} m wheelbase",
            h.x * 2.0,
            h.y * 2.0,
            h.z * 2.0,
            track * 2.0,
            base * 2.0
        );
        assert!(
            h.x > h.z,
            "the phase-29 car is no longer wider than it is long — the defect \
             this arm carries has been fixed, so delete the arm and close the \
             VEH1a ledger item that routed it"
        );
        assert!(
            base > h.z,
            "its wheels no longer stick out past its own bodywork ({base} m \
             against a half-length of {})",
            h.z
        );
    }

    /// **Authoring the same car twice is the same bytes** — the content-derived
    /// GUID rule, at the level a generator cares about.
    #[test]
    fn authoring_one_car_twice_produces_the_same_entities() {
        let def = island_vehicles().get("sedan").copied().expect("the sedan");
        let chassis = Uuid::from_u128(0x5EDA_1234);
        let ids = |doc: &SceneDoc| -> Vec<Uuid> {
            let mut v: Vec<Uuid> = doc
                .world()
                .world()
                .iter_entities()
                .filter_map(|e| e.get::<inf_ecs::components::Guid>().map(|g| g.0))
                .collect();
            v.sort();
            v
        };
        let spawn = VehicleSpawn {
            name: "Car",
            at: DVec3::ZERO,
            yaw_deg: 0.0,
            paint: Color::WHITE,
            clip: None,
        };
        let mut a = SceneDoc::new();
        spawn_vehicle(&mut a, chassis, &def, spawn);
        let mut b = SceneDoc::new();
        spawn_vehicle(&mut b, chassis, &def, spawn);
        assert_eq!(ids(&a), ids(&b));
        assert_eq!(
            ids(&a).len(),
            1 + def.body.parts().len() + 8,
            "a car is its chassis, its parts, four wheels and four tyres"
        );
    }
}
