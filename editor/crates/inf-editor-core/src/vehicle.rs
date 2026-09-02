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

use inf_ecs::math::Color;
use inf_ecs::vehicle::{rig_nodes, RigSpawn, VehicleDef};

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
///
/// **Ring 0's** since wave VEH2b: the runtime spawner paints the same tyre, and
/// a colour named twice is a colour that drifts.
pub use inf_ecs::vehicle::TYRE_COLOR;

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
/// # ONE RECIPE, TWO DOORS (wave VEH2b)
///
/// What a car is *made of* moved to Ring 0 as
/// [`inf_ecs::vehicle::rig_nodes`] — the chassis with its body, collider,
/// class and engine emitter; a child per `BodyPart`; four wheel sensors each
/// with a tyre. This function is now the half that is genuinely an *authoring*
/// concern: creating the entities through `SceneDoc::create_with_guid`, which
/// tracks creation order and the undo step. Wave VEH2b's traffic materializes
/// the same list straight into an `EcsWorld` through
/// [`inf_ecs::vehicle::spawn_rig`], and the two cannot disagree about what a
/// car is because there is only one list.
///
/// The **order** of that list is load-bearing here and nowhere else: a
/// `SceneDoc` writes its entities in creation order, so a walk in any other
/// order would produce a committed `.inf_lvl` with different bytes and the same
/// contents. `committed_sample_matches_generators` is what says it did not.
pub fn spawn_vehicle(
    doc: &mut SceneDoc,
    chassis: Uuid,
    def: &VehicleDef,
    spawn: VehicleSpawn<'_>,
) -> Uuid {
    let nodes = rig_nodes(
        chassis,
        def,
        &RigSpawn {
            name: spawn.name.to_string(),
            at: spawn.at,
            yaw_deg: spawn.yaw_deg,
            paint: spawn.paint,
            clip: spawn.clip,
            // An authored car keeps its engine emitter: it is the one a player
            // drives, and VEH1a's loop is addressed to it.
            engine_voice: true,
        },
    );
    for node in nodes {
        doc.create_with_guid(node.guid, SpawnKind::Empty, &node.name, node.parent);
        insert!(doc, node.guid, node.transform);
        if let Some(c) = node.body {
            insert!(doc, node.guid, c);
        }
        if let Some(c) = node.collider {
            insert!(doc, node.guid, c);
        }
        if let Some(c) = node.mesh {
            insert!(doc, node.guid, c);
        }
        if let Some(c) = node.material {
            insert!(doc, node.guid, c);
        }
        if let Some(c) = node.class {
            insert!(doc, node.guid, c);
        }
        if let Some(c) = node.audio {
            insert!(doc, node.guid, c);
        }
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
lateral_grip = 1.25
longitudinal_grip = 1.3
front_torque_split = 1.0
brake_bias = 0.63
cog_height_m = -0.34
stability_control = 0.4

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
lateral_grip = 1.12
longitudinal_grip = 1.22
diff_lock_rear = 0.4
cog_height_m = -0.42
drag_lateral_n_per_mps2 = 2.4
stability_control = 0.6

# ── the VEH2a fleet ─────────────────────────────────────────────────────────
#
# Three more silhouettes and three more DRIVES. The point of a catalogue is that
# the rows differ in ways a driver can feel, so each of these turns knobs the
# other two do not: the sports car is rear-drive with a revvy engine, a locked
# rear diff, real downforce and its aids turned down; the SUV is all-wheel-drive,
# tall, soft and heavily assisted; the van is a rear-drive diesel with a huge
# flank, a high centre of gravity and the strongest stability control of the
# five.

[sports]
label = \"Coupe\"

[sports.vehicle]
body = \"sports\"
drivetrain = \"rwd\"
half_width_m = 0.94
half_height_m = 0.58
half_length_m = 2.25
density_kg_m3 = 140.0
wheel_radius_m = 0.33
half_track_m = 0.82
half_wheelbase_m = 1.35
wheel_drop_m = -0.52
# A revvy engine: `torque_curve_bias` below 0.5 means it is soft low down and
# holds its torque to the limiter, which is the opposite end of the one knob the
# pickup's diesel turns the other way.
peak_torque_nm = 460.0
peak_torque_rpm = 4800.0
redline_rpm = 7600.0
idle_rpm = 900.0
idle_torque_frac = 0.6
redline_torque_frac = 0.86
torque_curve_bias = 0.4
engine_brake_nm = 55.0
final_drive = 3.95
gear_count = 7.0
gear_1_ratio = 4.4
gear_2_ratio = 2.9
gear_3_ratio = 1.9
gear_4_ratio = 1.45
gear_5_ratio = 1.15
gear_6_ratio = 0.95
gear_7_ratio = 0.8
reverse_ratio = 3.4
shift_time_s = 0.12
shift_up_rpm = 7200.0
shift_down_rpm = 3000.0
max_speed_mps = 62.0
max_engine_force_n = 24000.0
brake_force_n = 20000.0
handbrake_force_n = 9000.0
brake_bias = 0.66
diff_lock_rear = 0.75
lateral_grip = 1.45
longitudinal_grip = 1.5
tyre_long_peak_slip = 0.11
tyre_lat_peak_slip = 0.13
tyre_long_rise_bias = 0.8
tyre_lat_rise_bias = 0.78
tyre_slide_frac = 0.68
tyre_load_sensitivity = 0.18
wheel_inertia_kgm2 = 1.0
rest_length_m = 0.42
travel_m = 0.16
stiffness_n_per_m = 32000.0
damping_ns_per_m = 4200.0
rolling_resistance = 0.012
cog_height_m = -0.43
anti_roll_front_n_per_m = 22000.0
anti_roll_rear_n_per_m = 24000.0
downforce_n_per_mps2 = 0.45
downforce_centre_z = -0.5
drag_n_per_mps2 = 0.33
drag_lateral_n_per_mps2 = 1.1
max_steer_deg = 34.0
min_steer_deg = 7.0
steer_rate_deg_per_s = 320.0
steer_return_deg_per_s = 420.0
ackermann = 0.9
# A driver's car: the aids are present and they are turned DOWN.
abs_slip = 0.13
traction_control_slip = 0.16
stability_control = 0.25

[suv]
label = \"Wagon\"

[suv.vehicle]
body = \"suv\"
drivetrain = \"awd\"
half_width_m = 1.02
half_height_m = 0.86
half_length_m = 2.45
density_kg_m3 = 122.0
wheel_radius_m = 0.4
half_track_m = 0.9
half_wheelbase_m = 1.48
wheel_drop_m = -0.72
peak_torque_nm = 480.0
peak_torque_rpm = 3000.0
redline_rpm = 5600.0
idle_rpm = 750.0
idle_torque_frac = 0.62
redline_torque_frac = 0.62
torque_curve_bias = 0.62
engine_brake_nm = 48.0
final_drive = 4.0
shift_up_rpm = 5200.0
shift_down_rpm = 2000.0
max_speed_mps = 46.0
max_engine_force_n = 15000.0
brake_force_n = 19000.0
handbrake_force_n = 11000.0
brake_bias = 0.62
diff_lock_front = 0.25
diff_lock_rear = 0.45
lateral_grip = 1.05
longitudinal_grip = 1.15
tyre_long_peak_slip = 0.13
tyre_lat_peak_slip = 0.18
tyre_long_rise_bias = 0.7
tyre_lat_rise_bias = 0.68
tyre_slide_frac = 0.74
tyre_load_sensitivity = 0.26
wheel_inertia_kgm2 = 2.0
rest_length_m = 0.55
travel_m = 0.28
stiffness_n_per_m = 34000.0
damping_ns_per_m = 5200.0
rolling_resistance = 0.016
cog_height_m = -0.4
anti_roll_front_n_per_m = 16000.0
anti_roll_rear_n_per_m = 12000.0
downforce_n_per_mps2 = 0.05
drag_n_per_mps2 = 0.62
drag_lateral_n_per_mps2 = 1.9
max_steer_deg = 33.0
min_steer_deg = 8.0
steer_rate_deg_per_s = 190.0
steer_return_deg_per_s = 280.0
abs_slip = 0.16
traction_control_slip = 0.2
stability_control = 0.55

[van]
label = \"Box Van\"

[van.vehicle]
body = \"van\"
drivetrain = \"rwd\"
half_width_m = 1.05
half_height_m = 1.2
half_length_m = 3.0
density_kg_m3 = 109.0
wheel_radius_m = 0.42
half_track_m = 0.92
half_wheelbase_m = 1.9
wheel_drop_m = -1.02
peak_torque_nm = 620.0
peak_torque_rpm = 1900.0
redline_rpm = 3800.0
idle_rpm = 650.0
idle_torque_frac = 0.78
redline_torque_frac = 0.48
torque_curve_bias = 0.82
engine_brake_nm = 70.0
final_drive = 4.6
shift_up_rpm = 3300.0
shift_down_rpm = 1250.0
max_speed_mps = 32.0
max_engine_force_n = 18000.0
brake_force_n = 22000.0
handbrake_force_n = 13000.0
brake_bias = 0.66
diff_lock_rear = 0.35
lateral_grip = 0.95
longitudinal_grip = 1.0
tyre_long_peak_slip = 0.14
tyre_lat_peak_slip = 0.2
tyre_long_rise_bias = 0.66
tyre_lat_rise_bias = 0.64
tyre_slide_frac = 0.78
tyre_load_sensitivity = 0.3
wheel_inertia_kgm2 = 3.2
rest_length_m = 0.6
travel_m = 0.3
stiffness_n_per_m = 42000.0
damping_ns_per_m = 6800.0
rolling_resistance = 0.02
cog_height_m = -0.49
anti_roll_front_n_per_m = 14000.0
anti_roll_rear_n_per_m = 10000.0
downforce_n_per_mps2 = 0.0
drag_n_per_mps2 = 0.95
drag_lateral_n_per_mps2 = 3.2
max_steer_deg = 30.0
min_steer_deg = 6.0
steer_rate_deg_per_s = 150.0
steer_return_deg_per_s = 230.0
abs_slip = 0.18
traction_control_slip = 0.22
stability_control = 0.7
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
    // The recipe moved to Ring 0 at wave VEH2b, so the arms that read a spawned
    // car's components name the component types here rather than at the top of
    // the file: nothing outside this module builds one any more.
    use inf_ecs::components::{AudioSource, MeshRef, Primitive, Transform};
    use inf_ecs::math::Vec3d;
    use inf_ecs::vehicle::TYRE_ROLL_DEG;

    /// **The catalogue is five rows and all of them are cars** — and no two of
    /// them are the same car with two names.
    ///
    /// The last clause is the one worth having. A fleet whose rows differ only in
    /// colour is a fleet of one, so this walks the *tuning* and requires the five
    /// to differ in the things a driver feels: how fast, how heavy, which axle
    /// drives, how the engine makes its torque, how much grip, and how hard the
    /// aids intervene.
    #[test]
    fn the_island_catalogue_declares_a_fleet_and_no_two_rows_are_one_car() {
        let defs = island_vehicles();
        assert_eq!(defs.0.len(), 5);
        for (id, body) in [
            ("sedan", inf_ecs::vehicle::VehicleBody::Sedan),
            ("truck", inf_ecs::vehicle::VehicleBody::Truck),
            ("sports", inf_ecs::vehicle::VehicleBody::Sports),
            ("suv", inf_ecs::vehicle::VehicleBody::Suv),
            ("van", inf_ecs::vehicle::VehicleBody::Van),
        ] {
            assert_eq!(
                defs.get(id).unwrap_or_else(|| panic!("no `{id}` row")).body,
                body
            );
        }
        let sedan = defs.get("sedan").expect("the sedan row");
        let truck = defs.get("truck").expect("the truck row");
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
        // original rows are the same car with two names.
        assert!(truck.half_extents.z > sedan.half_extents.z);
        assert!(truck.class.max_speed_mps < sedan.class.max_speed_mps);
        assert!(truck.wheel_radius_m > sedan.wheel_radius_m);

        // **No two rows agree about how they drive.** Six axes, each of which a
        // driver feels: a row set that collided on any of them would be a fleet
        // wearing five names.
        type Axis = (&'static str, fn(&inf_ecs::vehicle::VehicleDef) -> f64);
        let axis: [Axis; 6] = [
            ("max_speed_mps", |d| d.class.max_speed_mps),
            ("peak_torque_nm", |d| d.class.peak_torque_nm),
            ("torque_curve_bias", |d| d.class.torque_curve_bias),
            ("lateral_grip", |d| d.class.lateral_grip),
            ("stability_control", |d| d.class.stability_control),
            ("mass", |d| {
                8.0 * d.half_extents.x * d.half_extents.y * d.half_extents.z * d.density_kg_m3
            }),
        ];
        for (name, read) in axis {
            let mut seen: Vec<(String, f64)> =
                defs.0.iter().map(|(k, d)| (k.clone(), read(d))).collect();
            seen.sort_by(|a, b| a.1.total_cmp(&b.1));
            for pair in seen.windows(2) {
                assert!(
                    pair[1].1 > pair[0].1,
                    "`{name}`: {} and {} both read {} — two rows that agree on \
                     every axis are one car with two names",
                    pair[0].0,
                    pair[1].0,
                    pair[0].1
                );
            }
        }
        // …and the three drivetrains are all represented, which is the axis that
        // is deliberately NOT all-different.
        let splits: std::collections::BTreeSet<u64> = defs
            .0
            .values()
            .map(|d| d.class.front_torque_split.to_bits())
            .collect();
        assert!(
            splits.len() >= 3,
            "the fleet has only {} distinct drivetrains",
            splits.len()
        );
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
