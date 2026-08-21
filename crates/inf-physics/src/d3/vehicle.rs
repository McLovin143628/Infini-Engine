//! **The vehicle fixed-step door** (P29.7): cast the wheel rays, apply what the
//! model asks for, write the wheels back.
//!
//! The `inf_physics` half of `inf_ecs::vehicle`, and the same split as
//! [`super::camera`] / `inf_ecs::camera` and [`super::movement`] /
//! `inf_ecs::movement`: everything here touches rapier and nothing here decides
//! anything. The arithmetic — the spring, the engine curve, the friction circle
//! — is on the other side of that wall and is unit-tested without a world.
//!
//! # One door, inside the movement door
//!
//! [`step_vehicles`] is called from the **last statement** of
//! [`super::movement::step_character_movement`], which is the one door both
//! hosts already call, in the one slot they already agree on. A sibling function
//! called separately by each host would be a hand-maintained mirror, and this
//! phase's own ledger records two defects of exactly that shape (the intent
//! feed and the publish order). Nothing about a vehicle needs its own slot: a
//! driver's controls are written by the character step immediately above, and
//! the forces must land before `bridge.step`, which is where the movement door
//! already sits.
//!
//! `O(vehicles)`, never `O(entities)`: the set is discovered inside the entity
//! walk `sync_from_world_sim` already makes, and this function walks the map.
//!
//! # Force ownership, which is not obvious
//!
//! A rapier force **persists** until `reset_forces` (P20.2's law), so a pass
//! that applies one must own the clear. The water pass resets and re-applies
//! every buoyant body at fixed-step stage 8, and this door runs at stage 12 — so
//! for a vehicle that is *also* buoyant, water's reset is already the clear and
//! resetting again here would delete this step's buoyancy before the solver ever
//! saw it. The rule is therefore one line with a real reason:
//! **reset iff the water pass does not already own this body**
//! ([`PhysicsBridge3D::is_buoyant`]).

use std::collections::BTreeSet;

use glam::DVec3;
use uuid::Uuid;

use inf_ecs::components::Transform;
use inf_ecs::math::Vec3d;
use inf_ecs::vehicle::{ChassisState, WheelContact, WheelForce};
use inf_ecs::EcsWorld;

use super::query::CastTargets;
use super::{ColliderId3D, PhysicsBridge3D};

/// What one vehicle's step did — returned so a test can assert on the
/// *decisions*, while the arms that matter assert on the WORLD.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VehicleOutcome {
    /// The chassis entity.
    pub chassis: Uuid,
    /// How many of its wheels found ground this step.
    pub wheels_grounded: usize,
    /// The suspension load summed over those wheels, newtons — a rig at rest on
    /// flat ground reports its own weight, which is the cheapest possible check
    /// that the springs are doing their job.
    pub load_n: f64,
    /// Forward speed, m/s, signed (negative is reversing).
    pub forward_mps: f64,
}

/// **Advance every vehicle one fixed step.**
///
/// Runs inside [`super::movement::step_character_movement`], after every
/// character has moved (so a driver's controls are this step's) and before the
/// solver.
pub fn step_vehicles(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    dt: f64,
) -> Vec<VehicleOutcome> {
    if !dt.is_finite() || dt <= 0.0 {
        return Vec::new();
    }
    let guids = bridge.vehicle_guids();
    if guids.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(guids.len());
    let mut forces: Vec<WheelForce> = Vec::new();
    for chassis in guids {
        if let Some(o) = step_one(world, bridge, chassis, dt, &mut forces) {
            out.push(o);
        }
    }
    out
}

fn step_one(
    world: &mut EcsWorld,
    bridge: &mut PhysicsBridge3D,
    chassis: Uuid,
    dt: f64,
    forces: &mut Vec<WheelForce>,
) -> Option<VehicleOutcome> {
    let body = bridge.body_of(chassis)?;
    let state = {
        let w = bridge.world();
        ChassisState {
            position: w.body_translation(body)?,
            rotation: w.body_rotation(body)?,
            linvel: w.body_linvel(body).unwrap_or(DVec3::ZERO),
            angvel: w.body_angvel(body).unwrap_or(DVec3::ZERO),
            mass_kg: w.body_mass(body).unwrap_or(0.0),
        }
    };
    let (_, _, up) = state.basis();
    // A wheel ray starts inside the vehicle it belongs to, so the exclusion is
    // structural rather than optional. The wheels themselves are NOT in this set
    // because they are not in rapier at all — the bridge consumes a wheel rather
    // than mirroring it (see `inf_ecs::vehicle`'s module docs).
    let mut exclude: BTreeSet<ColliderId3D> = BTreeSet::new();
    if let Some(c) = bridge.collider_of(chassis) {
        exclude.insert(c);
    }

    // ── 1. the rays. One per wheel, `O(wheels)`.
    let rays: Vec<(DVec3, f64, f64)> = {
        let v = bridge.vehicle_of(chassis)?;
        let rest = v.suspension_rest_m();
        v.rig()
            .wheels
            .iter()
            .map(|wm| {
                // The authored child transform is the wheel CENTRE at full
                // extension; the strut anchor is `rest` above it, and the ray
                // reaches one radius past the extended centre.
                let anchor_local = wm.mount_local.to_dvec3() + DVec3::Y * rest;
                (
                    state.position + state.rotation * anchor_local,
                    rest + wm.radius_m,
                    wm.radius_m,
                )
            })
            .collect()
    };
    let hits: Vec<Option<WheelContact>> = rays
        .iter()
        .map(|(origin, max_toi, _)| {
            bridge
                .world_mut()
                // **Everything solid**, not just static geometry: a car drives
                // over a crate and up a fractured chunk, and a suspension ray
                // that skipped dynamic bodies would put the wheel through them.
                .cast_ray_where(*origin, -up, *max_toi, &exclude, CastTargets::All)
                .map(|hit| WheelContact {
                    point: hit.point,
                    normal: hit.normal,
                    distance_m: hit.toi,
                })
        })
        .collect();

    // ── 2. the model.
    forces.clear();
    let (grounded, load_n, poses) = {
        let v = bridge.vehicle_mut(chassis)?;
        for (state, hit) in v.wheels_mut().iter_mut().zip(hits) {
            state.contact = hit;
        }
        v.solve(state, dt, forces);
        let grounded = v.wheels().iter().filter(|w| w.contact.is_some()).count();
        let load_n = v.wheels().iter().map(|w| w.load_n).sum::<f64>();
        let poses: Vec<(Uuid, Vec3d, f64, f64, f64)> = v
            .rig()
            .wheels
            .iter()
            .enumerate()
            .filter_map(|(i, wm)| {
                let (length, steer, spin) = v.wheel_pose(i)?;
                Some((wm.guid, wm.mount_local, length, steer, spin))
            })
            .collect();
        (grounded, load_n, poses)
    };

    // ── 3. the forces. The clear first — see this module's ownership note.
    if !bridge.is_buoyant(chassis) {
        bridge.world_mut().reset_forces(body);
    }
    for f in forces.iter() {
        if f.force != DVec3::ZERO {
            bridge
                .world_mut()
                .apply_force_at_point(body, f.force, f.point);
        }
    }

    // ── 4. the wheels, as they are drawn. A visual write only: nothing reads
    //    these back (the mount is taken once — see `reconcile_vehicles`), so a
    //    rig with no wheel meshes simulates identically to one with them.
    let rest = bridge.vehicle_of(chassis)?.suspension_rest_m();
    let mut moved = false;
    for (guid, mount_local, length, steer, spin) in poses {
        let Some(entity) = world.entity_of(guid) else {
            continue;
        };
        let Some(mut t) = world.world_mut().get_mut::<Transform>(entity) else {
            continue;
        };
        t.translation = Vec3d::new(
            mount_local.x,
            mount_local.y + (rest - length),
            mount_local.z,
        );
        // Euler YXZ degrees, the `Transform` convention: yaw is the steer and
        // pitch is the roll of the wheel about its own axle.
        t.rotation = Vec3d::new(spin, steer, 0.0);
        moved = true;
    }
    if moved {
        world.mark_dirty();
    }

    Some(VehicleOutcome {
        chassis,
        wheels_grounded: grounded,
        load_n,
        forward_mps: state.linvel.dot(state.basis().0),
    })
}

// ── the seat: enter, drive, exit (P29.7) ────────────────────────────────────

/// How close a character's feet must be to a seat to climb into it, metres.
///
/// Generous rather than exact: a door is not a pixel, and a refusal an author
/// cannot see the edge of reads as a broken control. `try_enter` answers the
/// nearest seat inside this radius and refuses everything else as a value.
pub const ENTER_REACH_M: f64 = 3.0;

/// How far to the side of the seat an exiting character is placed, in chassis
/// half-widths plus this, metres.
pub const EXIT_CLEARANCE_M: f64 = 0.6;

/// Where a vehicle's seat is in the world, the chassis rotation it is in, and
/// the chassis's own velocity — the three numbers the seat step needs.
pub fn seat_pose(bridge: &PhysicsBridge3D, chassis: Uuid) -> Option<(DVec3, glam::DQuat, DVec3)> {
    let body = bridge.body_of(chassis)?;
    let v = bridge.vehicle_of(chassis)?;
    let w = bridge.world();
    let pos = w.body_translation(body)?;
    let rot = w.body_rotation(body)?;
    let seat = pos + rot * v.rig().seat_local.to_dvec3();
    Some((seat, rot, w.body_linvel(body).unwrap_or(DVec3::ZERO)))
}

/// **Try to climb into the nearest vehicle** (P29.7).
///
/// Returns the chassis `Guid` if one was in reach. `O(vehicles)`, walked in
/// `Guid` order with a strict `<` on the distance, so a tie between two seats
/// at the same distance is broken by the lower guid rather than by a bevy
/// archetype layout.
///
/// A refusal is a value: no vehicle, none in reach, or one already occupied all
/// answer `None`, and the character does whatever it was going to do instead.
/// **Migrated onto the one interaction door** (island wave I5): this is now
/// `super::interact::nearest_seat`, which builds seat candidates and ranks them
/// with `inf_ecs::interact::resolve` — the same rule an authored
/// `Interactable` is ranked by.
///
/// Every property above is preserved *exactly*: the reach is still
/// [`ENTER_REACH_M`], the walk is still `Guid`-ordered with a strict `<`,
/// occupied seats are still skipped, and there is still **no view test** (the
/// seat candidate carries `NO_VIEW_TEST_DEG`, so the rule's view half is skipped
/// for it). What changed is that there is one implementation of "nearest thing
/// in reach" instead of two, so the day the reach or the tie-break moves, it
/// moves for both.
pub fn try_enter(bridge: &PhysicsBridge3D, feet: DVec3, occupied: &BTreeSet<Uuid>) -> Option<Uuid> {
    super::interact::nearest_seat(bridge, feet, occupied)
}

/// Park (or restore) the character's own collider while it is in a seat.
///
/// The **same door the ragdoll uses**, for the same reason: a character riding
/// on a chassis with its capsule still in the world is a capsule permanently
/// intersecting a dynamic body, which is a depenetration force with nowhere to
/// go. `set_collider_enabled` is a physics-world operation, so both hosts do it
/// identically and PIE parity sees it — which is what the brief asks of this
/// seam.
pub fn park_collider(bridge: &mut PhysicsBridge3D, guid: Uuid, parked: bool) {
    if let Some(c) = bridge.collider_of(guid) {
        bridge.world_mut().set_collider_enabled(c, !parked);
    }
}
