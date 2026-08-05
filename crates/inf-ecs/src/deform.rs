//! **Ground contacts → surface deformation** (P22.1): the one fixed-step rule
//! that turns the bodies resting on a terrain into footprints in it.
//!
//! [`inf_terrain::deform`] owns the *field* — how deep a layer presses, how it
//! recovers, how the sparse set is bounded. This module owns the *reading*: who
//! is touching the ground, how hard, and on which splat layer. It lives in
//! `inf-ecs` because that is the only Ring-0 crate that can see both halves —
//! the bodies (its own components) and the heightfield (`inf-terrain`, already a
//! dependency for the [`Terrain`] component) — and because putting it here makes
//! the two hosts' fixed steps call **one function** rather than keep two
//! byte-identical copies of a loop.
//!
//! That is the strongest form the MIRROR rule takes, and it is the same shape
//! [`crate::sky::advance_weather`] uses: `RuntimeSim::fixed_step` and
//! `SimSession::fixed_step` each contain a single call, so the editor preview and
//! the shipped player cannot disagree about where a footprint goes.
//!
//! # Where the field lives, and why it is a resource
//!
//! [`DeformFieldRes`] is a bevy **resource** on the sim world, not a component
//! and not a member of either host's struct.
//!
//! * It is not a component, so **no schema moves** — scene v19 and
//!   `.inf_terrain` v5 are both frozen in this batch, and the scene serializer
//!   walks entities and components, never resources. Nothing here can be saved,
//!   which is correct: a footprint is transient state, not authored content.
//!   (The P21 law — *in the editor the render store IS the save's staging
//!   source* — is answered structurally: there is no path from this resource to
//!   a file.)
//! * It is not a host member because both scene **projectors** read the sim
//!   world and neither reads the host struct: the player projects from
//!   `sim.world()`, and the editor viewport projects from `doc.world()` — which
//!   during Simulate *is* the world `SimSession::fixed_step` steps. Putting the
//!   field in the world is therefore the difference between the editor drawing
//!   what it simulates and the editor needing a Ring-2 fold to find out. P21.4
//!   paid for the alternative once already ("PIE previews the last *saved*
//!   cave"); this batch declines to pay it again.
//! * It is still sim state in every sense that matters: written only from the
//!   fixed step, read by everything else, a pure function of the step history.
//!
//! The resource is **absent** until something actually presses ground. A level
//! where nothing touches a terrain never allocates it, so the whole subsystem is
//! byte-identical to its pre-P22.1 self on every scene that does not use it.
//!
//! # Determinism
//!
//! Contacts are collected into a `Guid`-sorted vec before anything is mutated,
//! so the stamp order is a property of the level rather than of bevy's archetype
//! layout. The layer under a contact is read from the **resident** `TerrainData`
//! the world holds — the sim's working set, never the streamer's camera-driven
//! one ([`inf_terrain::wants`]) — so a contact over a tile the sim has not paged
//! in leaves no mark at all, which is fine: sim wants already cover every active
//! actor.
//!
//! Nothing in this file runs inside [`crate::SimSchedule`]; it is called from the
//! host's fixed step directly, exactly like the sky advance. The parallel
//! executor therefore cannot reorder it and the pool-size-invariance probes are
//! unaffected by construction.

use bevy_ecs::prelude::{Entity, Resource};
use glam::{DVec2, DVec3};
use inf_terrain::deform::{DeformField, PressureClass};
use uuid::Uuid;

use crate::components::{
    BodyKind3D, CharacterController3D, Collider3D, ColliderShape3DKind, GlobalTransform, Guid,
    RigidBody3D, Terrain, Transform,
};
use crate::world::EcsWorld;

/// How far **above** the ground a collider's underside may sit and still count
/// as resting on it, metres.
///
/// A character mover keeps a skin width (`CharacterController3D::offset`,
/// default 2 cm) and a dynamic body settles a solver epsilon off the surface, so
/// "touching" is a band, not an equality. 6 cm covers both with room for the
/// heightfield's own bilinear error between samples, and is far below any step
/// height, so a body standing on a ledge above the terrain does not stamp the
/// ground under it.
pub const CONTACT_BAND_ABOVE_M: f64 = 0.06;

/// How far **below** the ground a collider's underside may sit and still count
/// as resting on it, metres.
///
/// Penetration is normal — a capsule sinks a little under load, and a body
/// spawned slightly low takes a step or two to be pushed out. Half a metre is
/// generous enough to survive both and tight enough that a *buried* prop (a
/// foundation, a rock half-sunk into a hillside by an author) is not treated as
/// a thing that is pressing down on the surface every step for ever.
pub const CONTACT_BAND_BELOW_M: f64 = 0.50;

/// The deformation field, as a resource on the sim world. See the module docs.
#[derive(Resource, Default, Debug, Clone)]
pub struct DeformFieldRes(pub DeformField);

/// One ground contact, resolved and ready to stamp.
///
/// A plain value with no ECS in it, so [`ground_contacts`] is testable and the
/// stamp loop has nothing left to decide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundContact {
    /// The pressing entity — the sort key, and the only reason this is here.
    pub guid: Uuid,
    /// World XZ of the contact.
    pub center_xz: DVec2,
    /// Footprint radius, metres.
    pub radius_m: f64,
    /// What kind of thing is pressing.
    pub pressure: PressureClass,
    /// The terrain's dominant splat layer under the contact.
    pub layer: u8,
}

/// The lowest-`Guid` non-empty [`Terrain`] and its world origin — the same
/// "which terrain answers" rule the hosts' `terrain.height_at` seam applies, so
/// a footprint and a ground query can never pick different ground.
fn ground_terrain(world: &EcsWorld) -> Option<(Entity, DVec3)> {
    let w = world.world();
    let mut picked: Option<(Uuid, DVec3, Entity)> = None;
    for e in w.iter_entities() {
        let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
            continue;
        };
        let Some(t) = e.get::<Terrain>() else {
            continue;
        };
        if t.data.is_empty() {
            continue;
        }
        let origin = e
            .get::<GlobalTransform>()
            .map(|g| g.translation())
            .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
            .unwrap_or(DVec3::ZERO);
        if picked.as_ref().map(|(g, _, _)| guid < *g).unwrap_or(true) {
            picked = Some((guid, origin, e.id()));
        }
    }
    picked.map(|(_, origin, e)| (e, origin))
}

/// The world-space underside of a collider on its entity, and the radius of the
/// disc that prints the same area as its footprint.
///
/// Box footprints are rectangles and the field stamps discs, so the box maps to
/// the **equal-area** disc (`r = 2·√(hx·hz/π)`) rather than to its larger
/// half-extent: a 4 m × 0.5 m plank should not press a 4 m-radius circle.
fn footprint(c: &Collider3D, translation: DVec3) -> (f64, f64) {
    let y = translation.y + c.offset.y;
    match c.shape_kind {
        ColliderShape3DKind::Box => {
            let (hx, hy, hz) = (
                c.half_extents.x.abs(),
                c.half_extents.y.abs(),
                c.half_extents.z.abs(),
            );
            (
                y - hy,
                2.0 * (hx * hz / std::f64::consts::PI).sqrt().max(0.0),
            )
        }
        ColliderShape3DKind::Sphere => (y - c.radius, c.radius),
        // The contact patch of a vertical capsule is its cap, so the radius is
        // the radius; the segment half-length only moves the underside down.
        ColliderShape3DKind::Capsule => (y - (c.half_extents.y.abs() + c.radius), c.radius),
    }
}

/// Which pressure class a body presses with.
///
/// * a **character controller** is a person on foot — the reference class;
/// * a **dynamic** body is loose, so its weight arrives through whatever small
///   patch it happens to be resting on: a wheel, a crate corner, a barrel's rim;
/// * a **kinematic** body is *driven* — a vehicle chassis, a moving platform —
///   and is the heavy case;
/// * a **static** body has always been there. Level geometry does not press
///   ground; it is part of the level, and a static prop stamping the same
///   footprint every step for ever would pin a cell against the eviction bound
///   for nothing.
fn pressure_of(body: Option<&RigidBody3D>, character: bool) -> Option<PressureClass> {
    if character {
        return Some(PressureClass::Foot);
    }
    match body.map(|b| b.kind) {
        Some(BodyKind3D::Dynamic) => Some(PressureClass::Wheel),
        Some(BodyKind3D::Kinematic) => Some(PressureClass::Heavy),
        _ => None,
    }
}

/// Every ground contact in the world this step, **`Guid`-sorted**.
///
/// Pure: no mutation, no clock, no camera. Empty when the level has no terrain,
/// when nothing is near it, or when the tiles under everything are not
/// sim-resident.
pub fn ground_contacts(world: &EcsWorld) -> Vec<GroundContact> {
    let Some((terrain_entity, origin)) = ground_terrain(world) else {
        return Vec::new();
    };
    let w = world.world();
    let Some(terrain) = w.get::<Terrain>(terrain_entity) else {
        return Vec::new();
    };
    let mut out: Vec<GroundContact> = Vec::new();
    for e in w.iter_entities() {
        let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
            continue;
        };
        let Some(collider) = e.get::<Collider3D>() else {
            continue;
        };
        // A trigger volume detects; it does not push.
        if collider.sensor {
            continue;
        }
        let Some(pressure) = pressure_of(
            e.get::<RigidBody3D>(),
            e.get::<CharacterController3D>().is_some(),
        ) else {
            continue;
        };
        let Some(t) = e
            .get::<GlobalTransform>()
            .map(|g| g.translation())
            .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
        else {
            continue;
        };
        let center_xz = DVec2::new(t.x + collider.offset.x, t.z + collider.offset.z);
        let local = center_xz - DVec2::new(origin.x, origin.z);
        // Resident tiles only — an unloaded tile answers `None` and the contact
        // silently leaves no mark (see the module docs).
        let Some(ground) = terrain.data.height_at(local).map(|h| h + origin.y) else {
            continue;
        };
        let Some(layer) = terrain.data.dominant_layer_at(local) else {
            continue;
        };
        let (bottom, radius) = footprint(collider, t);
        let gap = bottom - ground;
        if !(-CONTACT_BAND_BELOW_M..=CONTACT_BAND_ABOVE_M).contains(&gap) {
            continue;
        }
        out.push(GroundContact {
            guid,
            center_xz,
            radius_m: radius,
            pressure,
            layer,
        });
    }
    out.sort_by_key(|c| c.guid);
    out
}

/// Snowfall accumulating on upward-facing surfaces right now, m/s.
///
/// A one-line forward to [`crate::sky::ResolvedSky::snow_accumulation_rate`],
/// which is where the rate is **calibrated** (`SNOW_ACCUMULATION_MAX_M_PER_S`,
/// ≈ 5 cm/hour at full blast) and whose doc comment says P22.1 consumes it.
/// P22.1 consumes it here and nowhere else, so there is exactly one definition
/// of how snowy it is.
pub fn snow_accumulation_rate(world: &EcsWorld) -> f64 {
    crate::sky::resolve_sky(world)
        .map(|s| s.snow_accumulation_rate() as f64)
        .unwrap_or(0.0)
}

/// **The fixed-step deformation slot.** Relax the field, then press this step's
/// ground contacts into it.
///
/// Call once per fixed step, immediately after the physics write-back and
/// `propagate` — bodies must be where the solver left them before anything asks
/// what they are standing on, and the transforms must have settled before the
/// footprint's XZ is read.
///
/// Relax before stamp, deliberately: a body that keeps resting then converges on
/// its class's target depth instead of oscillating a recovery step around it.
///
/// The field is created on first contact and never before, so a level that
/// deforms nothing runs one `Vec::new()` per step and allocates no resource at
/// all.
pub fn step_deformation(world: &mut EcsWorld, dt: f64) {
    let contacts = ground_contacts(world);
    let rate = snow_accumulation_rate(world);
    let w = world.world_mut();
    if !w.contains_resource::<DeformFieldRes>() {
        if contacts.is_empty() {
            return;
        }
        w.insert_resource(DeformFieldRes::default());
    }
    let mut field = w.resource_mut::<DeformFieldRes>();
    field.0.relax(dt, rate);
    for c in &contacts {
        field
            .0
            .stamp_contact(c.center_xz, c.radius_m, c.pressure, c.layer, dt);
    }
}

/// Read the deformation field, if this world has ever grown one.
pub fn deform_field(world: &EcsWorld) -> Option<&DeformField> {
    world.world().get_resource::<DeformFieldRes>().map(|r| &r.0)
}

/// The field's canonical bytes, or an empty vec when there is no field — the
/// shape a replay / PIE trace folds. See
/// [`inf_terrain::deform::DeformField::state_bytes`].
pub fn deform_state_bytes(world: &EcsWorld) -> Vec<u8> {
    deform_field(world)
        .map(|f| f.state_bytes())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_terrain::{TerrainData, DEFAULT_TILE_RESOLUTION};

    const DT: f64 = 1.0 / 60.0;
    const RES: u32 = DEFAULT_TILE_RESOLUTION;

    /// A flat terrain at y = 0 painted entirely with `layer`.
    fn flat_terrain(layer: u8) -> Terrain {
        let mut data = TerrainData::new(RES, 1.0);
        data.author_tile((0, 0), |_, _| 0.0);
        let tile = data.get_or_create_tile((0, 0));
        tile.ensure_weights(RES);
        let mut w = [0u8; 4];
        w[layer as usize] = 255;
        for j in 0..RES {
            for i in 0..RES {
                tile.set_weight_sample(RES, i, j, w);
            }
        }
        Terrain {
            data,
            ..Default::default()
        }
    }

    /// A world with a flat `layer` terrain and one capsule character standing at
    /// `(x, z)` with its underside exactly on the ground. Returns the walker.
    fn world_with_walker(layer: u8, x: f64, z: f64) -> (EcsWorld, Entity) {
        let mut world = EcsWorld::new();
        let t = world.spawn("terrain", None);
        world
            .world_mut()
            .entity_mut(t)
            .insert((flat_terrain(layer), Transform::default()));
        let a = world.spawn("walker", None);
        world.world_mut().entity_mut(a).insert((
            Transform {
                translation: crate::math::Vec3d::new(x, 0.75, z),
                ..Default::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Capsule,
                half_extents: crate::math::Vec3d::new(0.25, 0.5, 0.25),
                radius: 0.25,
                ..Default::default()
            },
            CharacterController3D::default(),
        ));
        world.propagate();
        (world, a)
    }

    #[test]
    fn a_walker_on_snow_leaves_a_contact_and_a_print() {
        let (mut world, _) = world_with_walker(3, 4.0, 4.0);
        let contacts = ground_contacts(&world);
        assert_eq!(contacts.len(), 1);
        assert_eq!(contacts[0].layer, 3);
        assert_eq!(contacts[0].pressure, PressureClass::Foot);
        assert!((contacts[0].radius_m - 0.25).abs() < 1e-9);

        assert!(deform_field(&world).is_none(), "nothing pressed yet");
        for _ in 0..60 {
            step_deformation(&mut world, DT);
        }
        let f = deform_field(&world).expect("a contact must grow the field");
        assert!(f.depth_at(DVec2::new(4.0, 4.0)) > 0.0);
    }

    #[test]
    fn a_level_that_deforms_nothing_allocates_nothing() {
        // The off-path property, at the ECS seam: no terrain at all.
        let mut world = EcsWorld::new();
        let a = world.spawn("floater", None);
        world.world_mut().entity_mut(a).insert((
            Transform::default(),
            Collider3D::default(),
            CharacterController3D::default(),
        ));
        world.propagate();
        for _ in 0..60 {
            step_deformation(&mut world, DT);
        }
        assert!(ground_contacts(&world).is_empty());
        assert!(deform_field(&world).is_none());
        assert!(deform_state_bytes(&world).is_empty());

        // …and a body far above the ground it could otherwise press.
        let (mut world, e) = world_with_walker(3, 4.0, 4.0);
        world
            .world_mut()
            .get_mut::<Transform>(e)
            .unwrap()
            .translation
            .y = 40.0;
        world.mark_dirty();
        world.propagate();
        for _ in 0..60 {
            step_deformation(&mut world, DT);
        }
        assert!(deform_field(&world).is_none());
    }

    #[test]
    fn a_static_body_never_presses_and_a_sensor_never_presses() {
        for make_static in [true, false] {
            let (mut world, e) = world_with_walker(3, 4.0, 4.0);
            let mut em = world.world_mut().entity_mut(e);
            em.remove::<CharacterController3D>();
            if make_static {
                em.insert(RigidBody3D::default()); // Static
            } else {
                em.insert(RigidBody3D {
                    kind: BodyKind3D::Dynamic,
                    ..Default::default()
                });
                world.world_mut().get_mut::<Collider3D>(e).unwrap().sensor = true;
            }
            world.mark_dirty();
            world.propagate();
            assert!(ground_contacts(&world).is_empty());
        }
    }

    #[test]
    fn a_dynamic_body_presses_harder_than_a_character() {
        let (mut world, e) = world_with_walker(3, 4.0, 4.0);
        let mut em = world.world_mut().entity_mut(e);
        em.remove::<CharacterController3D>();
        em.insert(RigidBody3D {
            kind: BodyKind3D::Dynamic,
            ..Default::default()
        });
        world.mark_dirty();
        world.propagate();
        let c = ground_contacts(&world);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].pressure, PressureClass::Wheel);
        assert!(c[0].pressure.depth_fraction() > PressureClass::Foot.depth_fraction());
    }

    #[test]
    fn the_step_is_a_pure_function_of_the_step_history() {
        let run = || {
            let (mut world, e) = world_with_walker(3, 4.0, 4.0);
            for s in 0..120 {
                {
                    let mut t = world.world_mut().get_mut::<Transform>(e).unwrap();
                    t.translation.x = 4.0 + s as f64 * 0.05;
                }
                world.mark_dirty();
                world.propagate();
                step_deformation(&mut world, DT);
            }
            (
                deform_state_bytes(&world),
                deform_field(&world).map(|f| f.cell_count()).unwrap_or(0),
            )
        };
        let (a, cells) = run();
        let (b, _) = run();
        assert_eq!(a, b);
        // Anti-vacuity: a 6 m walk really did lay a trail across more than one
        // 4 m cell, so the equality above is over a field with structure in it.
        assert!(cells > 1, "the walk left {cells} cells");
    }

    #[test]
    fn rock_ground_takes_no_print_but_still_reports_the_contact() {
        // The contact is real — the entity IS standing there — and the *layer*
        // is what refuses. Asserting both halves separately is what stops a
        // future "optimization" from dropping rock contacts and hiding a bug in
        // the contact scan behind a table entry.
        let (mut world, _) = world_with_walker(1, 4.0, 4.0);
        let c = ground_contacts(&world);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].layer, 1);
        for _ in 0..120 {
            step_deformation(&mut world, DT);
        }
        assert!(deform_field(&world).is_none_or(|f| f.is_empty()));
    }

    #[test]
    fn a_box_prints_its_equal_area_disc_not_its_longest_side() {
        let c = Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: crate::math::Vec3d::new(2.0, 0.1, 0.25),
            ..Default::default()
        };
        let (bottom, radius) = footprint(&c, DVec3::new(0.0, 1.0, 0.0));
        assert!((bottom - 0.9).abs() < 1e-9);
        let expect = 2.0 * (2.0f64 * 0.25 / std::f64::consts::PI).sqrt();
        assert!((radius - expect).abs() < 1e-9);
        assert!(radius < 2.0, "a plank must not press a 2 m circle");
    }

    #[test]
    fn the_snow_rate_reaches_the_field() {
        // The hook, end to end: no sky ⇒ zero, so the refill is inert on every
        // level that has not opted into weather.
        let (world, _) = world_with_walker(3, 4.0, 4.0);
        assert_eq!(snow_accumulation_rate(&world), 0.0);
    }
}
