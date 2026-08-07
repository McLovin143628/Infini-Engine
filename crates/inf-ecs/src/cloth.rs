//! **Garments simulate** (P24.4): the one fixed-step rule that turns a
//! [`ClothSim`] component into moving cloth.
//!
//! [`inf_anim::cloth`] owns the *solver* — the XPBD substeps, the constraint
//! sweep, the capsule projection. This module owns the *binding*: which entities
//! wear a garment, which `.inf_cloth` describes it, where its collision capsules
//! are this step, and which way is down in the wearer's own frame.
//!
//! It is verbatim the shape [`crate::deform`] records and [`crate::pose`]
//! repeats, and the reasons transfer without change:
//!
//! * **A bevy resource on the sim world**, not a component and not a host member.
//!   Not a component, so **no schema moves** — scene v21 is frozen for this
//!   batch, and the serializer walks entities and components, never resources, so
//!   there is no path from a simulated garment to a file. That is correct: a
//!   settled fold is transient state, not authored content, and the P21 law (*in
//!   the editor the render store IS the save's staging source*) is answered
//!   structurally rather than by discipline. Not a host member, because both
//!   scene projectors read the sim world and neither reads the host struct.
//! * **ONE Ring-0 function both fixed steps call**
//!   ([`step_cloth_simulation`]), so the editor's Simulate and the shipped player
//!   cannot disagree about how a coat falls. Written twice it would be two
//!   byte-identical loops, which is the shape the deform doctrine replaced.
//! * **Absent until something wears a garment.** A level with no `ClothSim`
//!   never allocates the resource and [`cloth_state_bytes`] returns an empty vec,
//!   so every trace recorded before this batch is byte-identical.
//!
//! # What the projectors read, and why it is only this
//!
//! [`LiveCloth`] carries the simulated positions **and the garment's triangle
//! list** (an `Arc`, cloned from the asset once at seed). A projector therefore
//! draws a garment out of the sim world alone — no asset store, no mesh lookup,
//! no second resolution path to keep in step with this one. That is what makes
//! the render half of P24.4 two small mirrored branches instead of a new pair of
//! stores.
//!
//! # Quality is NOT tier-driven, and that is a correction
//!
//! [`ClothSim::quality`]'s doc was written at P24.3 saying `0` "follows the
//! machine's capability tier". Reading it that way would put the **machine's**
//! tier into the substep count, into the particle positions, into
//! [`cloth_state_bytes`] and therefore into the trace two hosts are compared on —
//! which is exactly the P22.4 finding (*a preview must run what it previews*: the
//! debris budget was tier-clamped in embedded PIE and not in Simulate, and the
//! two ran different simulations on any Medium machine).
//!
//! So the ruling here, and the component's doc now says it too: `0` means **the
//! garment's own authored substep budget** ([`inf_anim::ClothMaterial::substeps`])
//! and a non-zero value overrides it per wearer. Both are properties of the
//! *content*, so two machines simulate the same coat.

use std::collections::BTreeMap;

use bevy_ecs::prelude::{Entity, Resource};
use glam::{DVec3, Vec3};
use inf_anim::cloth::{Capsule, ClothAsset, ClothState};
use inf_anim::SkeletonAsset;
use uuid::Uuid;

use crate::components::{ClothSim, GlobalTransform, Guid};
use crate::world::EcsWorld;

/// One wearer's live garment.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveCloth {
    /// The `.inf_cloth` GUID this garment was seeded from.
    ///
    /// Load-bearing, not bookkeeping — the same role [`crate::pose::EvaluatedPose::skeleton`]
    /// plays: re-binding a wearer to a different garment mid-session must re-seed
    /// rather than keep simulating the old particle set under the new topology.
    pub asset: Uuid,
    /// The solver's live state (positions, velocities, inverse masses, indices).
    pub state: ClothState,
}

/// **Every wearer's simulated garment**, keyed by [`Guid`] — a bevy resource,
/// exactly like [`crate::pose::PoseStoreRes`] and for the same reasons.
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct ClothStateRes(pub BTreeMap<Uuid, LiveCloth>);

/// The garment `guid` is wearing, if the sim is simulating one.
///
/// The **read door** both render projectors go through.
pub fn live_cloth(world: &EcsWorld, guid: Uuid) -> Option<&LiveCloth> {
    world
        .world()
        .get_resource::<ClothStateRes>()
        .and_then(|r| r.0.get(&guid))
}

/// How many garments the sim is simulating (`0` on a world that has never worn
/// one).
pub fn cloth_count(world: &EcsWorld) -> usize {
    world
        .world()
        .get_resource::<ClothStateRes>()
        .map(|r| r.0.len())
        .unwrap_or(0)
}

/// **Forget every garment.** Removes the resource outright, so the world is
/// byte-for-byte one that has never simulated cloth.
///
/// The editor calls this at BOTH ends of a Simulate session for the reason
/// [`crate::deform::clear_deformation`] documents: `SceneDoc`'s snapshot carries
/// entities and components and `EcsWorld::clear` despawns entities — neither
/// touches a resource, so without an explicit call a stopped session's settled
/// coat would still be draped over the author's character, and the next Simulate
/// would start from the previous run's fold instead of from the garment's rest
/// shape. That second half is a PIE-vs-shipping divergence by construction.
///
/// Idempotent, and a no-op on a world that never wore a garment.
pub fn clear_cloth(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<ClothStateRes>();
}

/// The simulated garments' canonical bytes, or an empty vec when nothing is
/// simulating — the shape a replay / PIE trace folds, exactly like
/// [`crate::pose::pose_state_bytes`].
///
/// Appended to each host's `state_bytes`, which is **hashed and never decoded**,
/// so this needs no version and no reader. A level with no garment produces an
/// empty vec and every pre-P24.4 trace is byte-identical.
///
/// # What is folded, and what is deliberately left out
///
/// Positions and the asset GUID. **Velocities are not folded**: they are an exact
/// function of the last substep's position delta (`v = (x − x_prev)/h`), so
/// hashing them would fold the same information twice — the same argument that
/// keeps sockets out of `pose_state_bytes`. The **asset GUID is** folded, because
/// two garments that happen to have coincident particles are still different
/// garments, and a re-bind that produced identical numbers would otherwise be
/// invisible to every gate.
///
/// `BTreeMap` order, so the bytes are a property of the level rather than of
/// bevy's archetype layout.
pub fn cloth_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(store) = world.world().get_resource::<ClothStateRes>() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (guid, live) in &store.0 {
        out.extend_from_slice(guid.as_bytes());
        out.extend_from_slice(live.asset.as_bytes());
        out.extend_from_slice(&(live.state.x.len() as u32).to_le_bytes());
        for p in &live.state.x {
            for v in p {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    out
}

/// **Which way is down, in the wearer's own frame** (m/s²).
///
/// The solver runs in model space (see [`inf_anim::cloth`]), so world-down has to
/// be rotated into it or a character lying on their back would have their coat
/// fall towards their feet. The conversion is the entity's own
/// [`GlobalTransform`], inverted — the same frame conversion
/// [`crate::pose::authored_ik_goals`] does for an IK target, and it is here for
/// the same reason: one place, so the two hosts cannot spell it differently.
///
/// A singular placement (a zero scale) has no model frame to convert into and
/// answers `None`; the caller skips the garment rather than simulating it under a
/// gravity full of infinities.
fn model_gravity(global: &glam::DAffine3) -> Option<Vec3> {
    let det = global.matrix3.determinant();
    if !det.is_finite() || det.abs() < 1.0e-12 {
        return None;
    }
    let down = global.inverse().transform_vector3(DVec3::NEG_Y);
    let len = down.length();
    if !len.is_finite() || len < 1.0e-9 {
        return None;
    }
    let d = down / len;
    let g = Vec3::new(d.x as f32, d.y as f32, d.z as f32) * inf_anim::cloth::GRAVITY_M_S2;
    g.is_finite().then_some(g)
}

/// The substep budget one wearer runs at — the asset's, unless the component pins
/// its own. See the module docs for why the machine's tier is not in this.
fn substeps_for(asset: &ClothAsset, sim: &ClothSim) -> u8 {
    if sim.quality == 0 {
        asset.material.substeps
    } else {
        sim.quality
    }
}

/// **The fixed-step cloth slot**: seed every newly-worn garment, place its
/// capsules on the pose this step evaluated, and advance the solver.
///
/// ONE function, called from both hosts' fixed steps — the strongest form the
/// MIRROR rule takes, and the same shape [`crate::deform::step_deformation`] and
/// [`crate::pose::step_pose_evaluation`] use.
///
/// Call it **after** the pose slot and after the attachment propagate: the
/// capsules are read off the pose the sim just published, and the model frame is
/// read off a `GlobalTransform` that has settled. Reading either earlier would
/// collide a garment against last step's body.
///
/// The two resolvers are the host's own registries, threaded in rather than
/// duplicated — `cloths` yields the `.inf_cloth` a component's `asset` GUID names,
/// `skeletons` the `.inf_skel` the wearer's [`crate::components::SkeletalMesh`]
/// names (needed because capsules are joint pairs and only the skeleton knows
/// where a joint is).
///
/// Rules, in order:
///  1. an entity with no [`ClothSim`], a disabled one, or one with no bound asset
///     wears nothing and is not in the store;
///  2. a garment whose `.inf_cloth` does not resolve, or does not
///     [`validate`](ClothAsset::validate), is **skipped** — the component is left
///     alone, exactly as an unresolvable skeleton leaves an `AnimStateMachine`
///     alone;
///  3. state is **carried** across steps for a wearer whose asset GUID has not
///     changed, and **re-seeded** the moment it does;
///  4. the store is rebuilt from the surviving entries each step, so a wearer that
///     stopped wearing a garment has no entry and a step that simulates nothing
///     removes the resource.
///
/// Deterministic: wearers are collected and sorted by `Guid` before anything is
/// written, and each garment's solve is independent, so the result is a property
/// of the level rather than of archetype iteration order.
pub fn step_cloth_simulation<'c>(
    world: &mut EcsWorld,
    dt: f64,
    cloths: &dyn Fn(Uuid) -> Option<&'c ClothAsset>,
    skeletons: &dyn Fn(Uuid) -> Option<&'c SkeletonAsset>,
) {
    // 1. Read pass — collect wearers so the write-back never overlaps the query.
    let mut wearers: Vec<(Entity, Uuid, ClothSim)> = Vec::new();
    {
        let w = world.world_mut();
        let mut q = w.query::<(Entity, &Guid, &ClothSim)>();
        for (e, g, cs) in q.iter(w) {
            if cs.enabled && cs.asset.is_some() {
                wearers.push((e, g.0, *cs));
            }
        }
    }
    if wearers.is_empty() {
        // Rule 4. Never *touch* a world that never grew a store, so a level with
        // no garment is byte-identical to its pre-P24.4 self.
        let w = world.world_mut();
        if w.contains_resource::<ClothStateRes>() {
            w.remove_resource::<ClothStateRes>();
        }
        return;
    }
    wearers.sort_by_key(|(_, guid, _)| *guid);

    // The previous step's store, lifted out whole: the solve below needs
    // `&mut EcsWorld` for nothing, but the publish does, and a borrow of the
    // resource would outlive it.
    let mut store = world
        .world_mut()
        .remove_resource::<ClothStateRes>()
        .map(|r| r.0)
        .unwrap_or_default();
    let mut next: BTreeMap<Uuid, LiveCloth> = BTreeMap::new();

    for (entity, guid, sim) in wearers {
        let Some(asset_id) = sim.asset else { continue };
        let Some(asset) = cloths(asset_id) else {
            continue;
        };

        // Rule 3: carry the state for the same garment, re-seed for a new one.
        let mut live = match store.remove(&guid) {
            Some(prev) if prev.asset == asset_id && prev.state.len() == asset.particle_count() => {
                prev
            }
            _ => match ClothState::seed(asset) {
                Ok(state) => LiveCloth {
                    asset: asset_id,
                    state,
                },
                // Rule 2: an invalid garment is skipped, by value.
                Err(_) => continue,
            },
        };

        let w = world.world();
        let Some(global) = w.get::<GlobalTransform>(entity).map(|g| g.0) else {
            // No placement, no model frame: the wearer has not been propagated
            // (or has no transform at all), so there is no honest "down".
            continue;
        };
        let Some(gravity) = model_gravity(&global) else {
            continue;
        };

        // The capsules ride the pose the sim published THIS step — read, never
        // re-evaluated (the P24.1 ruling), so the collision geometry and the
        // skinning cannot disagree about where an arm is.
        let capsules: Vec<Capsule> = crate::pose::evaluated_pose(world, guid)
            .and_then(|posed| {
                skeletons(posed.skeleton).map(|sk| {
                    let globals = inf_anim::global_transforms(&sk.skeleton, &posed.pose);
                    inf_anim::cloth::capsules_for(asset, &globals)
                })
            })
            .unwrap_or_default();

        // The per-wearer budget, applied by handing the solver a garment whose
        // substep count is this wearer's. Cloned rather than mutated in place
        // because the asset is a shared borrow of the host's registry.
        let stepped = substeps_for(asset, &sim);
        if stepped == asset.material.substeps {
            inf_anim::cloth::step_cloth(asset, &mut live.state, dt as f32, gravity, &capsules);
        } else {
            let mut tuned = asset.clone();
            tuned.material.substeps = stepped;
            inf_anim::cloth::step_cloth(&tuned, &mut live.state, dt as f32, gravity, &capsules);
        }
        next.insert(guid, live);
    }

    // 2. Publish (rule 4).
    let w = world.world_mut();
    if next.is_empty() {
        if w.contains_resource::<ClothStateRes>() {
            w.remove_resource::<ClothStateRes>();
        }
    } else {
        w.insert_resource(ClothStateRes(next));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{SkeletalMesh, Transform};
    use crate::math::Vec3d;
    use inf_anim::cloth::{ClothCapsule, ClothMaterial};
    use inf_anim::{Joint, JointTransform, Skeleton};

    const WEARER: Uuid = Uuid::from_u128(0x24_4001);
    const GARMENT: Uuid = Uuid::from_u128(0x24_4002);
    const SKEL: Uuid = Uuid::from_u128(0x24_4003);

    /// A 4×4 sheet pinned along its `j == 0` row — the smallest thing with
    /// hinges.
    ///
    /// **Horizontal at rest, deliberately.** The first cut hung it vertically in
    /// XY, which is already the equilibrium a `-Y` gravity produces: the hem
    /// moved by microns and "the garment fell" was unmeasurable. A flag pinned
    /// along one edge and released has somewhere to go.
    fn garment() -> ClothAsset {
        let mut pos = Vec::new();
        for j in 0..4u32 {
            for i in 0..4u32 {
                pos.push([i as f32 * 0.1, 1.0, j as f32 * 0.1]);
            }
        }
        let mut idx = Vec::new();
        for j in 0..3u32 {
            for i in 0..3u32 {
                let a = j * 4 + i;
                idx.extend_from_slice(&[a, a + 1, a + 4, a + 1, a + 5, a + 4]);
            }
        }
        ClothAsset::from_garment(
            *GARMENT.as_bytes(),
            &pos,
            &idx,
            &[0, 1, 2, 3],
            ClothMaterial::default(),
        )
        .expect("the fixture garment prepares")
        .with_capsules(vec![ClothCapsule {
            joint_a: 0,
            joint_b: 1,
            radius_m: 0.05,
        }])
    }

    fn skeleton() -> SkeletonAsset {
        SkeletonAsset::new(
            Skeleton::new(vec![
                Joint {
                    name: "root".into(),
                    parent: None,
                    inverse_bind: glam::Mat4::IDENTITY.to_cols_array(),
                    local_bind: JointTransform::IDENTITY,
                },
                Joint {
                    name: "tip".into(),
                    parent: Some(0),
                    inverse_bind: glam::Mat4::from_translation(glam::Vec3::new(0.0, -1.0, 0.0))
                        .to_cols_array(),
                    local_bind: JointTransform::from_trs(
                        glam::Vec3::Y,
                        glam::Quat::IDENTITY,
                        glam::Vec3::ONE,
                    ),
                },
            ])
            .unwrap(),
        )
    }

    struct Fixture {
        cloth: ClothAsset,
        skeleton: SkeletonAsset,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                cloth: garment(),
                skeleton: skeleton(),
            }
        }
        fn step(&self, world: &mut EcsWorld, dt: f64) {
            let cloths = |g: Uuid| (g == GARMENT).then_some(&self.cloth);
            let skels = |g: Uuid| (g == SKEL).then_some(&self.skeleton);
            step_cloth_simulation(world, dt, &cloths, &skels);
        }
    }

    /// A world with one garment-wearing character at `at`.
    fn world_with_wearer(at: Vec3d, enabled: bool) -> EcsWorld {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(WEARER, "Hero", None);
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: at,
                ..Default::default()
            },
            SkeletalMesh {
                mesh: Some(Uuid::from_u128(9)),
                skeleton: Some(SKEL),
            },
            ClothSim {
                asset: Some(GARMENT),
                enabled,
                quality: 0,
            },
        ));
        world.mark_dirty();
        world.propagate();
        world
    }

    /// **The headline gate**: a `ClothSim` component is READ — the garment moves,
    /// and it moves in the trace.
    #[test]
    fn a_cloth_sim_component_simulates_and_rides_the_trace() {
        let f = Fixture::new();
        let mut world = world_with_wearer(Vec3d::ZERO, true);
        assert_eq!(cloth_count(&world), 0, "nothing has stepped yet");
        assert!(cloth_state_bytes(&world).is_empty());

        f.step(&mut world, 1.0 / 60.0);
        assert_eq!(cloth_count(&world), 1, "the garment was seeded");
        let first = cloth_state_bytes(&world);
        assert!(!first.is_empty());
        for _ in 0..30 {
            f.step(&mut world, 1.0 / 60.0);
        }
        let later = cloth_state_bytes(&world);
        assert_ne!(first, later, "the garment never moved");

        // The pinned row really is pinned: the top four particles are where the
        // asset put them, so "it moved" is the free part moving.
        let live = live_cloth(&world, WEARER).unwrap();
        for i in 0..4 {
            assert_eq!(
                live.state.x[i], f.cloth.rest[i],
                "pinned particle {i} moved"
            );
        }
        assert!(
            live.state.x[15][1] < f.cloth.rest[15][1] - 0.001,
            "the hem did not fall"
        );
    }

    /// A level with no garment is byte-for-byte its pre-P24.4 self.
    #[test]
    fn a_world_with_no_garment_never_grows_a_store() {
        let mut world = EcsWorld::new();
        world.world_mut().spawn((Guid(Uuid::from_u128(4)),));
        world.reindex_guids();
        Fixture::new().step(&mut world, 1.0 / 60.0);
        assert_eq!(cloth_count(&world), 0);
        assert!(cloth_state_bytes(&world).is_empty());
        assert!(live_cloth(&world, WEARER).is_none());
    }

    /// Rule 1 and rule 2: a disabled garment, an unbound one and one whose asset
    /// does not resolve all simulate nothing — without touching the component.
    #[test]
    fn a_disabled_or_unresolvable_garment_simulates_nothing() {
        let f = Fixture::new();

        let mut off = world_with_wearer(Vec3d::ZERO, false);
        f.step(&mut off, 1.0 / 60.0);
        assert_eq!(cloth_count(&off), 0, "a disabled garment simulated");

        let mut ghost = world_with_wearer(Vec3d::ZERO, true);
        let e = ghost.entity_of(WEARER).unwrap();
        ghost.world_mut().get_mut::<ClothSim>(e).unwrap().asset = Some(Uuid::from_u128(0xDEAD));
        f.step(&mut ghost, 1.0 / 60.0);
        assert_eq!(cloth_count(&ghost), 0, "an unresolvable garment simulated");
        // The component survived: skipping is not unbinding.
        assert_eq!(
            ghost.world().get::<ClothSim>(e).unwrap().asset,
            Some(Uuid::from_u128(0xDEAD))
        );
    }

    /// Rule 4: a wearer that stops wearing leaves no stale fold behind, and the
    /// resource itself goes away.
    #[test]
    fn a_removed_garment_leaves_no_stale_fold() {
        let f = Fixture::new();
        let mut world = world_with_wearer(Vec3d::ZERO, true);
        f.step(&mut world, 1.0 / 60.0);
        assert_eq!(cloth_count(&world), 1);
        let e = world.entity_of(WEARER).unwrap();
        world.world_mut().get_mut::<ClothSim>(e).unwrap().enabled = false;
        f.step(&mut world, 1.0 / 60.0);
        assert_eq!(
            cloth_count(&world),
            0,
            "the store must be rebuilt, not merged"
        );
        assert!(cloth_state_bytes(&world).is_empty());
    }

    /// Rule 3: the fold is CARRIED across steps for the same garment, and
    /// **re-seeded** the moment the wearer is bound to a different one.
    #[test]
    fn the_fold_is_carried_and_a_rebind_reseeds_it() {
        let f = Fixture::new();
        let mut world = world_with_wearer(Vec3d::ZERO, true);
        for _ in 0..600 {
            f.step(&mut world, 1.0 / 60.0);
        }
        let settled = live_cloth(&world, WEARER).unwrap().state.x.clone();
        assert_ne!(
            settled, f.cloth.rest,
            "the garment never left its rest shape"
        );
        // One more step continues from the settled fold rather than from rest: a
        // state that had been re-seeded would jump the whole way back up.
        f.step(&mut world, 1.0 / 60.0);
        let next = &live_cloth(&world, WEARER).unwrap().state.x;
        let drift: f32 = settled
            .iter()
            .zip(next)
            .map(|(a, b)| (Vec3::from_array(*a) - Vec3::from_array(*b)).length())
            .fold(0.0, f32::max);
        // The distance a re-seed would have travelled, measured rather than
        // guessed — so the bound below is "much less than a re-seed" and not a
        // number somebody liked the look of.
        let reseed: f32 = settled
            .iter()
            .zip(&f.cloth.rest)
            .map(|(a, b)| (Vec3::from_array(*a) - Vec3::from_array(*b)).length())
            .fold(0.0, f32::max);
        assert!(
            drift < reseed * 0.1,
            "one more step moved the settled garment by {drift} m and a re-seed \
             would move it {reseed} m — the state is not being carried"
        );

        // The rebind half: point the component at a DIFFERENT garment and the
        // fold starts again from that garment's rest shape.
        let other = Uuid::from_u128(0x24_4009);
        let mut other_asset = garment();
        other_asset.garment = *other.as_bytes();
        let e = world.entity_of(WEARER).unwrap();
        world.world_mut().get_mut::<ClothSim>(e).unwrap().asset = Some(other);
        let cloths = |g: Uuid| {
            if g == other {
                Some(&other_asset)
            } else if g == GARMENT {
                Some(&f.cloth)
            } else {
                None
            }
        };
        let skels = |g: Uuid| (g == SKEL).then_some(&f.skeleton);
        step_cloth_simulation(&mut world, 1.0 / 60.0, &cloths, &skels);
        let fresh = live_cloth(&world, WEARER).unwrap();
        assert_eq!(fresh.asset, other, "the store kept the old garment's id");
        let jump: f32 = settled
            .iter()
            .zip(&fresh.state.x)
            .map(|(a, b)| (Vec3::from_array(*a) - Vec3::from_array(*b)).length())
            .fold(0.0, f32::max);
        assert!(
            jump > reseed * 0.5,
            "a rebind moved the garment by only {jump} m — the old fold was kept \
             under the new garment's topology"
        );
    }

    /// **Determinism**: two worlds stepped identically fold identical bytes, and
    /// the bytes MOVE (a constant compares equal too).
    #[test]
    fn simulation_is_deterministic() {
        let f = Fixture::new();
        let run = || {
            let mut w = world_with_wearer(Vec3d::ZERO, true);
            for _ in 0..45 {
                f.step(&mut w, 1.0 / 60.0);
            }
            cloth_state_bytes(&w)
        };
        let a = run();
        assert_eq!(a, run());
        assert!(!a.is_empty());
        let mut w = world_with_wearer(Vec3d::ZERO, true);
        for _ in 0..44 {
            f.step(&mut w, 1.0 / 60.0);
        }
        assert_ne!(cloth_state_bytes(&w), a, "the trace is a constant");
    }

    /// **Gravity is converted into the wearer's frame**, so a character rotated
    /// on their side has their garment fall the right way — the property that
    /// makes model-space simulation honest rather than a shortcut.
    #[test]
    fn gravity_is_read_in_the_wearers_own_frame() {
        let upright = glam::DAffine3::IDENTITY;
        let g = model_gravity(&upright).expect("an upright wearer has gravity");
        assert!(
            (g - Vec3::NEG_Y * inf_anim::cloth::GRAVITY_M_S2).length() < 1e-4,
            "{g:?}"
        );

        // Rolled +90° about +Z, which takes the wearer's +X onto world +Y. World
        // down is therefore the wearer's own **−X**, and their coat falls along
        // it — the whole point of converting rather than assuming −Y.
        let rolled = glam::DAffine3::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let g = model_gravity(&rolled).expect("a rolled wearer has gravity");
        assert!(
            (g - Vec3::NEG_X * inf_anim::cloth::GRAVITY_M_S2).length() < 1e-3,
            "a rolled character's garment must fall along its own -X, got {g:?}"
        );

        // A singular placement has no frame and is refused rather than inverted.
        assert!(model_gravity(&glam::DAffine3::from_scale(DVec3::ZERO)).is_none());
    }

    /// The per-wearer quality lever overrides the asset's substep budget, and
    /// `0` means "the garment's own" — **never the machine's tier** (module docs).
    #[test]
    fn quality_pins_the_substep_budget_per_wearer() {
        let a = garment();
        assert_eq!(a.material.substeps, ClothMaterial::default().substeps);
        let follow = ClothSim {
            asset: Some(GARMENT),
            enabled: true,
            quality: 0,
        };
        assert_eq!(substeps_for(&a, &follow), a.material.substeps);
        let pinned = ClothSim {
            quality: 3,
            ..follow
        };
        assert_eq!(substeps_for(&a, &pinned), 3);

        // …and it really changes the simulation, so the lever is not decoration.
        let f = Fixture::new();
        let mut lo = world_with_wearer(Vec3d::ZERO, true);
        let e = lo.entity_of(WEARER).unwrap();
        lo.world_mut().get_mut::<ClothSim>(e).unwrap().quality = 1;
        let mut hi = world_with_wearer(Vec3d::ZERO, true);
        for _ in 0..20 {
            f.step(&mut lo, 1.0 / 60.0);
            f.step(&mut hi, 1.0 / 60.0);
        }
        assert_ne!(
            cloth_state_bytes(&lo),
            cloth_state_bytes(&hi),
            "one substep and eight produced the same fold — the budget is not read"
        );
    }

    /// `clear_cloth` returns the world to one that has never simulated a garment
    /// — the Simulate start/stop door.
    #[test]
    fn clear_cloth_forgets_everything() {
        let f = Fixture::new();
        let mut world = world_with_wearer(Vec3d::ZERO, true);
        f.step(&mut world, 1.0 / 60.0);
        assert_eq!(cloth_count(&world), 1);
        clear_cloth(&mut world);
        assert_eq!(cloth_count(&world), 0);
        assert!(cloth_state_bytes(&world).is_empty());
        clear_cloth(&mut world); // idempotent
    }
}
