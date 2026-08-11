//! **Hair simulates** (P24.4): the one fixed-step rule that turns a
//! [`HairGuides`] component into moving strands.
//!
//! [`inf_anim::hair`] owns the *solver*; this module owns the *binding* — which
//! entities wear a hairstyle, which `.inf_hair` describes it, where each strand's
//! root is this step, and which way is down in the wearer's own frame.
//!
//! It is [`crate::cloth`] with the nouns changed, and deliberately so: same
//! resource-on-the-sim-world doctrine ([`crate::deform`]'s, third application),
//! same ONE-Ring-0-function-both-hosts-call rule, same "absent until something
//! wears one" property, same `quality`-is-content-not-tier ruling. Reading the two
//! side by side is the point — a difference between them would be a bug in one.

use std::collections::BTreeMap;

use bevy_ecs::prelude::{Entity, Resource};
use glam::Vec3;
use inf_anim::hair::{HairAsset, HairDetail, HairState};
use inf_anim::SkeletonAsset;
use uuid::Uuid;

use crate::components::{GlobalTransform, Guid, HairGuides};
use crate::world::EcsWorld;

/// One wearer's live hairstyle.
#[derive(Debug, Clone, PartialEq)]
pub struct LiveHair {
    /// The `.inf_hair` GUID this hairstyle was seeded from — load-bearing for the
    /// same reason [`crate::cloth::LiveCloth::asset`] is: a re-bind must re-seed.
    pub asset: Uuid,
    /// The solver's live state.
    pub state: HairState,
    /// The ribbon triangle list, rebuilt each step alongside the positions.
    ///
    /// Here rather than in the projector because it is a **pure function of the
    /// sim state** and there are two projectors: computed once, in the fixed step,
    /// it cannot be computed two different ways. (Its *positions* are rebuilt with
    /// it — see [`step_hair_simulation`].)
    ///
    /// Its *density* is the host's [`inf_anim::hair::HairDetail`], which is the
    /// one tier-derived number on this path — cards on a weak GPU, interpolated
    /// strands on a strong one, and the same particles either way.
    pub ribbon_positions: Vec<[f32; 3]>,
    /// The ribbon index list, shared across steps when the topology has not
    /// changed. Rebuilt with the positions; an `Arc` so a projector clones a
    /// pointer rather than the list.
    pub ribbon_indices: std::sync::Arc<Vec<u32>>,
}

/// **Every wearer's simulated hairstyle**, keyed by [`Guid`] — a bevy resource,
/// exactly like [`crate::cloth::ClothStateRes`].
#[derive(Resource, Default, Debug, Clone, PartialEq)]
pub struct HairStateRes(pub BTreeMap<Uuid, LiveHair>);

/// The hairstyle `guid` is wearing, if the sim is simulating one — the **read
/// door** both render projectors go through.
pub fn live_hair(world: &EcsWorld, guid: Uuid) -> Option<&LiveHair> {
    world
        .world()
        .get_resource::<HairStateRes>()
        .and_then(|r| r.0.get(&guid))
}

/// How many hairstyles the sim is simulating.
pub fn hair_count(world: &EcsWorld) -> usize {
    world
        .world()
        .get_resource::<HairStateRes>()
        .map(|r| r.0.len())
        .unwrap_or(0)
}

/// **Forget every hairstyle** — the Simulate start/stop door, for the reason
/// [`crate::cloth::clear_cloth`] documents. Idempotent.
pub fn clear_hair(world: &mut EcsWorld) {
    world.world_mut().remove_resource::<HairStateRes>();
}

/// The simulated hairstyles' canonical bytes, or an empty vec when nothing is
/// simulating.
///
/// Folds the **strand particle positions** and the asset GUID; the ribbon
/// geometry is deliberately absent because it is a pure function of those
/// positions, so hashing it would fold the same information twice — the argument
/// that keeps sockets out of [`crate::pose::pose_state_bytes`] and velocities out
/// of [`crate::cloth::cloth_state_bytes`].
pub fn hair_state_bytes(world: &EcsWorld) -> Vec<u8> {
    let Some(store) = world.world().get_resource::<HairStateRes>() else {
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

/// **The fixed-step hair slot**: seed every newly-worn hairstyle, anchor each
/// strand on the joint its root rides, advance the solver, and rebuild the
/// ribbons.
///
/// ONE function, called from both hosts' fixed steps. Call it in the same slot as
/// [`crate::cloth::step_cloth_simulation`] and for the same reason: the roots and
/// the capsules are read off the pose this step published, and the model frame off
/// a settled `GlobalTransform`.
///
/// Rules mirror the cloth slot's exactly: a disabled or unbound component wears
/// nothing (1); an unresolvable or invalid `.inf_hair` is skipped with the
/// component untouched (2); state is carried for the same asset and re-seeded on a
/// re-bind (3); the store is rebuilt each step and removed when nothing simulates
/// (4). Wearers are `Guid`-sorted before anything is written.
pub fn step_hair_simulation<'c>(
    world: &mut EcsWorld,
    dt: f64,
    hairs: &dyn Fn(Uuid) -> Option<&'c HairAsset>,
    skeletons: &dyn Fn(Uuid) -> Option<&'c SkeletonAsset>,
    detail: HairDetail,
) {
    let mut wearers: Vec<(Entity, Uuid, HairGuides)> = Vec::new();
    {
        let w = world.world_mut();
        let mut q = w.query::<(Entity, &Guid, &HairGuides)>();
        for (e, g, hg) in q.iter(w) {
            if hg.enabled && hg.asset.is_some() {
                wearers.push((e, g.0, *hg));
            }
        }
    }
    if wearers.is_empty() {
        let w = world.world_mut();
        if w.contains_resource::<HairStateRes>() {
            w.remove_resource::<HairStateRes>();
        }
        return;
    }
    wearers.sort_by_key(|(_, guid, _)| *guid);

    let mut store = world
        .world_mut()
        .remove_resource::<HairStateRes>()
        .map(|r| r.0)
        .unwrap_or_default();
    let mut next: BTreeMap<Uuid, LiveHair> = BTreeMap::new();

    for (entity, guid, hg) in wearers {
        let Some(asset_id) = hg.asset else { continue };
        let Some(asset) = hairs(asset_id) else {
            continue;
        };

        let mut live = match store.remove(&guid) {
            Some(prev) if prev.asset == asset_id && prev.state.len() == asset.particle_count() => {
                prev
            }
            _ => match HairState::seed(asset) {
                Ok(state) => LiveHair {
                    asset: asset_id,
                    state,
                    ribbon_positions: Vec::new(),
                    ribbon_indices: std::sync::Arc::new(Vec::new()),
                },
                Err(_) => continue,
            },
        };

        let Some(global) = world.world().get::<GlobalTransform>(entity).map(|g| g.0) else {
            continue;
        };
        let Some(gravity) = crate::cloth::model_gravity(&global) else {
            continue;
        };

        // Roots and capsules both ride the pose the sim published THIS step. With
        // no pose (no machine, or no resolvable skeleton) the strands keep their
        // REST roots and collide against nothing, which is a hairstyle hanging off
        // an unanimated head rather than a hairstyle that vanishes.
        let posed = crate::pose::evaluated_pose(world, guid)
            .and_then(|p| skeletons(p.skeleton).map(|sk| (sk, p)));
        let (roots, capsules) = match posed {
            Some((sk, p)) => {
                let globals = inf_anim::global_transforms(&sk.skeleton, &p.pose);
                (
                    inf_anim::hair::roots_for(asset, &globals),
                    inf_anim::hair::capsules_for_hair(asset, &globals),
                )
            }
            None => (
                asset
                    .strands
                    .iter()
                    .map(|s| Vec3::from_array(s.points[0]))
                    .collect(),
                Vec::new(),
            ),
        };

        let stepped = if hg.quality == 0 {
            asset.material.substeps
        } else {
            hg.quality
        };
        if stepped == asset.material.substeps {
            inf_anim::hair::step_hair(
                asset,
                &mut live.state,
                dt as f32,
                gravity,
                &roots,
                &capsules,
            );
        } else {
            let mut tuned = asset.clone();
            tuned.material.substeps = stepped;
            inf_anim::hair::step_hair(
                &tuned,
                &mut live.state,
                dt as f32,
                gravity,
                &roots,
                &capsules,
            );
        }
        // The ribbons, rebuilt HERE rather than in each projector: they are a pure
        // function of the state above, and two projectors computing them
        // separately is the mirror pair this doctrine exists to retire.
        //
        // `detail` is the ONE tier-derived quantity on this path (P24.4). It is
        // safe here — and only here — because the ribbons are not folded into
        // `hair_state_bytes`: the geometry changes with the machine, the trace
        // does not, which `the_detail_draws_differently_and_traces_identically`
        // measures rather than asserts.
        let (pos, idx) = inf_anim::hair::render_mesh(asset, &live.state, detail);
        if *live.ribbon_indices != idx {
            live.ribbon_indices = std::sync::Arc::new(idx);
        }
        live.ribbon_positions = pos;
        next.insert(guid, live);
    }

    let w = world.world_mut();
    if next.is_empty() {
        if w.contains_resource::<HairStateRes>() {
            w.remove_resource::<HairStateRes>();
        }
    } else {
        w.insert_resource(HairStateRes(next));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Transform;
    use crate::math::Vec3d;
    use inf_anim::hair::{HairGroom, HairMaterial, HairRoot};

    const WEARER: Uuid = Uuid::from_u128(0x24_4101);
    const STYLE: Uuid = Uuid::from_u128(0x24_4102);

    fn style() -> HairAsset {
        let roots: Vec<HairRoot> = (0..3)
            .map(|i| HairRoot {
                joint: 0,
                offset: [i as f32 * 0.02, 0.0, 0.0],
                direction: [1.0, 0.0, 0.0],
                clump: i as u16 / 2,
            })
            .collect();
        HairAsset::grow(
            *STYLE.as_bytes(),
            &roots,
            0.2,
            4,
            HairMaterial::default(),
            HairGroom::default(),
        )
        .expect("the fixture hairstyle grows")
    }

    fn world_with_wearer(enabled: bool) -> EcsWorld {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(WEARER, "Hero", None);
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: Vec3d::ZERO,
                ..Default::default()
            },
            HairGuides {
                asset: Some(STYLE),
                enabled,
                quality: 0,
            },
        ));
        world.mark_dirty();
        world.propagate();
        world
    }

    fn step(world: &mut EcsWorld, asset: &HairAsset) {
        step_at(world, asset, HairDetail::GUIDES);
    }

    fn step_at(world: &mut EcsWorld, asset: &HairAsset, detail: HairDetail) {
        let hairs = |g: Uuid| (g == STYLE).then_some(asset);
        let skels = |_: Uuid| None;
        step_hair_simulation(world, 1.0 / 60.0, &hairs, &skels, detail);
    }

    /// **The headline gate**: a `HairGuides` component is READ — the strands move,
    /// they ride the trace, and they build ribbons.
    #[test]
    fn a_hair_guides_component_simulates_and_rides_the_trace() {
        let a = style();
        let mut world = world_with_wearer(true);
        assert_eq!(hair_count(&world), 0);
        assert!(hair_state_bytes(&world).is_empty());

        step(&mut world, &a);
        assert_eq!(hair_count(&world), 1);
        let first = hair_state_bytes(&world);
        assert!(!first.is_empty());
        for _ in 0..40 {
            step(&mut world, &a);
        }
        assert_ne!(first, hair_state_bytes(&world), "the strands never moved");

        let live = live_hair(&world, WEARER).unwrap();
        assert_eq!(live.ribbon_positions.len(), a.particle_count() * 2);
        assert!(!live.ribbon_indices.is_empty());
        // The roots stayed on the (unposed) rest anchors and the tips fell.
        for s in 0..live.state.strand_count() {
            let base = live.state.starts[s] as usize;
            assert_eq!(live.state.x[base], a.strands[s].points[0]);
        }
        let tip = live.state.strand_count() * 5 - 1;
        assert!(live.state.x[tip][1] < -0.01, "no strand fell");
    }

    #[test]
    fn a_world_with_no_hair_never_grows_a_store() {
        let mut world = EcsWorld::new();
        world.world_mut().spawn((Guid(Uuid::from_u128(4)),));
        world.reindex_guids();
        step(&mut world, &style());
        assert_eq!(hair_count(&world), 0);
        assert!(hair_state_bytes(&world).is_empty());
    }

    #[test]
    fn a_disabled_or_unresolvable_hairstyle_simulates_nothing() {
        let a = style();
        let mut off = world_with_wearer(false);
        step(&mut off, &a);
        assert_eq!(hair_count(&off), 0);

        let mut ghost = world_with_wearer(true);
        let e = ghost.entity_of(WEARER).unwrap();
        ghost.world_mut().get_mut::<HairGuides>(e).unwrap().asset = Some(Uuid::from_u128(0xDEAD));
        step(&mut ghost, &a);
        assert_eq!(hair_count(&ghost), 0);
        assert_eq!(
            ghost.world().get::<HairGuides>(e).unwrap().asset,
            Some(Uuid::from_u128(0xDEAD)),
            "skipping is not unbinding"
        );
    }

    #[test]
    fn a_removed_hairstyle_leaves_no_stale_strands() {
        let a = style();
        let mut world = world_with_wearer(true);
        step(&mut world, &a);
        assert_eq!(hair_count(&world), 1);
        let e = world.entity_of(WEARER).unwrap();
        world.world_mut().get_mut::<HairGuides>(e).unwrap().enabled = false;
        step(&mut world, &a);
        assert_eq!(hair_count(&world), 0);
        assert!(hair_state_bytes(&world).is_empty());
    }

    #[test]
    fn simulation_is_deterministic() {
        let a = style();
        let run = || {
            let mut w = world_with_wearer(true);
            for _ in 0..45 {
                step(&mut w, &a);
            }
            hair_state_bytes(&w)
        };
        let first = run();
        assert_eq!(first, run());
        assert!(!first.is_empty());
        let mut w = world_with_wearer(true);
        for _ in 0..44 {
            step(&mut w, &a);
        }
        assert_ne!(hair_state_bytes(&w), first, "the trace is a constant");
    }

    /// **THE TIER LAW, measured** (P24.4): the render detail changes what is
    /// DRAWN and leaves the TRACE byte-identical.
    ///
    /// This is the whole reason a tier-derived number is allowed inside a fixed
    /// step at all. `ClothSim::quality` is content and never the machine, because
    /// a substep budget lands in `state_bytes`; the ribbon detail is the machine
    /// and never content, because ribbons do not. If a future edit folded the
    /// ribbons into `hair_state_bytes` — which would look like a harmless
    /// completeness fix — this arm goes red and says why.
    ///
    /// Both halves are asserted: the *cards* arm must differ from the *guides*
    /// arm (or "the detail changed nothing" would satisfy the trace half
    /// vacuously), and the two traces must be equal.
    #[test]
    fn the_detail_draws_differently_and_traces_identically() {
        let a = style();
        let run = |detail: HairDetail| {
            let mut w = world_with_wearer(true);
            for _ in 0..20 {
                step_at(&mut w, &a, detail);
            }
            let live = live_hair(&w, WEARER).expect("the wearer simulates").clone();
            (
                hair_state_bytes(&w),
                live.ribbon_positions,
                live.ribbon_indices,
            )
        };
        let (guide_trace, guide_pos, guide_idx) = run(HairDetail::GUIDES);
        let (card_trace, card_pos, card_idx) = run(HairDetail::CARDS);
        let (interp_trace, interp_pos, _) = run(HairDetail::INTERPOLATED);
        assert!(!guide_trace.is_empty(), "the fixture must simulate");
        assert!(!guide_pos.is_empty() && !guide_idx.is_empty());
        // Drawn differently…
        assert!(
            card_pos.len() < guide_pos.len(),
            "cards must draw less: {} vs {}",
            card_pos.len(),
            guide_pos.len()
        );
        assert!(
            interp_pos.len() > guide_pos.len(),
            "interpolation must draw more"
        );
        assert!(card_idx.len() < guide_idx.len());
        // …and traced identically.
        assert_eq!(card_trace, guide_trace, "a card changed the SIM");
        assert_eq!(interp_trace, guide_trace, "interpolation changed the SIM");
    }

    #[test]
    fn clear_hair_forgets_everything() {
        let a = style();
        let mut world = world_with_wearer(true);
        step(&mut world, &a);
        assert_eq!(hair_count(&world), 1);
        clear_hair(&mut world);
        assert_eq!(hair_count(&world), 0);
        assert!(hair_state_bytes(&world).is_empty());
        clear_hair(&mut world);
    }
}
