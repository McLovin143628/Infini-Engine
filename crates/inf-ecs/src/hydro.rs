//! **Hydrology seams over the ECS world** (P20.4) — the one derivation both
//! scene projectors need in order to let P19.1's flow maps reach P20's water.
//!
//! # Why this is here and not in either projector
//!
//! `project_water` is a MIRROR: it exists twice, character-for-character, in
//! `inf_viewport::host` and `inf_player::render`, because neither Ring-0 crate
//! can host it. Everything that could *silently* diverge between those two copies
//! is deliberately kept out of them and put in Ring 0 instead — that is what
//! [`sky::water_environment`](crate::sky::water_environment) is, and this is its
//! twin for the flow map. A host that walked the world for terrains itself would
//! be exactly the drift the mirror gate exists to stop, and this one is worse
//! than most: "which terrain answers here" is a *rule*, and two hosts with two
//! rules would foam two different rivers.
//!
//! # What the flow map is, and what it is allowed to do
//!
//! P19.1's erosion bake writes `DataMapKind::Flow` — the time-integrated volume
//! of water that left each cell, in m³ — so it peaks along the channels the water
//! carved. A river running down such a channel is a rapid; the same river across
//! an unmapped plain is not. That is the whole coupling, and it is deliberately
//! **additive only**: [`inf_water::flow_foam_gain`] returns exactly `1.0` where
//! there is no flow, no terrain, or no bake, so wiring it in changes nothing
//! about content that has no flow map. Every golden in the repo is proof.
//!
//! # Determinism
//!
//! [`TerrainFlow`] resolves terrains in ascending `Guid` order and answers from
//! the **first** one that has a height there — a rule, not a traversal artefact.
//! No clock, no camera, no RNG, no `HashMap` iteration. The gain is a pure `f64`
//! function of the map value, so two hosts, two runs and two machines agree.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_terrain::{DataMapKind, TerrainData};

use crate::components::{GlobalTransform, Terrain, Transform};
use crate::world::EcsWorld;
use crate::Guid;

/// The level's terrains, borrowed, ready to answer flow queries in world XZ.
///
/// Built once per projection by [`terrain_flow`] and thrown away with it; it
/// never clones a heightfield (a `TerrainData` is megabytes) and never outlives
/// the borrow of the world it came from.
pub struct TerrainFlow<'w> {
    /// `(entity origin, data)` per non-empty terrain, in ascending `Guid` order.
    terrains: Vec<(DVec3, &'w TerrainData)>,
}

impl<'w> TerrainFlow<'w> {
    /// Whether any terrain in this level carries a **non-default** flow map.
    ///
    /// The off-path test: a level whose terrains were never eroded — the default
    /// state of every terrain in the engine — answers `false`, and a caller can
    /// skip the per-frame query entirely. It is a claim about the data, not about
    /// the query, which is why it is asked once rather than inferred from a run
    /// of `1.0`s.
    pub fn is_mapped(&self) -> bool {
        self.terrains
            .iter()
            .any(|(_, d)| !d.data_maps_are_default())
    }

    /// Raw flow accumulation at a world XZ point, m³ — `None` over a hole, off
    /// every terrain's authored extent, or in a level with no terrain.
    ///
    /// First terrain (by `Guid`) that answers wins. That is the same "lowest-Guid
    /// terrain is the authority" rule the height query uses, extended only by
    /// letting a later terrain answer where an earlier one has no data — which is
    /// what multi-terrain levels (P16.5) need and what a strict first-only rule
    /// would break.
    pub fn flow_at(&self, world_xz: DVec2) -> Option<f64> {
        for (origin, data) in &self.terrains {
            let local = DVec2::new(world_xz.x - origin.x, world_xz.y - origin.z);
            if let Some(v) = data.data_map_at(DataMapKind::Flow, local) {
                return Some(v as f64);
            }
        }
        None
    }

    /// The **foam gain** a river frame at this point takes from the flow map.
    ///
    /// `1.0` wherever there is no answer, so this is total and never needs a
    /// caller-side fallback. See [`inf_water::flow_foam_gain`] for the curve and
    /// for why it can only ever add.
    #[inline]
    pub fn foam_gain_at(&self, world_xz: DVec2) -> f64 {
        match self.flow_at(world_xz) {
            Some(v) => inf_water::flow_foam_gain(v),
            None => 1.0,
        }
    }
}

/// Gather the level's terrains for flow queries (P20.4).
///
/// A world with no terrain — or none carrying data — yields an empty
/// [`TerrainFlow`] whose [`foam_gain_at`](TerrainFlow::foam_gain_at) is the
/// constant `1.0`, which is exactly what "no flow map" should mean.
pub fn terrain_flow(world: &EcsWorld) -> TerrainFlow<'_> {
    let w = world.world();
    // Collect (guid, entity) first and sort, so the answer is a function of the
    // level rather than of bevy's archetype layout.
    let mut found: Vec<(Uuid, bevy_ecs::entity::Entity)> = Vec::new();
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
        found.push((guid, e.id()));
    }
    found.sort_by_key(|(g, _)| *g);
    let terrains = found
        .into_iter()
        .filter_map(|(_, e)| {
            let data = &w.get::<Terrain>(e)?.data;
            let origin = w
                .get::<GlobalTransform>(e)
                .map(|g| g.translation())
                .or_else(|| w.get::<Transform>(e).map(|t| t.translation.to_dvec3()))
                .unwrap_or(DVec3::ZERO);
            Some((origin, data))
        })
        .collect();
    TerrainFlow { terrains }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3d;

    /// A one-tile terrain whose flow map is `value` over the whole tile.
    fn terrain_with_flow(value: f32) -> Terrain {
        const RES: u32 = 17;
        let mut t = Terrain {
            meters_per_sample: 1.0,
            tile_resolution: RES,
            data: inf_terrain::TerrainData::new(RES, 1.0),
            ..Terrain::default()
        };
        t.data.author_tile((0, 0), |_, _| 0.0);
        if value != 0.0 {
            let tile = t.data.get_tile_mut((0, 0)).unwrap();
            for j in 0..RES {
                for i in 0..RES {
                    tile.set_map_texel(RES, i, j, [value, 0.0, 0.0]);
                }
            }
        }
        t
    }

    fn world_with(terrains: &[(u128, Terrain, DVec3)]) -> EcsWorld {
        let mut w = EcsWorld::new();
        for (id, t, origin) in terrains {
            let guid = Uuid::from_u128(*id);
            let e = w.spawn_with_guid(guid, "terrain", None);
            w.world_mut().entity_mut(e).insert(t.clone());
            w.world_mut().entity_mut(e).insert(Transform {
                translation: Vec3d::new(origin.x, origin.y, origin.z),
                ..Transform::IDENTITY
            });
        }
        w.propagate();
        w
    }

    #[test]
    fn an_unmapped_level_gains_exactly_nothing() {
        let w = world_with(&[(1, terrain_with_flow(0.0), DVec3::ZERO)]);
        let flow = terrain_flow(&w);
        assert!(!flow.is_mapped(), "a never-eroded terrain is not mapped");
        // Every gain is the exact identity — not "close to 1".
        for i in 0..64 {
            let p = DVec2::new(i as f64 * 3.0, i as f64 * -2.0);
            assert_eq!(flow.foam_gain_at(p), 1.0, "at {p:?}");
        }
        // …and so is a level with no terrain at all.
        let bare = EcsWorld::new();
        let none = terrain_flow(&bare);
        assert!(!none.is_mapped());
        assert!(none.flow_at(DVec2::ZERO).is_none());
        assert_eq!(none.foam_gain_at(DVec2::new(500.0, -900.0)), 1.0);
    }

    #[test]
    fn a_mapped_terrain_boosts_the_gain_and_saturates() {
        let w = world_with(&[(
            1,
            terrain_with_flow(inf_water::FLOW_FOAM_REFERENCE_M3 as f32),
            DVec3::ZERO,
        )]);
        let flow = terrain_flow(&w);
        assert!(flow.is_mapped(), "the bake must be visible");
        let g = flow.foam_gain_at(DVec2::new(1.0, 1.0));
        assert!(g > 1.0, "gain {g}");
        assert!((g - inf_water::flow_foam_gain(inf_water::FLOW_FOAM_REFERENCE_M3)).abs() < 1e-12);
        // Off the authored extent there is no answer, and the gain is the identity.
        assert!(flow.flow_at(DVec2::new(1e6, 1e6)).is_none());
        assert_eq!(flow.foam_gain_at(DVec2::new(1e6, 1e6)), 1.0);
    }

    #[test]
    fn the_terrain_origin_is_honoured_and_the_guid_order_decides() {
        // Two terrains, the higher-Guid one flowing, offset a long way in X so
        // their extents do not overlap. The offset one must still answer, at its
        // own world position.
        let origin = DVec3::new(10_000.0, 0.0, 0.0);
        let w = world_with(&[
            (2, terrain_with_flow(500.0), origin),
            (1, terrain_with_flow(0.0), DVec3::ZERO),
        ]);
        let flow = terrain_flow(&w);
        assert!(flow.is_mapped());
        // At the origin, the lowest-Guid (unmapped) terrain answers 0.
        assert_eq!(flow.flow_at(DVec2::new(1.0, 1.0)), Some(0.0));
        assert_eq!(flow.foam_gain_at(DVec2::new(1.0, 1.0)), 1.0);
        // Over the offset terrain, the flow is found at its WORLD position —
        // which is the whole point of subtracting the origin.
        let there = DVec2::new(origin.x + 1.0, 1.0);
        assert_eq!(flow.flow_at(there), Some(500.0));
        assert!(flow.foam_gain_at(there) > 1.0);
        // Anti-vacuity: without the origin subtraction the query would have
        // landed off the tile and answered `None`.
        assert!(flow.flow_at(DVec2::new(origin.x + 1.0, 1e9)).is_none());
    }

    #[test]
    fn the_gather_is_a_pure_function_of_the_level() {
        let w = world_with(&[
            (7, terrain_with_flow(120.0), DVec3::new(0.0, 0.0, 0.0)),
            (3, terrain_with_flow(0.0), DVec3::new(0.0, 0.0, 5_000.0)),
        ]);
        let probe = || {
            let f = terrain_flow(&w);
            (0..50)
                .map(|i| f.foam_gain_at(DVec2::new(i as f64, i as f64)).to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(probe(), probe());
    }
}
