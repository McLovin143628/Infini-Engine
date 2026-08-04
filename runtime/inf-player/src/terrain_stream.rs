//! Camera-driven terrain streaming for the shipped player (P16.3b2).
//!
//! Binds the Ring-0 pieces — [`inf_terrain::wants`] (what to page) and
//! [`inf_terrain::stream`] (the cold stores + the streamer) — to a live
//! [`EcsWorld`]: one [`TerrainStreamer`] per `Terrain` entity whose
//! [`asset`](Terrain::asset) resolves to a `.inf_terrain` in the cooked pack (or
//! the dev-dir), driven from the two seams below.
//!
//! # THE DETERMINISM DOCTRINE, as wired here
//!
//! The fixed-step sim's results must **never** depend on camera-driven residency,
//! or the replay / PIE-==-shipping gates die. Two drivers, and the player calls
//! them from two different places:
//!
//! * [`sync_sim`](TerrainStreaming::sync_sim) runs **inside
//!   [`RuntimeSim::fixed_step`](crate::runtime_sim::RuntimeSim), at the top, before
//!   the step's own work**. Its wants come from [`sim_wants`] over the *sim's own*
//!   entity positions plus a fixed margin — no camera, no frame rate, no clock —
//!   and they are loaded synchronously in ascending key order into the
//!   `TerrainData` living in the ECS `Terrain` component: the one, and only,
//!   heightfield `terrain.height_at` can reach.
//! * [`sync_render`](TerrainStreaming::sync_render) runs at the **render-sync
//!   point** (`PlayerApp::frame`, immediately before the scene projection; the
//!   headless harness calls it at exactly the same place in its loop). Its wants
//!   are the camera's quadtree cut and they land in the streamer's *private*
//!   working set, which no entity holds a reference to.
//!
//! Because those are two different [`TerrainData`] values, "the camera changed a
//! sim result" is not a bug that can be introduced by a careless edit — it would
//! need a new reference from the world to a set the world cannot name. The
//! bounded duplication of tiles both want is the price.
//!
//! ## What bounds the sim set
//!
//! Sim wants are **never clamped to a budget** — a missing render page costs
//! detail, a missing sim page changes the simulation — so the only thing that
//! bounds them is *which entities contribute*. That is [`observes_terrain`]:
//! Blueprint actors, character movers and physics bodies, and nothing else. A
//! level's static scenery does not enlarge terrain residency, however much of it
//! there is. [`SIM_WANT_SOFT_CEILING`] is the advisory diagnostic for when the
//! observers themselves are spread wider than a streaming budget can serve.
//!
//! ## What a sim query outside its neighbourhood sees
//!
//! A blueprint asking `terrain.height_at` for a point beyond
//! [`SIM_MARGIN_TILES`] of every **observer** reads `0.0` — the same answer an
//! unauthored tile has always given — rather than paging the tile in. That is a
//! *deliberate* deterministic answer: it depends only on sim state, so it is
//! identical on every machine and every run.
//!
//! **But it is gameplay-visible, so treat these two knobs as content, not
//! configuration.** Widening [`SIM_MARGIN_TILES`] or [`observes_terrain`] changes
//! which pages are resident, and therefore turns some far `height_at` calls from
//! `0.0` into a real height. The result stays deterministic, but it is a
//! *different* simulation: replay traces and recorded goldens must be re-blessed
//! exactly as they would be for a physics-tuning bump.

use std::collections::BTreeSet;
use std::sync::Arc;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    ActorClass, CharacterController2D, CharacterController3D, GlobalTransform, RigidBody2D,
    RigidBody3D, Terrain,
};
use inf_ecs::{EcsWorld, Guid};
use inf_terrain::stream::{StreamBudget, TerrainStreamStats, TerrainStreamer};
use inf_terrain::wants::{sim_wants, RenderWantsParams, TileGrid};
use inf_terrain::{TerrainData, TileStore};

/// How far around each **terrain-observing** sim entity level-0 terrain is kept
/// resident, in level-0 tile spans. Two spans covers a tile plus its neighbours in
/// every direction, so `height_at`'s shared-edge fallback and `normal_at`'s central
/// differences never walk off the resident set.
pub const SIM_MARGIN_TILES: f64 = 2.0;

/// Soft ceiling on the level-0 pages one terrain's **sim** wants may ask for.
///
/// **Advisory only — the sim set is never clamped** (correctness first: a missing
/// render page costs detail, a missing sim page changes the simulation). Crossing
/// it means the observer predicate is admitting far more entities than intended,
/// or that observers are spread across more world than a streaming budget can
/// hold; either way it is a content/design signal, and it is logged loudly rather
/// than silently absorbed.
pub const SIM_WANT_SOFT_CEILING: usize = 256;

/// Refinement radius of a level-0 node, in level-0 tile spans — the anchor of the
/// geometric ladder [`RenderWantsParams::geometric`] builds. Matches the
/// renderer's clipmap ladder (`TERRAIN_LOD_SCALE`-class spacing), so the streamed
/// cut and the drawn rings stay in step.
pub const RENDER_LOD0_RADIUS_TILES: f64 = 2.5;

/// A store that can be shared with the job pool.
pub type SharedTileStore = Arc<dyn TileStore + Send + Sync>;

/// A resolved `.inf_terrain`: its cold store plus the **level-0 grid the asset
/// was built against**.
///
/// The grid travels with the store rather than being sniffed from a tile,
/// because it is the asset header's business: page origins are derived from it,
/// so adopting the level's (possibly stale) config instead would land every
/// streamed tile at the wrong world position.
pub struct TerrainSource {
    pub store: SharedTileStore,
    /// Samples per tile side, from the asset header.
    pub tile_resolution: u32,
    /// Level-0 metres per sample, from the asset header.
    pub meters_per_sample: f64,
}

/// One streamed terrain entity.
struct StreamedTerrain {
    /// The entity carrying the `Terrain` component.
    entity: Uuid,
    /// The `.inf_terrain` asset GUID it streams from.
    asset: Uuid,
    /// The entity's world XZ, so a world-space camera/position converts into the
    /// terrain-local frame the tile grid is anchored in.
    origin_xz: DVec2,
    store: SharedTileStore,
    streamer: TerrainStreamer,
    grid: TileGrid,
    /// Level-0 residency margin around each terrain observer, in metres.
    sim_margin_m: f64,
    /// Whether the last sync was already over [`SIM_WANT_SOFT_CEILING`], so a
    /// stable oversized set warns once instead of every fixed step.
    warned_sim_ceiling: bool,
    /// Which **simulation** tile versions have been mirrored into the render
    /// streamer (P21.4) — see [`TerrainStreaming::overlay_sim_edits`]. Empty for
    /// every level nothing carves at runtime.
    overlaid: std::collections::BTreeMap<inf_terrain::TileKey, u64>,
}

/// Whether `entity` can **observe** the terrain, i.e. whether the fixed step can
/// read a height (or need a collider) under it.
///
/// This predicate is what bounds the sim want set. Without it every entity with a
/// world transform contributed a [`SIM_MARGIN_TILES`] neighbourhood, so a level
/// with a few thousand scattered static props asked for most of the level-0 grid —
/// and because sim wants are (correctly) never clamped, that ceiling was
/// unenforceable. Residency must scale with the number of things that can *look*
/// at the ground, not with how much scenery the artist placed on it.
///
/// **Observers** (any one of these):
/// * [`ActorClass`] — the entity runs a Blueprint, which can call the
///   `terrain.height_at` host seam (the `RuntimeHost` dispatch in
///   [`crate::runtime_sim`]);
/// * [`CharacterController2D`] / [`CharacterController3D`] — a character mover
///   walks the surface and snaps to it;
/// * [`RigidBody2D`] / [`RigidBody3D`] — **future-proofing for terrain↔physics
///   coupling, which does not exist yet**: no terrain heightfield collider is fed
///   to the physics bridges today, so a body cannot currently read the terrain at
///   all. Admitting it is conservative in the safe direction — it can only make
///   the sim set larger than strictly needed, never smaller, and the day a
///   heightfield collider lands the pages it needs are already resident rather
///   than silently missing.
///
/// **Not observers** — deliberately: static decorative meshes, sprites, lights,
/// cameras (a camera is a *render* want, and letting it in here would be exactly
/// the coupling the doctrine forbids), the terrain entity itself, PCG volumes and
/// foliage (their scatter is baked against the terrain at load, not per step).
///
/// A world with no observers wants nothing, which is correct: nothing in the step
/// can tell the difference between an unloaded page and one that was never
/// queried. A blueprint probing a point far from **every** observer reads the
/// documented unauthored answer — deterministically, since the predicate reads
/// only sim state.
pub fn observes_terrain(world: &EcsWorld, entity: inf_ecs::Entity) -> bool {
    let w = world.world();
    w.get::<ActorClass>(entity).is_some()
        || w.get::<CharacterController2D>(entity).is_some()
        || w.get::<CharacterController3D>(entity).is_some()
        || w.get::<RigidBody2D>(entity).is_some()
        || w.get::<RigidBody3D>(entity).is_some()
}

/// Every streamed terrain in the loaded world.
///
/// Empty — and every method a no-op — for a world with no asset-backed terrain,
/// which is every level the editor writes today. Inline terrains keep the
/// pre-P16.3 behaviour exactly: fully resident, never touched here.
#[derive(Default)]
pub struct TerrainStreaming {
    entries: Vec<StreamedTerrain>,
    stats: TerrainStreamStats,
}

impl TerrainStreaming {
    /// Bind streamers to every `Terrain` entity whose `asset` GUID `resolve`
    /// can serve.
    ///
    /// Entities in `Guid` order (deterministic). A terrain whose asset does not
    /// resolve is left alone — its inline `data` stays authoritative, which is
    /// exactly the documented fallback when the referenced asset is missing
    /// (the cook already warns about a dangling `Terrain.asset`).
    ///
    /// The component's `TerrainData` is **re-configured from the asset header**
    /// when it is still empty and its grid disagrees, so streamed tiles land on
    /// the grid the asset was built against rather than on a stale level default.
    pub fn attach(
        world: &mut EcsWorld,
        budget: StreamBudget,
        mut resolve: impl FnMut(Uuid) -> Option<TerrainSource>,
    ) -> Self {
        let mut targets: Vec<(Uuid, Uuid, DVec3)> = {
            let w = world.world();
            let mut v: Vec<(Uuid, Uuid, DVec3)> = w
                .iter_entities()
                .filter_map(|e| {
                    let guid = e.get::<Guid>()?.0;
                    let terrain = e.get::<Terrain>()?;
                    let asset = terrain.asset?;
                    let origin = e
                        .get::<GlobalTransform>()
                        .map(|g| g.translation())
                        .unwrap_or(DVec3::ZERO);
                    Some((guid, asset, origin))
                })
                .collect();
            v.sort_by_key(|(g, _, _)| *g);
            v
        };

        let mut entries = Vec::new();
        for (entity, asset, origin) in targets.drain(..) {
            let Some(source) = resolve(asset) else {
                tracing::warn!(
                    "inf-player: terrain entity {entity} references .inf_terrain {asset}, \
                     which is not in the content — falling back to its inline data"
                );
                continue;
            };
            let TerrainSource {
                store,
                tile_resolution: res,
                meters_per_sample: mps,
            } = source;

            // The component's working set must sit on the asset's grid: a streamed
            // page's origin is derived from its coordinate, so a stale level
            // config would place every tile in the wrong place. A terrain with
            // authored inline tiles is left alone and reported.
            if let Some(e) = world.entity_of(entity) {
                if let Some(mut t) = world.world_mut().get_mut::<Terrain>(e) {
                    let stale =
                        t.data.tile_resolution() != res || t.data.meters_per_sample() != mps;
                    if stale && t.data.is_empty() {
                        t.data = TerrainData::new(res, mps);
                    } else if stale {
                        tracing::warn!(
                            "inf-player: terrain {entity} holds inline tiles on a {}² @ {} m grid \
                             but .inf_terrain {asset} is {res}² @ {mps} m — not streaming it",
                            t.data.tile_resolution(),
                            t.data.meters_per_sample()
                        );
                        continue;
                    }
                }
            }

            let catalog = inf_terrain::wants::TileCatalog::from_store(store.as_ref());
            let grid = TileGrid::new(res, mps);
            let params = RenderWantsParams::geometric(
                RENDER_LOD0_RADIUS_TILES * grid.level0_span(),
                inf_terrain::wants::TileIndex::max_lod(&catalog) + 1,
            );
            let streamer = TerrainStreamer::new(grid, catalog, params, budget, res, mps);
            tracing::info!(
                "inf-player: streaming terrain {entity} from .inf_terrain {asset} \
                 ({} page(s), {} LOD level(s), {res}² samples @ {mps} m)",
                streamer.catalog().len(),
                inf_terrain::wants::TileIndex::max_lod(streamer.catalog()) + 1,
            );
            entries.push(StreamedTerrain {
                entity,
                asset,
                origin_xz: DVec2::new(origin.x, origin.z),
                store,
                streamer,
                grid,
                sim_margin_m: SIM_MARGIN_TILES * grid.level0_span(),
                warned_sim_ceiling: false,
                overlaid: std::collections::BTreeMap::new(),
            });
        }
        Self {
            entries,
            stats: TerrainStreamStats::default(),
        }
    }

    /// Whether anything streams (a plain inline-terrain world streams nothing).
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Number of streamed terrains.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// The **render**-resident working set for `entity`, if it streams. The
    /// player's `project_terrain` draws this instead of the component's data.
    pub fn render_data(&self, entity: Uuid) -> Option<&TerrainData> {
        self.entries
            .iter()
            .find(|e| e.entity == entity)
            .map(|e| e.streamer.render_data())
    }

    /// The `.inf_terrain` asset GUID `entity` streams from.
    pub fn asset_of(&self, entity: Uuid) -> Option<Uuid> {
        self.entries
            .iter()
            .find(|e| e.entity == entity)
            .map(|e| e.asset)
    }

    /// Merged streaming counters across every streamed terrain.
    pub fn stats(&self) -> &TerrainStreamStats {
        &self.stats
    }

    /// The published render cut for `entity` (the resident-set trace the gate
    /// compares run to run).
    pub fn cut(&self, entity: Uuid) -> Option<&BTreeSet<inf_terrain::TileKey>> {
        self.entries
            .iter()
            .find(|e| e.entity == entity)
            .map(|e| e.streamer.cut())
    }

    /// **SIM (deterministic).** Page in the level-0 tiles the fixed step needs.
    ///
    /// Called at the top of every fixed step, **before** the step's own work, so
    /// blueprint `terrain.height_at` calls and physics heightfield queries within
    /// the step see a fully-settled neighbourhood. Wants come from sim state
    /// alone; loads are synchronous and ascending-key-ordered. Nothing about the
    /// camera, the frame rate, or how long a load took can reach this path.
    ///
    /// Only entities that can actually **observe** the terrain contribute — see
    /// [`observes_terrain`]. That is what keeps the sim set a handful of
    /// neighbourhoods rather than a function of how much scenery the level
    /// contains.
    pub fn sync_sim(&mut self, world: &mut EcsWorld) {
        if self.entries.is_empty() {
            return;
        }
        // Sim-state positions of the terrain OBSERVERS, `Guid`-sorted so the want
        // set is a pure function of the world's contents.
        let positions: Vec<DVec3> = {
            let mut v: Vec<(Uuid, DVec3)> = world
                .world()
                .iter_entities()
                .filter_map(|e| {
                    let g = e.get::<Guid>()?.0;
                    let t = e.get::<GlobalTransform>()?;
                    Some((g, e.id(), t.translation()))
                })
                .collect::<Vec<_>>()
                .into_iter()
                .filter(|(_, id, _)| observes_terrain(world, *id))
                .map(|(g, _, p)| (g, p))
                .collect();
            v.sort_by_key(|(g, _)| *g);
            v.into_iter().map(|(_, p)| p).collect()
        };

        for entry in &mut self.entries {
            let local: Vec<DVec2> = positions
                .iter()
                .map(|p| DVec2::new(p.x, p.z) - entry.origin_xz)
                .collect();
            let wants = sim_wants(entry.grid, &local, entry.sim_margin_m);
            // Advisory ceiling: never clamped (correctness first), but a sim set
            // this large means the observers are spread wider than a streaming
            // budget can serve, or that something is contributing that should not.
            // Logged on the way up only, so a stable large set says it once.
            let over = wants.len() > SIM_WANT_SOFT_CEILING;
            if over && !entry.warned_sim_ceiling {
                tracing::warn!(
                    "inf-player: terrain {} sim residency wants {} level-0 page(s) from {} \
                     terrain observer(s) — over the {} soft ceiling. The sim set is NEVER \
                     clamped (a missing sim page changes the simulation), so this is a memory \
                     cost, not a correctness one; check that the observers are not scattered \
                     across the whole world.",
                    entry.entity,
                    wants.len(),
                    positions.len(),
                    SIM_WANT_SOFT_CEILING
                );
            }
            entry.warned_sim_ceiling = over;

            let Some(e) = world.entity_of(entry.entity) else {
                continue;
            };
            let Some(mut terrain) = world.world_mut().get_mut::<Terrain>(e) else {
                continue;
            };
            // Paging a page in is not an edit: `sync_residency` stamps the tile
            // (so a consumer re-reads it) but never marks it dirty, so no
            // write-back is scheduled and the authored bytes stay untouched.
            entry
                .streamer
                .sync_sim(&mut terrain.data, &wants, entry.store.as_ref());
        }
        self.refresh_stats();
    }

    /// **RENDER (camera-driven).** Advance every streamed terrain's cut toward
    /// the camera's target and apply the (bounded, deterministic) load batch.
    ///
    /// Call at exactly one point per frame — the render-sync point — in both the
    /// windowed and the headless loops, so a scripted camera path produces the
    /// same resident-set trace either way.
    /// **Mirror the SIMULATION's runtime terrain edits into the render streamer**
    /// (P21.4). Returns how many tiles were pinned or released.
    ///
    /// A runtime carve opens a **hole** in the sim world's `Terrain.data`
    /// (`inf_voxel::apply_surface_cut`, from `runtime_voxel_op`), so gameplay stops
    /// standing on ground that is no longer there. On an **asset-backed** terrain —
    /// the only kind that can persist a hole mask at all, and therefore the
    /// configuration every carved level ships in — the render side streams its own
    /// tiles out of the `.inf_terrain` and never sees that edit: the clipmap keeps
    /// drawing solid ground over a mouth the player just blew open, and can walk
    /// through it.
    ///
    /// This is the player's twin of the editor's
    /// `inf_editor_core::terrain_stream::TerrainStreaming::overlay_document_edits`,
    /// with the sim world in place of the document. A **dirty** level-0 tile is
    /// pinned into the streamer, which then neither evicts it nor re-reads it from
    /// the asset. Dirty is a function of what was written, never of what any camera
    /// paged, so the direction of dependency is `sim → render` alone.
    ///
    /// # The pin set is bounded by the CUT, and that is not a detail
    ///
    /// The editor pins every dirty tile and releases them all on save. A *player*
    /// never saves, and a runtime hole mask is never cleared — so pinning every
    /// dirty tile meant a pin set that only ever grew. Past
    /// `StreamBudget::max_resident_tiles` (1024) that is not a memory cost, it is
    /// a **stall**: `TerrainStreamer::pin_ceiling` clamps the camera's cut to
    /// `.max(1)` once the pins fill the budget, so the terrain silently stops
    /// streaming around the player and nothing says why.
    ///
    /// So the pin follows the **cut** — the tiles the streamer is actually drawing.
    /// A dirty tile inside the cut is pinned; a pinned tile that leaves the cut is
    /// released (and dropped, so the next visit pages it fresh and re-pins it from
    /// the sim, which still holds the hole). The pin set is therefore bounded by
    /// the cut, the cut is bounded by the budget, and a carve fifty kilometres
    /// behind the player costs nothing.
    ///
    /// Called **after** `sync_render`, so "the cut" is this frame's rather than
    /// last frame's — the same ordering, and the same reason, as the voxel
    /// overlay's fourth act in `crate::render::sync_voxel_store`.
    pub fn overlay_sim_edits(&mut self, world: &EcsWorld) -> usize {
        if self.entries.is_empty() {
            return 0;
        }
        let mut mirrored = 0usize;
        for entry in &mut self.entries {
            let Some(e) = world.entity_of(entry.entity) else {
                continue;
            };
            let Some(terrain) = world.world().get::<Terrain>(e) else {
                continue;
            };

            // Release pins the cut has moved off. `unpin_and_evict` rather than a
            // bare unpin: leaving the tile resident-but-unpinned would let the
            // camera page the asset's un-holed bytes over it later with nothing
            // watching, and the tile is outside the cut so dropping it draws
            // nothing away.
            let stale: Vec<inf_terrain::TileKey> = entry
                .overlaid
                .keys()
                .copied()
                .filter(|k| !entry.streamer.cut().contains(k))
                .collect();
            for key in stale {
                entry.streamer.unpin_and_evict(key);
                entry.overlaid.remove(&key);
                mirrored += 1;
            }

            for key in terrain.data.dirty_tiles() {
                if !key.is_lod0() || !entry.streamer.cut().contains(&key) {
                    continue;
                }
                // A dirty key with no tile is a deletion; the streamer must drop
                // and unpin it, or a phantom page survives that the store has
                // nothing to page over. (The editor's twin carries the same arm,
                // for the same reason.)
                let Some(tile) = terrain.data.get_tile(key.coord) else {
                    if entry.overlaid.remove(&key).is_some() || entry.streamer.is_pinned(key) {
                        entry.streamer.unpin_and_evict(key);
                        mirrored += 1;
                    }
                    continue;
                };
                let stamp = terrain.data.tile_version(key);
                if entry.overlaid.get(&key) == Some(&stamp) && entry.streamer.is_pinned(key) {
                    continue;
                }
                entry.streamer.pin_tile(key, tile.clone());
                entry.overlaid.insert(key, stamp);
                mirrored += 1;
            }
        }
        if mirrored > 0 {
            self.refresh_stats();
        }
        mirrored
    }

    /// How many tiles of `entity` have been mirrored from the sim — the engagement
    /// counter a gate reads instead of pixels.
    pub fn overlaid_len(&self, entity: Uuid) -> usize {
        self.entries
            .iter()
            .find(|e| e.entity == entity)
            .map(|e| e.overlaid.len())
            .unwrap_or(0)
    }

    /// **RENDER (camera-driven).** Advance every streamed terrain's cut toward the
    /// camera's target and apply the (bounded, deterministic) load batch.
    ///
    /// Call at exactly one point per frame — the render-sync point — in both the
    /// windowed and the headless loops, so a scripted camera path produces the
    /// same resident-set trace either way.
    pub fn sync_render(&mut self, camera_world: DVec3) {
        if self.entries.is_empty() {
            return;
        }
        let cam = DVec2::new(camera_world.x, camera_world.z);
        for entry in &mut self.entries {
            let store = entry.store.clone();
            entry
                .streamer
                .sync_render(cam - entry.origin_xz, store.as_ref());
        }
        self.refresh_stats();
    }

    fn refresh_stats(&mut self) {
        let mut merged = TerrainStreamStats::default();
        for e in &self.entries {
            merged.merge(e.streamer.stats());
        }
        self.stats = merged;
    }
}
