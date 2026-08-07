//! World-partition **cell streaming** for the shipped player (P16.5).
//!
//! Binds the Ring-0 partition pieces — [`inf_scene::partition`] (how a level is
//! binned, and the `.inf_part` container) — to a live [`EcsWorld`]: entities
//! spawn and despawn a cell at a time as the simulation's own streaming sources
//! move.
//!
//! # THE DETERMINISM DOCTRINE, extended
//!
//! Terrain streaming (P16.3b2, [`crate::terrain_stream`]) could afford a
//! camera-driven half, because a missing terrain page only costs detail. **Cell
//! streaming cannot**: an unloaded cell means its entities *do not exist*, and
//! the fixed step can see that. So the doctrine tightens, and the tightening is
//! structural rather than a rule people remember:
//!
//! * **Wants come from sim entities, never a camera.** The want set is derived
//!   from entities carrying [`StreamingSource`] — the possessed character, a
//!   scripted convoy, whatever the level designates — read at the fixed-step
//!   boundary in `Guid` order. There is no camera argument to
//!   [`sync_sim`](CellStreaming::sync_sim) to pass one by accident. (The editor
//!   viewport's free camera *is* a legitimate source while authoring; the editor
//!   stays single-document in v1, so nothing streams there at all.)
//! * **Activation and deactivation happen only at the deterministic sync point**,
//!   in ascending cell order, inside [`sync_sim`](CellStreaming::sync_sim) at the
//!   very top of the fixed step — before anything in that step can observe the
//!   world. Two runs of one scripted walk produce the same activation trace,
//!   frame for frame.
//! * **Loading may be early; activation may not be late.** Decoding a cell's blob
//!   is allowed to happen ahead of need (the prefetch margin) or in a bounded
//!   batch, but a cell that reaches its activation step still unloaded is loaded
//!   **synchronously, blocking the step**. That is the v1 semantic, stated
//!   plainly: correctness first, and the prefetch margin is the mitigation. It is
//!   also what makes the margin provably free of simulation consequences — the
//!   gate runs one scripted walk under two different margins and compares the sim
//!   traces byte for byte.
//!
//! ## What streams, and what does not
//!
//! Only **cook-partitioned** (statically-placed) entities stream. The manager
//! remembers exactly which `Guid`s it spawned for a cell, and deactivation
//! despawns exactly those. An entity a Blueprint or a mod spawned at runtime is
//! in no cell's list and is therefore **never** despawned by streaming — v1's
//! rule, and the conservative one: a runtime-spawned entity has no authored home
//! to return to, so evicting it would destroy state nothing could rebuild.
//!
//! The corollary is honest too: a *statically-placed* entity that a blueprint
//! moves far from its birth cell still despawns when that **birth** cell
//! deactivates, because residency is a function of where the cook put it. Content
//! that moves an entity across the world should mark it
//! [`AlwaysLoaded`](inf_ecs::components::AlwaysLoaded) — that is what the marker
//! is for.
//!
//! ## Blueprint actors, and the v1 boundary
//!
//! [`RuntimeSim`](crate::runtime_sim::RuntimeSim) assigns blueprint actor ids
//! once, at construction, from the entities present then — which is why the
//! **persistent cell is spawned by the level builder, before actor binding**,
//! rather than by this manager. A persistent entity's `ActorClass` therefore
//! ticks exactly as it does in an unpartitioned level.
//!
//! **An entity that streams IN does not gain a ticking blueprint in v1.** Its
//! components — mesh, colliders, lights, audio, animation — are all live; only
//! the actor map is fixed at construction, and making it dynamic is a
//! `RuntimeSim` change (blueprint entity handles are stable `i64`s assigned in
//! `Guid` order) that this batch deliberately did not make. Content whose logic
//! must run marks the entity [`AlwaysLoaded`](inf_ecs::components::AlwaysLoaded)
//! — or gives it a [`StreamingSource`], which also makes it persistent — and the
//! partitioner then keeps it in the persistent cell where binding works. Stated
//! here, and in the sample's README, rather than discovered.
//!
//! ## In-editor Simulate
//!
//! The editor's in-process Simulate runs the **whole document**, unpartitioned,
//! because the editor is single-document in v1 and there is nothing to stream out
//! of. Play-in-editor (the subprocess player) and a shipped build both stream, and
//! it is those two the PIE-==-shipping gate compares.
//!
//! ## Cross-cell references
//!
//! A cook-time reference from cell A to an entity in cell B is a cook **warning**
//! (`inf_scene::partition::cross_cell_refs`), not an error and not a promotion.
//! Promoting would make residency depend on the reference graph, which turns a
//! streaming budget into a guess; so the hazard is surfaced where it is cheap to
//! fix and the author decides — exactly the dangling-terrain-ref precedent.
//!
//! At **runtime** the same hazard has an observable state:
//! [`CellStreamStats::unresolved_refs`] (P16.6) counts the references that are
//! disconnected *right now* — an `AttachedTo` whose target is not resident (the
//! follower freezes where it stands, per `inf_ecs::attach::update_attachments`) or
//! a joint whose other body is not resident (the physics bridge drops it). The
//! cook warns at build time about the ones it can see statically; this is what the
//! author reads when it actually happens, and what the P16.6 budget gate asserts
//! is zero on a well-authored scene.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use uuid::Uuid;

use inf_ecs::components::{
    AttachedTo, GlobalTransform, Joint2D, Joint3D, StreamingSource, Transform,
};
use inf_ecs::{EcsWorld, Guid};
use inf_scene::partition::{
    cell_distance, cells_within, CellCoord, CellDirEntry, CellKey, PartitionAssetHeader,
    PartitionAssetReader, PartitionPlan, PartitionSettings,
};
use inf_scene::RuntimeEntity;

/// The fixed salt XORed into a level GUID to derive its `.inf_part` GUID.
///
/// **Kept in sync with `inf_packager::cook::PARTITION_ID_SALT`** (duplicated here
/// so the shipped player does not depend on the cook pipeline — the same pattern
/// as [`crate::vmesh::derived_vmesh_id`] and [`crate::level::PACK_FILE`]). A
/// drift test asserts the two agree.
const PARTITION_ID_SALT: u128 = 0x7016_0500_5041_5254_9d4e_2c7a_b31f_60e8;

/// Derive the deterministic `.inf_part` asset id for a level id. XOR with a
/// constant is a bijection, so distinct levels always yield distinct partition
/// ids; mirrors `inf_packager::derived_partition_id`.
pub fn derived_partition_id(level_id: Uuid) -> Uuid {
    Uuid::from_u128(level_id.as_u128() ^ PARTITION_ID_SALT)
}

// ── budget ──────────────────────────────────────────────────────────────────

/// Bounds on the streaming work one [`sync_sim`](CellStreaming::sync_sim) does.
///
/// Note what is **not** here: there is no ceiling on resident cells. A cell the
/// activation radius wants is a cell the simulation needs, and clamping it would
/// change results rather than cost detail — the same "sim wants are never
/// clamped" rule terrain streaming states. The only knob is how much *look-ahead*
/// decoding one step may do, which by construction cannot change a result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellStreamBudget {
    /// Ceiling on how many **prefetch** (not-yet-needed) cells one sync may
    /// decode. `0` disables look-ahead entirely, which is legal and simply means
    /// every activation blocks — a useful setting for proving that the prefetch
    /// margin cannot move a sim trace.
    pub max_prefetch_per_sync: usize,
}

impl Default for CellStreamBudget {
    fn default() -> Self {
        Self {
            max_prefetch_per_sync: 4,
        }
    }
}

/// Advisory ceiling on how many cells one sync may **activate**.
///
/// Crossing it is not clamped (activation is never clamped — see
/// [`CellStreamBudget`]); it means a streaming source teleported, or the
/// activation radius is wider than the cell size makes sensible. Logged once on
/// the way up so a stable oversized set says it one time.
pub const ACTIVATION_SOFT_CEILING: usize = 64;

// ── diagnostics ─────────────────────────────────────────────────────────────

/// Cell-residency counters for the debug/stats path and the P16.6 budget
/// ratchets — the cell mirror of `inf_terrain::TerrainStreamStats`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CellStreamStats {
    /// Streamed grid cells currently active (entities spawned).
    pub cells_resident: usize,
    /// Cells decoded and held ready but not yet activated (the prefetch buffer).
    pub cells_loaded: usize,
    /// Cells the activation radius wants that are not resident yet — the backlog
    /// a step would have to block on. Normally `0`, because activation blocks.
    pub cells_pending: usize,
    /// Entities currently spawned by streaming (persistent cell excluded).
    pub entities_resident: usize,
    /// Bytes of cell blob held resident + prefetched.
    pub bytes_resident: u64,
    /// Cell activations since the manager was created.
    pub activations: u64,
    /// Cell deactivations since the manager was created.
    pub deactivations: u64,
    /// Activations that had to load their cell synchronously (a prefetch miss).
    /// **Not an error** — it is the documented v1 semantic — but it is the number
    /// the prefetch margin exists to drive down.
    pub blocking_loads: u64,
    /// Cells whose blob failed to decode. Excluded from every subsequent want, so
    /// a corrupt cell is a hole in the world rather than a crash loop.
    pub failed: u64,
    /// **Cross-cell references currently disconnected** (P16.6): live entities
    /// whose `AttachedTo.target`, `Joint2D.other` or `Joint3D.other` names a
    /// `Guid` that does not resolve in the world right now.
    ///
    /// This is a *live* count, recomputed at the deterministic sync point — it
    /// rises when the cell holding the target streams out and falls when it comes
    /// back. It is **not an error count**: the behaviour is defined either way (an
    /// `AttachedTo` whose target is gone leaves the follower frozen where it
    /// stands; a joint whose other body is gone is skipped by the physics bridge).
    /// It is the number that tells an author their level depends on a reference
    /// across a streaming boundary — the runtime face of the cook's
    /// `cross_cell_refs` warning.
    ///
    /// A pure function of world state, like every other counter here, so it cannot
    /// make a run non-reproducible.
    pub unresolved_refs: usize,
}

impl CellStreamStats {
    /// A one-line summary for the diagnostics log (the `tracing` seam terrain
    /// streaming already uses — no new panel, no new IPC channel).
    pub fn summary(&self) -> String {
        format!(
            "cell streaming: {} cell(s) resident ({} entities), {} loaded, {} pending, \
             {} activations / {} deactivations / {} blocking / {} failed, {:.1} KiB, \
             {} unresolved ref(s)",
            self.cells_resident,
            self.entities_resident,
            self.cells_loaded,
            self.cells_pending,
            self.activations,
            self.deactivations,
            self.blocking_loads,
            self.failed,
            self.bytes_resident as f64 / 1024.0,
            self.unresolved_refs,
        )
    }
}

/// One activation/deactivation, in the order it happened — the trace the gates
/// compare run to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct CellEvent {
    /// The fixed step this happened on.
    pub step: u64,
    /// `true` = activated (spawned), `false` = deactivated (despawned).
    pub activated: bool,
    /// The grid cell.
    pub coord: CellCoord,
}

// ── cold stores ─────────────────────────────────────────────────────────────

/// Where a level's streamed cells come from.
///
/// Two implementations, one interface, and that is the PIE-==-shipping seam: a
/// cooked run slices cells out of the pack mapping, an editor/dev-dir run bins
/// the level's own entities in memory with the *same* Ring-0 function the cook
/// used. Same cells, same order, same activation timeline.
pub trait CellStore: Send + Sync {
    /// Every **streamed grid** coordinate this store can serve, ascending.
    fn grid_coords(&self) -> Vec<CellCoord>;
    /// The persistent cell's entities (always resident).
    fn persistent(&self) -> Result<Vec<RuntimeEntity>, String>;
    /// One grid cell's entities; `None` when the store has no such cell.
    fn cell(&self, coord: CellCoord) -> Result<Option<Vec<RuntimeEntity>>, String>;
    /// A cell's on-disk blob size, for the residency byte counter.
    fn cell_bytes_len(&self, coord: CellCoord) -> u64;
    /// The grid's cell edge length, from the source's own header — never from the
    /// level's (possibly stale) settings, for the same reason a streamed terrain
    /// adopts its asset's grid: a cell's world footprint is derived from its
    /// coordinate, so disagreeing about the size puts every cell in the wrong
    /// place.
    fn cell_size_m(&self) -> f64;
}

/// A [`CellStore`] over a `.inf_part` entry of a **cooked pack**.
///
/// The header + cell directory are parsed once at construction; every fetch after
/// that is a binary search plus a sub-slice of [`PackReader::read_ref`]'s
/// borrowed mapping, then one bincode decode of that cell alone — which is the
/// entire reason partitions cook uncompressed
/// ([`PackWriter::compresses_kind`](inf_asset::PackWriter::compresses_kind)).
///
/// A pack that nonetheless stores the entry compressed (an older cook, or a
/// hand-built pack) still works: the payload is decoded **once** at construction
/// into an owned buffer and served from there. Slower to open, identical
/// afterwards — mirroring `inf_terrain::PackTileStore`'s fallback, and for the
/// same reason: a stale pack degrades in speed, never in correctness.
pub struct PackCellStore {
    backing: PackBacking,
    header: PartitionAssetHeader,
    dir: Vec<CellDirEntry>,
}

enum PackBacking {
    /// Zero-copy: slice straight out of the pack mapping on every fetch.
    Pack(Arc<inf_asset::PackReader>, inf_asset::AssetId),
    /// The compressed-entry fallback: the decoded payload, owned.
    Owned(Vec<u8>),
}

impl PackCellStore {
    /// Open the `.inf_part` asset `guid` inside `reader`.
    pub fn open(
        reader: Arc<inf_asset::PackReader>,
        guid: inf_asset::AssetId,
    ) -> Result<Self, String> {
        let payload = reader
            .read_ref(guid)
            .map_err(|e| format!("read partition asset {guid}: {e}"))?;
        let borrowed = matches!(payload, std::borrow::Cow::Borrowed(_));
        let (header, dir) = {
            let r = PartitionAssetReader::new(payload.as_ref())
                .map_err(|e| format!("partition asset {guid}: {e}"))?;
            (*r.header(), r.directory().to_vec())
        };
        let backing = if borrowed {
            drop(payload);
            PackBacking::Pack(reader, guid)
        } else {
            PackBacking::Owned(payload.into_owned())
        };
        Ok(Self {
            backing,
            header,
            dir,
        })
    }

    /// The payload bytes, borrowed from whichever backing holds them.
    fn payload(&self) -> Option<&[u8]> {
        match &self.backing {
            // Verified once at construction; a compressed entry took the Owned
            // arm, so this is always the borrowed mapping.
            PackBacking::Pack(reader, guid) => match reader.read_ref(*guid) {
                Ok(std::borrow::Cow::Borrowed(b)) => Some(b),
                _ => None,
            },
            PackBacking::Owned(v) => Some(v.as_slice()),
        }
    }

    fn entry(&self, key: CellKey) -> Option<&CellDirEntry> {
        self.dir
            .binary_search_by(|e| e.key.cmp(&key))
            .ok()
            .map(|i| &self.dir[i])
    }

    fn blob(&self, key: CellKey) -> Option<&[u8]> {
        let e = self.entry(key)?;
        // Bounds were validated when the directory was parsed.
        self.payload()?
            .get(e.offset as usize..(e.offset + e.len) as usize)
    }

    fn decode(&self, key: CellKey) -> Result<Option<Vec<RuntimeEntity>>, String> {
        let Some(bytes) = self.blob(key) else {
            return Ok(None);
        };
        inf_scene::partition::decode_cell(bytes)
            .map(Some)
            .map_err(|e| format!("{key}: {e}"))
    }
}

impl std::fmt::Debug for PackCellStore {
    /// Summarizes; never dumps the (possibly enormous) payload.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackCellStore")
            .field(
                "backing",
                &match &self.backing {
                    PackBacking::Pack(_, guid) => format!("pack:{guid}"),
                    PackBacking::Owned(v) => format!("owned:{} bytes", v.len()),
                },
            )
            .field("cell_count", &self.dir.len())
            .field("cell_size_m", &self.header.cell_size_m)
            .finish()
    }
}

impl CellStore for PackCellStore {
    fn grid_coords(&self) -> Vec<CellCoord> {
        self.dir.iter().filter_map(|e| e.key.coord()).collect()
    }

    fn persistent(&self) -> Result<Vec<RuntimeEntity>, String> {
        Ok(self.decode(CellKey::PERSISTENT)?.unwrap_or_default())
    }

    fn cell(&self, coord: CellCoord) -> Result<Option<Vec<RuntimeEntity>>, String> {
        self.decode(CellKey::grid(coord))
    }

    fn cell_bytes_len(&self, coord: CellCoord) -> u64 {
        self.entry(CellKey::grid(coord)).map(|e| e.len).unwrap_or(0)
    }

    fn cell_size_m(&self) -> f64 {
        self.header.cell_size_m
    }
}

/// A [`CellStore`] over an **in-memory** [`PartitionPlan`] — the loose-`.inf_lvl`
/// and PIE path.
///
/// A level that has not been cooked still carries all its entities inline, so the
/// runtime bins them with the very same Ring-0 function the cook uses
/// (`inf_scene::partition::partition_entities`) and streams the result. That is
/// what makes PIE == shipping *structural* here rather than a coincidence two
/// code paths are asked to maintain: one binning rule, two sources.
///
/// Byte sizes reported here are the canonical blob encodings, so the residency
/// counters mean the same thing on both paths.
#[derive(Debug)]
pub struct MemoryCellStore {
    persistent: Vec<RuntimeEntity>,
    cells: BTreeMap<CellCoord, Vec<RuntimeEntity>>,
    sizes: BTreeMap<CellCoord, u64>,
    cell_size_m: f64,
}

impl MemoryCellStore {
    /// Bin `entities` with `settings` and serve the result.
    pub fn from_entities(entities: &[RuntimeEntity], settings: &PartitionSettings) -> Self {
        Self::from_plan(
            inf_scene::partition::partition_entities(entities, settings),
            settings.effective_cell_size(),
        )
    }

    /// Serve an already-binned plan.
    pub fn from_plan(plan: PartitionPlan, cell_size_m: f64) -> Self {
        let sizes = plan
            .cells
            .iter()
            .map(|(&c, list)| {
                (
                    c,
                    inf_scene::partition::encode_cell(list)
                        .map(|b| b.len() as u64)
                        .unwrap_or(0),
                )
            })
            .collect();
        Self {
            persistent: plan.persistent,
            cells: plan.cells,
            sizes,
            cell_size_m,
        }
    }
}

impl CellStore for MemoryCellStore {
    fn grid_coords(&self) -> Vec<CellCoord> {
        self.cells.keys().copied().collect()
    }

    fn persistent(&self) -> Result<Vec<RuntimeEntity>, String> {
        Ok(self.persistent.clone())
    }

    fn cell(&self, coord: CellCoord) -> Result<Option<Vec<RuntimeEntity>>, String> {
        Ok(self.cells.get(&coord).cloned())
    }

    fn cell_bytes_len(&self, coord: CellCoord) -> u64 {
        self.sizes.get(&coord).copied().unwrap_or(0)
    }

    fn cell_size_m(&self) -> f64 {
        self.cell_size_m
    }
}

// ── the manager ─────────────────────────────────────────────────────────────

/// Cell streaming for one loaded world.
///
/// Empty — and every method a no-op — for a world whose level is not partitioned,
/// which is every level the editor writes by default, so unpartitioned behaviour
/// is bit-identical to the pre-P16.5 player.
#[derive(Default)]
pub struct CellStreaming {
    store: Option<Arc<dyn CellStore>>,
    settings: PartitionSettings,
    budget: CellStreamBudget,
    /// The grid's cell size, taken from the **store's** header.
    cell_size_m: f64,
    /// Every streamed cell the store can serve.
    available: BTreeSet<CellCoord>,
    /// Active cells → the entity `Guid`s streaming spawned for them. Membership
    /// here is what makes despawn precise: a runtime-spawned entity is in no
    /// list, so it is never evicted.
    resident: BTreeMap<CellCoord, Vec<Uuid>>,
    /// Decoded-but-not-yet-activated cells (the prefetch buffer).
    loaded: BTreeMap<CellCoord, Vec<RuntimeEntity>>,
    /// Cells whose blob failed to decode — excluded from every subsequent want,
    /// so a corrupt cell is a hole in the world, not a crash loop.
    blocked: BTreeSet<CellCoord>,
    stats: CellStreamStats,
    /// The activation trace (the gate's comparison unit).
    events: Vec<CellEvent>,
    /// Whether the last sync was already over [`ACTIVATION_SOFT_CEILING`], so a
    /// stable oversized set warns once instead of every fixed step.
    warned_ceiling: bool,
    /// Referrer `Guid`s whose broken cross-cell reference has already been logged
    /// (P16.6) — once per GUID, not once per step, so a cell that stays out of
    /// residency for a thousand steps says it one time.
    logged_unresolved: BTreeSet<Uuid>,
}

impl CellStreaming {
    /// Attach cell streaming over `store`.
    ///
    /// The **persistent cell is not spawned here** — it is spawned by
    /// [`InfSceneWorldBuilder`](crate::level::InfSceneWorldBuilder), which runs
    /// before actor binding, because the persistent cell *is* the level's world
    /// at step 0 and its blueprints have to bind like any other level's. This
    /// manager owns only what streams.
    pub fn attach(
        store: Arc<dyn CellStore>,
        settings: PartitionSettings,
        budget: CellStreamBudget,
    ) -> Self {
        let cell_size_m = store.cell_size_m();
        let available: BTreeSet<CellCoord> = store.grid_coords().into_iter().collect();
        tracing::info!(
            "inf-player: cell streaming {} grid cell(s) @ {cell_size_m} m \
             (activation {} m, prefetch margin {} m)",
            available.len(),
            settings.effective_activation_radius(),
            settings.effective_prefetch_margin(),
        );
        Self {
            store: Some(store),
            settings,
            budget,
            cell_size_m,
            available,
            resident: BTreeMap::new(),
            loaded: BTreeMap::new(),
            blocked: BTreeSet::new(),
            stats: CellStreamStats::default(),
            events: Vec::new(),
            warned_ceiling: false,
            logged_unresolved: BTreeSet::new(),
        }
    }

    /// Whether anything streams (an unpartitioned world streams nothing).
    pub fn is_empty(&self) -> bool {
        self.store.is_none()
    }

    /// The level's partition settings, as the manager applies them.
    pub fn settings(&self) -> &PartitionSettings {
        &self.settings
    }

    /// The grid's cell edge length (metres), from the source's header.
    pub fn cell_size_m(&self) -> f64 {
        self.cell_size_m
    }

    /// Every streamed cell the source can serve, ascending.
    pub fn available(&self) -> impl Iterator<Item = CellCoord> + '_ {
        self.available.iter().copied()
    }

    /// The currently active cells, ascending — the residency trace.
    pub fn resident(&self) -> impl Iterator<Item = CellCoord> + '_ {
        self.resident.keys().copied()
    }

    /// Whether `coord` is active.
    pub fn is_resident(&self, coord: CellCoord) -> bool {
        self.resident.contains_key(&coord)
    }

    /// Residency counters (budget hooks for P16.6).
    pub fn stats(&self) -> &CellStreamStats {
        &self.stats
    }

    /// The activation/deactivation trace, in order.
    pub fn events(&self) -> &[CellEvent] {
        &self.events
    }

    /// **SIM (deterministic).** Reconcile cell residency with the streaming
    /// sources' positions.
    ///
    /// Called at the top of every fixed step, **before** the step's own work, so
    /// everything downstream — physics sync, Blueprint Tick, animation — sees a
    /// settled world. The sequence, all of it a pure function of sim state:
    ///
    /// 1. read source positions (`Guid`-sorted);
    /// 2. compute the activation want set and the (wider) prefetch want set;
    /// 3. decode a **bounded batch** of prefetch cells — pure look-ahead;
    /// 4. **deactivate** `resident − wants`, ascending;
    /// 5. **activate** `wants − resident`, ascending, blocking on any cell the
    ///    prefetch did not reach;
    /// 6. propagate the hierarchy once.
    ///
    /// Steps 4 and 5 are the only ones the simulation can observe, and neither
    /// consults a clock, a camera, or what finished in time.
    pub fn sync_sim(&mut self, world: &mut EcsWorld, step: u64) {
        let Some(store) = self.store.clone() else {
            return;
        };

        // 1. Sources: sim-state positions + per-source radii, `Guid`-sorted so
        //    the want set is a pure function of the world's contents.
        let sources = streaming_sources(world);

        // 2. Wants. A cell the store cannot serve is never wanted, and a cell
        //    whose blob failed to decode is permanently excluded (see `blocked`).
        let level_radius = self.settings.effective_activation_radius();
        let margin = self.settings.effective_prefetch_margin();
        let mut activate: BTreeSet<CellCoord> = BTreeSet::new();
        let mut prefetch: BTreeSet<CellCoord> = BTreeSet::new();
        for (pos, radius) in &sources {
            let r = radius.max(level_radius);
            cells_within(pos.x, pos.z, r, self.cell_size_m, &mut activate);
            cells_within(pos.x, pos.z, r + margin, self.cell_size_m, &mut prefetch);
        }
        activate.retain(|c| self.available.contains(c) && !self.blocked.contains(c));
        prefetch.retain(|c| self.available.contains(c) && !self.blocked.contains(c));

        let over = activate.len() > ACTIVATION_SOFT_CEILING;
        if over && !self.warned_ceiling {
            tracing::warn!(
                "inf-player: cell streaming wants {} cell(s) active from {} streaming source(s) \
                 — over the {} soft ceiling. Activation is NEVER clamped (a missing cell changes \
                 the simulation), so this is a memory cost, not a correctness one; check the \
                 activation radius against the {} m cell size.",
                activate.len(),
                sources.len(),
                ACTIVATION_SOFT_CEILING,
                self.cell_size_m
            );
        }
        self.warned_ceiling = over;

        // 3. Prefetch: decode a bounded batch of cells wanted **soon but not
        //    now** — strictly `prefetch − activate`. Pure look-ahead: nothing
        //    below reads `loaded` for a decision, only to avoid re-decoding.
        //
        //    Excluding the cells this step is about to activate is what keeps the
        //    semantic honest. Loading a cell you need *right now* is not
        //    prefetching, it is blocking the step, and it is counted as such
        //    (`blocking_loads`) in step 5 — which is exactly the number the
        //    prefetch margin exists to drive down, and the number that would
        //    otherwise silently read zero.
        let pending: Vec<CellCoord> = prefetch
            .iter()
            .copied()
            .filter(|c| {
                !activate.contains(c)
                    && !self.loaded.contains_key(c)
                    && !self.resident.contains_key(c)
            })
            .take(self.budget.max_prefetch_per_sync)
            .collect();
        if !pending.is_empty() {
            for (coord, result) in pending
                .iter()
                .copied()
                .zip(load_cells(store.as_ref(), &pending))
            {
                match result {
                    Ok(Some(entities)) => {
                        self.loaded.insert(coord, entities);
                    }
                    Ok(None) => {
                        // The directory said it existed; it does not. Treat it
                        // like a decode failure rather than looping on it.
                        self.block(coord, "cell vanished from the store");
                    }
                    Err(e) => self.block(coord, &e),
                }
            }
        }

        // 4. Deactivate, ascending. Despawning exactly the `Guid`s we spawned is
        //    what keeps runtime-spawned entities safe.
        let leaving: Vec<CellCoord> = self
            .resident
            .keys()
            .copied()
            .filter(|c| !activate.contains(c))
            .collect();
        for coord in leaving {
            let Some(guids) = self.resident.remove(&coord) else {
                continue;
            };
            // Reverse order so a subtree's root is despawned last: `despawn`
            // removes a subtree, and the partitioner keeps a hierarchy in one
            // cell, so this is belt-and-braces against a `Guid` that is already
            // gone (handled below either way).
            for guid in guids.iter().rev() {
                if let Some(e) = world.entity_of(*guid) {
                    world.despawn(e);
                }
            }
            self.stats.deactivations += 1;
            self.events.push(CellEvent {
                step,
                activated: false,
                coord,
            });
            tracing::debug!("inf-player: cell ({},{}) deactivated", coord.0, coord.1);
        }

        // 5. Activate, ascending. A cell the prefetch did not reach is loaded
        //    here, **blocking the step** — the documented v1 semantic.
        let entering: Vec<CellCoord> = activate
            .iter()
            .copied()
            .filter(|c| !self.resident.contains_key(c))
            .collect();
        for coord in entering {
            let entities = match self.loaded.remove(&coord) {
                Some(e) => e,
                None => {
                    self.stats.blocking_loads += 1;
                    match store.cell(coord) {
                        Ok(Some(e)) => e,
                        Ok(None) => {
                            self.block(coord, "cell vanished from the store");
                            continue;
                        }
                        Err(e) => {
                            self.block(coord, &e);
                            continue;
                        }
                    }
                }
            };
            let guids = crate::level::spawn_entities(world, entities);
            self.stats.activations += 1;
            self.events.push(CellEvent {
                step,
                activated: true,
                coord,
            });
            tracing::debug!(
                "inf-player: cell ({},{}) activated — {} entity(ies)",
                coord.0,
                coord.1,
                guids.len()
            );
            self.resident.insert(coord, guids);
        }

        // 6. One propagate for the whole reconcile.
        world.propagate();
        self.refresh_stats(&activate);
        // 7. Cross-cell reference health (P16.6). A read-only scan of sim state
        //    AFTER the reconcile, so the count describes the world the step is
        //    about to run against — and, being a pure function of that state, it
        //    can never make a run non-reproducible.
        self.stats.unresolved_refs = self.scan_unresolved_refs(world);
    }

    /// Count — and, once per referrer, log — the cross-cell references that are
    /// disconnected in `world` right now (P16.6).
    ///
    /// Walks the entities carrying a persisted entity reference and checks each
    /// target `Guid` against the live world. Deterministic: entities are visited
    /// in `Guid` order, so the log lines (and the count) are a pure function of
    /// world state.
    ///
    /// Runtime-spawned entities are included deliberately: a blueprint that
    /// attached something to a streamed entity has exactly the same hazard as a
    /// cook-time reference, and the cook could not have warned about it.
    fn scan_unresolved_refs(&mut self, world: &EcsWorld) -> usize {
        let w = world.world();
        // (referrer, field, target) — collected then sorted, so the log order is
        // stable rather than archetype-order.
        let mut broken: Vec<(Uuid, &'static str, Uuid)> = Vec::new();
        for e in w.iter_entities() {
            let Some(guid) = e.get::<Guid>().map(|g| g.0) else {
                continue;
            };
            let mut check = |field: &'static str, target: Option<Uuid>| {
                if let Some(t) = target {
                    if world.entity_of(t).is_none() {
                        broken.push((guid, field, t));
                    }
                }
            };
            check(
                "attached_to.target",
                e.get::<AttachedTo>().map(|a| a.target),
            );
            check(
                "joint_2d.other",
                e.get::<Joint2D>().and_then(|j| j.other.get()),
            );
            check(
                "joint_3d.other",
                e.get::<Joint3D>().and_then(|j| j.other.get()),
            );
        }
        broken.sort_unstable();
        for &(referrer, field, target) in &broken {
            if self.logged_unresolved.insert(referrer) {
                tracing::debug!(
                    "inf-player: entity {referrer}'s {field} names {target}, which is not \
                     resident — the reference is disconnected until that cell streams back in \
                     (an AttachedTo follower freezes in place; a joint is skipped). Mark the \
                     target AlwaysLoaded, or keep the pair in one cell, if the link must hold."
                );
            }
        }
        broken.len()
    }

    /// World-space bounds of a cell: `(min_xz, max_xz)` in metres. The debug
    /// overlay strokes these.
    pub fn cell_bounds(&self, coord: CellCoord) -> ([f64; 2], [f64; 2]) {
        let s = self.cell_size_m;
        let min = [coord.0 as f64 * s, coord.1 as f64 * s];
        ([min[0], min[1]], [min[0] + s, min[1] + s])
    }

    /// A cell's state, for the debug overlay's colour.
    pub fn cell_state(&self, coord: CellCoord) -> CellState {
        if self.blocked.contains(&coord) {
            CellState::Failed
        } else if self.resident.contains_key(&coord) {
            CellState::Active
        } else if self.loaded.contains_key(&coord) {
            CellState::Loaded
        } else {
            CellState::Cold
        }
    }

    /// Exclude a cell from every future want and count it. Loud once, then quiet:
    /// a corrupt cell must not turn into a per-step log flood.
    fn block(&mut self, coord: CellCoord, why: &str) {
        if self.blocked.insert(coord) {
            self.stats.failed += 1;
            tracing::error!(
                "inf-player: partition cell ({},{}) failed to load ({why}) — it will stay \
                 unpopulated for the rest of this run",
                coord.0,
                coord.1
            );
        }
    }

    fn refresh_stats(&mut self, wanted: &BTreeSet<CellCoord>) {
        let store = self.store.as_ref();
        self.stats.cells_resident = self.resident.len();
        self.stats.cells_loaded = self.loaded.len();
        self.stats.cells_pending = wanted
            .iter()
            .filter(|c| !self.resident.contains_key(c))
            .count();
        self.stats.entities_resident = self.resident.values().map(Vec::len).sum();
        self.stats.bytes_resident = match store {
            Some(s) => self
                .resident
                .keys()
                .chain(self.loaded.keys())
                .map(|c| s.cell_bytes_len(*c))
                .sum(),
            None => 0,
        };
    }
}

/// What a cell is doing right now (the debug overlay's colour key).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Not loaded, not active.
    Cold,
    /// Decoded and ready, waiting for the activation radius.
    Loaded,
    /// Entities spawned.
    Active,
    /// Its blob failed to load; permanently excluded.
    Failed,
}

/// Every streaming source's world XZ position + its own activation radius, in
/// `Guid` order.
///
/// `GlobalTransform` is preferred (a source parented to something must follow
/// it); the local `Transform` is the fallback for a world that has not propagated
/// yet — which is exactly the state `attach` leaves behind before the first step.
pub fn streaming_sources(world: &EcsWorld) -> Vec<(glam::DVec3, f64)> {
    let w = world.world();
    let mut v: Vec<(Uuid, glam::DVec3, f64)> = w
        .iter_entities()
        .filter_map(|e| {
            let guid = e.get::<Guid>()?.0;
            let src = e.get::<StreamingSource>()?;
            let pos = e
                .get::<GlobalTransform>()
                .map(|g| g.translation())
                .or_else(|| e.get::<Transform>().map(|t| t.translation.to_dvec3()))
                .unwrap_or(glam::DVec3::ZERO);
            Some((guid, pos, src.radius_m))
        })
        .collect();
    v.sort_by_key(|(g, _, _)| *g);
    v.into_iter().map(|(_, p, r)| (p, r)).collect()
}

/// Decode `coords` (already ascending) into entity lists.
///
/// Runs on the [`inf_core`] job pool for a batch worth parallelizing; the pool's
/// `parallel_map_ref` is a **deterministic in-order pure map**, so the output is a
/// function of the input alone and the caller applies it in coordinate order.
/// Small batches — and the browser player, which has no thread pool — take the
/// serial path and produce the identical result.
fn load_cells(
    store: &(dyn CellStore + '_),
    coords: &[CellCoord],
) -> Vec<Result<Option<Vec<RuntimeEntity>>, String>> {
    // wasm32 has no thread pool (rayon would build one and panic in a browser),
    // so the web player always takes the serial path — which produces the
    // identical result, `parallel_map_ref` being a pure in-order map.
    #[cfg(not(target_arch = "wasm32"))]
    {
        /// Below this many cells the pool's dispatch costs more than the decode.
        const PARALLEL_THRESHOLD: usize = 2;
        if coords.len() >= PARALLEL_THRESHOLD {
            return inf_core::parallel_map_ref(coords, |c| store.cell(*c));
        }
    }
    coords.iter().map(|c| store.cell(*c)).collect()
}

/// The per-axis distance from a world XZ point to a cell — re-exported so the
/// gate can assert residency bounds without re-deriving the rule.
pub fn distance_to_cell(coord: CellCoord, x: f64, z: f64, cell_size_m: f64) -> f64 {
    cell_distance(coord, x, z, cell_size_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::components::{AlwaysLoaded, MeshRef};

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn rec(g: u128, name: &str, at: (f64, f64)) -> RuntimeEntity {
        let mut e = RuntimeEntity {
            guid: guid(g),
            name: name.into(),
            parent: None,
            transform: Transform::from_translation(glam::DVec3::new(at.0, 0.0, at.1)),
            visible: true,
            mesh: Some(MeshRef::default()),
            material: None,
            light: None,
            camera: None,
            sprite: None,
            tilemap: None,
            nine_slice: None,
            text2d: None,
            light_2d: None,
            rigid_body_2d: None,
            collider_2d: None,
            character_controller_2d: None,
            rigid_body_3d: None,
            collider_3d: None,
            character_controller_3d: None,
            actor: None,
            terrain: None,
            pcg_volume: None,
            skeletal_mesh: None,
            anim_player: None,
            anim_state_machine: None,
            root_motion: None,
            attached_to: None,
            joint_2d: None,
            joint_3d: None,
            audio_source: None,
            audio_listener: None,
            decal: None,
            volume: None,
            spline: None,
            foliage: None,
            streaming_source: None,
            always_loaded: None,
            time_of_day: None,
            sky_atmosphere: None,
            water_body: None,
            buoyancy: None,
            voxel_volume: None,
            destructible: None,
            ik_target: None,
            cloth_sim: None,
            hair_guides: None,
        };
        e.visible = true;
        e
    }

    fn settings() -> PartitionSettings {
        PartitionSettings {
            enabled: true,
            cell_size_m: 100.0,
            activation_radius_m: 10.0,
            prefetch_margin_m: 0.0,
        }
    }

    /// A 3×1 strip of cells plus a persistent manager and a streaming source.
    fn fixture() -> (EcsWorld, CellStreaming) {
        let mut entities = vec![
            RuntimeEntity {
                always_loaded: Some(AlwaysLoaded),
                mesh: None,
                ..rec(0xF0, "GameMode", (0.0, 0.0))
            },
            RuntimeEntity {
                streaming_source: Some(StreamingSource { radius_m: 0.0 }),
                ..rec(0xF1, "Player", (50.0, 0.0))
            },
        ];
        for i in 0..3i32 {
            entities.push(rec(
                0x100 + i as u128,
                "Prop",
                (i as f64 * 100.0 + 50.0, 0.0),
            ));
        }
        let store: Arc<dyn CellStore> =
            Arc::new(MemoryCellStore::from_entities(&entities, &settings()));
        // The level builder spawns the persistent cell; this stands in for it.
        let mut world = EcsWorld::new();
        crate::level::spawn_entities(&mut world, store.persistent().unwrap());
        world.propagate();
        let s = CellStreaming::attach(
            store,
            settings(),
            CellStreamBudget {
                max_prefetch_per_sync: 0,
            },
        );
        (world, s)
    }

    fn move_source(world: &mut EcsWorld, x: f64) {
        let e = world.entity_of(guid(0xF1)).expect("source");
        inf_ecs::sim::set_translation(world, e, inf_ecs::Vec3d::new(x, 0.0, 0.0));
        world.propagate();
    }

    #[test]
    fn attach_spawns_only_the_persistent_cell() {
        let (world, s) = fixture();
        // GameMode (AlwaysLoaded) + Player (StreamingSource) are persistent.
        assert!(world.entity_of(guid(0xF0)).is_some());
        assert!(world.entity_of(guid(0xF1)).is_some());
        // No prop is spawned before the first sync.
        for i in 0..3u128 {
            assert!(world.entity_of(guid(0x100 + i)).is_none());
        }
        assert_eq!(s.stats().cells_resident, 0);
        assert_eq!(s.available().count(), 3);
    }

    #[test]
    fn cells_activate_and_deactivate_as_the_source_walks() {
        let (mut world, mut s) = fixture();

        s.sync_sim(&mut world, 0);
        assert!(world.entity_of(guid(0x100)).is_some(), "cell 0 activated");
        assert!(world.entity_of(guid(0x101)).is_none(), "cell 1 is far");
        assert_eq!(s.stats().cells_resident, 1);
        assert_eq!(
            s.stats().blocking_loads,
            1,
            "no prefetch → activation blocks"
        );

        move_source(&mut world, 150.0);
        s.sync_sim(&mut world, 1);
        assert!(world.entity_of(guid(0x100)).is_none(), "cell 0 evicted");
        assert!(world.entity_of(guid(0x101)).is_some(), "cell 1 activated");
        assert_eq!(s.stats().cells_resident, 1);
        assert_eq!(s.stats().activations, 2);
        assert_eq!(s.stats().deactivations, 1);

        // The persistent cell never moves.
        assert!(world.entity_of(guid(0xF0)).is_some());

        // The trace is exactly what happened, in order.
        assert_eq!(
            s.events(),
            &[
                CellEvent {
                    step: 0,
                    activated: true,
                    coord: (0, 0)
                },
                CellEvent {
                    step: 1,
                    activated: false,
                    coord: (0, 0)
                },
                CellEvent {
                    step: 1,
                    activated: true,
                    coord: (1, 0)
                },
            ]
        );
    }

    #[test]
    fn a_reactivated_cell_respawns_with_its_authored_state() {
        let (mut world, mut s) = fixture();
        s.sync_sim(&mut world, 0);
        let before = world
            .world_translation(world.entity_of(guid(0x100)).unwrap())
            .unwrap();

        move_source(&mut world, 250.0);
        s.sync_sim(&mut world, 1);
        assert!(world.entity_of(guid(0x100)).is_none());

        move_source(&mut world, 50.0);
        s.sync_sim(&mut world, 2);
        let after = world
            .world_translation(world.entity_of(guid(0x100)).unwrap())
            .unwrap();
        assert_eq!(before, after, "a reloaded cell respawns identically");
    }

    #[test]
    fn runtime_spawned_entities_are_never_despawned_by_streaming() {
        let (mut world, mut s) = fixture();
        s.sync_sim(&mut world, 0);
        // A blueprint spawns something inside the resident cell's footprint.
        let spawned = world.spawn_with_guid(guid(0xBEEF), "Bullet", None);
        inf_ecs::sim::set_translation(&mut world, spawned, inf_ecs::Vec3d::new(50.0, 0.0, 0.0));
        world.propagate();

        move_source(&mut world, 250.0);
        s.sync_sim(&mut world, 1);
        assert!(
            world.entity_of(guid(0x100)).is_none(),
            "the authored prop streamed out"
        );
        assert!(
            world.entity_of(guid(0xBEEF)).is_some(),
            "a runtime-spawned entity must survive its cell's eviction"
        );
    }

    #[test]
    fn prefetch_changes_when_work_happens_but_not_what_activates() {
        // Two managers over the same content, one with look-ahead and one
        // without: the residency trace must be identical.
        let run = |margin: f64, budget: usize| {
            let mut entities = vec![RuntimeEntity {
                streaming_source: Some(StreamingSource { radius_m: 0.0 }),
                ..rec(0xF1, "Player", (50.0, 0.0))
            }];
            for i in 0..4i32 {
                entities.push(rec(
                    0x100 + i as u128,
                    "Prop",
                    (i as f64 * 100.0 + 50.0, 0.0),
                ));
            }
            let set = PartitionSettings {
                prefetch_margin_m: margin,
                ..settings()
            };
            let store: Arc<dyn CellStore> =
                Arc::new(MemoryCellStore::from_entities(&entities, &set));
            let mut world = EcsWorld::new();
            crate::level::spawn_entities(&mut world, store.persistent().unwrap());
            world.propagate();
            let mut s = CellStreaming::attach(
                store,
                set,
                CellStreamBudget {
                    max_prefetch_per_sync: budget,
                },
            );
            for step in 0..8u64 {
                move_source(&mut world, 50.0 + step as f64 * 50.0);
                s.sync_sim(&mut world, step);
            }
            (s.events().to_vec(), s.stats().blocking_loads)
        };

        let (cold_events, cold_blocks) = run(0.0, 0);
        let (warm_events, warm_blocks) = run(400.0, 8);
        assert_eq!(
            cold_events, warm_events,
            "the prefetch margin changed WHICH cells activated — it may only change when \
             loading happens"
        );
        assert!(cold_blocks > 0, "the cold run must actually block");
        assert!(
            warm_blocks < cold_blocks,
            "prefetch bought nothing: {warm_blocks} vs {cold_blocks} blocking loads"
        );
    }

    #[test]
    fn a_world_with_no_streaming_source_holds_only_the_persistent_cell() {
        let entities = vec![
            RuntimeEntity {
                always_loaded: Some(AlwaysLoaded),
                mesh: None,
                ..rec(0xF0, "GameMode", (0.0, 0.0))
            },
            rec(0x100, "Prop", (50.0, 0.0)),
        ];
        let store: Arc<dyn CellStore> =
            Arc::new(MemoryCellStore::from_entities(&entities, &settings()));
        let mut world = EcsWorld::new();
        crate::level::spawn_entities(&mut world, store.persistent().unwrap());
        world.propagate();
        let mut s = CellStreaming::attach(store, settings(), CellStreamBudget::default());
        s.sync_sim(&mut world, 0);
        assert_eq!(s.stats().cells_resident, 0, "nothing wants a cell");
        assert!(world.entity_of(guid(0xF0)).is_some());
        assert!(world.entity_of(guid(0x100)).is_none());
    }

    /// **P16.6 — `unresolved_refs` tracks the live cross-cell hazard.**
    ///
    /// A persistent entity is attached to a prop in a far cell. While that cell is
    /// cold the reference is disconnected and the counter says so; walking the
    /// source over reconnects it and the counter falls back to zero. The behaviour
    /// itself is unchanged (the follower simply keeps its transform) — what is new
    /// is that the author can see it.
    #[test]
    fn unresolved_refs_counts_disconnected_cross_cell_references() {
        use inf_ecs::components::AttachedTo;
        use inf_ecs::Vec3d;

        let far = guid(0x102); // the prop in cell (2,0)
        let mut entities = vec![
            RuntimeEntity {
                always_loaded: Some(AlwaysLoaded),
                mesh: None,
                // A persistent follower riding a socket on a STREAMED prop.
                attached_to: Some(AttachedTo::new(far, "", Vec3d::ZERO)),
                ..rec(0xF2, "Turret", (0.0, 0.0))
            },
            RuntimeEntity {
                streaming_source: Some(StreamingSource { radius_m: 0.0 }),
                ..rec(0xF1, "Player", (50.0, 0.0))
            },
            // …and a joint whose other body is a prop in cell (1,0).
            RuntimeEntity {
                always_loaded: Some(AlwaysLoaded),
                joint_3d: Some(inf_ecs::components::Joint3D {
                    other: inf_ecs::refs::EntityRef::new(guid(0x101)),
                    ..Default::default()
                }),
                ..rec(0xF3, "Winch", (0.0, 0.0))
            },
        ];
        for i in 0..3i32 {
            entities.push(rec(
                0x100 + i as u128,
                "Prop",
                (i as f64 * 100.0 + 50.0, 0.0),
            ));
        }
        let store: Arc<dyn CellStore> =
            Arc::new(MemoryCellStore::from_entities(&entities, &settings()));
        let mut world = EcsWorld::new();
        crate::level::spawn_entities(&mut world, store.persistent().unwrap());
        world.propagate();
        let mut s = CellStreaming::attach(store, settings(), CellStreamBudget::default());

        // Step 0: the source is in cell (0,0); both targets are cold.
        s.sync_sim(&mut world, 0);
        assert!(
            world.entity_of(far).is_none(),
            "the far prop is not resident"
        );
        assert_eq!(
            s.stats().unresolved_refs,
            2,
            "both the attachment and the joint are disconnected"
        );
        assert!(s.stats().summary().contains("unresolved ref"));

        // Walk the source into cell (1,0): the joint's body streams in.
        let player = world.entity_of(guid(0xF1)).expect("player");
        inf_ecs::sim::set_translation(&mut world, player, Vec3d::new(150.0, 0.0, 0.0));
        world.propagate();
        s.sync_sim(&mut world, 1);
        assert!(world.entity_of(guid(0x101)).is_some());
        assert_eq!(s.stats().unresolved_refs, 1, "the joint reconnected");

        // …and onto the (1,0)/(2,0) boundary, where BOTH cells are inside the
        // activation radius: every reference reconnects.
        inf_ecs::sim::set_translation(&mut world, player, Vec3d::new(200.0, 0.0, 0.0));
        world.propagate();
        s.sync_sim(&mut world, 2);
        assert!(world.entity_of(far).is_some());
        assert!(world.entity_of(guid(0x101)).is_some());
        assert_eq!(s.stats().unresolved_refs, 0, "everything is connected");

        // Walking away disconnects it again — the counter is live, not sticky.
        inf_ecs::sim::set_translation(&mut world, player, Vec3d::new(50.0, 0.0, 0.0));
        world.propagate();
        s.sync_sim(&mut world, 3);
        assert_eq!(s.stats().unresolved_refs, 2);

        // A world with no such references never counts anything.
        let (mut w2, mut s2) = fixture();
        s2.sync_sim(&mut w2, 0);
        assert_eq!(s2.stats().unresolved_refs, 0);
    }

    #[test]
    fn an_unpartitioned_manager_is_an_inert_no_op() {
        let mut world = EcsWorld::new();
        let mut s = CellStreaming::default();
        assert!(s.is_empty());
        s.sync_sim(&mut world, 0);
        assert_eq!(s.stats(), &CellStreamStats::default());
        assert!(s.events().is_empty());
    }

    #[test]
    fn cell_state_and_bounds_describe_the_grid() {
        let (mut world, mut s) = fixture();
        s.sync_sim(&mut world, 0);
        assert_eq!(s.cell_state((0, 0)), CellState::Active);
        assert_eq!(s.cell_state((2, 0)), CellState::Cold);
        let (min, max) = s.cell_bounds((1, 0));
        assert_eq!(min, [100.0, 0.0]);
        assert_eq!(max, [200.0, 100.0]);
        assert!(s.stats().summary().contains("cell streaming"));
    }

    #[test]
    fn a_source_radius_wider_than_the_level_default_pulls_more_world() {
        let mut entities = vec![RuntimeEntity {
            streaming_source: Some(StreamingSource { radius_m: 250.0 }),
            ..rec(0xF1, "Scout", (50.0, 0.0))
        }];
        for i in 0..4i32 {
            entities.push(rec(
                0x100 + i as u128,
                "Prop",
                (i as f64 * 100.0 + 50.0, 0.0),
            ));
        }
        let store: Arc<dyn CellStore> =
            Arc::new(MemoryCellStore::from_entities(&entities, &settings()));
        let mut world = EcsWorld::new();
        crate::level::spawn_entities(&mut world, store.persistent().unwrap());
        world.propagate();
        let mut s = CellStreaming::attach(store, settings(), CellStreamBudget::default());
        s.sync_sim(&mut world, 0);
        // 250 m around x=50 reaches cells 0..=3 (level default is only 10 m).
        assert_eq!(s.stats().cells_resident, 4);
    }
}
