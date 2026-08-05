//! World partition: the cook-time cell split and the `.inf_part` container
//! (P16.5).
//!
//! A partitioned level does not ship its entities inside the `.inf_lvl` blob.
//! They are binned by world position into a **grid of cells**, plus one
//! **persistent cell** that is always resident, and written to a random-access
//! `.inf_part` asset the runtime pages a cell at a time.
//!
//! ```text
//! ┌ header (64 B, all little-endian) ─────────────────────────────────┐
//! │  magic         [u8; 8]   b"INFPARTN"                              │
//! │  schema_ver    u32       PARTITION_ASSET_SCHEMA_VERSION           │
//! │  reserved      u32       always 0                                 │
//! │  cell_size     f64       cell edge length, metres                 │
//! │  origin        [f64; 3]  world anchor of cell (0,0)'s min corner  │
//! │  cell_count    u32       directory entries (incl. the persistent) │
//! │  entity_count  u32       entities across every cell               │
//! │  blob_base     u64       absolute offset of the blob section      │
//! ├ cell directory (cell_count × 32 B, sorted by CellKey) ────────────┤
//! │  kind u32 · cx i32 · cz i32 · entities u32 · offset u64 · len u64 │
//! ├ blob section (each cell's bincode bytes, 16-byte-aligned) ────────┤
//! │  … cell blobs, zero-padded up to each 16-byte boundary …          │
//! └───────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Why this shape (and why it is not a new pack feature)
//!
//! This is deliberately the **same layout** `.inf_terrain` uses
//! ([`inf_terrain::asset`]), for the same reasons: a flat sorted directory is
//! binary-searchable, the 16-byte blob alignment matches
//! [`inf_asset::BLOB_ALIGN`] so a slice taken out of an mmap'd pack is aligned by
//! address too, and `BTreeMap`-ordered emission with deterministic zero padding
//! makes two builds of one level byte-identical.
//!
//! Each cell blob is a `bincode` `Vec<`[`RuntimeEntity`]`>` in the *same*
//! `standard()` configuration `.inf_lvl` uses — so a cell's bytes are exactly
//! the bytes those entities would have had inside the level payload.
//!
//! ## The kind decision
//!
//! The partition rides in its **own** pack entry
//! ([`inf_asset::AssetKind::Partition`], `.inf_part`), not inside the level's.
//! That is forced, not stylistic: a cell must be reachable through
//! [`PackReader::read_ref`](inf_asset::PackReader::read_ref) as a borrowed slice,
//! which requires an *uncompressed* entry, and `AssetKind::Level` is (correctly)
//! compressed — a level payload is one blob nobody pages. Flipping `Level` to raw
//! to make room would have moved the bytes of every pack ever cooked and broken
//! the non-partitioned regression gate. So `Partition` joins `MeshletMesh` and
//! `Terrain` on the streaming-class side of `PackWriter::compresses_kind`, and
//! `kind_code` stays the exhaustive, append-only table it was.
//!
//! # There is exactly one writer: [`PartitionAssetBuilder::build`]
//!
//! Like [`inf_terrain::TerrainAsset`], [`PartitionAsset`] implements neither
//! [`AssetPayload`](inf_asset::AssetPayload) nor `Serialize`/`Deserialize`: the
//! bytes on disk are the **raw image**, and a generic `inf_asset::encode` would
//! prepend a bincode length varint that knocks every cell off its 16-byte
//! boundary. The door is closed at compile time rather than documented and hoped
//! for.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::RuntimeEntity;

/// Magic at the head of every `.inf_part` payload.
pub const PARTITION_ASSET_MAGIC: [u8; 8] = *b"INFPARTN";

/// Current `.inf_part` payload schema version.
pub const PARTITION_ASSET_SCHEMA_VERSION: u32 = 1;

/// Cell blobs start on multiples of this many bytes (see the module docs) —
/// the same constant, and the same reasoning, as `inf_terrain::TILE_ALIGN`.
pub const CELL_ALIGN: u64 = 16;

/// Bytes of the fixed header.
pub const HEADER_LEN: u64 = 64;

/// Bytes of one cell-directory entry.
pub const DIR_ENTRY_LEN: u64 = 32;

/// Default cell edge length (metres). 256 m is UE's World Partition default and
/// a sane compromise: small enough that one cell is a spawn budget, large enough
/// that a walking character crosses one every few seconds, not every step.
pub const DEFAULT_CELL_SIZE_M: f64 = 256.0;

/// Default activation radius (metres) — one cell in every direction.
pub const DEFAULT_ACTIVATION_RADIUS_M: f64 = 256.0;

/// Default prefetch margin (metres) beyond the activation radius. Loading is
/// allowed to happen anywhere inside `activation + margin`; **activation** is
/// what is deterministic, so this knob buys latency, never a different result.
pub const DEFAULT_PREFETCH_MARGIN_M: f64 = 256.0;

/// Smallest legal cell size. A degenerate (zero/negative/NaN) cell size would
/// make the floor-division binning meaningless, so it is clamped rather than
/// trusted.
pub const MIN_CELL_SIZE_M: f64 = 1.0;

// ── settings ────────────────────────────────────────────────────────────────

/// A level's world-partition configuration (`.inf_lvl` schema v10) — the block
/// the World Settings panel edits and the cook + the runtime both read.
///
/// `enabled: false` (the default) means the level cooks and loads **exactly** as
/// it did before P16.5: no `.inf_part` entry, no cell streaming, byte-identical
/// packs. That is the regression contract.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PartitionSettings {
    /// Whether this level is partitioned at cook time and streamed at runtime.
    #[serde(default)]
    pub enabled: bool,
    /// Cell edge length in **metres** (square cells on the world XZ plane).
    #[serde(default = "default_cell_size_m")]
    pub cell_size_m: f64,
    /// How close a [`StreamingSource`](inf_ecs::components::StreamingSource) must
    /// come to a cell's footprint, in **metres**, before that cell's entities are
    /// spawned. A source's own `radius_m` wins when it is larger.
    #[serde(default = "default_activation_radius_m")]
    pub activation_radius_m: f64,
    /// Extra **metres** beyond the activation radius within which a cell may be
    /// *loaded* (decoded) ahead of need.
    ///
    /// This knob can only change **when** work happens, never **what** the
    /// simulation does: a cell that reaches its activation step unloaded is
    /// loaded synchronously, blocking the step. Widening the margin removes
    /// hitches; it must never move a sim trace, and the gate asserts exactly that.
    #[serde(default = "default_prefetch_margin_m")]
    pub prefetch_margin_m: f64,
}

fn default_cell_size_m() -> f64 {
    DEFAULT_CELL_SIZE_M
}
fn default_activation_radius_m() -> f64 {
    DEFAULT_ACTIVATION_RADIUS_M
}
fn default_prefetch_margin_m() -> f64 {
    DEFAULT_PREFETCH_MARGIN_M
}

impl Default for PartitionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            cell_size_m: DEFAULT_CELL_SIZE_M,
            activation_radius_m: DEFAULT_ACTIVATION_RADIUS_M,
            prefetch_margin_m: DEFAULT_PREFETCH_MARGIN_M,
        }
    }
}

impl PartitionSettings {
    /// The cell size actually used for binning — clamped to [`MIN_CELL_SIZE_M`]
    /// and finite, so a hand-edited or corrupt level cannot produce a nonsense
    /// (or panicking) grid.
    #[inline]
    pub fn effective_cell_size(&self) -> f64 {
        if self.cell_size_m.is_finite() && self.cell_size_m >= MIN_CELL_SIZE_M {
            self.cell_size_m
        } else {
            MIN_CELL_SIZE_M.max(DEFAULT_CELL_SIZE_M.min(self.cell_size_m.abs()))
        }
    }

    /// The activation radius actually used — non-negative and finite.
    #[inline]
    pub fn effective_activation_radius(&self) -> f64 {
        if self.activation_radius_m.is_finite() && self.activation_radius_m >= 0.0 {
            self.activation_radius_m
        } else {
            DEFAULT_ACTIVATION_RADIUS_M
        }
    }

    /// The prefetch margin actually used — non-negative and finite.
    #[inline]
    pub fn effective_prefetch_margin(&self) -> f64 {
        if self.prefetch_margin_m.is_finite() && self.prefetch_margin_m >= 0.0 {
            self.prefetch_margin_m
        } else {
            DEFAULT_PREFETCH_MARGIN_M
        }
    }
}

// ── cell identity ───────────────────────────────────────────────────────────

/// A cell's grid coordinate on the world XZ plane (floor-division of world
/// metres by the cell size — the `TileKey` precedent).
pub type CellCoord = (i32, i32);

/// Which cell a blob belongs to.
///
/// `Ord` puts the persistent cell **first** (`kind = 0`), then grid cells in
/// `(cx, cz)` order. That ordering is the directory's order, the activation
/// order, and the trace order — one comparison drives all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellKey {
    /// `0` = the persistent cell, `1` = a grid cell. A plain integer (rather
    /// than a `bool`) because it is what the directory stores and what the sort
    /// key is built from.
    kind: u32,
    /// Grid coordinate; `(0, 0)` and meaningless for the persistent cell.
    coord: CellCoord,
}

impl CellKey {
    /// The persistent cell — always resident, never streamed.
    pub const PERSISTENT: CellKey = CellKey {
        kind: 0,
        coord: (0, 0),
    };

    /// A grid cell at `coord`.
    #[inline]
    pub fn grid(coord: CellCoord) -> Self {
        Self { kind: 1, coord }
    }

    /// Whether this is the persistent cell.
    #[inline]
    pub fn is_persistent(self) -> bool {
        self.kind == 0
    }

    /// The grid coordinate (`None` for the persistent cell).
    #[inline]
    pub fn coord(self) -> Option<CellCoord> {
        (!self.is_persistent()).then_some(self.coord)
    }

    /// The raw directory fields (`kind`, `cx`, `cz`).
    #[inline]
    pub fn raw(self) -> (u32, i32, i32) {
        (self.kind, self.coord.0, self.coord.1)
    }
}

impl std::fmt::Display for CellKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.coord() {
            Some((x, z)) => write!(f, "cell({x},{z})"),
            None => write!(f, "cell(persistent)"),
        }
    }
}

/// The cell a world XZ position falls in — floor division, so the grid is
/// continuous through the origin and across negative coordinates (the `TileKey`
/// precedent, and the reason a plain `as i32` truncation would be wrong).
#[inline]
pub fn cell_of(x: f64, z: f64, cell_size_m: f64) -> CellCoord {
    let s = if cell_size_m.is_finite() && cell_size_m >= MIN_CELL_SIZE_M {
        cell_size_m
    } else {
        MIN_CELL_SIZE_M
    };
    // A NaN/±∞ coordinate has no cell. Answering `(0, 0)` is the deterministic
    // choice: it is stable across machines, and a corrupt transform lands
    // somewhere reachable rather than 2 billion cells away where nothing would
    // ever activate it (Rust's saturating float→int cast would silently give it
    // `0` anyway; this states the intent).
    let axis = |v: f64| {
        if !v.is_finite() {
            return 0;
        }
        (v / s).floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32
    };
    (axis(x), axis(z))
}

/// The XZ distance (metres) from `(x, z)` to a cell's footprint — `0.0` inside
/// it. **Per-axis (Chebyshev)**, matching how the terrain streamer bounds its
/// wants: the want set is the cells covering the axis-aligned box `p ± radius`,
/// so a corner cell is legitimately `radius·√2` away in Euclidean terms.
#[inline]
pub fn cell_distance(coord: CellCoord, x: f64, z: f64, cell_size_m: f64) -> f64 {
    let s = if cell_size_m.is_finite() && cell_size_m >= MIN_CELL_SIZE_M {
        cell_size_m
    } else {
        MIN_CELL_SIZE_M
    };
    let min_x = coord.0 as f64 * s;
    let min_z = coord.1 as f64 * s;
    let dx = (min_x - x).max(x - (min_x + s)).max(0.0);
    let dz = (min_z - z).max(z - (min_z + s)).max(0.0);
    dx.max(dz)
}

/// Every grid cell whose footprint comes within `radius_m` of `(x, z)`, in
/// ascending order. A pure function of its arguments — no clock, no camera, no
/// iteration order to leak.
pub fn cells_within(
    x: f64,
    z: f64,
    radius_m: f64,
    cell_size_m: f64,
    out: &mut BTreeSet<CellCoord>,
) {
    if !(x.is_finite() && z.is_finite() && radius_m.is_finite()) || radius_m < 0.0 {
        return;
    }
    let s = if cell_size_m.is_finite() && cell_size_m >= MIN_CELL_SIZE_M {
        cell_size_m
    } else {
        MIN_CELL_SIZE_M
    };
    let (lo_x, lo_z) = cell_of(x - radius_m, z - radius_m, s);
    let (hi_x, hi_z) = cell_of(x + radius_m, z + radius_m, s);
    // Guard against an absurd radius asking for billions of cells.
    let span_x = (hi_x as i64 - lo_x as i64).unsigned_abs();
    let span_z = (hi_z as i64 - lo_z as i64).unsigned_abs();
    if span_x
        .saturating_add(1)
        .saturating_mul(span_z.saturating_add(1))
        > MAX_WANT_CELLS
    {
        return;
    }
    for cz in lo_z..=hi_z {
        for cx in lo_x..=hi_x {
            out.insert((cx, cz));
        }
    }
}

/// Ceiling on how many cells one [`cells_within`] call may enumerate. A radius
/// wide enough to exceed it is a content bug (or a corrupt level), and silently
/// allocating a billion coordinates would be a far worse failure than wanting
/// nothing and saying so upstream.
const MAX_WANT_CELLS: u64 = 64 * 1024;

// ── the cook-time split ─────────────────────────────────────────────────────

/// The result of binning a level's entities.
///
/// `cells` is a `BTreeMap`, and within a cell entities keep the level's original
/// (parents-first) order, so the plan is a pure function of the entity list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PartitionPlan {
    /// Entities that exist for the whole run (see [`persistent_reason`]).
    pub persistent: Vec<RuntimeEntity>,
    /// Streamed grid cells, keyed by coordinate.
    pub cells: BTreeMap<CellCoord, Vec<RuntimeEntity>>,
}

impl PartitionPlan {
    /// Total entities across every cell.
    pub fn entity_count(&self) -> usize {
        self.persistent.len() + self.cells.values().map(Vec::len).sum::<usize>()
    }

    /// Every key present, persistent first.
    pub fn keys(&self) -> impl Iterator<Item = CellKey> + '_ {
        std::iter::once(CellKey::PERSISTENT).chain(self.cells.keys().copied().map(CellKey::grid))
    }

    /// The entities of one cell.
    pub fn cell(&self, key: CellKey) -> Option<&[RuntimeEntity]> {
        match key.coord() {
            None => Some(&self.persistent),
            Some(c) => self.cells.get(&c).map(Vec::as_slice),
        }
    }
}

/// Why an entity is not streamed. Returned by [`persistent_reason`] so the cook
/// report (and a future editor overlay) can explain a partition rather than
/// merely apply it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistentReason {
    /// Carries the [`AlwaysLoaded`](inf_ecs::components::AlwaysLoaded) marker.
    AlwaysLoaded,
    /// Carries [`StreamingSource`](inf_ecs::components::StreamingSource) —
    /// residency is derived *from* this entity, so streaming it out would be a
    /// bootstrap paradox: nothing would remain to reactivate its own cell.
    StreamingSource,
    /// Occupies no space: no renderable, physical, audio, animation or world
    /// component at all. A manager / settings / event-hub entity, which has a
    /// transform only because every entity does.
    Unplaced,
    /// A member of this entity's hierarchy is persistent for one of the reasons
    /// above, and a hierarchy always streams as one unit.
    HierarchyMember,
}

/// Whether an entity **occupies space** — the component-based rule the
/// partitioner uses to tell a placed actor from a manager.
///
/// A placed entity carries at least one component whose meaning is spatial:
///
/// * renderable — `mesh`, `sprite`, `tilemap`, `nine_slice`, `text2d`,
///   `skeletal_mesh`, `foliage`, `decal`, `spline`;
/// * lighting / view — `light`, `light_2d`, `camera`;
/// * physical — the six 2D/3D body / collider / controller slots, both joint
///   slots, `volume`;
/// * world — `terrain`, `pcg_volume`;
/// * audible — `audio_source`, `audio_listener`;
/// * animated / attached — `anim_player`, `anim_state_machine`, `root_motion`,
///   `attached_to`.
///
/// Everything else — an entity with a name, a transform and at most an `actor`
/// blueprint binding — is **unplaced**: its position means nothing to anybody,
/// so binning it into a cell would be an arbitrary decision with real gameplay
/// consequences (a game-mode manager that despawns when the player walks away).
/// Those go to the persistent cell.
///
/// Note `actor` deliberately does **not** make an entity placed. A blueprint on a
/// mesh streams with the mesh; a blueprint on a bare transform is a manager.
///
/// Nor do the v11 sky slots (`time_of_day` / `sky_atmosphere`, P17.1). The level
/// clock is a world-wide singleton whose transform means nothing, so it falls
/// through to [`PersistentReason::Unplaced`] and cooks into the persistent cell —
/// which is the only correct answer: a sun that streams out when the player walks
/// away would take the whole sky with it.
pub fn occupies_space(e: &RuntimeEntity) -> bool {
    e.mesh.is_some()
        || e.sprite.is_some()
        || e.tilemap.is_some()
        || e.nine_slice.is_some()
        || e.text2d.is_some()
        || e.skeletal_mesh.is_some()
        || e.foliage.is_some()
        || e.decal.is_some()
        || e.spline.is_some()
        || e.light.is_some()
        || e.light_2d.is_some()
        || e.camera.is_some()
        || e.rigid_body_2d.is_some()
        || e.collider_2d.is_some()
        || e.character_controller_2d.is_some()
        || e.rigid_body_3d.is_some()
        || e.collider_3d.is_some()
        || e.character_controller_3d.is_some()
        || e.joint_2d.is_some()
        || e.joint_3d.is_some()
        || e.volume.is_some()
        || e.terrain.is_some()
        || e.pcg_volume.is_some()
        || e.audio_source.is_some()
        || e.audio_listener.is_some()
        || e.anim_player.is_some()
        || e.anim_state_machine.is_some()
        || e.root_motion.is_some()
        || e.attached_to.is_some()
}

/// Why this **single** entity would be persistent, ignoring its hierarchy.
/// [`partition_entities`] then promotes whole trees ([`PersistentReason::HierarchyMember`]).
pub fn persistent_reason(e: &RuntimeEntity) -> Option<PersistentReason> {
    if e.always_loaded.is_some() {
        return Some(PersistentReason::AlwaysLoaded);
    }
    if e.streaming_source.is_some() {
        return Some(PersistentReason::StreamingSource);
    }
    if !occupies_space(e) {
        return Some(PersistentReason::Unplaced);
    }
    None
}

/// Bin a level's entities into the persistent cell + a grid of streamed cells.
///
/// **A pure function of `entities` and `settings`** — the whole determinism story
/// rests on that, because the *same* function runs in two places: the cook (which
/// writes the result to `.inf_part`) and the runtime's in-memory path (a loose
/// `.inf_lvl`, or the PIE handoff, which has no cooked asset). One binning rule,
/// two sources, therefore PIE == shipping by construction rather than by
/// agreement.
///
/// ## The rules, in order
///
/// 1. **A hierarchy is one unit.** Every entity is assigned the cell of its
///    **root ancestor**, so a parent and its children never split. Splitting them
///    would mean spawning a child whose parent does not exist — `populate_world`
///    resolves parents by GUID *within the batch it is given*, so a cross-cell
///    parent link would silently drop the child to the world root.
/// 2. **A tree with any persistent member is wholly persistent** (rule 1's
///    corollary): if the root, or any descendant, hits [`persistent_reason`], the
///    entire tree lands in the persistent cell.
/// 3. **Otherwise the root's world XZ decides**, by floor division
///    ([`cell_of`]). A root's local translation *is* its world translation, which
///    is why the root is the one that is asked.
/// 4. **Order is preserved** inside every cell (the level's parents-first
///    order), so a cell spawns exactly as the level would have.
///
/// A cycle in the parent links (which the editor's reparent guard makes
/// unreachable, but a hand-forged file could contain) resolves to the entity
/// itself rather than looping.
pub fn partition_entities(
    entities: &[RuntimeEntity],
    settings: &PartitionSettings,
) -> PartitionPlan {
    let cell_size = settings.effective_cell_size();

    // guid → index, for the root walk.
    let index: BTreeMap<Uuid, usize> = entities
        .iter()
        .enumerate()
        .map(|(i, e)| (e.guid, i))
        .collect();

    // Root ancestor of each entity (depth-guarded against a forged cycle).
    let root_of: Vec<usize> = (0..entities.len())
        .map(|start| {
            let mut cur = start;
            for _ in 0..entities.len() {
                let Some(parent) = entities[cur].parent else {
                    break;
                };
                let Some(&next) = index.get(&parent) else {
                    break; // dangling parent → this entity is its own root
                };
                if next == cur {
                    break;
                }
                cur = next;
            }
            cur
        })
        .collect();

    // Which roots are persistent: the root's own reason, or any member's.
    let mut persistent_roots: BTreeSet<usize> = BTreeSet::new();
    for (i, e) in entities.iter().enumerate() {
        if persistent_reason(e).is_some() {
            persistent_roots.insert(root_of[i]);
        }
    }

    let mut plan = PartitionPlan::default();
    for (i, e) in entities.iter().enumerate() {
        let root = root_of[i];
        if persistent_roots.contains(&root) {
            plan.persistent.push(e.clone());
            continue;
        }
        let t = entities[root].transform.translation;
        let coord = cell_of(t.x, t.z, cell_size);
        plan.cells.entry(coord).or_default().push(e.clone());
    }
    plan
}

/// A cook-time reference that crosses a cell boundary — the honest surfacing of a
/// dangling-at-runtime hazard (the `dangling_terrain_refs` precedent).
///
/// An entity in cell A naming an entity in cell B is legal to author and legal to
/// cook, but at runtime the target may simply not be spawned when the referrer
/// is: the reference resolves to nothing. v1 does **not** promote either cell —
/// promoting would make residency depend on the reference graph, which is exactly
/// the kind of implicit coupling that turns a streaming budget into a guess.
/// Instead the cook says so and the author decides (mark one `AlwaysLoaded`, or
/// reparent them into one hierarchy, which rule 1 keeps together).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CrossCellRef {
    /// The referring entity.
    pub from: Uuid,
    /// Its cell.
    pub from_cell: CellKey,
    /// The referenced entity.
    pub to: Uuid,
    /// The target's cell.
    pub to_cell: CellKey,
    /// Which component field carries the reference (`"joint_3d.other"`, …).
    pub field: &'static str,
}

/// Every entity-to-entity reference in `plan` that crosses a cell boundary.
///
/// The fields checked are the persisted entity refs: `joint_2d.other`,
/// `joint_3d.other` and `attached_to.target`. (Hierarchy parent links cannot
/// cross a boundary — rule 1 of [`partition_entities`] — and asset GUID refs are
/// not entity refs.) A reference **into** the persistent cell is never reported:
/// that target is always resident, which is the sanctioned way to fix one.
/// Sorted + deduplicated, so a cook report stays deterministic.
pub fn cross_cell_refs(plan: &PartitionPlan) -> Vec<CrossCellRef> {
    let mut cell_of_entity: BTreeMap<Uuid, CellKey> = BTreeMap::new();
    for e in &plan.persistent {
        cell_of_entity.insert(e.guid, CellKey::PERSISTENT);
    }
    for (&coord, list) in &plan.cells {
        for e in list {
            cell_of_entity.insert(e.guid, CellKey::grid(coord));
        }
    }

    let mut out: BTreeSet<CrossCellRef> = BTreeSet::new();
    let mut check = |from: Uuid, to: Option<Uuid>, field: &'static str| {
        let (Some(to), Some(&from_cell)) = (to, cell_of_entity.get(&from)) else {
            return;
        };
        let Some(&to_cell) = cell_of_entity.get(&to) else {
            return; // a ref to an entity the level does not contain: not our story
        };
        // Persistent targets are always resident; persistent referrers reach
        // streamed targets, which IS a hazard and is reported.
        if to_cell.is_persistent() || from_cell == to_cell {
            return;
        }
        out.insert(CrossCellRef {
            from,
            from_cell,
            to,
            to_cell,
            field,
        });
    };

    let all = plan.persistent.iter().chain(plan.cells.values().flatten());
    for e in all {
        if let Some(j) = &e.joint_2d {
            check(e.guid, j.other.get(), "joint_2d.other");
        }
        if let Some(j) = &e.joint_3d {
            check(e.guid, j.other.get(), "joint_3d.other");
        }
        if let Some(a) = &e.attached_to {
            check(e.guid, Some(a.target), "attached_to.target");
        }
    }
    out.into_iter().collect()
}

/// A Blueprint-bearing entity that landed in a **streamed** cell — and therefore
/// will never tick.
///
/// See [`streamed_actors`] for why this is a diagnostic rather than a fixup.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StreamedActor {
    /// The entity carrying the `ActorClass` binding.
    pub entity: Uuid,
    /// The streamed cell it was binned into.
    pub cell: CellKey,
    /// The `.inf_act` blueprint-class asset GUID it is bound to.
    pub class: Uuid,
}

/// Every entity in a **streamed** cell that carries a Blueprint `actor` binding.
///
/// # Why this has to be a cook diagnostic
///
/// The runtime assigns Blueprint actor ids **once**, at `RuntimeSim`
/// construction, from the entities present then — which is the persistent cell.
/// An entity that streams in later is fully live in every other respect (mesh,
/// colliders, lights, audio, animation) but its `ActorClass` **never runs**.
///
/// That combination — a mesh plus an `ActorClass` — is not an exotic authoring
/// mistake, it is the single most common gameplay shape there is: an enemy, a
/// door, a pickup, a patrolling guard. Placed in a world with partitioning on, it
/// is *guaranteed* to bin into a streamed cell (a mesh occupies space, so
/// [`occupies_space`] admits it) and *guaranteed* to sit there inert. Nothing
/// crashes; the entity simply stands still forever. Without this warning the only
/// signal is a designer noticing that their level is full of statues.
///
/// v1 does **not** auto-promote such an entity to the persistent cell. Promotion
/// would make residency depend on which entities happen to carry a script — so a
/// level's memory ceiling would silently move every time gameplay logic was added,
/// which is exactly the kind of implicit coupling that turns a streaming budget
/// into a guess. The cook says so, names the remedy, and the author decides:
/// mark it [`AlwaysLoaded`](inf_ecs::components::AlwaysLoaded).
///
/// Sorted + deduplicated (by `BTreeSet`), so a cook report stays deterministic.
///
/// **This diagnostic retires the day the actor map becomes dynamic** — at which
/// point the negative test in `runtime/inf-player/tests/partitioned_world.rs`
/// fails first, which is the intended order.
pub fn streamed_actors(plan: &PartitionPlan) -> Vec<StreamedActor> {
    let mut out: BTreeSet<StreamedActor> = BTreeSet::new();
    for (&coord, list) in &plan.cells {
        for e in list {
            if let Some(class) = e.actor {
                out.insert(StreamedActor {
                    entity: e.guid,
                    cell: CellKey::grid(coord),
                    class,
                });
            }
        }
    }
    out.into_iter().collect()
}

/// A `Terrain` entity that landed in a **streamed** cell — and therefore takes
/// the ground with it when that cell deactivates.
///
/// See [`streamed_terrains`] for why this is a diagnostic rather than a fixup.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StreamedTerrain {
    /// The entity carrying the `Terrain` component.
    pub entity: Uuid,
    /// The streamed cell it was binned into.
    pub cell: CellKey,
    /// Its `.inf_terrain` asset GUID, if it streams from one (`None` = inline).
    pub asset: Option<Uuid>,
}

/// Every entity in a **streamed** cell that carries a `Terrain` component.
///
/// # Why this has to be a cook diagnostic
///
/// A terrain "occupies space" ([`occupies_space`]) — it has a transform and it is
/// visible, so by the ordinary rule it bins into the one cell containing its
/// entity origin. But a heightfield is not *at* its origin: it spans kilometres
/// from it, and a streamed one spans as much of the world as its `.inf_terrain`
/// covers. So the cell that owns it is a single 128 m–2 km square, and the moment
/// the player walks out of that square **the ground despawns under them** — the
/// terrain entity is gone, `terrain.height_at` reads the unauthored `0.0`
/// everywhere, and the whole world renders as sky. Nothing crashes; the level
/// simply stops having a floor.
///
/// That is not an exotic authoring mistake either: enabling partitioning on a
/// level that already has terrain produces it every single time, with no symptom
/// until somebody walks far enough.
///
/// v1 does **not** auto-promote a terrain to the persistent cell, for the same
/// reason [`streamed_actors`] does not promote a Blueprint actor: a fixup here
/// would make residency depend on which *components* an entity happens to carry,
/// so a level's memory ceiling would move silently as content is added. It is
/// also not always wrong to stream a terrain — a small inline patch of ground
/// genuinely local to one cell is a legitimate (if unusual) thing to author. The
/// cook says what will happen, names the remedy, and the author decides: mark it
/// [`AlwaysLoaded`](inf_ecs::components::AlwaysLoaded).
///
/// Sorted + deduplicated (by `BTreeSet`), so a cook report stays deterministic.
///
/// **This diagnostic narrows the day terrain entities carry their real world
/// extent into the partitioner** — at which point a terrain could be binned by its
/// footprint (or split across cells) instead of by its origin, and only a terrain
/// that really is cell-local would stream.
pub fn streamed_terrains(plan: &PartitionPlan) -> Vec<StreamedTerrain> {
    let mut out: BTreeSet<StreamedTerrain> = BTreeSet::new();
    for (&coord, list) in &plan.cells {
        for e in list {
            if let Some(t) = &e.terrain {
                out.insert(StreamedTerrain {
                    entity: e.guid,
                    cell: CellKey::grid(coord),
                    asset: t.asset,
                });
            }
        }
    }
    out.into_iter().collect()
}

// ── errors ──────────────────────────────────────────────────────────────────

/// A failure building or reading a `.inf_part` payload.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PartitionAssetError {
    #[error("partition asset is shorter than its fixed header")]
    TooShort,
    #[error("not an .inf_part payload (bad magic)")]
    BadMagic,
    #[error("partition asset schema v{found} is newer than this build (v{current})")]
    SchemaTooNew { found: u32, current: u32 },
    #[error("partition asset is malformed: {0}")]
    Malformed(String),
    #[error("partition cell {key} failed to decode: {message}")]
    CellDecode { key: CellKey, message: String },
    #[error("partition cell {key} added twice")]
    DuplicateCell { key: CellKey },
}

type Result<T> = std::result::Result<T, PartitionAssetError>;

/// Round `n` up to the next multiple of [`CELL_ALIGN`].
#[inline]
fn align_up(n: u64) -> u64 {
    n.next_multiple_of(CELL_ALIGN)
}

/// The bincode configuration cell blobs use — the **same** `standard()` the
/// `.inf_lvl` path uses, so an entity's bytes are identical in both containers.
fn cell_config() -> impl bincode::config::Config {
    bincode::config::standard()
}

/// Encode one cell's entity list to its canonical blob bytes.
pub fn encode_cell(entities: &[RuntimeEntity]) -> Result<Vec<u8>> {
    bincode::serde::encode_to_vec(entities, cell_config())
        .map_err(|e| PartitionAssetError::Malformed(format!("cell encode: {e}")))
}

/// Decode one cell's entity list from its canonical blob bytes.
pub fn decode_cell(bytes: &[u8]) -> std::result::Result<Vec<RuntimeEntity>, String> {
    bincode::serde::decode_from_slice(bytes, cell_config())
        .map(|(v, _)| v)
        .map_err(|e| e.to_string())
}

// ── header + directory ──────────────────────────────────────────────────────

/// The decoded fixed header of a `.inf_part` payload.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartitionAssetHeader {
    /// Payload schema version.
    pub schema_version: u32,
    /// Cell edge length in metres.
    pub cell_size_m: f64,
    /// World anchor of cell `(0, 0)`'s minimum corner (zero in v1 — the grid is
    /// anchored at the world origin; the field exists so a future rebased world
    /// does not need a format bump).
    pub origin: [f64; 3],
    /// Number of directory entries (including the persistent cell).
    pub cell_count: u32,
    /// Entities across every cell.
    pub entity_count: u32,
    /// Absolute offset of the blob section within the payload.
    pub blob_base: u64,
}

/// One cell-directory entry: where a cell's blob lives inside the payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellDirEntry {
    /// Which cell this is.
    pub key: CellKey,
    /// How many entities the blob holds (so a caller can budget before decoding).
    pub entity_count: u32,
    /// Absolute offset of the blob **within the payload** (a multiple of
    /// [`CELL_ALIGN`]) — usable directly as a `seek` target on a loose file.
    pub offset: u64,
    /// Blob length in bytes (unpadded).
    pub len: u64,
}

// ── builder ─────────────────────────────────────────────────────────────────

/// Assembles a `.inf_part` payload cell by cell.
///
/// Cells land in a `BTreeMap` keyed by [`CellKey`], so insertion order cannot
/// affect the output — [`build`](Self::build) is a pure function of the cell set.
#[derive(Debug, Clone)]
pub struct PartitionAssetBuilder {
    cell_size_m: f64,
    origin: [f64; 3],
    cells: BTreeMap<CellKey, (u32, Vec<u8>)>,
}

impl PartitionAssetBuilder {
    /// A builder for a grid of the given cell size, anchored at the world origin.
    pub fn new(cell_size_m: f64) -> Self {
        Self {
            cell_size_m: if cell_size_m.is_finite() && cell_size_m >= MIN_CELL_SIZE_M {
                cell_size_m
            } else {
                DEFAULT_CELL_SIZE_M
            },
            origin: [0.0; 3],
            cells: BTreeMap::new(),
        }
    }

    /// Add a cell, encoding its entity list to the canonical blob. Errors on a
    /// duplicate key.
    pub fn insert(&mut self, key: CellKey, entities: &[RuntimeEntity]) -> Result<()> {
        let bytes = encode_cell(entities)?;
        let count = u32::try_from(entities.len())
            .map_err(|_| PartitionAssetError::Malformed("more than u32::MAX entities".into()))?;
        if self.cells.insert(key, (count, bytes)).is_some() {
            return Err(PartitionAssetError::DuplicateCell { key });
        }
        Ok(())
    }

    /// Number of cells staged.
    pub fn len(&self) -> usize {
        self.cells.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Serialize the payload image.
    pub fn build(self) -> Result<PartitionAsset> {
        let count = u32::try_from(self.cells.len())
            .map_err(|_| PartitionAssetError::Malformed("more than u32::MAX cells".into()))?;
        let entity_total: u64 = self.cells.values().map(|(n, _)| *n as u64).sum();
        let entity_count = u32::try_from(entity_total)
            .map_err(|_| PartitionAssetError::Malformed("more than u32::MAX entities".into()))?;

        // Offsets first: the blob section starts after the header + full
        // directory (both already multiples of CELL_ALIGN, so no padding is ever
        // needed there — `align_up` states the invariant rather than relying on
        // it), and every cell after it starts on the next aligned boundary.
        let blob_base = align_up(HEADER_LEN + DIR_ENTRY_LEN * count as u64);
        let mut offsets = Vec::with_capacity(self.cells.len());
        let mut offset = blob_base;
        for (_, blob) in self.cells.values() {
            offsets.push(offset);
            offset = align_up(offset + blob.len() as u64);
        }
        let total = offset;

        let mut out = Vec::with_capacity(total as usize);
        out.extend_from_slice(&PARTITION_ASSET_MAGIC);
        out.extend_from_slice(&PARTITION_ASSET_SCHEMA_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved, always zero
        out.extend_from_slice(&self.cell_size_m.to_le_bytes());
        for v in self.origin {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&count.to_le_bytes());
        out.extend_from_slice(&entity_count.to_le_bytes());
        out.extend_from_slice(&blob_base.to_le_bytes());
        debug_assert_eq!(out.len() as u64, HEADER_LEN);

        for ((key, (n, blob)), &off) in self.cells.iter().zip(&offsets) {
            let (kind, cx, cz) = key.raw();
            out.extend_from_slice(&kind.to_le_bytes());
            out.extend_from_slice(&cx.to_le_bytes());
            out.extend_from_slice(&cz.to_le_bytes());
            out.extend_from_slice(&n.to_le_bytes());
            out.extend_from_slice(&off.to_le_bytes());
            out.extend_from_slice(&(blob.len() as u64).to_le_bytes());
        }
        debug_assert_eq!(out.len() as u64, HEADER_LEN + DIR_ENTRY_LEN * count as u64);

        // Blob section, zero-padded up to each offset (and to the end, so the
        // payload length is itself a multiple of CELL_ALIGN).
        for ((_, blob), &off) in self.cells.values().zip(&offsets) {
            out.resize(off as usize, 0);
            out.extend_from_slice(blob);
        }
        out.resize(total as usize, 0);

        PartitionAsset::from_bytes(out)
    }
}

/// Build a `.inf_part` payload from a binned [`PartitionPlan`].
///
/// The persistent cell is **always** written, even when empty: a partitioned
/// level with no persistent entities is legal, and a directory that sometimes
/// omits an entry is a directory whose reader has to guess.
pub fn build_partition_asset(
    plan: &PartitionPlan,
    settings: &PartitionSettings,
) -> Result<PartitionAsset> {
    let mut b = PartitionAssetBuilder::new(settings.effective_cell_size());
    b.insert(CellKey::PERSISTENT, &plan.persistent)?;
    for (&coord, list) in &plan.cells {
        b.insert(CellKey::grid(coord), list)?;
    }
    b.build()
}

// ── the payload ─────────────────────────────────────────────────────────────

/// A validated `.inf_part` payload image.
///
/// Owns the bytes; [`reader`](Self::reader) borrows a random-access view over
/// them. See the module docs for why [`as_bytes`](Self::as_bytes) — not
/// `inf_asset::encode` — is what goes into a pack.
#[derive(Clone, PartialEq)]
pub struct PartitionAsset {
    bytes: Vec<u8>,
    header: PartitionAssetHeader,
}

impl PartitionAsset {
    /// The current payload schema version (published beside the bytes, since the
    /// generic `AssetPayload` door is deliberately closed — see the module docs).
    pub const SCHEMA_VERSION: u32 = PARTITION_ASSET_SCHEMA_VERSION;

    /// Validate and take ownership of a payload image.
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self> {
        let header = parse(&bytes)?.0;
        Ok(Self { bytes, header })
    }

    /// The canonical payload bytes — what is packed as a `.inf_part` entry.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume into the payload bytes.
    #[inline]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// The decoded header.
    #[inline]
    pub fn header(&self) -> &PartitionAssetHeader {
        &self.header
    }

    /// A random-access view over the payload.
    pub fn reader(&self) -> PartitionAssetReader<&[u8]> {
        // Already validated in `from_bytes`; re-parsing cannot fail.
        PartitionAssetReader::new(self.bytes.as_slice()).expect("validated payload")
    }
}

impl std::fmt::Debug for PartitionAsset {
    /// Summarizes; never dumps the (possibly enormous) cell blobs.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartitionAsset")
            .field("schema_version", &self.header.schema_version)
            .field("cell_size_m", &self.header.cell_size_m)
            .field("cell_count", &self.header.cell_count)
            .field("entity_count", &self.header.entity_count)
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

// ── reader ──────────────────────────────────────────────────────────────────

/// Random access over a `.inf_part` payload.
///
/// Generic over the byte source so the *same* reader serves every backing: an
/// owned `Vec<u8>` (a loose file read), a `&[u8]` borrowed from a
/// [`PackReader::read_ref`](inf_asset::PackReader::read_ref) mapping, or a `Cow`
/// straight off it.
///
/// The header + directory are parsed and validated once at construction; every
/// [`cell_bytes`](Self::cell_bytes) after that is a binary search plus a slice —
/// no decode, no allocation.
#[derive(Debug, Clone)]
pub struct PartitionAssetReader<B> {
    bytes: B,
    header: PartitionAssetHeader,
    dir: Vec<CellDirEntry>,
}

/// A [`PartitionAssetReader`] borrowing its bytes (the pack-mapping case).
pub type PartitionAssetView<'a> = PartitionAssetReader<&'a [u8]>;

impl<B: AsRef<[u8]>> PartitionAssetReader<B> {
    /// Parse + validate a payload. Rejects a bad magic, a newer schema, an
    /// out-of-bounds, misaligned or overlapping blob, and an out-of-order
    /// directory (which would break both the binary search and the
    /// byte-determinism guarantee).
    pub fn new(bytes: B) -> Result<Self> {
        let (header, dir) = parse(bytes.as_ref())?;
        Ok(Self { bytes, header, dir })
    }

    /// The whole payload image.
    #[inline]
    pub fn bytes(&self) -> &[u8] {
        self.bytes.as_ref()
    }

    /// The decoded header.
    #[inline]
    pub fn header(&self) -> &PartitionAssetHeader {
        &self.header
    }

    /// Cell edge length in metres.
    #[inline]
    pub fn cell_size_m(&self) -> f64 {
        self.header.cell_size_m
    }

    /// Number of cells (including the persistent one).
    #[inline]
    pub fn cell_count(&self) -> usize {
        self.dir.len()
    }
    pub fn is_empty(&self) -> bool {
        self.dir.is_empty()
    }

    /// Entities across every cell.
    #[inline]
    pub fn entity_count(&self) -> u32 {
        self.header.entity_count
    }

    /// The cell directory, persistent-first then in `(cx, cz)` order.
    #[inline]
    pub fn directory(&self) -> &[CellDirEntry] {
        &self.dir
    }

    /// Every key present, in directory order.
    pub fn keys(&self) -> impl Iterator<Item = CellKey> + '_ {
        self.dir.iter().map(|e| e.key)
    }

    /// Every **streamed** grid coordinate present, ascending.
    pub fn grid_coords(&self) -> impl Iterator<Item = CellCoord> + '_ {
        self.dir.iter().filter_map(|e| e.key.coord())
    }

    /// The directory entry for `key`.
    pub fn entry(&self, key: CellKey) -> Option<&CellDirEntry> {
        self.dir
            .binary_search_by(|e| e.key.cmp(&key))
            .ok()
            .map(|i| &self.dir[i])
    }

    /// A cell's blob, **borrowed from the payload** — 16-byte aligned within it.
    pub fn cell_bytes(&self, key: CellKey) -> Option<&[u8]> {
        let e = self.entry(key)?;
        // Bounds + alignment were validated in `new`.
        Some(&self.bytes.as_ref()[e.offset as usize..(e.offset + e.len) as usize])
    }

    /// Decode a cell's entity list.
    pub fn cell(&self, key: CellKey) -> Result<Option<Vec<RuntimeEntity>>> {
        let Some(bytes) = self.cell_bytes(key) else {
            return Ok(None);
        };
        decode_cell(bytes)
            .map(Some)
            .map_err(|message| PartitionAssetError::CellDecode { key, message })
    }
}

/// Parse + validate the header and directory of a payload image.
fn parse(data: &[u8]) -> Result<(PartitionAssetHeader, Vec<CellDirEntry>)> {
    if (data.len() as u64) < HEADER_LEN {
        return Err(PartitionAssetError::TooShort);
    }
    if data[0..8] != PARTITION_ASSET_MAGIC {
        return Err(PartitionAssetError::BadMagic);
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let u64_at = |o: usize| u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
    let f64_at = |o: usize| f64::from_le_bytes(data[o..o + 8].try_into().unwrap());

    let schema_version = u32_at(8);
    if schema_version > PARTITION_ASSET_SCHEMA_VERSION {
        return Err(PartitionAssetError::SchemaTooNew {
            found: schema_version,
            current: PARTITION_ASSET_SCHEMA_VERSION,
        });
    }
    let cell_size_m = f64_at(16);
    if !(cell_size_m.is_finite() && cell_size_m >= MIN_CELL_SIZE_M) {
        return Err(PartitionAssetError::Malformed(format!(
            "cell_size {cell_size_m} is not a sane cell edge"
        )));
    }
    let origin = [f64_at(24), f64_at(32), f64_at(40)];
    if !origin.iter().all(|v| v.is_finite()) {
        return Err(PartitionAssetError::Malformed(
            "origin is not finite".into(),
        ));
    }
    let cell_count = u32_at(48);
    let entity_count = u32_at(52);
    let blob_base = u64_at(56);

    let dir_end = HEADER_LEN + DIR_ENTRY_LEN * cell_count as u64;
    if (data.len() as u64) < dir_end {
        return Err(PartitionAssetError::Malformed(
            "truncated in the cell directory".into(),
        ));
    }
    if blob_base != align_up(dir_end) || blob_base > data.len() as u64 {
        return Err(PartitionAssetError::Malformed(format!(
            "blob_base {blob_base} does not follow the directory"
        )));
    }

    let mut dir = Vec::with_capacity(cell_count as usize);
    let mut prev: Option<CellKey> = None;
    // Blobs are laid out in directory order, so each must start at or after the
    // previous one's end. This is the **overlap** check: two entries sharing
    // bytes would let one cell alias another, and a reader cannot notice.
    let mut prev_end = blob_base;
    for i in 0..cell_count as u64 {
        let base = (HEADER_LEN + DIR_ENTRY_LEN * i) as usize;
        let kind = u32::from_le_bytes(data[base..base + 4].try_into().unwrap());
        if kind > 1 {
            return Err(PartitionAssetError::Malformed(format!(
                "cell kind {kind} is neither persistent (0) nor grid (1)"
            )));
        }
        let coord = (
            i32::from_le_bytes(data[base + 4..base + 8].try_into().unwrap()),
            i32::from_le_bytes(data[base + 8..base + 12].try_into().unwrap()),
        );
        if kind == 0 && coord != (0, 0) {
            return Err(PartitionAssetError::Malformed(
                "the persistent cell must carry coordinate (0, 0)".into(),
            ));
        }
        let key = if kind == 0 {
            CellKey::PERSISTENT
        } else {
            CellKey::grid(coord)
        };
        let entities = u32::from_le_bytes(data[base + 12..base + 16].try_into().unwrap());
        let offset = u64::from_le_bytes(data[base + 16..base + 24].try_into().unwrap());
        let len = u64::from_le_bytes(data[base + 24..base + 32].try_into().unwrap());
        if let Some(p) = prev {
            if key <= p {
                return Err(PartitionAssetError::Malformed(
                    "cell directory is not strictly ascending".into(),
                ));
            }
        }
        prev = Some(key);
        if offset % CELL_ALIGN != 0 {
            return Err(PartitionAssetError::Malformed(format!(
                "cell blob at {offset} is not {CELL_ALIGN}-byte aligned"
            )));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| PartitionAssetError::Malformed("cell length overflow".into()))?;
        if offset < blob_base || end > data.len() as u64 {
            return Err(PartitionAssetError::Malformed(format!(
                "cell blob [{offset}, {end}) out of bounds"
            )));
        }
        if offset < prev_end {
            return Err(PartitionAssetError::Malformed(format!(
                "cell blob [{offset}, {end}) overlaps the previous blob (ends at {prev_end})"
            )));
        }
        prev_end = end;
        dir.push(CellDirEntry {
            key,
            entity_count: entities,
            offset,
            len,
        });
    }

    Ok((
        PartitionAssetHeader {
            schema_version,
            cell_size_m,
            origin,
            cell_count,
            entity_count,
            blob_base,
        },
        dir,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_ecs::components::{
        AlwaysLoaded, Camera, Joint3D, Light, MeshRef, StreamingSource, Transform,
    };
    use inf_ecs::math::Vec3d;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    /// A bare entity record with every slot `None`.
    fn rec(g: u128, name: &str, at: (f64, f64, f64)) -> RuntimeEntity {
        RuntimeEntity {
            guid: guid(g),
            name: name.into(),
            parent: None,
            transform: Transform::from_translation(
                inf_ecs::math::Vec3d::new(at.0, at.1, at.2).to_dvec3(),
            ),
            visible: true,
            mesh: None,
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
        }
    }

    /// A placed prop (carries a mesh, so it occupies space).
    fn prop(g: u128, at: (f64, f64, f64)) -> RuntimeEntity {
        RuntimeEntity {
            mesh: Some(MeshRef::default()),
            ..rec(g, "Prop", at)
        }
    }

    fn settings(cell: f64) -> PartitionSettings {
        PartitionSettings {
            enabled: true,
            cell_size_m: cell,
            ..Default::default()
        }
    }

    // ── binning ─────────────────────────────────────────────────────────────

    #[test]
    fn cell_of_floors_through_the_origin_and_into_negatives() {
        assert_eq!(cell_of(0.0, 0.0, 256.0), (0, 0));
        assert_eq!(cell_of(255.9, 0.0, 256.0), (0, 0));
        assert_eq!(cell_of(256.0, 0.0, 256.0), (1, 0));
        // The reason floor division is required: `as i32` truncates toward zero,
        // so -0.1 and +0.1 would share a cell and the grid would be broken.
        assert_eq!(cell_of(-0.1, -0.1, 256.0), (-1, -1));
        assert_eq!(cell_of(-256.0, -256.0, 256.0), (-1, -1));
        assert_eq!(cell_of(-256.1, 0.0, 256.0), (-2, 0));
        // A degenerate cell size is clamped, never divided by.
        assert_eq!(cell_of(5.0, 5.0, 0.0), (5, 5));
        // A non-finite coordinate is answered deterministically, not saturated
        // into an unreachable corner of the grid.
        assert_eq!(cell_of(f64::NAN, 0.0, 256.0), (0, 0));
        assert_eq!(cell_of(f64::INFINITY, f64::NEG_INFINITY, 256.0), (0, 0));
    }

    #[test]
    fn cell_distance_is_zero_inside_and_per_axis_outside() {
        assert_eq!(cell_distance((0, 0), 10.0, 10.0, 100.0), 0.0);
        assert_eq!(cell_distance((0, 0), 150.0, 50.0, 100.0), 50.0);
        assert_eq!(cell_distance((0, 0), -20.0, 50.0, 100.0), 20.0);
        // A corner: per-axis (Chebyshev), so 50, not 50·√2.
        assert_eq!(cell_distance((0, 0), 150.0, 150.0, 100.0), 50.0);
    }

    #[test]
    fn cells_within_is_a_pure_box_query() {
        let mut got = BTreeSet::new();
        cells_within(50.0, 50.0, 0.0, 100.0, &mut got);
        assert_eq!(got, BTreeSet::from([(0, 0)]));

        let mut got = BTreeSet::new();
        cells_within(50.0, 50.0, 60.0, 100.0, &mut got);
        // x,z ∈ [-10, 110] → cells -1..=1 on both axes minus the ones outside…
        assert!(got.contains(&(0, 0)) && got.contains(&(-1, -1)) && got.contains(&(1, 1)));
        assert_eq!(got.len(), 9);

        // An absurd radius yields nothing rather than allocating the universe.
        let mut got = BTreeSet::new();
        cells_within(0.0, 0.0, 1e12, 1.0, &mut got);
        assert!(got.is_empty());
    }

    #[test]
    fn placed_entities_bin_by_position_and_managers_go_persistent() {
        let entities = vec![
            prop(1, (10.0, 0.0, 10.0)),          // cell (0,0)
            prop(2, (300.0, 0.0, 10.0)),         // cell (1,0)
            prop(3, (-10.0, 0.0, -300.0)),       // cell (-1,-2)
            rec(4, "GameMode", (0.0, 0.0, 0.0)), // unplaced → persistent
        ];
        let plan = partition_entities(&entities, &settings(256.0));
        assert_eq!(plan.persistent.len(), 1);
        assert_eq!(plan.persistent[0].name, "GameMode");
        assert_eq!(plan.cells.len(), 3);
        assert_eq!(plan.cells[&(0, 0)].len(), 1);
        assert_eq!(plan.cells[&(1, 0)].len(), 1);
        assert_eq!(plan.cells[&(-1, -2)].len(), 1);
        assert_eq!(plan.entity_count(), entities.len());
    }

    #[test]
    fn always_loaded_and_streaming_sources_never_stream() {
        let entities = vec![
            RuntimeEntity {
                always_loaded: Some(AlwaysLoaded),
                ..prop(1, (5000.0, 0.0, 5000.0))
            },
            RuntimeEntity {
                streaming_source: Some(StreamingSource::default()),
                ..prop(2, (5000.0, 0.0, 5000.0))
            },
            prop(3, (5000.0, 0.0, 5000.0)),
        ];
        let plan = partition_entities(&entities, &settings(256.0));
        assert_eq!(plan.persistent.len(), 2);
        assert_eq!(plan.cells.len(), 1);
        assert_eq!(plan.cells[&(19, 19)].len(), 1);

        assert_eq!(
            persistent_reason(&entities[0]),
            Some(PersistentReason::AlwaysLoaded)
        );
        assert_eq!(
            persistent_reason(&entities[1]),
            Some(PersistentReason::StreamingSource)
        );
        assert_eq!(persistent_reason(&entities[2]), None);
    }

    #[test]
    fn a_hierarchy_never_splits_across_cells() {
        // The child sits 1 km from its parent, but streams with it.
        let parent = prop(1, (10.0, 0.0, 10.0));
        let child = RuntimeEntity {
            parent: Some(guid(1)),
            ..prop(2, (1000.0, 0.0, 1000.0))
        };
        let grandchild = RuntimeEntity {
            parent: Some(guid(2)),
            ..prop(3, (-1000.0, 0.0, 0.0))
        };
        let plan = partition_entities(&[parent, child, grandchild], &settings(256.0));
        assert_eq!(
            plan.cells.len(),
            1,
            "{:?}",
            plan.cells.keys().collect::<Vec<_>>()
        );
        assert_eq!(plan.cells[&(0, 0)].len(), 3);
    }

    #[test]
    fn one_persistent_member_promotes_the_whole_tree() {
        let root = prop(1, (5000.0, 0.0, 5000.0));
        let child = RuntimeEntity {
            parent: Some(guid(1)),
            always_loaded: Some(AlwaysLoaded),
            ..prop(2, (5000.0, 0.0, 5000.0))
        };
        let plan = partition_entities(&[root, child], &settings(256.0));
        assert_eq!(plan.persistent.len(), 2);
        assert!(plan.cells.is_empty());
    }

    #[test]
    fn a_forged_parent_cycle_terminates() {
        let a = RuntimeEntity {
            parent: Some(guid(2)),
            ..prop(1, (0.0, 0.0, 0.0))
        };
        let b = RuntimeEntity {
            parent: Some(guid(1)),
            ..prop(2, (0.0, 0.0, 0.0))
        };
        let plan = partition_entities(&[a, b], &settings(256.0));
        assert_eq!(plan.entity_count(), 2);
    }

    #[test]
    fn occupies_space_covers_the_documented_component_set() {
        assert!(!occupies_space(&rec(1, "Bare", (0.0, 0.0, 0.0))));
        // An actor binding alone is a manager, not a placed thing.
        assert!(!occupies_space(&RuntimeEntity {
            actor: Some(guid(9)),
            ..rec(1, "Manager", (0.0, 0.0, 0.0))
        }));
        assert!(occupies_space(&prop(1, (0.0, 0.0, 0.0))));
        // The v11 sky authority is a world singleton, not a placed thing: it must
        // land in the persistent cell so the sun can never stream out.
        let sky = RuntimeEntity {
            time_of_day: Some(inf_ecs::components::TimeOfDay::default()),
            sky_atmosphere: Some(inf_ecs::components::SkyAtmosphere::default()),
            ..rec(1, "Sky", (5_000.0, 0.0, 5_000.0))
        };
        assert!(!occupies_space(&sky));
        assert_eq!(persistent_reason(&sky), Some(PersistentReason::Unplaced));
        assert!(occupies_space(&RuntimeEntity {
            light: Some(Light::default()),
            ..rec(1, "L", (0.0, 0.0, 0.0))
        }));
        assert!(occupies_space(&RuntimeEntity {
            camera: Some(Camera::default()),
            ..rec(1, "C", (0.0, 0.0, 0.0))
        }));
        // A terrain occupies space — and that is exactly why `streamed_terrains`
        // exists: it bins by its ORIGIN while its heightfield spans kilometres, so
        // an unmarked terrain in a partitioned level takes the ground with it.
        assert!(occupies_space(&RuntimeEntity {
            terrain: Some(inf_ecs::components::Terrain::default()),
            ..rec(1, "T", (0.0, 0.0, 0.0))
        }));
    }

    #[test]
    fn cross_cell_references_are_reported_but_never_promote_a_cell() {
        let a = RuntimeEntity {
            joint_3d: Some(Joint3D {
                other: inf_ecs::EntityRef::new(guid(2)),
                ..Default::default()
            }),
            ..prop(1, (10.0, 0.0, 10.0))
        };
        let b = prop(2, (400.0, 0.0, 10.0));
        // A ref into the persistent cell is fine (always resident).
        let manager = rec(3, "Manager", (0.0, 0.0, 0.0));
        let c = RuntimeEntity {
            attached_to: Some(inf_ecs::components::AttachedTo::new(
                guid(3),
                "",
                Vec3d::ZERO,
            )),
            ..prop(4, (10.0, 0.0, 10.0))
        };
        let plan = partition_entities(&[a, b, manager, c], &settings(256.0));
        let refs = cross_cell_refs(&plan);
        assert_eq!(refs.len(), 1, "{refs:?}");
        assert_eq!(refs[0].from, guid(1));
        assert_eq!(refs[0].to, guid(2));
        assert_eq!(refs[0].field, "joint_3d.other");
        assert_eq!(refs[0].from_cell, CellKey::grid((0, 0)));
        assert_eq!(refs[0].to_cell, CellKey::grid((1, 0)));
        // Reporting does not change the plan.
        assert_eq!(plan.cells.len(), 2);
    }

    /// A Blueprint on a **streamed** entity is reported — the mesh + `ActorClass`
    /// shape is the commonest gameplay actor there is, and in a partitioned level
    /// it is guaranteed to bin into a cell and guaranteed never to tick.
    #[test]
    fn streamed_blueprint_actors_are_reported_and_persistent_ones_are_not() {
        let class = guid(0xAC7);
        let entities = vec![
            // The dangerous shape: a mesh (so it occupies space → streams) that
            // also runs a blueprint.
            RuntimeEntity {
                actor: Some(class),
                ..prop(1, (10.0, 0.0, 10.0))
            },
            // Same binding, but pinned persistent — this one binds and ticks.
            RuntimeEntity {
                actor: Some(class),
                always_loaded: Some(AlwaysLoaded),
                ..prop(2, (400.0, 0.0, 10.0))
            },
            // A manager: unplaced, so persistent by the `Unplaced` rule.
            RuntimeEntity {
                actor: Some(class),
                ..rec(3, "GameMode", (0.0, 0.0, 0.0))
            },
            // A streamed prop with no blueprint at all — not our story.
            prop(4, (700.0, 0.0, 10.0)),
        ];
        let plan = partition_entities(&entities, &settings(256.0));
        let reported = streamed_actors(&plan);

        assert_eq!(reported.len(), 1, "{reported:?}");
        assert_eq!(reported[0].entity, guid(1));
        assert_eq!(reported[0].cell, CellKey::grid((0, 0)));
        assert_eq!(reported[0].class, class);

        // The two persistent bindings are silent: they bind and tick normally.
        assert_eq!(plan.persistent.len(), 2);
        // Reporting does not change the plan — v1 never auto-promotes, because
        // residency must not depend on which entities carry a script.
        assert_eq!(plan.cells.len(), 2);
        assert!(plan.cells[&(0, 0)].iter().any(|e| e.guid == guid(1)));
    }

    /// A level with no blueprints at all — and an unpartitioned one — report
    /// nothing, so the advisory never fires spuriously.
    #[test]
    fn streamed_actors_is_silent_without_blueprints() {
        let plan = partition_entities(
            &[prop(1, (10.0, 0.0, 10.0)), prop(2, (400.0, 0.0, 10.0))],
            &settings(256.0),
        );
        assert!(streamed_actors(&plan).is_empty());
        assert!(streamed_actors(&PartitionPlan::default()).is_empty());
    }

    /// A `Terrain` on a **streamed** entity is reported — a heightfield spans far
    /// more world than the cell holding its origin, so streaming it out takes the
    /// ground with it. An `AlwaysLoaded` terrain is silent, which is what every
    /// committed sample does.
    #[test]
    fn streamed_terrains_are_reported_and_always_loaded_ones_are_not() {
        let asset = guid(0xAA);
        let entities = vec![
            // The dangerous shape: a terrain with no marker at all.
            RuntimeEntity {
                terrain: Some(inf_ecs::components::Terrain {
                    asset: Some(asset),
                    ..Default::default()
                }),
                mesh: None,
                ..rec(1, "Terrain", (10.0, 0.0, 10.0))
            },
            // An INLINE terrain, also unmarked — same hazard, different source.
            RuntimeEntity {
                terrain: Some(inf_ecs::components::Terrain::default()),
                mesh: None,
                ..rec(2, "Inline Terrain", (400.0, 0.0, 10.0))
            },
            // Marked persistent — the sanctioned authoring, and silent.
            RuntimeEntity {
                terrain: Some(inf_ecs::components::Terrain::default()),
                always_loaded: Some(AlwaysLoaded),
                mesh: None,
                ..rec(3, "Good Terrain", (700.0, 0.0, 10.0))
            },
            // A streamed prop with no terrain — not our story.
            prop(4, (10.0, 0.0, 10.0)),
        ];
        let plan = partition_entities(&entities, &settings(256.0));
        let reported = streamed_terrains(&plan);

        assert_eq!(reported.len(), 2, "{reported:?}");
        // Sorted + deduplicated by `BTreeSet`, so the report is deterministic.
        assert_eq!(reported[0].entity, guid(1));
        assert_eq!(reported[0].cell, CellKey::grid((0, 0)));
        assert_eq!(reported[0].asset, Some(asset), "the .inf_terrain is named");
        assert_eq!(reported[1].entity, guid(2));
        assert_eq!(reported[1].cell, CellKey::grid((1, 0)));
        assert_eq!(reported[1].asset, None, "an inline terrain has no asset");

        // Reporting does not change the plan — v1 never auto-promotes, because
        // residency must not depend on which components an entity carries.
        assert!(plan.cells[&(0, 0)].iter().any(|e| e.guid == guid(1)));
        assert!(plan.persistent.iter().any(|e| e.guid == guid(3)));
    }

    /// A level with no terrain — and an unpartitioned one — report nothing.
    #[test]
    fn streamed_terrains_is_silent_without_terrain() {
        let plan = partition_entities(
            &[prop(1, (10.0, 0.0, 10.0)), prop(2, (400.0, 0.0, 10.0))],
            &settings(256.0),
        );
        assert!(streamed_terrains(&plan).is_empty());
        assert!(streamed_terrains(&PartitionPlan::default()).is_empty());
    }

    // ── the container ───────────────────────────────────────────────────────

    fn sample_plan() -> PartitionPlan {
        let mut entities = vec![rec(0xF0, "GameMode", (0.0, 0.0, 0.0))];
        for i in 0..9i32 {
            let (cx, cz) = (i % 3, i / 3);
            entities.push(prop(
                0x100 + i as u128,
                (cx as f64 * 256.0 + 8.0, 0.0, cz as f64 * 256.0 + 8.0),
            ));
        }
        partition_entities(&entities, &settings(256.0))
    }

    fn sample_asset() -> PartitionAsset {
        build_partition_asset(&sample_plan(), &settings(256.0)).unwrap()
    }

    #[test]
    fn header_and_directory_round_trip() {
        let asset = sample_asset();
        let r = asset.reader();
        assert_eq!(r.header().schema_version, PARTITION_ASSET_SCHEMA_VERSION);
        assert_eq!(r.cell_size_m(), 256.0);
        assert_eq!(r.cell_count(), 10, "9 grid cells + the persistent one");
        assert_eq!(r.entity_count(), 10);
        // Persistent first, then ascending grid order.
        let keys: Vec<CellKey> = r.keys().collect();
        assert!(keys[0].is_persistent());
        assert!(keys.windows(2).all(|w| w[0] < w[1]), "strictly ascending");
        assert_eq!(r.grid_coords().count(), 9);
    }

    #[test]
    fn every_cell_round_trips_bit_identically() {
        let plan = sample_plan();
        let asset = build_partition_asset(&plan, &settings(256.0)).unwrap();
        let r = asset.reader();
        for key in plan.keys() {
            let want = plan.cell(key).unwrap();
            let got = r.cell(key).unwrap().expect("present");
            assert_eq!(got, want, "{key}");
            // Bytes, not just values: the blob is the canonical bincode.
            assert_eq!(r.cell_bytes(key).unwrap(), encode_cell(want).unwrap());
        }
        assert!(r.cell(CellKey::grid((99, 99))).unwrap().is_none());
        assert!(r.cell_bytes(CellKey::grid((-5, -5))).is_none());
    }

    #[test]
    fn every_cell_blob_is_16_byte_aligned_inside_the_payload() {
        let asset = sample_asset();
        let r = asset.reader();
        for e in r.directory() {
            assert_eq!(e.offset % CELL_ALIGN, 0, "{} misaligned", e.key);
            assert!(e.offset >= r.header().blob_base);
            assert!(e.offset + e.len <= asset.as_bytes().len() as u64);
        }
        // The header + directory are themselves multiples of the alignment, so
        // the blob section needs no leading padding at all.
        assert_eq!(HEADER_LEN % CELL_ALIGN, 0);
        assert_eq!(DIR_ENTRY_LEN % CELL_ALIGN, 0);
        assert_eq!(
            r.header().blob_base,
            HEADER_LEN + DIR_ENTRY_LEN * r.cell_count() as u64
        );
        // And the payload as a whole ends on a boundary.
        assert_eq!(asset.as_bytes().len() as u64 % CELL_ALIGN, 0);
    }

    #[test]
    fn rebuilds_are_byte_identical_and_order_independent() {
        assert_eq!(sample_asset().as_bytes(), sample_asset().as_bytes());

        // Insertion order cannot leak into the bytes (BTreeMap-ordered directory).
        let plan = sample_plan();
        let mut fwd = PartitionAssetBuilder::new(256.0);
        let mut rev = PartitionAssetBuilder::new(256.0);
        let keys: Vec<CellKey> = plan.keys().collect();
        for k in &keys {
            fwd.insert(*k, plan.cell(*k).unwrap()).unwrap();
        }
        for k in keys.iter().rev() {
            rev.insert(*k, plan.cell(*k).unwrap()).unwrap();
        }
        assert_eq!(fwd.build().unwrap(), rev.build().unwrap());
    }

    #[test]
    fn the_persistent_cell_is_always_written_even_when_empty() {
        let plan = partition_entities(&[prop(1, (0.0, 0.0, 0.0))], &settings(256.0));
        assert!(plan.persistent.is_empty());
        let asset = build_partition_asset(&plan, &settings(256.0)).unwrap();
        let r = asset.reader();
        assert_eq!(r.cell_count(), 2);
        assert_eq!(r.cell(CellKey::PERSISTENT).unwrap(), Some(Vec::new()));
    }

    #[test]
    fn duplicate_cells_are_rejected() {
        let mut b = PartitionAssetBuilder::new(256.0);
        b.insert(CellKey::grid((0, 0)), &[]).unwrap();
        assert_eq!(
            b.insert(CellKey::grid((0, 0)), &[]),
            Err(PartitionAssetError::DuplicateCell {
                key: CellKey::grid((0, 0))
            })
        );
    }

    #[test]
    fn corrupt_payloads_are_rejected_not_trusted() {
        let asset = sample_asset();
        assert_eq!(
            PartitionAsset::from_bytes(vec![0u8; 8]).unwrap_err(),
            PartitionAssetError::TooShort
        );
        let mut bad = asset.as_bytes().to_vec();
        bad[0] = b'X';
        assert_eq!(
            PartitionAsset::from_bytes(bad).unwrap_err(),
            PartitionAssetError::BadMagic
        );
        let mut future = asset.as_bytes().to_vec();
        future[8..12].copy_from_slice(&(PARTITION_ASSET_SCHEMA_VERSION + 7).to_le_bytes());
        assert!(matches!(
            PartitionAsset::from_bytes(future).unwrap_err(),
            PartitionAssetError::SchemaTooNew { .. }
        ));
        // A blob pointing past the end.
        let mut oob = asset.as_bytes().to_vec();
        oob[HEADER_LEN as usize + 16..HEADER_LEN as usize + 24]
            .copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            PartitionAsset::from_bytes(oob).unwrap_err(),
            PartitionAssetError::Malformed(_)
        ));
        // A misaligned blob offset.
        let mut mis = asset.as_bytes().to_vec();
        let off = u64::from_le_bytes(
            mis[HEADER_LEN as usize + 16..HEADER_LEN as usize + 24]
                .try_into()
                .unwrap(),
        );
        mis[HEADER_LEN as usize + 16..HEADER_LEN as usize + 24]
            .copy_from_slice(&(off + 1).to_le_bytes());
        assert!(matches!(
            PartitionAsset::from_bytes(mis).unwrap_err(),
            PartitionAssetError::Malformed(_)
        ));
        // Truncation inside the directory.
        assert!(matches!(
            PartitionAsset::from_bytes(asset.as_bytes()[..HEADER_LEN as usize + 8].to_vec())
                .unwrap_err(),
            PartitionAssetError::Malformed(_)
        ));
        // An unknown cell kind.
        let mut kind = asset.as_bytes().to_vec();
        kind[HEADER_LEN as usize..HEADER_LEN as usize + 4].copy_from_slice(&7u32.to_le_bytes());
        assert!(matches!(
            PartitionAsset::from_bytes(kind).unwrap_err(),
            PartitionAssetError::Malformed(_)
        ));
        // Two directory entries sharing bytes.
        let r = asset.reader();
        assert!(r.cell_count() >= 2);
        let (first, second) = (r.directory()[0], r.directory()[1]);
        let mut overlap = asset.as_bytes().to_vec();
        let len_at = HEADER_LEN as usize + 24;
        overlap[len_at..len_at + 8]
            .copy_from_slice(&(second.offset - first.offset + 1).to_le_bytes());
        let err = PartitionAsset::from_bytes(overlap).unwrap_err();
        assert!(
            matches!(&err, PartitionAssetError::Malformed(m) if m.contains("overlaps")),
            "expected an overlap rejection, got {err}"
        );
    }

    /// The **fail-loud divergence** regression, mirroring `inf_terrain::asset`'s.
    ///
    /// A `.inf_part` is the raw image; a generic `inf_asset::encode` would frame
    /// it with a bincode length prefix and knock every cell off its 16-byte
    /// boundary. That door is closed at compile time — `PartitionAsset`
    /// implements neither `AssetPayload` nor `Serialize`/`Deserialize` — but a
    /// future refactor could re-open it, so it is pinned here as a *loud* failure
    /// rather than a quiet wrong answer.
    #[test]
    fn framed_and_raw_forms_are_not_interchangeable() {
        let asset = sample_asset();
        let image = asset.as_bytes();
        let framed = bincode::serde::encode_to_vec(image, bincode::config::standard()).unwrap();
        assert_ne!(framed, image, "framing really does change the bytes");
        assert_eq!(
            PartitionAssetReader::new(framed.as_slice()).unwrap_err(),
            PartitionAssetError::BadMagic,
            "a framed payload must fail the magic check, not be parsed"
        );
    }

    #[test]
    fn settings_clamp_degenerate_values() {
        let s = PartitionSettings {
            enabled: true,
            cell_size_m: f64::NAN,
            activation_radius_m: -5.0,
            prefetch_margin_m: f64::INFINITY,
        };
        assert!(s.effective_cell_size().is_finite());
        assert!(s.effective_cell_size() >= MIN_CELL_SIZE_M);
        assert_eq!(s.effective_activation_radius(), DEFAULT_ACTIVATION_RADIUS_M);
        assert_eq!(s.effective_prefetch_margin(), DEFAULT_PREFETCH_MARGIN_M);
    }
}
