//! Undo / redo (P3.4).
//!
//! Every editor mutation is an [`EditCommand`] with an `apply` (redo) and a
//! `revert` (undo) that both go through `SceneDoc`'s **raw** (non-recording)
//! mutations — so undo/redo never re-enter the recorder. Commands group into a
//! [`Transaction`]: a gizmo drag opens one, streams transform edits into it, and
//! commits a single undo entry (P3.4.2). Structural inverses (create/delete)
//! reuse the P3.5 [`EntityRecord`] so a deleted subtree round-trips exactly.

use std::collections::BTreeMap;

use inf_ecs::components::{FoliageInstance, Sprite, Transform};
use inf_ecs::PropValue;
use inf_terrain::{BiomeDelta, DataMapDelta, HeightDelta, HoleDelta, HoleDeltaBuilder, SplatDelta};
use inf_voxel::{
    SpoilPlan, VoxelDelta, VoxelDeltaBuilder, VoxelOp, VoxelOpKind, VoxelShape, MATERIAL_COUNT,
};
use uuid::Uuid;

use glam::DVec3;

use crate::scene::serialize::{EntityRecord, LevelSettings};
use crate::scene::SceneDoc;
use crate::voxel_store::SharedVoxelVolumes;

/// The default history depth (well past the phase gate's 50 steps).
pub const HISTORY_LIMIT: usize = 256;

/// **The history's byte ceiling** (Hardening D) — 256 MiB of undo records.
///
/// [`HISTORY_LIMIT`] bounds the history by *count*, which is a bound on entries
/// and not on memory: one sculpt stroke over a 1024² terrain patch is megabytes,
/// and 256 of them is the count-bound being honoured while the editor holds a
/// gigabyte of undo. Whichever ceiling binds first wins; the byte one is
/// deliberately generous, because dropping undo history is a worse outcome than
/// holding it and the number exists to stop the *pathological* case.
pub const HISTORY_BYTE_LIMIT: usize = 256 * 1024 * 1024;

/// What one [`EntityRecord`] is charged in [`EditCommand::memory_bytes`].
///
/// A record has no size accessor and its real cost is dominated by the
/// components it carries (a `Terrain`'s tiles, a `Tilemap`'s chunks), so this is
/// an ESTIMATE and is named rather than buried: an entity's own scalar
/// components, generously. Closing it properly means a `memory_bytes` on
/// `EntityRecord`, which is a walk over every component slot — recorded as the
/// honest remainder rather than left as an implied claim of accuracy.
const RECORD_ESTIMATE: usize = 4096;

pub(crate) enum EditCommand {
    /// `at` is the entity's slot in the creation-order list. The record is boxed
    /// so this variant doesn't bloat the whole enum (an `EntityRecord` grew with
    /// the P8.2b 2D component slots).
    Create {
        at: usize,
        record: Box<EntityRecord>,
    },
    /// A deleted subtree (records with their original order slots) + the top
    /// GUIDs actually removed.
    Delete {
        items: Vec<(usize, EntityRecord)>,
        tops: Vec<Uuid>,
    },
    Rename {
        guid: Uuid,
        before: String,
        after: String,
    },
    Reparent {
        guid: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
    },
    SetVisible {
        guid: Uuid,
        before: bool,
        after: bool,
    },
    SetTransform {
        guid: Uuid,
        before: Transform,
        after: Transform,
    },
    SetProp {
        guid: Uuid,
        type_path: String,
        field: String,
        before: PropValue,
        after: PropValue,
    },
    /// Whole-component `Sprite` swap (P8.2a). The `Sprite` fields the slicer
    /// writes (`texture`, `atlas_rect`) aren't reflection-addressable, so the
    /// component round-trips as a value; `None` means "no `Sprite` component".
    SetSprite {
        guid: Uuid,
        before: Option<Sprite>,
        after: Option<Sprite>,
    },
    /// One tile-painting stroke (P8.2b). Stores the pre/post index of **only the
    /// touched cells** — never a whole chunk map — so a stroke over a large map
    /// stays cheap. `cells` is `(x, y, before, after)`.
    SetTiles {
        guid: Uuid,
        cells: Vec<(i32, i32, u32, u32)>,
    },
    /// Whole-component `ActorClass` swap (P9.5): the blueprint-class binding GUID
    /// (a non-reflected identity link) round-trips as a value; `None` = unbound.
    SetActor {
        guid: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
    },
    /// The `Material.asset` `.inf_mat` binding (P26.3b · scene v22): a
    /// `#[reflect(ignore)]` identity link, so it round-trips as a value here
    /// exactly as [`SetActor`](Self::SetActor) does rather than through the
    /// property path. `None` = unbound, which is the scalars-only surface every
    /// pre-v22 level had.
    SetMaterialAsset {
        guid: Uuid,
        before: Option<Uuid>,
        after: Option<Uuid>,
    },
    /// One terrain sculpt stroke (P10.2b): a sparse before/after height-sample
    /// record ([`HeightDelta`]). The live stroke already mutated the terrain when
    /// this is recorded, so `apply`/`revert` here are pure redo/undo — redo
    /// replays the `after` samples, undo replays `before` (and drops any tiles the
    /// stroke authored from nothing). Boxed so the (potentially large) delta
    /// doesn't bloat every other command variant.
    SculptTerrain { guid: Uuid, delta: Box<HeightDelta> },
    /// A whole-file level-settings swap (R-P4): gravity / sim rate / the persisted
    /// render (post/exposure/lighting) block round-trip as one value. `Copy` +
    /// small, so this variant is stored inline (no boxing).
    SetLevelSettings {
        old: LevelSettings,
        new: LevelSettings,
    },
    /// One terrain splat-paint stroke (P10.4): a sparse before/after weight-sample
    /// record ([`SplatDelta`]). Like [`SculptTerrain`](EditCommand::SculptTerrain)
    /// the live stroke already mutated the weights, so `apply`/`revert` are pure
    /// redo/undo — redo replays `after` weights, undo replays `before` (and drops
    /// any weight buffers the stroke materialized from the sparse default). Boxed.
    PaintSplat { guid: Uuid, delta: Box<SplatDelta> },
    /// One terrain **biome-paint** stroke (P19.2): a sparse before/after record of
    /// the per-sample biome ids it wrote ([`BiomeDelta`]).
    ///
    /// A sibling of [`PaintSplat`](EditCommand::PaintSplat), not a field on it,
    /// for the same reason `PaintSplat` is a sibling of
    /// [`SculptTerrain`](EditCommand::SculptTerrain): the payloads are genuinely
    /// different layers with different producers, and merging them would put an
    /// always-empty buffer inside every stroke on the undo stack. Like both, the
    /// live stroke already mutated the terrain, so `apply`/`revert` are pure
    /// redo/undo — undo replays `before` and drops any id buffers the stroke
    /// materialized from the sparse default. Boxed.
    PaintBiome { guid: Uuid, delta: Box<BiomeDelta> },
    /// One erosion bake's **data-map** half (P19.1): a sparse before/after record
    /// of the flow / deposition / wear accumulators it moved ([`DataMapDelta`]).
    ///
    /// Recorded *beside* the bake's [`SculptTerrain`](EditCommand::SculptTerrain)
    /// inside a single transaction, so one Ctrl+Z restores both layers. It is a
    /// separate command rather than a field on the height delta because the two
    /// layers have different producers: every sculpt brush writes heights, only
    /// erosion writes data maps, and folding them together would put an
    /// always-empty map buffer inside every stroke on the undo stack. Boxed.
    WriteDataMaps {
        guid: Uuid,
        delta: Box<DataMapDelta>,
    },
    /// Add / remove a whole component (E-P1). `before`/`after` are full entity
    /// component snapshots ([`EntityRecord`] via `record_of`) — the record is the
    /// complete truth, so replaying either side re-inserts what it holds and
    /// removes every optional component it leaves `None`
    /// (`raw_apply_record_components`). Boxed so the (large) record doesn't bloat
    /// the other variants. Covers add (`before` lacks the component, `after` has
    /// it) and remove (the reverse) with one code path.
    SwapComponents {
        guid: Uuid,
        before: Box<EntityRecord>,
        after: Box<EntityRecord>,
    },
    /// One foliage scatter stroke (E-P6): the instances added and/or the
    /// `(index, instance)` pairs removed. Like the terrain deltas the live stroke
    /// already mutated the component, so `apply`/`revert` are pure redo/undo — redo
    /// removes the `removed` indices (descending) then pushes `added`; undo pops
    /// `added` off the end then re-inserts `removed` at their original indices.
    /// A stroke is add-XOR-erase, so one vector is always empty.
    PaintFoliage {
        guid: Uuid,
        added: Vec<FoliageInstance>,
        removed: Vec<(usize, FoliageInstance)>,
    },
    /// **One voxel carve/fill and the heightfield holes it opened** (P21.2) —
    /// deliberately ONE command rather than a pair, so a single Ctrl+Z takes back
    /// the rock *and* the ground above it.
    ///
    /// Splitting them into two commands inside a transaction was the obvious
    /// alternative and is wrong here: the two halves are not independent edits
    /// that happen to be grouped, they are two representations of one cut, and a
    /// history that could ever replay one without the other would leave a cave
    /// sealed behind ground or a hole in the sky over solid rock.
    ///
    /// * `volume` — the `VoxelVolume` entity whose chunks moved.
    /// * `volumes` — the shared working set the chunks live in. The document
    ///   holds no voxel data (schema v19 keeps it out in the `.inf_voxel`), and
    ///   `SceneDoc::undo` runs on the Ring-2 command thread with no viewport, so
    ///   the store reaches the history through this handle. See
    ///   [`SharedVoxelVolumes`] for why that, and not a journal or a move into
    ///   the document.
    /// * `holes` — the heightfield half, **per terrain**: a tunnel may cross more
    ///   than one (P16.6 multi-terrain), and an arbitrary tie-break would silently
    ///   drop the mouths on all but one of them. Empty for a cut that never
    ///   reached a surface, which is every cave dug below the ground.
    ///
    /// Both payloads are boxed/owned rather than inline — a stroke's delta is
    /// dense sub-boxes and would bloat every other variant.
    CarveVoxels {
        volume: Uuid,
        volumes: SharedVoxelVolumes,
        delta: Box<VoxelDelta>,
        holes: Vec<(Uuid, HoleDelta)>,
    },
}

impl EditCommand {
    /// **Approximate heap footprint of this record, in bytes** (Hardening D).
    ///
    /// The history used to be bounded by [`HISTORY_LIMIT`] alone — a *count* —
    /// and the memory diagnostics charged a flat 512 bytes per entry. Both are
    /// wrong in the same direction and by orders of magnitude: 256 sculpt strokes
    /// over a 1024² terrain is hundreds of megabytes reported as 128 KiB, and the
    /// count-only bound means the undo stack's real ceiling is
    /// "256 × whatever the largest stroke was".
    ///
    /// Every payload that can be large already had a `memory_bytes()` and none of
    /// them had a caller: `HeightDelta`, `SplatDelta` (added here),
    /// `BiomeDelta`, `DataMapDelta`, `HoleDelta` (added here) and `VoxelDelta`.
    /// This is the one place they are summed.
    ///
    /// The small variants are charged their `size_of` and no more — the estimate
    /// is for *budgeting*, and a rename's `String` is noise beside a stroke. The
    /// one that is neither small nor delta-shaped is an [`EntityRecord`], which
    /// has no size accessor and whose cost is dominated by whatever components it
    /// carries; it is charged its `size_of` too, and that is stated rather than
    /// hidden — see [`RECORD_ESTIMATE`].
    pub(crate) fn memory_bytes(&self) -> usize {
        let base = std::mem::size_of::<EditCommand>();
        let payload = match self {
            EditCommand::Create { .. } => RECORD_ESTIMATE,
            EditCommand::Delete { items, .. } => items.len().saturating_mul(RECORD_ESTIMATE),
            EditCommand::SwapComponents { .. } => 2 * RECORD_ESTIMATE,
            EditCommand::SculptTerrain { delta, .. } => delta.memory_bytes(),
            EditCommand::PaintSplat { delta, .. } => delta.memory_bytes(),
            EditCommand::PaintBiome { delta, .. } => delta.memory_bytes(),
            EditCommand::WriteDataMaps { delta, .. } => delta.memory_bytes(),
            EditCommand::SetTiles { cells, .. } => cells
                .len()
                .saturating_mul(std::mem::size_of::<(i32, i32, u32, u32)>()),
            EditCommand::PaintFoliage { added, removed, .. } => added
                .len()
                .saturating_add(removed.len())
                .saturating_mul(std::mem::size_of::<FoliageInstance>()),
            EditCommand::CarveVoxels { delta, holes, .. } => delta
                .memory_bytes()
                .saturating_add(holes.iter().map(|(_, h)| h.memory_bytes()).sum::<usize>()),
            // Value swaps: the enum's own footprint is the whole cost.
            EditCommand::Rename { .. }
            | EditCommand::Reparent { .. }
            | EditCommand::SetVisible { .. }
            | EditCommand::SetTransform { .. }
            | EditCommand::SetProp { .. }
            | EditCommand::SetSprite { .. }
            | EditCommand::SetActor { .. }
            | EditCommand::SetMaterialAsset { .. }
            | EditCommand::SetLevelSettings { .. } => 0,
        };
        base.saturating_add(payload)
    }

    /// Do (or redo) the edit.
    pub(crate) fn apply(&self, doc: &mut SceneDoc) {
        match self {
            EditCommand::Create { at, record } => doc.raw_spawn_record(record, *at),
            EditCommand::Delete { tops, .. } => doc.raw_delete(tops),
            EditCommand::Rename { guid, after, .. } => doc.raw_rename(*guid, after),
            EditCommand::Reparent { guid, after, .. } => {
                doc.raw_reparent(*guid, *after);
            }
            EditCommand::SetVisible { guid, after, .. } => doc.raw_set_visible(*guid, *after),
            EditCommand::SetTransform { guid, after, .. } => doc.raw_set_transform(*guid, *after),
            EditCommand::SetProp {
                guid,
                type_path,
                field,
                after,
                ..
            } => {
                doc.raw_write_prop(*guid, type_path, field, after);
            }
            EditCommand::SetSprite { guid, after, .. } => {
                doc.raw_set_sprite(*guid, after.clone());
            }
            EditCommand::SetTiles { guid, cells } => {
                let after: Vec<(i32, i32, u32)> =
                    cells.iter().map(|&(x, y, _, a)| (x, y, a)).collect();
                doc.raw_set_tiles(*guid, &after);
            }
            EditCommand::SetActor { guid, after, .. } => {
                doc.raw_set_actor(*guid, *after);
            }
            EditCommand::SetMaterialAsset { guid, after, .. } => {
                doc.raw_set_material_asset(*guid, *after);
            }
            EditCommand::SetLevelSettings { new, .. } => {
                doc.raw_set_settings(*new);
            }
            EditCommand::SculptTerrain { guid, delta } => {
                doc.raw_apply_terrain_delta(*guid, delta);
            }
            EditCommand::PaintSplat { guid, delta } => {
                doc.raw_apply_splat_delta(*guid, delta);
            }
            EditCommand::PaintBiome { guid, delta } => {
                doc.raw_apply_biome_delta(*guid, delta);
            }
            EditCommand::WriteDataMaps { guid, delta } => {
                doc.raw_apply_data_map_delta(*guid, delta);
            }
            EditCommand::SwapComponents { guid, after, .. } => {
                doc.raw_apply_record_components(*guid, after);
            }
            EditCommand::PaintFoliage {
                guid,
                added,
                removed,
            } => {
                doc.raw_apply_foliage(*guid, added, removed);
            }
            EditCommand::CarveVoxels {
                volume,
                volumes,
                delta,
                holes,
            } => {
                doc.raw_write_carve(*volume, volumes, delta, holes, false);
            }
        }
    }

    /// Undo the edit.
    pub(crate) fn revert(&self, doc: &mut SceneDoc) {
        match self {
            EditCommand::Create { record, .. } => doc.raw_delete(&[record.guid]),
            EditCommand::Delete { items, .. } => {
                // Two passes so the hierarchy survives regardless of the order
                // slots: (1) respawn every record at its slot — a record whose
                // parent sits at a LATER slot (a reparent under a later-created
                // node) spawns to the root because its parent isn't back yet;
                // (2) fix up every parent link now that all GUIDs exist again.
                // The second pass is a no-op for the common parents-precede-
                // children ordering.
                let mut items: Vec<&(usize, EntityRecord)> = items.iter().collect();
                items.sort_by_key(|(at, _)| *at);
                for (at, record) in &items {
                    doc.raw_spawn_record(record, *at);
                }
                for (_, record) in &items {
                    doc.raw_fixup_parent(record.guid, record.parent);
                }
            }
            EditCommand::Rename { guid, before, .. } => doc.raw_rename(*guid, before),
            EditCommand::Reparent { guid, before, .. } => {
                doc.raw_reparent(*guid, *before);
            }
            EditCommand::SetVisible { guid, before, .. } => doc.raw_set_visible(*guid, *before),
            EditCommand::SetTransform { guid, before, .. } => doc.raw_set_transform(*guid, *before),
            EditCommand::SetProp {
                guid,
                type_path,
                field,
                before,
                ..
            } => {
                doc.raw_write_prop(*guid, type_path, field, before);
            }
            EditCommand::SetSprite { guid, before, .. } => {
                doc.raw_set_sprite(*guid, before.clone());
            }
            EditCommand::SetTiles { guid, cells } => {
                let before: Vec<(i32, i32, u32)> =
                    cells.iter().map(|&(x, y, b, _)| (x, y, b)).collect();
                doc.raw_set_tiles(*guid, &before);
            }
            EditCommand::SetActor { guid, before, .. } => {
                doc.raw_set_actor(*guid, *before);
            }
            EditCommand::SetMaterialAsset { guid, before, .. } => {
                doc.raw_set_material_asset(*guid, *before);
            }
            EditCommand::SetLevelSettings { old, .. } => {
                doc.raw_set_settings(*old);
            }
            EditCommand::SculptTerrain { guid, delta } => {
                doc.raw_revert_terrain_delta(*guid, delta);
            }
            EditCommand::PaintSplat { guid, delta } => {
                doc.raw_revert_splat_delta(*guid, delta);
            }
            EditCommand::PaintBiome { guid, delta } => {
                doc.raw_revert_biome_delta(*guid, delta);
            }
            EditCommand::WriteDataMaps { guid, delta } => {
                doc.raw_revert_data_map_delta(*guid, delta);
            }
            EditCommand::SwapComponents { guid, before, .. } => {
                doc.raw_apply_record_components(*guid, before);
            }
            EditCommand::PaintFoliage {
                guid,
                added,
                removed,
            } => {
                doc.raw_revert_foliage(*guid, added, removed);
            }
            EditCommand::CarveVoxels {
                volume,
                volumes,
                delta,
                holes,
            } => {
                doc.raw_write_carve(*volume, volumes, delta, holes, true);
            }
        }
    }
}

// ── the coupled voxel carve (P21.2) ─────────────────────────────────────────
//
// A carve is two edits to two containers that must behave as one: SDF samples in
// a `.inf_voxel` and hole bits in a `.inf_terrain`. Everything below exists to
// keep them one — one verdict before the gesture starts, one stroke that
// accumulates both records, one `EditCommand` on the history.
//
// It lives here rather than in `doc.rs` because it *is* the undo record under
// construction, and because the document owns only half of it: the heightfield.

/// What the tools are allowed to do with a proposed cut, decided **before**
/// anything is written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CarveVerdict {
    /// The cut stays clear of every heightfield surface — a cave under a hill, a
    /// brush swung in mid-air, a level with no terrain at all. Legal everywhere:
    /// no hole is needed, so no container has to be able to persist one.
    NoSurface,
    /// It breaks through, and every terrain it breaks through is asset-backed and
    /// can persist the mouths. `terrains` is in document order.
    Opens { terrains: Vec<Uuid> },
    /// **Refused.** It breaks through an *inline* terrain — one whose heightfield
    /// is stored in the `.inf_lvl` itself.
    ///
    /// Schema v19 is frozen, so `TerrainData`'s wire form is pinned at tile
    /// generation 3 and an inline terrain **cannot persist a hole mask** (see
    /// `TerrainTileFrozenV3`). An author who punched a cave mouth here would lose
    /// it on the next save+reload, which breaks the house law that *a committed
    /// level must not depend on which tools the session visited*.
    ///
    /// The **whole op** is refused, voxels included — not "carve the rock and
    /// skip the holes". A partial success leaves a cave with no mouth: geometry
    /// the author cannot see, cannot reach, and has no reason to suspect. A clean
    /// refusal that names the fix is a better surprise than a silent one.
    RefusedInline { terrain: Uuid },
}

impl CarveVerdict {
    /// `true` when the tools may proceed.
    pub fn allowed(&self) -> bool {
        !matches!(self, CarveVerdict::RefusedInline { .. })
    }

    /// The terrains whose hole masks this cut will move (empty unless
    /// [`Opens`](CarveVerdict::Opens)).
    pub fn terrains(&self) -> &[Uuid] {
        match self {
            CarveVerdict::Opens { terrains } => terrains,
            _ => &[],
        }
    }
}

/// Why a surface-crossing carve was refused, verbatim, for the tool status seam.
///
/// One string so the viewport, a future command and any test quote the same
/// sentence — and it names the fix, because a refusal the author cannot act on is
/// just a broken tool with better manners.
pub const INLINE_TERRAIN_CARVE_REFUSAL: &str =
    "Carve refused: this cut breaks through an INLINE terrain, and a terrain stored in the \
     level cannot save cave mouths (the level schema pins its tiles at a layout with no hole \
     mask). Nothing was carved. Convert the terrain to asset-backed — import or export it as \
     a .inf_terrain — and the same cut will work.";

/// …and why a cut was refused because its target volume has no chunks to cut.
///
/// Distinct from the tool's own "no volume is loaded" line (which is about
/// *choosing* a target before the gesture starts): this one fires **mid-gesture**,
/// when the volume that was there at mouse-down has since been released — its
/// asset reference was cleared, its entity was deleted, or the projection dropped
/// it. Nothing was cut, so nothing above it may be opened either.
pub const VOLUME_NOT_LOADED_CARVE_REFUSAL: &str =
    "Carve refused: the target voxel volume has no loaded chunks, so there is nothing to cut. \
     Its .inf_voxel may have been released or its asset reference cleared since the gesture \
     started. Nothing was carved, and no cave mouth was opened.";

/// …and why a cut was refused because the shared working set cannot be read.
///
/// A poisoned mutex means a thread panicked while holding the volumes — the
/// chunks are in an unknown state, so a carve must not write to them and, above
/// all, must not open a mouth in the ground over rock it could not touch.
pub const POISONED_STORE_CARVE_REFUSAL: &str =
    "Carve refused: the voxel working set is unreadable — a thread panicked while holding it, \
     so the loaded chunks are in an unknown state. Nothing was carved, and no cave mouth was \
     opened. Save your work and restart the editor.";

/// …and why a **dig** was refused before it started: it is simply too big.
///
/// A dig is judged whole (`a4e5844`), so the size gate has to answer *before*
/// the first sample moves — which is what
/// [`VoxelShape::affected_sample_count`](inf_voxel::VoxelShape::affected_sample_count)
/// is for. The number it bounds is not only the cut: the spoil that has to
/// balance it is bounded by the same count, and both are held in memory as one
/// undo record.
pub const DIG_TOO_LARGE_REFUSAL: &str =
    "Dig refused: this excavation is larger than one transaction may move. Nothing was cut. \
     Reduce the pit's footprint or its depth, or dig it in several passes — a dig is committed \
     whole, so half a foundation pit is not an option the editor offers.";

/// …and why a dig was refused because its spoil could not be placed.
///
/// **Never seen in practice** — the pile's search region grows until it holds
/// the count. It exists because "the spoil did not fit" must be a refusal the
/// author reads rather than a transaction that silently fails conservation.
pub const SPOIL_SHORTFALL_REFUSAL: &str =
    "Spoil refused: the excavated soil could not be placed — the spoil site is buried in solid \
     rock for as far as the pile could search, or the stroke removed more than one heap can \
     hold. The cut itself is committed (Ctrl+Z takes it back); the material was discarded. Move \
     the spoil site to open ground, or dig in smaller passes.";

/// …and why a **brush stroke**'s spoil was refused: the stroke removed more than
/// one pile may hold.
///
/// A distinct sentence from [`DIG_TOO_LARGE_REFUSAL`] because the situation is
/// the opposite one. A box or trench cut is judged *before* it cuts, so its
/// refusal can honestly say "nothing was cut". A brush stroke's dabs are already
/// in the world by the time anyone can count them — `CarveStroke::dab`
/// accumulates across frames with no ceiling — so the only thing left to refuse
/// is the heap, and telling the author "nothing was cut" would be a lie about a
/// hole they are looking at.
pub const SPOIL_TOO_LARGE_REFUSAL: &str =
    "Spoil skipped: this stroke removed more material than one heap can hold, so the soil was \
     discarded rather than piled. The cut is committed and Ctrl+Z takes it back. Dig in several \
     passes if you want the spoil.";

/// The most lattice samples one dig transaction may move, cut and spoil
/// together.
///
/// Two million samples is a **58.5 m** cube at half-metre voxels (`∛2e6 = 126`
/// samples on a side) — comfortably past any foundation pit an author drags, and
/// the point where the undo record, the spoil search and the re-mesh stop being
/// interactive. The bound is checked against
/// [`affected_sample_count`](inf_voxel::VoxelShape::affected_sample_count),
/// which is the op's whole affected region.
///
/// **A dig at the ceiling is not interactive**, and that is a decided trade
/// rather than an oversight: mouse-up on a 2 M-sample box cut spends ≈1.3 s
/// under the shared-volumes lock (4 M would be ≈3.1 s), because the cut, the
/// spoil search and the re-mesh all run there. The alternative — moving the
/// re-mesh off the lock — is a bigger change than this batch should make to the
/// store's threading, and lowering the ceiling would refuse pits an author can
/// legitimately want. So the ceiling stays, the number is written down, and
/// "make a big dig incremental" is the ledgered follow-up. See the P21.3 status
/// block.
pub const MAX_DIG_SAMPLES: u64 = 2_000_000;

/// **Why a carve did not happen** — every refusal the coupled transaction can
/// hand back, as a value rather than a `None` the caller has to guess about.
///
/// Before this existed the carve doors returned `Option`, so the viewport
/// reported [`INLINE_TERRAIN_CARVE_REFUSAL`] for *any* empty answer — including
/// the two below, which have nothing to do with inline terrain and name a
/// completely different fix. A verdict readout that explains the wrong problem is
/// worse than a silent one: the author converts a terrain that was never at
/// fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CarveRefusal {
    /// The cut breaks through an inline terrain — [`CarveVerdict::RefusedInline`].
    InlineTerrain { terrain: Uuid },
    /// The target volume has no loaded chunks (released, unresolved, or deleted).
    VolumeNotLoaded,
    /// The shared voxel working set is poisoned.
    PoisonedStore,
    /// The dig would move more than [`MAX_DIG_SAMPLES`] samples (P21.3).
    TooLarge {
        /// The bound the ops actually asked for.
        samples: u64,
    },
    /// A **brush stroke** removed more than one pile may hold, so the soil was
    /// discarded (P21.3 audit). The cut itself is committed.
    SpoilTooLarge {
        /// Voxels the stroke removed.
        voxels: u64,
    },
    /// The spoil pile could not hold what the cut removed (P21.3).
    SpoilShortfall,
}

impl CarveRefusal {
    /// The sentence the tool status seam quotes, verbatim.
    pub fn message(&self) -> &'static str {
        match self {
            CarveRefusal::InlineTerrain { .. } => INLINE_TERRAIN_CARVE_REFUSAL,
            CarveRefusal::VolumeNotLoaded => VOLUME_NOT_LOADED_CARVE_REFUSAL,
            CarveRefusal::PoisonedStore => POISONED_STORE_CARVE_REFUSAL,
            CarveRefusal::TooLarge { .. } => DIG_TOO_LARGE_REFUSAL,
            CarveRefusal::SpoilTooLarge { .. } => SPOIL_TOO_LARGE_REFUSAL,
            CarveRefusal::SpoilShortfall => SPOIL_SHORTFALL_REFUSAL,
        }
    }
}

/// Where a dig's excavated material goes (P21.3).
///
/// Three answers and no fourth, because "somewhere sensible" is not a rule an
/// author can predict or a test can pin:
///
/// * [`Discard`](SpoilChoice::Discard) — the material is removed from the world.
///   The P21.2 behaviour, and still the right one for a cave (nobody carries
///   the spoil out of a tunnel in a level editor).
/// * [`Auto`](SpoilChoice::Auto) — the documented default site: east of the
///   cut, clear of its rim, dropped onto the ground there.
/// * [`At`](SpoilChoice::At) — the author picked a spot in the viewport.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SpoilChoice {
    /// Remove the material without piling it anywhere.
    #[default]
    Discard,
    /// Pile it at the deterministic default site.
    Auto,
    /// Pile it at this world point (metres). A non-finite point falls back to
    /// discarding rather than writing a pile at infinity.
    At(DVec3),
}

/// Running totals of a carve stroke, for the tool's live readout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CarveTally {
    /// Voxel samples whose SDF or material changed.
    pub touched: u64,
    /// Voxel samples that went solid → empty.
    pub carved: u64,
    /// Voxel samples that went empty → solid.
    pub filled: u64,
    /// Chunks materialized from nothing.
    pub created_chunks: u64,
    /// Heightfield samples whose hole bit changed.
    pub holes: u64,
    /// Samples removed, **by the material they carried** (P21.3).
    ///
    /// The excavation ledger's left-hand side. `carved` is its sum, kept
    /// separately because the readout quotes both and a caller that only wants
    /// "how much did I dig" should not have to sum an array.
    pub carved_by_material: [u64; MATERIAL_COUNT],
    /// Samples placed as **spoil**, by material — the ledger's right-hand side.
    ///
    /// Equal to [`carved_by_material`](Self::carved_by_material) element for
    /// element whenever the dig displaced its soil (see [`Self::conserved`]).
    /// All zero when spoil is off, which is a deliberate discard and not a
    /// failure of conservation — the author asked for the material to go away.
    pub spoiled_by_material: [u64; MATERIAL_COUNT],
}

impl CarveTally {
    /// `true` when the stroke has changed nothing at all.
    pub fn is_noop(&self) -> bool {
        self.touched == 0 && self.holes == 0
    }

    /// Cubic metres of material removed minus added, at `voxel_size_m`. The
    /// number the readout quotes, and the one the soil-displacement
    /// conservation gate subtracts against.
    pub fn net_removed_m3(&self, voxel_size_m: f64) -> f64 {
        let cell = voxel_size_m * voxel_size_m * voxel_size_m;
        (self.carved as f64 - self.filled as f64) * cell
    }

    /// Total samples placed as spoil.
    pub fn spoiled(&self) -> u64 {
        self.spoiled_by_material.iter().sum()
    }

    /// Cubic metres of spoil placed, at `voxel_size_m`.
    pub fn spoiled_m3(&self, voxel_size_m: f64) -> f64 {
        self.spoiled() as f64 * voxel_size_m.powi(3)
    }

    /// Cubic metres removed, at `voxel_size_m` — the excavation readout's
    /// left-hand number (gross, not net: a dig quotes what came out of the hole,
    /// and the spoil it became is the other number beside it).
    pub fn removed_m3(&self, voxel_size_m: f64) -> f64 {
        self.carved as f64 * voxel_size_m.powi(3)
    }

    /// **The conservation predicate.** `true` when every voxel removed was
    /// placed again, per material, exactly.
    ///
    /// Answers `true` for a dig that removed nothing (nothing to conserve) and
    /// `false` the moment a single voxel of a single material goes missing —
    /// there is no tolerance here on purpose, because an integer identity that
    /// is allowed to be nearly true is not an identity (the P10 erosion mass
    /// gates' rule).
    pub fn conserved(&self) -> bool {
        self.carved_by_material == self.spoiled_by_material
    }

    /// A one-line per-material breakdown for the tool readout, e.g.
    /// `"layer 0: 412, layer 2: 96"`. Empty when nothing was removed.
    pub fn material_breakdown(&self) -> String {
        let parts: Vec<String> = self
            .carved_by_material
            .iter()
            .enumerate()
            .filter(|(_, &n)| n > 0)
            .map(|(m, &n)| format!("layer {m}: {n}"))
            .collect();
        parts.join(", ")
    }
}

/// One carve/fill gesture in progress — a brush stroke or a spline tunnel —
/// accumulating both halves of the reversible record.
///
/// Held by the viewport thread between mouse-down and mouse-up exactly as an
/// `inf_terrain::Stroke` is, and for the same reason: the dabs mutate live (so
/// the cave appears under the cursor) and the *record* is what reaches the
/// history at commit.
pub struct CarveStroke {
    volume: Uuid,
    volumes: SharedVoxelVolumes,
    voxels: VoxelDeltaBuilder,
    /// One builder per terrain the stroke has broken through, in entity order.
    holes: BTreeMap<Uuid, HoleDeltaBuilder>,
    tally: CarveTally,
    /// World AABB of every shape the stroke has cut, `None` until the first dab
    /// — what the **default spoil rule** measures from (P21.3).
    ///
    /// Accumulated here rather than recomputed by the tools because the default
    /// site has to be a function of the whole gesture: a spoil heap placed east
    /// of the *last* dab of a hundred-metre trench would sit in the middle of
    /// the trench.
    bounds: Option<(DVec3, DVec3)>,
    /// Whether each dab re-meshes immediately (P21.4).
    ///
    /// `true` for the live **brush**, whose whole point is that the cave appears
    /// under the cursor as it is dug and whose dabs are small. `false` for a
    /// one-shot **dig** — a box cut or a trench committed on mouse-up — where the
    /// re-mesh is the one part of the transaction that does not have to happen
    /// under the shared-volumes guard: `VoxelMeshCache` is keyed on the chunk
    /// versions the cut just bumped, so the viewport's next per-frame
    /// `sync_camera` rebuilds exactly the same chunks. See
    /// `EditorVoxelVolumes::carve_into_deferred` for the measurement.
    remesh: bool,
}

impl CarveStroke {
    /// Open a stroke against `volume`'s chunks in `volumes`, re-meshing each dab
    /// as it lands — the live-brush shape.
    pub fn begin(volume: Uuid, volumes: SharedVoxelVolumes) -> Self {
        Self {
            volume,
            volumes,
            voxels: VoxelDeltaBuilder::new(),
            holes: BTreeMap::new(),
            tally: CarveTally::default(),
            bounds: None,
            remesh: true,
        }
    }

    /// [`begin`](Self::begin) with the per-dab re-mesh **deferred** to the
    /// viewport's next frame — the one-shot-dig shape (P21.4).
    pub fn begin_deferred(volume: Uuid, volumes: SharedVoxelVolumes) -> Self {
        Self {
            remesh: false,
            ..Self::begin(volume, volumes)
        }
    }

    /// The volume entity this stroke is cutting.
    pub fn volume(&self) -> Uuid {
        self.volume
    }

    /// Totals so far.
    pub fn tally(&self) -> CarveTally {
        self.tally
    }

    /// The world AABB of everything this stroke has cut so far, `None` before
    /// the first dab.
    pub fn bounds(&self) -> Option<(DVec3, DVec3)> {
        self.bounds
    }

    /// Displace this stroke's removed material to `site` as a spoil pile,
    /// **into the same undo record** (P21.3).
    ///
    /// Returns the refusal when the pile could not hold the count; the caller
    /// then abandons the transaction rather than committing an unbalanced one.
    /// On success the tally's
    /// [`spoiled_by_material`](CarveTally::spoiled_by_material) equals its
    /// [`carved_by_material`](CarveTally::carved_by_material) exactly, which is
    /// what [`CarveTally::conserved`] reports and what the gate asserts.
    ///
    /// Idempotence is **not** claimed and is not wanted: calling this twice
    /// would place the soil twice. It is called once, by the one door that
    /// closes a dig.
    pub fn spoil(&mut self, site: DVec3) -> Result<(), CarveRefusal> {
        let counts = self.tally.carved_by_material;
        if counts.iter().all(|&n| n == 0) {
            return Ok(()); // a dig that removed nothing spoils nothing
        }
        // **The heap may never place back into the hole it came out of** (P21.3
        // audit): the default site's clearance is the pile's *analytic* radius
        // and the real footprint is wider, so a big dig used to refill part of
        // its own pit — with conservation balancing perfectly while it did,
        // because a refilled pit is an impeccably conserved place to put soil.
        // The stroke's own bounds are the exclusion, so the guarantee is
        // structural rather than arithmetic.
        let mut plan = SpoilPlan::new(counts, site);
        if let Some((lo, hi)) = self.bounds {
            plan = plan.excluding(lo, hi);
        }
        let mut volumes = self
            .volumes
            .lock()
            .map_err(|_| CarveRefusal::PoisonedStore)?;
        let report = if self.remesh {
            volumes.spoil_into(self.volume, &plan, &mut self.voxels)
        } else {
            volumes.spoil_into_deferred(self.volume, &plan, &mut self.voxels)
        }
        .ok_or(CarveRefusal::VolumeNotLoaded)?;
        if !report.is_exact() {
            return Err(CarveRefusal::SpoilShortfall);
        }
        self.tally.spoiled_by_material = report.placed;
        self.tally.filled += report.total_placed();
        self.tally.touched += report.total_placed();
        self.tally.created_chunks += report.created_chunks;
        Ok(())
    }

    /// Lay one dab: cut the voxels, then open (or close) the heightfield samples
    /// the cut crosses on every `terrain` it reaches.
    ///
    /// `terrains` is the verdict's list, resolved once when the gesture started —
    /// re-deriving it per dab would let a stroke wander onto a terrain the author
    /// was never told about (and, if that one were inline, onto ground whose
    /// mouths the save would silently seal).
    ///
    /// **The caller must have paged the cut's footprint into the document first**
    /// for a streamed terrain, exactly as the sculpt brush does: the hole rule
    /// only sees authored tiles, and a mouth cannot be punched in a page that is
    /// not in memory.
    ///
    /// # The hole half is strictly conditional on the voxel half (P21.2 audit)
    ///
    /// `Err` — and **not one hole bit written** — when the rock could not be cut
    /// at all. This used to run the two halves independently: the voxel write was
    /// wrapped in `if let Ok(…)` / `if let Some(…)` and the heightfield loop below
    /// ran regardless, so a released volume or a poisoned working set produced a
    /// mouth in the ground over solid rock — the exact "cave with no mouth"
    /// failure `RefusedInline` exists to prevent, arrived at from the other side,
    /// and this one is *saveable*: the hole mask rides the `.inf_terrain`, so a
    /// Ctrl+S commits a hole into ground nothing ever hollowed.
    ///
    /// The refusal is a value rather than a log line because the tools put it on
    /// `viewport://tool-status`, where the author is looking.
    pub fn dab(
        &mut self,
        doc: &mut SceneDoc,
        op: &VoxelOp,
        terrains: &[Uuid],
    ) -> Result<CarveTally, CarveRefusal> {
        let mut moved = false;
        // The voxel half FIRST, and its answer decides whether the ground is
        // allowed to open. Both failure returns are refusals, not empty edits:
        // `lock` fails when a thread panicked holding the chunks, and
        // `carve_into` answers `None` when this volume has no loaded chunks.
        let mut volumes = self
            .volumes
            .lock()
            .map_err(|_| CarveRefusal::PoisonedStore)?;
        let report = if self.remesh {
            volumes.carve_into(self.volume, op, &mut self.voxels)
        } else {
            volumes.carve_into_deferred(self.volume, op, &mut self.voxels)
        }
        .ok_or(CarveRefusal::VolumeNotLoaded)?;
        self.tally.touched += report.touched;
        self.tally.carved += report.total_carved();
        self.tally.filled += report.total_filled();
        self.tally.created_chunks += report.created_chunks;
        for m in 0..MATERIAL_COUNT {
            // The excavation ledger's left-hand side, accumulated per dab so a
            // hundred-dab drag spoils exactly what the whole stroke removed.
            self.tally.carved_by_material[m] += report.carved[m];
        }
        // The gesture's own footprint, for the default spoil rule. Taken from
        // the shape and not from the report, so it is the region the author
        // described rather than the subset that happened to contain rock.
        //
        // **Only for a VALID shape** (P21.3 audit): `aabb_m` answers
        // `(ZERO, ZERO)` for a degenerate one — a zero-length trench, a NaN from
        // the first frame of a drag — and unioning that box would drag the
        // stroke's bounds (and therefore the spoil exclusion and the default
        // site) all the way to the world origin.
        let (lo, hi) = op.shape.aabb_m(0.0);
        if op.shape.is_valid() && lo.is_finite() && hi.is_finite() {
            self.bounds = Some(match self.bounds {
                Some((l, h)) => (l.min(lo), h.max(hi)),
                None => (lo, hi),
            });
        }
        moved |= !report.is_noop();
        // Released before the heightfield half so the two locks are never held at
        // once — the fixed order is document first, volumes second, and this
        // function already holds the document.
        drop(volumes);
        // A carve opens the surface; a fill closes it again. The `open` flag is
        // the op's kind and nothing else, which is what makes the fill of a
        // region clear exactly the bits its carve set.
        let open = matches!(op.kind, VoxelOpKind::Carve);
        for &terrain in terrains {
            let Some(origin) = doc.terrain_data_and_origin(terrain).map(|(_, o)| o) else {
                continue;
            };
            let builder = self.holes.entry(terrain).or_default();
            let changed = doc
                .with_terrain_data_mut(terrain, |data| {
                    inf_voxel::touch_surface_cut(data, origin, &op.shape, open, builder)
                })
                .unwrap_or(0);
            self.tally.holes += changed as u64;
            moved |= changed > 0;
        }
        if moved {
            // The document version is bumped for a **voxel-only** cut too, not
            // just for one that moved the hole mask. The viewport's projection is
            // version-gated, so a cave dug entirely below the surface — which
            // touches no terrain tile at all — would otherwise be meshed in the
            // store and never reach the screen until an unrelated edit happened
            // to bump the document. `with_terrain_data_mut` is the streamer's
            // residency door and is deliberately non-touching, so this seam owns
            // both the bump and the dirty flag either way.
            doc.world_mut().mark_dirty();
            doc.touch();
        }
        Ok(self.tally)
    }

    /// Close the stroke into the reversible pair, or `None` when it changed
    /// nothing (a click that missed, a re-carve of ground already hollow).
    fn finish(self, doc: &SceneDoc) -> Option<(VoxelDelta, Vec<(Uuid, HoleDelta)>)> {
        let delta = self
            .volumes
            .lock()
            .ok()
            .and_then(|v| v.finish_carve(self.volume, &self.voxels))
            .unwrap_or_default();
        let holes: Vec<(Uuid, HoleDelta)> = self
            .holes
            .iter()
            .filter_map(|(&guid, builder)| {
                let data = doc.terrain_data_and_origin(guid)?.0;
                let d = builder.finalize(data);
                (!d.is_empty()).then_some((guid, d))
            })
            .collect();
        if delta.is_empty() && holes.is_empty() {
            return None;
        }
        Some((delta, holes))
    }
}

impl SceneDoc {
    /// Decide what a cut of `shape` (world metres) is allowed to do to this
    /// document's terrains — **the gate every carve tool takes first**.
    ///
    /// Walks the document's terrains in creation order and asks
    /// [`inf_voxel::cut_crosses_surface`] about each. Reading the *document's*
    /// working set and not the viewport's render cut is deliberate: the document
    /// is what gets saved, so the terrain that has to be able to persist a mouth
    /// is the one whose tiles are about to be written.
    ///
    /// A cut over a streamed terrain whose footprint has not been paged in yet
    /// reads as [`NoSurface`](CarveVerdict::NoSurface). That is not a hole in the
    /// gate — the same paging the sculpt brush already does before its first dab
    /// is what makes the answer true, and it is why the tools page before they
    /// ask.
    pub fn carve_verdict(&self, shape: &VoxelShape) -> CarveVerdict {
        let mut terrains = Vec::new();
        for &guid in self.order() {
            let Some((data, origin)) = self.terrain_data_and_origin(guid) else {
                continue;
            };
            if !inf_voxel::cut_crosses_surface(data, origin, shape) {
                continue;
            }
            if self.terrain_asset_of(guid).is_none() {
                return CarveVerdict::RefusedInline { terrain: guid };
            }
            terrains.push(guid);
        }
        if terrains.is_empty() {
            CarveVerdict::NoSurface
        } else {
            CarveVerdict::Opens { terrains }
        }
    }

    /// Commit a [`CarveStroke`] as **one** undo entry. Returns whether anything
    /// was recorded (an empty stroke records nothing, like every other brush).
    ///
    /// The stroke's dabs already mutated both containers, so this only finalizes
    /// and records — the shape `edit_commit_sculpt` established one layer down.
    pub fn edit_commit_carve(&mut self, stroke: CarveStroke) -> bool {
        let volume = stroke.volume;
        let volumes = stroke.volumes.clone();
        // The undo label names what the author did, which is not always a carve:
        // a dig that displaced its soil is an *excavation*, and "Undo Carve
        // Voxels" over a step that also removed a spoil heap describes half of
        // itself.
        let label = if stroke.tally().spoiled() > 0 {
            "Excavate"
        } else {
            "Carve Voxels"
        };
        let Some((delta, holes)) = stroke.finish(self) else {
            return false;
        };
        self.record_edit(
            label,
            EditCommand::CarveVoxels {
                volume,
                volumes,
                delta: Box::new(delta),
                holes,
            },
        );
        true
    }

    /// Close a brush stroke as a **dig**: displace its soil, then commit both
    /// halves as one undo entry (P21.3).
    ///
    /// The carve brush's door. It cannot go through [`edit_dig`](Self::edit_dig)
    /// — a brush stroke's dabs were already cut, live, one frame at a time —
    /// so this is the tail of that function with the two passes it has already
    /// run removed.
    ///
    /// Returns the **final** ledger (spoil included, which is why the caller
    /// cannot read it off the stroke it just gave away), whether an undo entry
    /// was recorded, and the refusal if the soil could not be placed. A failed
    /// spoil still commits the cut: rock the author can see must be undoable
    /// (`a4e5844`), and the refusal is what the readout quotes instead of a
    /// balance it did not achieve.
    pub fn edit_commit_dig(
        &mut self,
        mut stroke: CarveStroke,
        spoil: SpoilChoice,
    ) -> (CarveTally, bool, Option<CarveRefusal>) {
        let voxel_size_m = stroke
            .volumes
            .lock()
            .ok()
            .and_then(|v| v.slot(stroke.volume).map(|s| s.data.voxel_size_m()))
            .unwrap_or_else(|| inf_ecs::components::VoxelVolume::default().voxel_size_m);
        let refusal = self.spoil_stroke(&mut stroke, spoil, voxel_size_m).err();
        let tally = stroke.tally();
        (tally, self.edit_commit_carve(stroke), refusal)
    }

    /// The one-shot carve: apply `op` and record it, holes included. Returns the
    /// tally, or the [`CarveRefusal`] that stopped it.
    ///
    /// The seam a scripted or single-click cut uses; a brush goes through
    /// [`CarveStroke`] so its dabs merge.
    pub fn edit_carve(
        &mut self,
        volume: Uuid,
        volumes: &SharedVoxelVolumes,
        op: &VoxelOp,
    ) -> Result<CarveTally, CarveRefusal> {
        self.edit_carve_path(volume, volumes, std::slice::from_ref(op))
    }

    /// Cut a **chain** of ops as ONE transaction: judge every one first, then cut
    /// every one. `Err` — and nothing written at all — if any is refused.
    ///
    /// The spline tunnel's door, and the reason it is here rather than in the
    /// viewport host is the atomicity: judging as it went would leave the *first
    /// half* of a tunnel in the volume with no undo entry describing it, which is
    /// a partial success that is not even reversible — strictly worse than the
    /// one [`CarveVerdict::RefusedInline`] exists to prevent. Ring 1 also means
    /// all three CI legs run it, while `inf_viewport::host` is
    /// `#[cfg(any(windows, macos))]`.
    ///
    /// **The caller pages first.** The verdict reads authored tiles only, so on a
    /// streamed terrain an unpaged footprint answers "this reaches no surface"
    /// and waves a breakthrough through unchecked.
    pub fn edit_carve_path(
        &mut self,
        volume: Uuid,
        volumes: &SharedVoxelVolumes,
        ops: &[VoxelOp],
    ) -> Result<CarveTally, CarveRefusal> {
        self.edit_dig(volume, volumes, ops, SpoilChoice::Discard)
    }

    /// **The excavation transaction** (P21.3): cut a chain of ops and displace
    /// what they removed, as ONE undo entry.
    ///
    /// The dig tools' single door — box cut, spline trench and the carve brush's
    /// committed stroke all arrive here — and the superset of
    /// [`edit_carve_path`](Self::edit_carve_path), which is now this with
    /// [`SpoilChoice::Discard`].
    ///
    /// # Judged whole, cut whole, spoiled whole
    ///
    /// Four questions are answered **before a single sample moves**:
    ///
    /// 1. is the dig within [`MAX_DIG_SAMPLES`]? (a pure function of the shapes)
    /// 2. is the working set readable?
    /// 3. is the target volume loaded?
    /// 4. does any leg break through an **inline** terrain?
    ///
    /// Only then does anything cut. That ordering is `a4e5844`'s ruling applied
    /// to a pit: a refusal discovered halfway through a foundation excavation
    /// would leave a half-dug hole the author never asked for, and the size gate
    /// in particular *cannot* be discovered any other way — you find out a dig
    /// is too big by doing it.
    ///
    /// The spoil runs last, inside the same stroke, so the pit, its cave mouths
    /// and the heap it produced are one `EditCommand` and one Ctrl+Z. A spoil
    /// that cannot be placed is a refusal **after** the cut, and the transaction
    /// is committed anyway with the refusal returned — the same ruling the
    /// mid-chain refusal above gets, and for the same reason: rock the author
    /// can see, with no undo entry describing it, is the worst of the three
    /// outcomes.
    pub fn edit_dig(
        &mut self,
        volume: Uuid,
        volumes: &SharedVoxelVolumes,
        ops: &[VoxelOp],
        spoil: SpoilChoice,
    ) -> Result<CarveTally, CarveRefusal> {
        // Pass 0 — SIZE. A pure function of the shapes, so it can answer before
        // anything is read, let alone written.
        let voxel_size_m = volumes
            .lock()
            .ok()
            .and_then(|v| v.slot(volume).map(|s| s.data.voxel_size_m()))
            .unwrap_or_else(|| inf_ecs::components::VoxelVolume::default().voxel_size_m);
        let mut samples = 0u64;
        for op in ops {
            samples = samples.saturating_add(op.shape.affected_sample_count(voxel_size_m));
        }
        if samples > MAX_DIG_SAMPLES {
            return Err(CarveRefusal::TooLarge { samples });
        }
        // Pass 1 — judge. Nothing is cut yet, so a refusal here really is a
        // refusal of the whole op and not of its tail.
        //
        // **The voxel half is judged in the same pass** (P21.2 audit): a chain
        // that only discovered a released volume on its third leg would already
        // have cut two, and `dab`'s per-leg refusal is a backstop rather than the
        // gate. Asking here — once, before anything moves — is what makes the
        // refusal that actually happens (a volume whose asset never resolved, or
        // one the projection released) atomic like the inline one.
        match volumes.lock() {
            Err(_) => return Err(CarveRefusal::PoisonedStore),
            Ok(v) if v.slot(volume).is_none() => return Err(CarveRefusal::VolumeNotLoaded),
            Ok(_) => {}
        }
        let mut plan: Vec<Vec<Uuid>> = Vec::with_capacity(ops.len());
        for op in ops {
            match self.carve_verdict(&op.shape) {
                CarveVerdict::RefusedInline { terrain } => {
                    return Err(CarveRefusal::InlineTerrain { terrain })
                }
                v => plan.push(v.terrains().to_vec()),
            }
        }
        // Pass 2 — cut, into one stroke, so the chain is one undo entry.
        //
        // **Deferred** (P21.4): a dig is committed on mouse-up as one shape (or a
        // handful), so nobody is watching the surface appear under a cursor — and
        // the re-mesh is the part of the transaction with no reason to hold the
        // shared-volumes guard. The mesh cache is stamp-keyed on the chunk
        // versions this cut bumps, so the viewport's next `sync_camera` rebuilds
        // exactly the same chunks, on the render thread, outside this lock.
        let mut stroke = CarveStroke::begin_deferred(volume, volumes.clone());
        for (op, terrains) in ops.iter().zip(&plan) {
            match stroke.dab(self, op, terrains) {
                Ok(_) => {}
                Err(refusal) => {
                    // A leg refused after earlier legs cut — the volume was
                    // released or its store poisoned *between* two legs, which
                    // pass 1 cannot rule out because another thread owns the
                    // working set. Commit what is already in the world anyway:
                    // `a4e5844` established that an un-undoable partial tunnel is
                    // strictly worse than a partial one, and this leg opened no
                    // ground (`dab` refuses before the heightfield half), so the
                    // entry describes exactly what happened.
                    self.edit_commit_carve(stroke);
                    return Err(refusal);
                }
            }
        }
        // Pass 3 — displace the soil, into the SAME stroke.
        if let Err(refusal) = self.spoil_stroke(&mut stroke, spoil, voxel_size_m) {
            self.edit_commit_carve(stroke);
            return Err(refusal);
        }
        let tally = stroke.tally();
        self.edit_commit_carve(stroke);
        Ok(tally)
    }

    /// Resolve a [`SpoilChoice`] against `stroke`'s own footprint and place the
    /// pile.
    ///
    /// Split out because the brush's committed stroke reaches it from
    /// `edit_commit_dig` while the box and trench cuts reach it from
    /// [`edit_dig`](Self::edit_dig), and a second copy of "where does the soil
    /// go" is how the two would come to answer differently.
    pub(crate) fn spoil_stroke(
        &self,
        stroke: &mut CarveStroke,
        spoil: SpoilChoice,
        voxel_size_m: f64,
    ) -> Result<(), CarveRefusal> {
        // **The size gate the brush path had no other place to get** (P21.3
        // audit, B1). `edit_dig` bounds its ops before it cuts, but a brush
        // stroke's dabs are already in the world and `CarveStroke::dab`
        // accumulates across frames with no ceiling — so an hour of dragging
        // arrives here as one unbounded count, and the spoil search reacts to it
        // by growing 32 times and multiplying three `i32`-clamped spans into a
        // panic. Under the volumes guard. Mid-transaction. This is the bound.
        let removed: u64 = stroke.tally().carved_by_material.iter().sum();
        if removed > MAX_DIG_SAMPLES {
            return Err(CarveRefusal::SpoilTooLarge { voxels: removed });
        }
        let Some(site) = self.spoil_site(stroke, spoil, voxel_size_m) else {
            return Ok(());
        };
        stroke.spoil(site)
    }

    /// Where this stroke's soil goes, or `None` when it is discarded / there is
    /// nothing to displace.
    ///
    /// The `Auto` rule is Ring 0's [`inf_voxel::default_spoil_site`] — east of
    /// the cut, clear of its rim — with **one** thing added that Ring 0 cannot
    /// know: the pile is dropped onto the ground under that point when this
    /// level has terrain there. A heap standing at the height of the pit's rim
    /// on sloping ground would float or bury itself, and neither is what an
    /// author dragging a pit on a hillside means.
    pub fn spoil_site(
        &self,
        stroke: &CarveStroke,
        spoil: SpoilChoice,
        voxel_size_m: f64,
    ) -> Option<DVec3> {
        let total: u64 = stroke.tally().carved_by_material.iter().sum();
        if total == 0 {
            return None;
        }
        match spoil {
            SpoilChoice::Discard => None,
            SpoilChoice::At(p) if p.is_finite() => Some(p),
            SpoilChoice::At(_) => None,
            SpoilChoice::Auto => {
                let (lo, hi) = stroke.bounds()?;
                let mut site = inf_voxel::default_spoil_site(lo, hi, total, voxel_size_m);
                if let Some(y) = self.ground_surface_y(site.x, site.z) {
                    site.y = y;
                }
                Some(site)
            }
        }
    }

    /// The **topmost** terrain surface at world XZ, over this document's
    /// terrains in creation order, or `None` where no terrain answers.
    ///
    /// Topmost and not nearest, because this answers "what would a heap stand
    /// on?". `height_at` is the poisoned bilinear query, so a hole reads as no
    /// ground — which is right: a pile must not be stood on the lid of a cave
    /// that is not there.
    pub fn ground_surface_y(&self, x: f64, z: f64) -> Option<f64> {
        let mut best: Option<f64> = None;
        for &guid in self.order() {
            let Some((data, origin)) = self.terrain_data_and_origin(guid) else {
                continue;
            };
            let local = glam::DVec2::new(x - origin.x, z - origin.z);
            let Some(h) = data.height_at(local) else {
                continue;
            };
            let y = origin.y + h;
            if best.is_none_or(|b| y > b) {
                best = Some(y);
            }
        }
        best
    }

    /// Redo (`revert = false`) or undo (`revert = true`) a carve: both halves,
    /// in one call, so neither can be replayed without the other.
    ///
    /// The voxel half goes through the shared store (which re-meshes what moved);
    /// the heightfield half through the document's own tiles, healing every
    /// touched mask so an undone carve leaves tiles byte-identical to ones
    /// nothing ever carved.
    ///
    /// # The coupling rule runs backwards too (P21.2 audit)
    ///
    /// If the rock half cannot be replayed — the volume was released, or its
    /// working set is poisoned — the heightfield half **does not run either**, and
    /// the failure is logged rather than swallowed. Healing the mouths over rock
    /// that is still carved is the same "the two halves came apart" state this
    /// command exists to make impossible, and it is the half that a save would
    /// then commit into the `.inf_terrain`.
    ///
    /// An **empty** `delta` is not a refusal: a cut that opened a mouth without
    /// moving a sample (a brush swung through air over a hillside) has no rock
    /// half to replay, and its mouths are the whole record.
    pub(crate) fn raw_write_carve(
        &mut self,
        volume: Uuid,
        volumes: &SharedVoxelVolumes,
        delta: &VoxelDelta,
        holes: &[(Uuid, HoleDelta)],
        revert: bool,
    ) {
        let what = if revert { "undo" } else { "redo" };
        let voxels_replayed = match volumes.lock() {
            Ok(mut v) => {
                if revert {
                    v.revert_delta(volume, delta)
                } else {
                    v.apply_delta(volume, delta)
                }
            }
            // The line has to match what the code three statements down actually
            // does, or the log is a second, contradictory specification: the
            // `!delta.is_empty()` guard below only stops the mouths when there was
            // a rock half to lose, so a holes-only record is replayed here despite
            // the poison — correctly, since it never had one.
            Err(_) if delta.is_empty() => {
                tracing::warn!(
                    "inf-editor-core: carve {what} could not read the voxel working set (a \
                     thread panicked holding it), but this record moved no voxel samples — \
                     its {} cave-mouth record(s) are replayed as usual",
                    holes.len()
                );
                false
            }
            Err(_) => {
                tracing::error!(
                    "inf-editor-core: carve {what} could not read the voxel working set (a \
                     thread panicked holding it) — neither the rock nor the {} cave mouth \
                     record(s) were replayed, so the two halves stay consistent",
                    holes.len()
                );
                false
            }
        };
        if !delta.is_empty() && !voxels_replayed {
            tracing::error!(
                "inf-editor-core: carve {what} found no loaded volume {volume} — its {} \
                 sample patch(es) were not replayed, so the {} cave-mouth record(s) above \
                 them were skipped too rather than leaving holes over solid rock",
                delta.patches.len(),
                holes.len()
            );
            self.world_mut().mark_dirty();
            self.touch();
            return;
        }
        for (guid, hole) in holes {
            // **Both answers are checked** (P21.3 audit). `with_terrain_data_mut`
            // returns `None` when the entity has no terrain at all — a deleted
            // terrain, a cleared component — and the inner call returns how many
            // patches it had to skip because their tile is no longer resident.
            // Dropping either one turns "the cave mouths were not put back" into
            // silence, on the exact path whose whole job is keeping the rock and
            // the ground consistent.
            let skipped = self.with_terrain_data_mut(*guid, |data| {
                if revert {
                    data.revert_hole_delta(hole)
                } else {
                    data.apply_hole_delta(hole)
                }
            });
            match skipped {
                None => tracing::error!(
                    "inf-editor-core: carve {what} found no terrain {guid} — its {} cave-mouth \
                     patch(es) were not replayed, so the ground and the rock have come apart. \
                     Prefer re-carving over saving this level.",
                    hole.patches.len()
                ),
                Some(n) if n > 0 => tracing::warn!(
                    "inf-editor-core: carve {what} skipped {n} of {} cave-mouth patch(es) on \
                     terrain {guid} — their tiles are not paged in, so those mouths keep their \
                     current state until the tiles return.",
                    hole.patches.len()
                ),
                Some(_) => {}
            }
        }
        self.world_mut().mark_dirty();
        self.touch();
    }
}

pub(crate) struct Transaction {
    pub label: String,
    pub commands: Vec<EditCommand>,
}

impl Transaction {
    /// Approximate heap footprint of this entry — the sum of its commands' (see
    /// [`EditCommand::memory_bytes`]).
    fn memory_bytes(&self) -> usize {
        self.commands
            .iter()
            .map(EditCommand::memory_bytes)
            .fold(self.label.len(), usize::saturating_add)
    }
}

/// The undo/redo stacks plus the currently-open transaction.
pub struct EditHistory {
    undo: Vec<Transaction>,
    redo: Vec<Transaction>,
    open: Option<Transaction>,
    /// Open-transaction nesting depth. `begin` increments it, `commit`
    /// decrements it; the transaction closes only when the OUTERMOST commit
    /// brings this back to zero, so begin/begin/commit/commit nests correctly.
    depth: u32,
    limit: usize,
    /// Byte ceiling for the two stacks together (see [`HISTORY_BYTE_LIMIT`]).
    byte_limit: usize,
}

impl Default for EditHistory {
    fn default() -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            open: None,
            depth: 0,
            limit: HISTORY_LIMIT,
            byte_limit: HISTORY_BYTE_LIMIT,
        }
    }
}

impl EditHistory {
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Number of undo entries currently on the stack (bounded by [`HISTORY_LIMIT`]).
    /// Surfaced by the memory diagnostics (P15) — the "undo stack depth" budget.
    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    /// Number of redo entries currently on the stack.
    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Label of the next undo/redo entry (for the Edit menu: "Undo Rename").
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|t| t.label.as_str())
    }

    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|t| t.label.as_str())
    }

    pub(crate) fn begin(&mut self, label: &str) {
        // Nested begins fold into the outer transaction; the depth counter tracks
        // the nesting so only the matching outermost commit closes it.
        if self.open.is_none() {
            self.open = Some(Transaction {
                label: label.to_string(),
                commands: Vec::new(),
            });
        }
        self.depth += 1;
    }

    /// Record a command: append to the open transaction, or commit it as a
    /// standalone entry labelled `label`. Clears the redo stack (a new edit
    /// forks history).
    pub(crate) fn record(&mut self, label: &str, cmd: EditCommand) {
        if let Some(open) = self.open.as_mut() {
            open.commands.push(cmd);
        } else {
            self.push(Transaction {
                label: label.to_string(),
                commands: vec![cmd],
            });
        }
    }

    pub(crate) fn commit(&mut self) {
        // Only the outermost commit closes the transaction; inner commits just
        // unwind one level of nesting.
        if self.depth > 1 {
            self.depth -= 1;
            return;
        }
        self.depth = 0;
        if let Some(txn) = self.open.take() {
            if !txn.commands.is_empty() {
                self.push(txn);
            }
        }
    }

    /// `true` while a transaction is open (nesting depth ≥ 1).
    pub(crate) fn has_open(&self) -> bool {
        self.open.is_some()
    }

    /// **Force-close a stranded transaction, whatever its nesting depth.**
    ///
    /// Returns `true` when one was closed *and* had commands to record.
    ///
    /// # The failure this exists for — session-wide undo death
    ///
    /// [`begin`](Self::begin) increments `depth` and [`commit`](Self::commit)
    /// decrements it, and only the *outermost* commit closes the transaction. So
    /// one `begin` with no matching `commit` leaves `open = Some(..)` and
    /// `depth = 1` **forever**: every later begin/commit pair bounces 1 → 2 → 1,
    /// every subsequent edit is appended to the stranded transaction instead of
    /// pushed as its own entry, `undo_len()` stops growing, and **Ctrl+Z is
    /// silently dead for the rest of the session**. Nothing surfaces it — the
    /// edits all land in the world, the document is dirty, the save works.
    ///
    /// It is reachable today: the win32 pump opens `"Move"` when a gizmo drag
    /// begins and commits it on release, both inside the tool-gated select
    /// branch, so *hold a translate handle → Ctrl+Shift+P → `tool.sculpt` →
    /// release* strands one. The foliage stroke has the same shape.
    ///
    /// Committing rather than discarding, for the settlement pattern's reason:
    /// the edits are in the world and the author can see them, so the only
    /// question is whether Ctrl+Z can reach them.
    pub(crate) fn settle_open(&mut self) -> bool {
        self.depth = 0;
        match self.open.take() {
            Some(txn) if !txn.commands.is_empty() => {
                self.push(txn);
                true
            }
            _ => false,
        }
    }

    /// **Approximate heap footprint of both stacks** (Hardening D).
    ///
    /// Walked rather than cached, because the alternative — a running total —
    /// has to be kept in step with `take_undo`/`push_redo`/`take_redo`/`clear`
    /// and every path that moves an entry between the stacks, and a byte counter
    /// that drifts is worse than one that costs a walk. The walk is over at most
    /// `2 × HISTORY_LIMIT` entries of arithmetic (`memory_bytes` sums patch
    /// *counts*, it does not touch the buffers), and its two callers are `push`
    /// and a diagnostics command.
    pub fn memory_bytes(&self) -> usize {
        self.undo
            .iter()
            .chain(&self.redo)
            .map(Transaction::memory_bytes)
            .fold(0usize, usize::saturating_add)
    }

    fn push(&mut self, txn: Transaction) {
        self.redo.clear();
        self.undo.push(txn);
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
        // **And the byte ceiling** — the count bound is a bound on entries, not
        // on memory: one sculpt stroke over a large terrain patch is megabytes,
        // and 256 of them honour `HISTORY_LIMIT` perfectly. Oldest-first, like
        // the count eviction, and never past the last entry: a single edit that
        // is on its own over the ceiling stays, because a history that can throw
        // away the thing you just did is not a history.
        while self.undo.len() > 1 && self.memory_bytes() > self.byte_limit {
            self.undo.remove(0);
        }
    }

    pub(crate) fn take_undo(&mut self) -> Option<Transaction> {
        self.undo.pop()
    }

    pub(crate) fn take_redo(&mut self) -> Option<Transaction> {
        self.redo.pop()
    }

    pub(crate) fn push_redo(&mut self, txn: Transaction) {
        self.redo.push(txn);
    }

    pub(crate) fn push_undo(&mut self, txn: Transaction) {
        self.undo.push(txn);
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
        self.open = None;
        self.depth = 0;
    }
}

#[cfg(test)]
mod tests {
    use crate::ipc::{SceneNode, SpawnKind};
    use crate::scene::SceneDoc;
    use inf_ecs::components::Transform;
    use inf_ecs::math::Vec3d;

    /// Order-independent scene comparison (name/kind/parent/visibility).
    fn fingerprint(doc: &mut SceneDoc) -> Vec<SceneNode> {
        let mut nodes = doc.snapshot().nodes;
        nodes.sort_by(|a, b| a.guid.cmp(&b.guid));
        for n in &mut nodes {
            n.children.sort();
        }
        nodes
    }

    #[test]
    fn undo_redo_restores_each_mutation() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let before = fingerprint(&mut doc);

        // A rename, a child create, a reparent, a visibility toggle.
        doc.edit_rename(a, "Renamed");
        let b = doc.edit_create(SpawnKind::Cube, "B", Some(a));
        doc.edit_reparent(b, None);
        doc.edit_set_visible(a, false);

        // Four independent edits → four undo steps back to `before`.
        for _ in 0..4 {
            assert!(doc.undo());
        }
        assert_eq!(fingerprint(&mut doc), before, "undo did not restore state");

        // Redo them all forward again.
        for _ in 0..4 {
            assert!(doc.redo());
        }
        assert!(!doc.can_redo());
    }

    #[test]
    fn fifty_step_undo_redo_is_clean() {
        let mut doc = SceneDoc::new();
        let base = fingerprint(&mut doc);
        let mut guids = Vec::new();
        for i in 0..50 {
            guids.push(doc.edit_create(SpawnKind::Cube, &format!("C{i}"), None));
        }
        let full = fingerprint(&mut doc);

        for _ in 0..50 {
            assert!(doc.undo());
        }
        assert_eq!(fingerprint(&mut doc), base, "50 undos must reach the start");
        assert!(!doc.can_undo());

        for _ in 0..50 {
            assert!(doc.redo());
        }
        assert_eq!(
            fingerprint(&mut doc),
            full,
            "50 redos must restore everything"
        );
    }

    fn translation_x(doc: &SceneDoc, guid: uuid::Uuid) -> f64 {
        let props = doc.entity_props(guid);
        let t = props.iter().find(|p| p.display == "Transform").unwrap();
        match &t
            .fields
            .iter()
            .find(|f| f.name == "translation")
            .unwrap()
            .value
        {
            inf_ecs::PropValue::Vec3(v) => v[0],
            _ => panic!("translation not a vec3"),
        }
    }

    #[test]
    fn transaction_groups_into_one_step() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);

        // A gizmo drag: many transform edits stream into one transaction.
        doc.begin_transaction("drag");
        for i in 1..=10 {
            doc.edit_set_transform(
                a,
                Transform {
                    translation: Vec3d::new(i as f64, 0.0, 0.0),
                    ..Transform::IDENTITY
                },
            );
        }
        doc.commit_transaction();
        assert_eq!(translation_x(&doc, a), 10.0);

        // A single undo reverts the whole drag back to the origin (not just the
        // last of the ten edits).
        assert!(doc.undo());
        assert_eq!(translation_x(&doc, a), 0.0, "the drag is one undo step");
        // The remaining undo entry is the create.
        assert!(doc.undo());
        assert!(!doc.can_undo());
    }

    #[test]
    fn delete_undo_restores_subtree() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Empty, "A", None);
        let _b = doc.edit_create(SpawnKind::Cube, "B", Some(a));
        let _c = doc.edit_create(SpawnKind::Sphere, "C", Some(a));
        let full = fingerprint(&mut doc);

        doc.edit_delete(&[a]);
        assert!(doc.snapshot().nodes.is_empty());

        assert!(doc.undo());
        assert_eq!(
            fingerprint(&mut doc),
            full,
            "deleted subtree must round-trip"
        );
    }

    /// Deleting a reparented pair together and undoing restores the hierarchy
    /// even when the child sits at an EARLIER order slot than its parent (A
    /// created before B, then A reparented under B). The two-pass respawn
    /// (spawn-all, then fix-up-parents) re-attaches A under B instead of
    /// silently rooting it.
    #[test]
    fn delete_undo_restores_reparent_under_later_node() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);
        let b = doc.edit_create(SpawnKind::Empty, "B", None); // later order slot
        assert!(doc.edit_reparent(a, Some(b)));

        doc.edit_delete(&[a, b]);
        assert!(doc.snapshot().nodes.is_empty(), "both deleted");
        assert!(doc.undo());

        // A is restored UNDER B (not as a root).
        let ea = doc.entity_of(a).expect("A restored by undo");
        let parent = doc
            .world()
            .parent_of(ea)
            .and_then(|p| doc.world().guid_of(p));
        assert_eq!(parent, Some(b), "A stays under B after delete→undo");
        assert!(doc.entity_of(b).is_some(), "B restored by undo");
    }

    /// Editing the level settings dirties + bumps the version, and undo/redo
    /// round-trip the whole settings block (gravity + render/post) exactly (R-P4).
    #[test]
    fn level_settings_edit_undo_redo_round_trips() {
        use crate::scene::serialize::LevelSettings;

        let mut doc = SceneDoc::new();
        let base = doc.settings();
        let v0 = doc.version();
        assert!(!doc.is_dirty());

        let mut edited = base;
        edited.render.exposure = 2.5;
        edited.render.bloom_enabled = true;
        edited.sim_hz = 120.0;
        doc.edit_settings(edited);

        assert!(doc.is_dirty(), "a settings edit dirties the document");
        assert!(doc.version() > v0, "a settings edit bumps the version");
        assert_eq!(doc.settings(), edited);

        // Undo restores the original settings exactly …
        assert!(doc.undo());
        assert_eq!(doc.settings(), base);
        // … and redo re-applies the edited block.
        assert!(doc.redo());
        assert_eq!(doc.settings(), edited);

        // An idempotent edit (same value) records nothing.
        let before_len = doc.undo_len();
        doc.edit_settings(edited);
        assert_eq!(doc.undo_len(), before_len, "no-op edit records nothing");
        assert_eq!(doc.settings(), LevelSettings { ..edited });
    }

    /// P17.1: editing the time of day from World Settings is **one** undo step,
    /// and the first edit creates the sky authority inside that same step — so a
    /// single Ctrl+Z takes the whole opt-in back out, entity and all.
    #[test]
    fn time_of_day_edit_creates_the_authority_in_one_undo_step() {
        use inf_ecs::components::TimeOfDay;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        assert!(doc.time_of_day().is_none(), "a new level has no clock");

        let authored = TimeOfDay {
            seconds: 43_200.0,
            day_of_year: 355,
            latitude_deg: -33.9,
            longitude_deg: 151.2,
            rate: 60.0,
        };
        let guid = doc.edit_time_of_day(authored, true).expect("created");

        assert_eq!(doc.time_of_day(), Some(authored));
        assert_eq!(doc.sky_authority(), Some(guid));
        assert_eq!(doc.order().len(), entities + 1, "one Sky actor appeared");
        assert_eq!(
            doc.undo_len(),
            undos + 1,
            "create + two components + five fields collapse into ONE step"
        );
        assert!(doc.is_dirty());

        // Undo removes the whole opt-in …
        assert!(doc.undo());
        assert!(doc.time_of_day().is_none());
        assert_eq!(doc.order().len(), entities);
        // … and redo restores it exactly.
        assert!(doc.redo());
        assert_eq!(doc.time_of_day(), Some(authored));

        // A second edit re-uses the authority (no new entity) and is still one step.
        let undos = doc.undo_len();
        let later = TimeOfDay {
            seconds: 0.0,
            ..authored
        };
        assert_eq!(doc.edit_time_of_day(later, false), Some(guid));
        assert_eq!(doc.order().len(), entities + 1);
        assert_eq!(doc.undo_len(), undos + 1);
        assert_eq!(doc.time_of_day(), Some(later));

        // An idempotent edit records nothing.
        let undos = doc.undo_len();
        doc.edit_time_of_day(later, true);
        assert_eq!(doc.undo_len(), undos, "no-op edit records nothing");
    }

    /// P17.1 finding-3 guard: `create: false` on a level with **no** clock must be
    /// a total no-op — no entity, no undo entry, no version bump, no dirty flag.
    ///
    /// This is what stops an unrelated World Settings write (gravity, sim rate, the
    /// partition block) from conjuring a `Sky` actor out of the time-of-day
    /// *preview* values the panel round-trips. The flag used to live only in the
    /// Ring-2 command, where nothing could test it.
    #[test]
    fn time_of_day_without_create_never_conjures_an_authority() {
        use inf_ecs::components::TimeOfDay;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        let version = doc.version();
        assert!(doc.time_of_day().is_none());

        let previewed = TimeOfDay {
            seconds: 1234.0,
            ..TimeOfDay::default()
        };
        assert_eq!(doc.edit_time_of_day(previewed, false), None);

        assert!(doc.time_of_day().is_none(), "no clock was created");
        assert!(doc.sky_authority().is_none());
        assert_eq!(doc.order().len(), entities, "no entity appeared");
        assert_eq!(doc.undo_len(), undos, "nothing was recorded");
        assert_eq!(doc.version(), version, "the version did not bump");
        assert!(!doc.is_dirty(), "the document was not dirtied");

        // …and `create: true` on the same doc does the whole opt-in in one step.
        let guid = doc.edit_time_of_day(previewed, true).expect("created");
        assert_eq!(doc.sky_authority(), Some(guid));
        assert_eq!(doc.time_of_day(), Some(previewed));
        assert_eq!(doc.order().len(), entities + 1);
        assert_eq!(doc.undo_len(), undos + 1);
        assert!(doc.is_dirty());

        // Once an authority exists, `create: false` still writes it — the flag
        // governs creation only.
        let later = TimeOfDay {
            seconds: 4321.0,
            ..previewed
        };
        assert_eq!(doc.edit_time_of_day(later, false), Some(guid));
        assert_eq!(doc.time_of_day(), Some(later));
        assert_eq!(
            doc.order().len(),
            entities + 1,
            "still exactly one Sky actor"
        );
    }

    /// Nested `begin`/`commit` collapse into ONE undo step: an inner commit must
    /// not close the outer transaction (only the outermost commit does).
    #[test]
    fn nested_transactions_close_on_outermost_commit() {
        let mut doc = SceneDoc::new();
        let a = doc.edit_create(SpawnKind::Cube, "A", None);

        let at = |x: f64| Transform {
            translation: Vec3d::new(x, 0.0, 0.0),
            ..Transform::IDENTITY
        };
        doc.begin_transaction("outer");
        doc.edit_set_transform(a, at(1.0));
        doc.begin_transaction("inner");
        doc.edit_set_transform(a, at(2.0));
        doc.commit_transaction(); // inner: must NOT close the transaction
        doc.edit_set_transform(a, at(3.0));
        doc.commit_transaction(); // outer: closes now
        assert_eq!(translation_x(&doc, a), 3.0);

        // One undo reverts all three edits (a single grouped step), not just the
        // last — the discriminator against the old fold-on-first-commit bug.
        assert!(doc.undo());
        assert_eq!(
            translation_x(&doc, a),
            0.0,
            "nested begins collapse into one undo step"
        );
        // The remaining entry is the create.
        assert!(doc.undo());
        assert!(!doc.can_undo());
    }

    /// P17.2: the atmosphere is edited through the same one-step door as the
    /// clock, on the same authority — and the first edit on a clockless level
    /// creates BOTH components inside that single step, because an atmosphere
    /// with no clock beside it is the inert shape `inf_ecs::sky` warns about.
    #[test]
    fn sky_atmosphere_edit_creates_the_authority_in_one_undo_step() {
        use inf_ecs::components::SkyAtmosphere;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        assert!(doc.sky_atmosphere().is_none(), "a new level has no sky");

        let authored = SkyAtmosphere {
            physical: true,
            turbidity: 2.5,
            fog_density: 6.0e-4,
            fog_height: 120.0,
            star_intensity: 0.4,
            ..SkyAtmosphere::default()
        };
        let guid = doc.edit_sky_atmosphere(authored, true).expect("created");

        assert_eq!(doc.sky_atmosphere(), Some(authored));
        assert_eq!(doc.sky_authority(), Some(guid));
        // The clock came with it, at its defaults — the level is now a complete
        // sky authority, not half of one.
        assert_eq!(
            doc.time_of_day(),
            Some(inf_ecs::components::TimeOfDay::default())
        );
        assert_eq!(doc.order().len(), entities + 1, "one Sky actor appeared");
        assert_eq!(
            doc.undo_len(),
            undos + 1,
            "create + both components + every field collapse into ONE step"
        );

        // One Ctrl+Z takes the whole opt-in back out; redo restores it exactly.
        assert!(doc.undo());
        assert!(doc.sky_atmosphere().is_none());
        assert_eq!(doc.order().len(), entities);
        assert!(doc.redo());
        assert_eq!(doc.sky_atmosphere(), Some(authored));

        // An idempotent edit records nothing.
        let undos = doc.undo_len();
        doc.edit_sky_atmosphere(authored, true);
        assert_eq!(doc.undo_len(), undos, "no-op edit records nothing");
    }

    /// The same finding-3 guard the clock has: `create: false` on a level with no
    /// authority must be a total no-op, so a World Settings write that merely
    /// echoes the previewed atmosphere defaults back cannot conjure a `Sky` actor
    /// while somebody is editing gravity.
    #[test]
    fn sky_atmosphere_without_create_never_conjures_an_authority() {
        use inf_ecs::components::SkyAtmosphere;

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();
        let version = doc.version();

        assert_eq!(
            doc.edit_sky_atmosphere(SkyAtmosphere::default(), false),
            None
        );
        assert_eq!(doc.order().len(), entities);
        assert_eq!(doc.undo_len(), undos);
        assert_eq!(doc.version(), version, "a no-op must not bump the version");
        assert!(!doc.is_dirty());

        // With `create`, the same call opts in …
        let guid = doc
            .edit_sky_atmosphere(SkyAtmosphere::default(), true)
            .expect("opted in");
        assert_eq!(doc.sky_authority(), Some(guid));

        // … and afterwards `create: false` still writes the existing authority.
        let hazy = SkyAtmosphere {
            turbidity: 4.0,
            ..SkyAtmosphere::default()
        };
        assert_eq!(doc.edit_sky_atmosphere(hazy, false), Some(guid));
        assert_eq!(doc.sky_atmosphere(), Some(hazy));
    }

    /// P17.4: a weather edit is **one** undo step, labelled for what the user did,
    /// and it opts a clockless level in exactly like the other two entry points.
    #[test]
    fn weather_edit_is_one_undo_step_and_creates_the_authority() {
        use inf_ecs::components::{SkyAtmosphere, WeatherPreset};

        let mut doc = SceneDoc::new();
        let entities = doc.order().len();
        let undos = doc.undo_len();

        // Without `create` on a clockless level: a total no-op.
        assert_eq!(doc.edit_weather(SkyAtmosphere::default(), false), None);
        assert_eq!(doc.order().len(), entities);
        assert_eq!(doc.undo_len(), undos);
        assert!(
            !doc.is_dirty(),
            "a refused write must not dirty the document"
        );

        // With `create`: the Sky actor + both components + eleven field writes
        // collapse into ONE entry.
        let storm = SkyAtmosphere {
            weather_enabled: true,
            weather_target: WeatherPreset::Storm,
            ..SkyAtmosphere::default()
        };
        let guid = doc.edit_weather(storm, true).expect("opted in");
        assert_eq!(doc.sky_authority(), Some(guid));
        assert_eq!(doc.undo_len(), undos + 1, "one undo entry, not thirteen");
        assert_eq!(doc.order().len(), entities + 1);
        let a = doc.sky_atmosphere().expect("atmosphere");
        assert!(a.weather_enabled);
        assert_eq!(a.weather_target, WeatherPreset::Storm);

        // A single Ctrl+Z takes the whole opt-in back out …
        assert!(doc.undo());
        assert_eq!(doc.order().len(), entities);
        assert_eq!(doc.sky_authority(), None);
        // … and redo restores it, enum field included (the reflect variant name
        // really applied — a wrong spelling would silently leave `Clear`).
        assert!(doc.redo());
        assert_eq!(
            doc.sky_atmosphere().unwrap().weather_target,
            WeatherPreset::Storm
        );

        // Re-writing the same values records nothing.
        let before = doc.undo_len();
        assert_eq!(doc.edit_weather(storm, false), Some(guid));
        assert_eq!(doc.undo_len(), before, "a no-op must record nothing");
    }

    /// The two blocks share one component, so writing either must leave the other
    /// alone — the composition the Ring-2 command depends on when it posts the
    /// whole settings DTO on every edit.
    #[test]
    fn weather_and_atmosphere_edits_compose_rather_than_overwrite() {
        use crate::ipc::{WeatherDto, WeatherPresetDto};
        use inf_ecs::components::{SkyAtmosphere, WeatherPreset};

        let mut doc = SceneDoc::new();
        doc.edit_sky_atmosphere(
            SkyAtmosphere {
                turbidity: 4.0,
                fog_height: 120.0,
                ..SkyAtmosphere::default()
            },
            true,
        )
        .expect("created");

        // A weather write through the DTO overlay path the command uses.
        let base = doc.sky_atmosphere().unwrap();
        let dto = WeatherDto::from_doc(&doc).snapped_to(WeatherPresetDto::Snow);
        doc.edit_weather(dto.to_component(base), true);

        let a = doc.sky_atmosphere().unwrap();
        assert_eq!(a.turbidity, 4.0, "the atmosphere half must survive");
        assert_eq!(a.fog_height, 120.0);
        assert!(a.weather_enabled);
        assert_eq!(a.weather_target, WeatherPreset::Snow);
        assert_eq!(a.weather_params(), WeatherPreset::Snow.params());
        assert_eq!(a.weather_blend_remaining, 0.0, "a preset button snaps");

        // …and the reverse order: an atmosphere write must not reset the weather.
        let base = doc.sky_atmosphere().unwrap();
        doc.edit_sky_atmosphere(
            SkyAtmosphere {
                turbidity: 1.5,
                ..base
            },
            false,
        );
        let a = doc.sky_atmosphere().unwrap();
        assert_eq!(a.turbidity, 1.5);
        assert_eq!(a.weather_target, WeatherPreset::Snow);
        assert_eq!(
            a.weather_precipitation,
            WeatherPreset::Snow.params().precipitation
        );
    }

    /// Hostile DTO input is clamped, never stored — the same contract
    /// `atmosphere_dto_clamps_hostile_input` holds one block over.
    #[test]
    fn weather_dto_clamps_hostile_input() {
        use crate::ipc::{WeatherDto, WeatherPresetDto};
        use inf_ecs::components::SkyAtmosphere;

        let hostile = WeatherDto {
            present: true,
            enabled: true,
            preset: WeatherPresetDto::Fog,
            blend_seconds: f32::NAN,
            blend_remaining: -5.0,
            coverage: 9.0,
            cloud_type: -3.0,
            wind_x: f32::INFINITY,
            wind_z: -1e9,
            fog_density: -1.0,
            precipitation: 7.5,
            snowiness: f32::NAN,
        };
        let a = hostile.to_component(SkyAtmosphere::default());
        assert_eq!(
            a.weather_blend_seconds, 8.0,
            "NaN falls back to the default"
        );
        assert_eq!(a.weather_blend_remaining, 0.0, "negative clamps to settled");
        // The blend bound is the SHARED Ring-0 constant, not a repeated literal:
        // `sky::set_weather` is the other door into these two fields and clamps to
        // the same value, and the bound is arithmetic (past it the f32 countdown
        // stops making progress and the blend never settles), so two copies of the
        // number would be two chances to arm the blender forever.
        let huge = WeatherDto {
            blend_seconds: 5e6,
            blend_remaining: 5e6,
            ..hostile
        };
        let b = huge.to_component(SkyAtmosphere::default());
        assert_eq!(b.weather_blend_seconds, inf_ecs::sky::MAX_WEATHER_BLEND_S);
        assert_eq!(b.weather_blend_remaining, inf_ecs::sky::MAX_WEATHER_BLEND_S);
        assert_eq!(a.weather_coverage, 1.0);
        assert_eq!(a.weather_cloud_type, 0.0);
        assert_eq!(a.weather_wind_x, 0.0, "infinite wind falls back to still");
        assert_eq!(a.weather_wind_z, -200.0);
        assert_eq!(a.weather_fog_density, 0.0);
        assert_eq!(a.weather_precipitation, 1.0);
        assert_eq!(a.weather_snowiness, 0.0);
        // The atmosphere half of the base is untouched.
        assert_eq!(a.cloud_coverage, SkyAtmosphere::default().cloud_coverage);
    }

    /// The DTO deliberately carries no colours, so a World Settings write must
    /// leave a Details-authored sun/moon/gradient colour alone. This is the
    /// overlay contract that `SkyAtmosphereDto::to_component` implements and the
    /// Ring-2 command relies on.
    #[test]
    fn atmosphere_dto_overlay_preserves_authored_colours() {
        use crate::ipc::SkyAtmosphereDto;
        use inf_ecs::components::SkyAtmosphere;
        use inf_ecs::math::Color;

        let mut doc = SceneDoc::new();
        let authored_sun = Color::new(1.0, 0.4, 0.1, 1.0);
        doc.edit_sky_atmosphere(
            SkyAtmosphere {
                sun_color: authored_sun,
                ..SkyAtmosphere::default()
            },
            true,
        )
        .expect("created");
        // …except the DTO cannot carry that colour, so write it through the
        // component path the Details grid uses.
        let guid = doc.sky_authority().unwrap();
        let tp = doc
            .world()
            .registry()
            .type_path_for("SkyAtmosphere")
            .unwrap();
        doc.edit_set_prop(
            guid,
            tp,
            "sun_color",
            &inf_ecs::props::PropValue::Color([1.0, 0.4, 0.1, 1.0]),
        );
        assert_eq!(doc.sky_atmosphere().unwrap().sun_color, authored_sun);

        // Now do what the panel does: project, change one number, write back.
        let mut dto = SkyAtmosphereDto::from_doc(&doc);
        assert!(dto.present);
        dto.fog_density = 5.0e-4;
        let base = doc.sky_atmosphere().unwrap();
        doc.edit_sky_atmosphere(dto.to_component(base), dto.present);

        let after = doc.sky_atmosphere().unwrap();
        assert_eq!(after.fog_density, 5.0e-4);
        assert_eq!(
            after.sun_color, authored_sun,
            "a fog edit erased the authored sun colour"
        );
    }

    /// Nonsense from a hand-crafted IPC payload is clamped, never stored — a
    /// `NaN` turbidity would blank the sky and a negative fog density would make
    /// the exponential integral blow up.
    #[test]
    fn atmosphere_dto_clamps_hostile_input() {
        use crate::ipc::SkyAtmosphereDto;
        use inf_ecs::components::SkyAtmosphere;

        let hostile = SkyAtmosphereDto {
            present: true,
            enabled: true,
            physical: true,
            sky_intensity: f32::NAN,
            turbidity: -5.0,
            mie_anisotropy: 12.0,
            sun_disc_deg: -1.0,
            moon_disc_deg: 1e9,
            star_intensity: f32::INFINITY,
            tint_strength: 4.0,
            aerial_perspective: -2.0,
            fog_density: -1.0,
            fog_falloff: f32::NAN,
            fog_height: 0.0,
            clouds_enabled: true,
            cloud_coverage: 9.0,
            cloud_type: -3.0,
            cloud_bottom: f32::NAN,
            cloud_top: 1e12,
            cloud_density: -1.0,
            cloud_detail: f32::NAN,
            cloud_seed: u32::MAX,
            cloud_wind_x: 1e9,
            cloud_wind_z: f32::NAN,
            cloud_phase_g: 5.0,
            cloud_shadow: -1.0,
            cloud_ambient: 1e9,
        };
        let a = hostile.to_component(SkyAtmosphere::default());
        assert_eq!(a.sky_intensity, 1.0, "NaN fell back to the default");
        assert_eq!(a.turbidity, 0.0);
        assert_eq!(a.mie_anisotropy, 0.95);
        assert_eq!(a.sun_disc_deg, 0.0);
        assert_eq!(a.moon_disc_deg, 90.0);
        assert_eq!(a.star_intensity, 1.0);
        assert_eq!(a.tint_strength, 1.0);
        assert_eq!(a.aerial_perspective, 0.0);
        assert_eq!(a.fog_density, 0.0);
        assert_eq!(a.fog_falloff, 0.002);
        // ── clouds (P17.3) ──
        assert_eq!(a.cloud_coverage, 1.0);
        assert_eq!(a.cloud_type, 0.0);
        assert_eq!(a.cloud_bottom, 1500.0, "NaN fell back to the default");
        assert_eq!(a.cloud_top, 50_000.0);
        assert_eq!(a.cloud_density, 0.0);
        assert_eq!(a.cloud_detail, 0.6, "NaN fell back to the default");
        // The seed is MASKED, not clamped: only the low 24 bits survive the f32
        // uniform, so anything else would display one sky and render another.
        assert_eq!(a.cloud_seed, 0x00ff_ffff);
        assert_eq!(a.cloud_wind_x, 200.0);
        assert_eq!(a.cloud_wind_z, 2.0, "NaN fell back to the default");
        assert_eq!(a.cloud_phase_g, 0.95);
        assert_eq!(a.cloud_shadow, 0.0);
        assert_eq!(a.cloud_ambient, 4.0);
        // Everything the DTO does not carry is untouched — including the cloud
        // block's own `Color`, which stays in the Details grid with the other five.
        assert_eq!(a.zenith, SkyAtmosphere::default().zenith);
        assert_eq!(a.cloud_color, SkyAtmosphere::default().cloud_color);
    }

    /// The Phase-17 "done when": a **new** level's sky is the physical one. The
    /// default scene carries the authority pair, and — the part that is easy to
    /// get wrong — it does NOT also carry a hand-authored directional sun, which
    /// would light the level twice from two directions.
    #[test]
    fn the_default_scene_has_a_physical_sky_and_only_one_sun() {
        use inf_ecs::components::{Light, LightKind};

        let doc = SceneDoc::with_demo();
        let sky = doc.sky_atmosphere().expect("the default level has a sky");
        assert!(sky.physical, "the default sky is the physical one");
        assert!(sky.enabled, "the default sun lights the scene");
        assert_eq!(
            doc.time_of_day(),
            Some(inf_ecs::components::TimeOfDay::default()),
            "the default clock is the documented 10:00 default"
        );
        assert_eq!(
            doc.time_of_day().unwrap().rate,
            0.0,
            "an idle editor must not move the sun"
        );

        // No authored directional light: the time of day IS the sun.
        let world = doc.world().world();
        let directional = world
            .iter_entities()
            .filter_map(|e| e.get::<Light>())
            .filter(|l| l.kind == LightKind::Directional)
            .count();
        assert_eq!(
            directional, 0,
            "the default scene has both a sky sun and an authored one — it would be lit twice"
        );

        // The authority is the actor World Settings would have created.
        let guid = doc.sky_authority().expect("authority");
        assert_eq!(doc.display_name(guid), "Sky");
    }

    /// P17.3: a **new** level boots with clouds, while the component default —
    /// which is also what every v12 level lifts to — stays off.
    ///
    /// The two must disagree in exactly this direction and nowhere else. If the
    /// component default ever flipped true, every existing level would silently
    /// grow a sky it was never authored against on its next load, which is the
    /// one thing the frozen-record scheme cannot undo. If the *demo* default
    /// flipped false, the `editor_default` golden's one P17.3 re-bless would have
    /// been for nothing.
    #[test]
    fn the_default_scene_opts_into_clouds_while_the_component_does_not() {
        use inf_ecs::components::SkyAtmosphere;

        let doc = SceneDoc::with_demo();
        let sky = doc.sky_atmosphere().expect("the default level has a sky");
        assert!(
            sky.clouds_enabled,
            "a new level should boot with clouds — that is the P17.3 look"
        );
        assert!(
            !SkyAtmosphere::default().clouds_enabled,
            "the COMPONENT default must stay off, or every v12 level grows clouds \
             it was never authored against when it is lifted"
        );
        // ...and nothing else in the block was privately tuned: the demo diverges
        // by exactly one boolean, so what the golden pictures is the documented
        // defaults.
        let defaults = SkyAtmosphere::default();
        assert_eq!(
            sky,
            SkyAtmosphere {
                clouds_enabled: true,
                ..defaults
            },
            "the default scene tuned something other than `clouds_enabled`"
        );
    }
}
