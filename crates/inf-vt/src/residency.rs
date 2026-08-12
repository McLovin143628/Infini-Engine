//! **Residency**: which virtual tiles are in the physical pool right now, which
//! slot each one occupies, and what every *other* virtual tile falls back to.
//!
//! # The state, in three sentences
//!
//! A pool is a flat array of interchangeable slots. Each registered texture has
//! one indirection entry per virtual tile, holding the `(slot, mip)` of the
//! **finest resident ancestor** of that tile — itself, when it is resident. The
//! coarsest level of every texture is admitted at registration and pinned, so
//! that walk always terminates and **no entry is ever empty**.
//!
//! # Fallbacks are maintained, not searched
//!
//! The naive implementation resolves a fallback by walking up the pyramid at
//! sample time. That is the wrong side of the seam — the *shader* would have to
//! do the walk, one dependent read per level, and the walk's length would depend
//! on residency. Instead every entry is kept **already resolved**: an admit
//! writes its own address into itself and into the non-resident subtree beneath
//! it; an evict writes its parent's entry into itself and into the same subtree.
//! Each is bounded by the affected subtree rather than by the pyramid, and the
//! shader does one read.
//!
//! Maintained state is state that can drift, so [`VtResidency::resolve`] is
//! checked against an independent brute-force walk after **every** transaction of
//! a seeded churn in `tests/churn.rs` — the whole table, not a sample of it.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::address::{DescError, TileCoord, VtTextureDesc};
use crate::pool::{plan_pool, VtAdvisory, VtPoolConfig, VtPoolGeometry};
use crate::table::{pack_entry, unpack_entry, TableImage, MAX_SLOT_INDEX};

/// Source of residency stamps, **process-global and monotone** — the same
/// construction, and the same reason, as `inf_vgeom::stream`'s
/// `NEXT_RESIDENCY_STAMP` and `inf_terrain`'s `NEXT_TILE_VERSION`.
///
/// A GPU mirror caches "the indirection block I last uploaded for texture T was
/// generation N" and re-uploads only when the generation moved. A per-residency
/// counter restarting at 1 would let a *freshly created* residency mint a
/// generation a stale cache already holds — after a budget change drops the pool,
/// or after a level switch — and the mirror would then keep serving a table
/// describing slots that no longer hold what it thinks. A global counter never
/// decreases, so a new residency's generation is strictly greater than any
/// generation any cache can be holding.
///
/// **Its own domain, for now.** `inf-vgeom`'s and `inf-terrain`'s counters are
/// crate-private statics, so there is nothing to share without making one of them
/// public and giving three streamers a reason to contend on one cache line. P28.3
/// ("one stamp domain") is where that unification belongs; until then this is a
/// third counter with identical semantics, and the three of them agree on the one
/// property that matters — never decreasing.
static NEXT_VT_STAMP: AtomicU64 = AtomicU64::new(1);

/// The next stamp. Never 0, so 0 is safe as "nothing uploaded yet".
fn next_stamp() -> u64 {
    NEXT_VT_STAMP.fetch_add(1, Ordering::Relaxed)
}

/// A registered virtual texture. Index into the residency's texture list, and the
/// `tex_id` a shader uses to find the texture's block in the indirection table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VtTextureHandle(pub u32);

impl VtTextureHandle {
    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// "This tile, please." The whole input language of residency — deliberately no
/// priority, no camera and no budget: see the crate docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtWant {
    pub texture: VtTextureHandle,
    pub tile: TileCoord,
}

impl VtWant {
    pub const fn new(texture: VtTextureHandle, tile: TileCoord) -> Self {
        Self { texture, tile }
    }
}

/// A page entering the pool: write these bytes into this slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtAdmit {
    pub slot: u32,
    pub texture: VtTextureHandle,
    pub tile: TileCoord,
}

/// A page leaving the pool.
///
/// **There is nothing to do on the GPU for one.** The slot's texels simply become
/// unreachable — no live indirection entry names the slot any more — and they are
/// overwritten by the next admit that takes it. This record exists so a mirror can
/// account for the slot and a debug view can colour it, not so anything can be
/// erased.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtEvict {
    pub slot: u32,
    pub texture: VtTextureHandle,
    pub tile: TileCoord,
}

/// Where a virtual tile actually lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtResolved {
    pub slot: u32,
    /// The resident tile whose texels serve this address — the finest resident
    /// ancestor. Equal to the queried address when that tile is itself resident.
    pub tile: TileCoord,
}

/// The complete difference between two residency states.
///
/// **Deterministic and stamp-free.** Nothing in here is drawn from the global
/// counter, so two runs of one want sequence produce byte-identical
/// [`trace`](Self::trace) output — which is exactly what the determinism gate
/// pins, and what a stamp in this struct would silently destroy.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VtTransaction {
    /// Pages to write into the atlas, in payload order (roots first).
    pub admits: Vec<VtAdmit>,
    /// Pages whose slots became unreachable, in the order they were taken.
    pub evicts: Vec<VtEvict>,
    /// Textures whose indirection block changed, ascending. The mirror re-writes
    /// exactly these blocks.
    pub tables: Vec<VtTextureHandle>,
    /// Whether the table's *layout* moved (a texture was registered), so the
    /// mirror must re-create and re-upload the whole buffer rather than patch it.
    pub layout_rebuilt: bool,
    /// Wants that could not be admitted because every slot was pinned, admitted
    /// or touched this transaction. Never a silent cap: this is the number, and
    /// [`VtStats::budget_clamped`] is the flag.
    pub deferred: u32,
    /// Wants naming a tile outside their texture's grid — a caller bug, counted
    /// rather than panicked on, because a want set is computed from a camera.
    pub out_of_range: u32,
}

impl VtTransaction {
    /// Whether this transaction changes anything at all.
    pub fn is_empty(&self) -> bool {
        self.admits.is_empty()
            && self.evicts.is_empty()
            && self.tables.is_empty()
            && !self.layout_rebuilt
    }

    /// A canonical, **stamp-free** rendering of the transaction — the thing a
    /// determinism gate compares byte for byte between two runs.
    pub fn trace(&self) -> String {
        let mut s = String::new();
        if self.layout_rebuilt {
            s.push_str("layout\n");
        }
        for a in &self.admits {
            s.push_str(&format!(
                "admit slot={} tex={} mip={} x={} y={}\n",
                a.slot, a.texture.0, a.tile.mip, a.tile.x, a.tile.y
            ));
        }
        for e in &self.evicts {
            s.push_str(&format!(
                "evict slot={} tex={} mip={} x={} y={}\n",
                e.slot, e.texture.0, e.tile.mip, e.tile.x, e.tile.y
            ));
        }
        for t in &self.tables {
            s.push_str(&format!("table tex={}\n", t.0));
        }
        if self.deferred > 0 {
            s.push_str(&format!("defer {}\n", self.deferred));
        }
        if self.out_of_range > 0 {
            s.push_str(&format!("oor {}\n", self.out_of_range));
        }
        s
    }
}

/// Counters for the debug/stats path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VtStats {
    pub textures: usize,
    /// Slots the pool has.
    pub slots: u32,
    /// Slots holding a page.
    pub resident: u32,
    /// Slots holding a pinned root — the mandatory floor, in slots.
    pub roots: u32,
    /// Wants the last transaction was given (after dedup).
    pub wanted: usize,
    /// Wants the last transaction could not seat.
    pub deferred: u32,
    /// Pages admitted since the residency was created.
    pub admits: u64,
    /// Pages evicted since the residency was created.
    pub evicts: u64,
    /// Whether the last transaction had to defer a want for want of a slot.
    pub budget_clamped: bool,
}

impl VtStats {
    /// A one-line human summary — the same dump
    /// `inf_terrain::TerrainStreamStats` and `inf_voxel` ship, in the same shape
    /// so three streamers read alike in one log.
    ///
    /// **Ships before its caller**, deliberately: P26.5's residency heat-map and
    /// pool-budget knobs are what read it, and the Output Log line beside them.
    /// Pinned by [`tests::the_stats_line_says_what_it_counts`] meanwhile, so it
    /// cannot rot into a string that names none of its numbers.
    pub fn summary(&self) -> String {
        format!(
            "vt residency: {} textures, {}/{} slots resident ({} pinned roots), \
             {} wanted / {} deferred, {} admits / {} evicts{}",
            self.textures,
            self.resident,
            self.slots,
            self.roots,
            self.wanted,
            self.deferred,
            self.admits,
            self.evicts,
            if self.budget_clamped {
                " [budget-clamped]"
            } else {
                ""
            }
        )
    }
}

/// A registration this pool cannot honour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VtError {
    #[error("the descriptor is not a virtual texture: {0}")]
    Desc(#[from] DescError),
    #[error(
        "the texture's stored tile is {desc}² texels and the pool's slots are {pool}² — one \
         pool holds one slot size"
    )]
    PoolGeometryMismatch { desc: u32, pool: u32 },
    #[error(
        "the mandatory floor of {roots} always-resident root pages ({floor_bytes} B) does not \
         fit the {budget_bytes} B page budget ({slots} slots). Every registered texture keeps \
         its coarsest level resident for ever, so this is a floor and not a preference: raise \
         the budget, or register fewer textures"
    )]
    MandatoryFloorExceedsBudget {
        roots: u32,
        floor_bytes: u64,
        budget_bytes: u64,
        slots: u32,
    },
}

/// One physical slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot {
    /// `(texture index, tile)` — `None` when free.
    occupant: Option<(u32, TileCoord)>,
    /// Last touched. **A stamp, not a measurement** (see [`NEXT_VT_STAMP`]).
    stamp: u64,
    /// A root: never evicted, for any reason, by anyone.
    pinned: bool,
}

/// One registered texture's residency.
#[derive(Debug, Clone)]
struct Texture {
    desc: VtTextureDesc,
    /// Exact-resident slot per virtual tile, indexed by `desc.entry_index`.
    resident: Vec<Option<u32>>,
    generation: u64,
    /// Whether this texture's root pages have been emitted in a transaction (and
    /// therefore actually exist in the atlas, not merely in the table).
    warm: bool,
}

/// The residency of one physical page pool.
#[derive(Debug, Clone)]
pub struct VtResidency {
    cfg: VtPoolConfig,
    geometry: VtPoolGeometry,
    page_bytes: u64,
    slots: Vec<Slot>,
    free: BTreeSet<u32>,
    textures: Vec<Texture>,
    table: TableImage,
    pending_roots: Vec<(VtTextureHandle, TileCoord)>,
    pending_evicts: Vec<VtEvict>,
    layout_generation: u64,
    layout_dirty: bool,
    stats: VtStats,
}

impl VtResidency {
    /// A pool sized from `cfg`, with the advisories planning it raised.
    ///
    /// The advisories are returned rather than logged, so a host decides whether
    /// they are a warning, a fatal, or a line in an import report — the
    /// `texture_import_advisories` shape.
    pub fn new(cfg: VtPoolConfig) -> (Self, Vec<VtAdvisory>) {
        let (geometry, advisories) = plan_pool(cfg);
        let page_bytes = cfg.format.page_bytes(geometry.stored_tile_size);
        let n = geometry.slot_count();
        debug_assert!(n <= MAX_SLOT_INDEX + 1, "a slot index must fit 16 bits");
        let table = TableImage::layout(&[], geometry.slots_x, geometry.stored_tile_size);
        let this = Self {
            cfg,
            geometry,
            page_bytes,
            slots: vec![
                Slot {
                    occupant: None,
                    stamp: 0,
                    pinned: false,
                };
                n as usize
            ],
            free: (0..n).collect(),
            textures: Vec::new(),
            table,
            pending_roots: Vec::new(),
            pending_evicts: Vec::new(),
            layout_generation: next_stamp(),
            layout_dirty: false,
            stats: VtStats {
                slots: n,
                ..Default::default()
            },
        };
        (this, advisories)
    }

    #[inline]
    pub fn config(&self) -> VtPoolConfig {
        self.cfg
    }
    #[inline]
    pub fn geometry(&self) -> VtPoolGeometry {
        self.geometry
    }
    /// Bytes one page occupies.
    #[inline]
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }
    /// The VRAM the atlas costs — `slots × page_bytes`, and **never** more than
    /// [`VtPoolConfig::budget_bytes`].
    #[inline]
    pub fn capacity_bytes(&self) -> u64 {
        u64::from(self.geometry.slot_count()) * self.page_bytes
    }
    /// Bytes currently holding a page.
    #[inline]
    pub fn resident_bytes(&self) -> u64 {
        u64::from(self.stats.resident) * self.page_bytes
    }
    #[inline]
    pub fn stats(&self) -> &VtStats {
        &self.stats
    }
    #[inline]
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }
    /// One texture's address space.
    pub fn desc(&self, tex: VtTextureHandle) -> Option<&VtTextureDesc> {
        self.textures.get(tex.index()).map(|t| &t.desc)
    }

    /// **Register a virtual texture and seat its root pages.**
    ///
    /// The roots are seated *here*, not at the first want, because the mandatory
    /// floor is a claim on the budget that has to be settled before anything else
    /// spends it — and because the "every tile resolves to something" law has to
    /// hold from the instant a texture exists, not from the instant it is first
    /// sampled. Their *bytes* are emitted by the next
    /// [`apply_wants`](Self::apply_wants) (as the leading admits), which is what
    /// keeps a page upload to exactly one door.
    ///
    /// Seating roots may evict cached non-root pages — roots outrank cache — and
    /// those evictions ride along in the next transaction.
    pub fn register_texture(&mut self, desc: VtTextureDesc) -> Result<VtTextureHandle, VtError> {
        desc.validate()?;
        if desc.stored_tile_size() != self.geometry.stored_tile_size {
            return Err(VtError::PoolGeometryMismatch {
                desc: desc.stored_tile_size(),
                pool: self.geometry.stored_tile_size,
            });
        }
        let roots = desc.root_tiles();
        let floor = self.stats.roots + roots.len() as u32;
        if floor > self.geometry.slot_count() {
            return Err(VtError::MandatoryFloorExceedsBudget {
                roots: floor,
                floor_bytes: u64::from(floor) * self.page_bytes,
                budget_bytes: self.cfg.budget_bytes,
                slots: self.geometry.slot_count(),
            });
        }

        let handle = VtTextureHandle(self.textures.len() as u32);
        let tiles = desc.tile_count() as usize;
        self.textures.push(Texture {
            desc,
            resident: vec![None; tiles],
            generation: next_stamp(),
            warm: false,
        });
        // The directory sits before the blocks, so a new texture moves every
        // block after it: the image is laid out again and refilled from the live
        // residency (a coarse-to-fine sweep, which is also the only place the
        // entries are computed from scratch).
        self.relayout();

        // Seat the roots. The floor check above guarantees this succeeds: at most
        // `slot_count` slots are pinned, so at least `roots.len()` are free or
        // evictable.
        let mut protected = BTreeSet::new();
        for tile in roots {
            let (slot, evicted) = self
                .acquire_slot(&protected)
                .expect("the mandatory-floor check guarantees a slot for every root");
            if let Some(e) = evicted {
                self.pending_evicts.push(e);
            }
            self.seat(slot, handle.0, tile, true);
            protected.insert(slot);
            self.pending_roots.push((handle, tile));
        }
        self.stats.textures = self.textures.len();
        Ok(handle)
    }

    /// **The sync point.** Advance residency one deterministic step toward what
    /// `wants` asks for, and hand back the work.
    ///
    /// Call once per frame. The result is a pure function of `(state, wants)` —
    /// see the crate docs for the four things that makes true.
    pub fn apply_wants(&mut self, wants: &[VtWant]) -> VtTransaction {
        let mut txn = VtTransaction {
            layout_rebuilt: std::mem::take(&mut self.layout_dirty),
            evicts: std::mem::take(&mut self.pending_evicts),
            ..Default::default()
        };
        let mut dirty: BTreeSet<VtTextureHandle> = txn
            .evicts
            .iter()
            .map(|e| e.texture)
            .chain(self.pending_roots.iter().map(|(h, _)| *h))
            .collect();

        // ── 1. root pages first, always ──
        //
        // They are already seated (registration did that); this emits their bytes
        // exactly once, before anything else can take a slot. A protected set,
        // even though roots are pinned and pinning already forbids it — a safety
        // property should not rest on a second mechanism agreeing with it.
        let mut protected: BTreeSet<u32> = BTreeSet::new();
        for (handle, tile) in std::mem::take(&mut self.pending_roots) {
            let t = &self.textures[handle.index()];
            let Some(idx) = t.desc.entry_index(tile) else {
                continue;
            };
            let Some(slot) = t.resident[idx as usize] else {
                continue;
            };
            protected.insert(slot);
            self.touch(slot);
            txn.admits.push(VtAdmit {
                slot,
                texture: handle,
                tile,
            });
            self.stats.admits += 1;
        }
        for t in &mut self.textures {
            t.warm = true;
        }

        // ── 2. normalize the want set ──
        //
        // Sorted by (texture, PAYLOAD order) and deduplicated, so the result does
        // not depend on how the caller assembled the frame — and so a batch of
        // admits is a forward scan of the container rather than a scatter. The
        // derived `Ord` on `TileCoord` would sort (mip, x, y) and walk the file
        // sideways; that is the P26.1 audit's finding, used rather than merely
        // recorded.
        let mut sorted: Vec<VtWant> = wants
            .iter()
            .copied()
            .filter(|w| w.texture.index() < self.textures.len())
            .collect();
        sorted.sort_by_key(|w| (w.texture.0, w.tile.payload_order()));
        sorted.dedup();
        self.stats.wanted = sorted.len();

        // ── 3. touch what is already there, collect the misses ──
        let mut misses: Vec<VtWant> = Vec::new();
        for w in &sorted {
            let t = &self.textures[w.texture.index()];
            let Some(idx) = t.desc.entry_index(w.tile) else {
                txn.out_of_range += 1;
                continue;
            };
            match t.resident[idx as usize] {
                Some(slot) => {
                    protected.insert(slot);
                    self.touch(slot);
                }
                None => misses.push(*w),
            }
        }

        // ── 4. admit the misses, in order, against the slots that are left ──
        let mut deferred = 0u32;
        for (i, w) in misses.iter().enumerate() {
            let Some((slot, evicted)) = self.acquire_slot(&protected) else {
                // Nothing changes between iterations once acquisition fails, so
                // every remaining miss fails too. Counting them without retrying
                // is the same answer for O(1) instead of O(n·slots).
                deferred = (misses.len() - i) as u32;
                break;
            };
            if let Some(e) = evicted {
                dirty.insert(e.texture);
                txn.evicts.push(e);
                self.stats.evicts += 1;
            }
            self.seat(slot, w.texture.0, w.tile, false);
            protected.insert(slot);
            dirty.insert(w.texture);
            txn.admits.push(VtAdmit {
                slot,
                texture: w.texture,
                tile: w.tile,
            });
            self.stats.admits += 1;
        }

        txn.deferred = deferred;
        txn.tables = dirty.into_iter().collect();
        self.stats.deferred = deferred;
        self.stats.budget_clamped = deferred > 0;
        self.stats.resident = self.slots.iter().filter(|s| s.occupant.is_some()).count() as u32;
        txn
    }

    /// Where a virtual tile actually lands — the finest resident ancestor.
    ///
    /// Reads the maintained table, exactly as the shader will, and then re-derives
    /// the resolved tile's address by walking the same clamped chain. `None` only
    /// for an unknown handle or an address outside the grid.
    pub fn resolve(&self, tex: VtTextureHandle, at: TileCoord) -> Option<VtResolved> {
        let t = self.textures.get(tex.index())?;
        let word = *self
            .table
            .words
            .get(self.table.entry_word(tex.index(), &t.desc, at)?)?;
        let e = unpack_entry(word);
        Some(VtResolved {
            slot: e.slot,
            tile: t.desc.ancestor(at, e.mip)?,
        })
    }

    /// Whether `at` is itself resident (as opposed to resolving to an ancestor).
    pub fn is_resident(&self, tex: VtTextureHandle, at: TileCoord) -> bool {
        self.textures.get(tex.index()).is_some_and(|t| {
            t.desc
                .entry_index(at)
                .is_some_and(|i| t.resident[i as usize].is_some())
        })
    }

    /// What a slot holds, if anything.
    pub fn slot_occupant(&self, slot: u32) -> Option<(VtTextureHandle, TileCoord)> {
        self.slots
            .get(slot as usize)?
            .occupant
            .map(|(t, c)| (VtTextureHandle(t), c))
    }

    /// Whether a slot holds a pinned root.
    pub fn slot_is_root(&self, slot: u32) -> bool {
        self.slots.get(slot as usize).is_some_and(|s| s.pinned)
    }

    /// Whether this texture's root pages have been emitted in an applied
    /// transaction — i.e. whether the atlas really holds what the table claims.
    ///
    /// P26.3 must not sample a texture that is not warm: its table is complete
    /// from registration, but the pages behind it exist only once the transaction
    /// carrying them has been applied.
    pub fn is_warm(&self, tex: VtTextureHandle) -> bool {
        self.textures.get(tex.index()).is_some_and(|t| t.warm)
    }

    // ── the table, for the mirror ───────────────────────────────────────────

    /// The whole indirection image.
    #[inline]
    pub fn table_words(&self) -> &[u32] {
        &self.table.words
    }

    /// One texture's block: `(word offset, words)`.
    pub fn table_block(&self, tex: VtTextureHandle) -> Option<(usize, &[u32])> {
        let b = self.table.blocks.get(tex.index())?;
        Some((b.base, &self.table.words[b.base..b.base + b.len]))
    }

    /// The layout generation — moves when a texture is registered, so a mirror
    /// knows its buffer is the wrong shape. **A stamp, not a measurement.**
    #[inline]
    pub fn layout_generation(&self) -> u64 {
        self.layout_generation
    }

    /// One texture's block generation — moves when any of its entries change.
    /// **A stamp, not a measurement.**
    pub fn generation(&self, tex: VtTextureHandle) -> Option<u64> {
        self.textures.get(tex.index()).map(|t| t.generation)
    }

    // ── internals ───────────────────────────────────────────────────────────

    fn relayout(&mut self) {
        let descs: Vec<VtTextureDesc> = self.textures.iter().map(|t| t.desc.clone()).collect();
        self.table = TableImage::layout(
            &descs,
            self.geometry.slots_x,
            self.geometry.stored_tile_size,
        );
        for t in 0..self.textures.len() {
            self.recompute_entries(t);
        }
        self.layout_generation = next_stamp();
        self.layout_dirty = true;
    }

    /// Fill one texture's entries from scratch, coarsest level first, so a level's
    /// non-resident tiles can simply copy their parent's already-final entry.
    ///
    /// The only from-scratch computation in the crate; everything else patches.
    fn recompute_entries(&mut self, t: usize) {
        let desc = self.textures[t].desc.clone();
        for mip in (0..desc.mip_count()).rev() {
            let m = desc.mips[mip as usize];
            for y in 0..m.tiles_y {
                for x in 0..m.tiles_x {
                    let at = TileCoord::new(mip, x, y);
                    let idx = desc.entry_index(at).expect("in grid");
                    let word = match self.textures[t].resident[idx as usize] {
                        Some(slot) => pack_entry(slot, mip),
                        None => match desc.parent(at) {
                            Some(p) => {
                                self.table.words
                                    [self.table.entry_word(t, &desc, p).expect("parent in grid")]
                            }
                            // The coarsest level is always resident, so this is
                            // reachable only between `relayout` and the seating of
                            // a brand-new texture's roots — a window inside
                            // `register_texture` with no observer.
                            None => 0,
                        },
                    };
                    let w = self.table.entry_word(t, &desc, at).expect("in grid");
                    self.table.words[w] = word;
                }
            }
        }
    }

    /// Write `word` into `from`'s entry and into every **non-resident** descendant
    /// whose fallback ran through it. A resident descendant stops the walk: its
    /// own subtree resolves to it, not through `from`.
    fn propagate(&mut self, t: usize, from: TileCoord, word: u32) {
        let desc = self.textures[t].desc.clone();
        let mut stack = vec![from];
        while let Some(v) = stack.pop() {
            let w = self.table.entry_word(t, &desc, v).expect("in grid");
            self.table.words[w] = word;
            if v.mip == 0 {
                continue;
            }
            for cy in desc.child_y_range(v.mip, v.y) {
                for cx in desc.child_x_range(v.mip, v.x) {
                    let c = TileCoord::new(v.mip - 1, cx, cy);
                    let idx = desc.entry_index(c).expect("child in grid");
                    if self.textures[t].resident[idx as usize].is_none() {
                        stack.push(c);
                    }
                }
            }
        }
    }

    fn touch(&mut self, slot: u32) {
        if let Some(s) = self.slots.get_mut(slot as usize) {
            s.stamp = next_stamp();
        }
    }

    /// Put `tile` of texture `t` in `slot` and re-point the fallbacks beneath it.
    fn seat(&mut self, slot: u32, t: u32, tile: TileCoord, pinned: bool) {
        self.slots[slot as usize] = Slot {
            occupant: Some((t, tile)),
            stamp: next_stamp(),
            pinned,
        };
        let ti = t as usize;
        let idx = self.textures[ti]
            .desc
            .entry_index(tile)
            .expect("seating a tile that exists");
        self.textures[ti].resident[idx as usize] = Some(slot);
        self.propagate(ti, tile, pack_entry(slot, tile.mip));
        self.textures[ti].generation = next_stamp();
        if pinned {
            self.stats.roots += 1;
        }
    }

    /// The lowest free slot, or the least-recently-stamped evictable one.
    ///
    /// Never returns a pinned root and never returns a `protected` slot (one this
    /// transaction has already admitted or touched), so **a transaction can never
    /// evict what it just brought in**.
    fn acquire_slot(&mut self, protected: &BTreeSet<u32>) -> Option<(u32, Option<VtEvict>)> {
        if let Some(&slot) = self.free.iter().next() {
            self.free.remove(&slot);
            return Some((slot, None));
        }
        let victim = lru_victim(self.slots.iter().enumerate().filter_map(|(i, s)| {
            let slot = i as u32;
            (s.occupant.is_some() && !s.pinned && !protected.contains(&slot))
                .then_some((s.stamp, slot))
        }))?;
        let evict = self.unseat(victim);
        self.free.remove(&victim);
        Some((victim, Some(evict)))
    }

    /// Empty `slot`, re-pointing its tile (and the subtree that fell back through
    /// it) at its parent's entry.
    fn unseat(&mut self, slot: u32) -> VtEvict {
        let (t, tile) = self.slots[slot as usize]
            .occupant
            .take()
            .expect("a victim is occupied");
        debug_assert!(
            !self.slots[slot as usize].pinned,
            "a pinned root must never be unseated"
        );
        let ti = t as usize;
        let desc = self.textures[ti].desc.clone();
        let idx = desc.entry_index(tile).expect("seated tile exists");
        self.textures[ti].resident[idx as usize] = None;
        // A root is never unseated, so a victim always has a parent.
        let parent = desc.parent(tile).expect("a non-root tile has a parent");
        let word = self.table.words[self.table.entry_word(ti, &desc, parent).expect("in grid")];
        self.propagate(ti, tile, word);
        self.textures[ti].generation = next_stamp();
        self.free.insert(slot);
        VtEvict {
            slot,
            texture: VtTextureHandle(t),
            tile,
        }
    }
}

/// The eviction rule: the smallest `(stamp, slot)`.
///
/// A free function so the tie-break can be exercised directly. Stamps come from a
/// global monotone counter, so two live slots cannot in fact hold the same one —
/// which is exactly why the tie-break must not be left implicit. Ordering that
/// depended on the counter's uniqueness would be an ordering that stops being
/// total the day the counter is shared, wrapped or replaced (P28.3 proposes all
/// three), and the failure mode would be an eviction order that varies run to run.
pub(crate) fn lru_victim(candidates: impl Iterator<Item = (u64, u32)>) -> Option<u32> {
    candidates
        .min_by_key(|&(stamp, slot)| (stamp, slot))
        .map(|(_, s)| s)
}

/// A `BTreeMap` of every texture's resolved table, for tests and debug views.
///
/// Ships (rather than living in a test) because the mirror's readback arm
/// compares GPU bytes against exactly this.
pub fn resolved_table(res: &VtResidency) -> BTreeMap<(u32, TileCoord), VtResolved> {
    let mut out = BTreeMap::new();
    for t in 0..res.texture_count() {
        let handle = VtTextureHandle(t as u32);
        let Some(desc) = res.desc(handle) else {
            continue;
        };
        for mip in 0..desc.mip_count() {
            let m = desc.mips[mip as usize];
            for y in 0..m.tiles_y {
                for x in 0..m.tiles_x {
                    let at = TileCoord::new(mip, x, y);
                    if let Some(r) = res.resolve(handle, at) {
                        out.insert((t as u32, at), r);
                    }
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::address::full_pyramid;
    use crate::pool::PageFormat;

    fn pool(pages: u64) -> VtResidency {
        let (r, _) = VtResidency::new(VtPoolConfig {
            format: PageFormat::Bc1,
            stored_tile_size: 136,
            budget_bytes: PageFormat::Bc1.page_bytes(136) * pages,
            max_texture_dim: 8192,
        });
        r
    }

    #[test]
    fn the_lru_tie_break_is_the_slot_index() {
        // Equal stamps cannot arise from the global counter — which is why this
        // is exercised on the rule directly rather than through the pool.
        assert_eq!(
            lru_victim([(7u64, 9u32), (7, 3), (7, 5)].into_iter()),
            Some(3)
        );
        assert_eq!(
            lru_victim([(9u64, 1u32), (2, 8), (5, 4)].into_iter()),
            Some(8)
        );
        assert_eq!(lru_victim(std::iter::empty()), None);
    }

    #[test]
    fn registration_seats_the_roots_and_the_first_transaction_emits_them() {
        let mut r = pool(16);
        let h = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .expect("fits");
        assert_eq!(r.stats().roots, 1, "a full pyramid has one root tile");
        assert!(!r.is_warm(h), "the bytes are not in the atlas yet");
        // Every virtual tile already resolves — to the root.
        let root = TileCoord::new(8, 0, 0);
        assert_eq!(
            r.resolve(h, TileCoord::new(0, 2, 1)),
            Some(VtResolved {
                slot: 0,
                tile: root
            })
        );

        let txn = r.apply_wants(&[]);
        assert!(txn.layout_rebuilt);
        assert_eq!(
            txn.admits,
            vec![VtAdmit {
                slot: 0,
                texture: h,
                tile: root
            }]
        );
        assert!(r.is_warm(h));
        // …and a second transaction does not emit it again.
        assert!(r.apply_wants(&[]).admits.is_empty());
    }

    #[test]
    fn a_mip_less_texture_pins_its_whole_grid() {
        let mut r = pool(16);
        let desc = crate::VtTextureDesc {
            tile_size: 128,
            border: 4,
            srgb: false,
            mips: vec![crate::VtMipDesc {
                width: 320,
                height: 192,
                tiles_x: 3,
                tiles_y: 2,
            }],
        };
        let h = r.register_texture(desc).expect("fits in 16 pages");
        assert_eq!(r.stats().roots, 6);
        for slot in 0..6 {
            assert!(r.slot_is_root(slot), "slot {slot} must be pinned");
        }
        assert_eq!(r.apply_wants(&[]).admits.len(), 6);
        // Its only level is the coarsest, so everything is resident exactly.
        assert!(r.is_resident(h, TileCoord::new(0, 2, 1)));
    }

    #[test]
    fn the_mandatory_floor_is_refused_by_name() {
        let mut r = pool(4);
        let big = crate::VtTextureDesc {
            tile_size: 128,
            border: 4,
            srgb: false,
            mips: vec![crate::VtMipDesc {
                width: 1024,
                height: 1024,
                tiles_x: 8,
                tiles_y: 8,
            }],
        };
        let err = r.register_texture(big).expect_err("64 roots, 4 slots");
        assert_eq!(
            err,
            VtError::MandatoryFloorExceedsBudget {
                roots: 64,
                floor_bytes: 64 * 9_248,
                budget_bytes: 4 * 9_248,
                slots: 4,
            }
        );
        assert!(
            err.to_string().contains("mandatory floor"),
            "the refusal names itself: {err}"
        );
        // The refusal is total: nothing was registered, nothing was seated.
        assert_eq!(r.texture_count(), 0);
        assert_eq!(r.stats().roots, 0);
    }

    #[test]
    fn a_texture_whose_tiles_are_the_wrong_size_is_refused() {
        let mut r = pool(16);
        let err = r
            .register_texture(full_pyramid(320, 192, 128, 8, false))
            .expect_err("152² tiles in a 136² pool");
        assert_eq!(
            err,
            VtError::PoolGeometryMismatch {
                desc: 144,
                pool: 136
            }
        );
    }

    #[test]
    fn a_want_outside_the_grid_is_counted_not_admitted() {
        let mut r = pool(16);
        let h = r
            .register_texture(full_pyramid(320, 192, 128, 4, false))
            .unwrap();
        r.apply_wants(&[]);
        let txn = r.apply_wants(&[
            VtWant::new(h, TileCoord::new(0, 99, 0)),
            VtWant::new(h, TileCoord::new(99, 0, 0)),
            VtWant::new(VtTextureHandle(7), TileCoord::new(0, 0, 0)),
        ]);
        assert_eq!(txn.out_of_range, 2, "an unknown handle is filtered earlier");
        assert!(txn.admits.is_empty());
        // …and it reaches the trace, which is the only place a caller watching a
        // transaction stream would ever see it.
        assert_eq!(txn.trace(), "oor 2\n");
    }

    /// The stats line names every number it carries, and the clamp flag only
    /// when it is set.
    #[test]
    fn the_stats_line_says_what_it_counts() {
        let mut r = pool(4);
        let h = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .unwrap();
        r.apply_wants(&[]);
        let quiet = r.stats().summary();
        assert!(quiet.contains("vt residency"), "{quiet}");
        assert!(quiet.contains("1/4 slots resident"), "{quiet}");
        assert!(quiet.contains("1 pinned roots"), "{quiet}");
        assert!(
            !quiet.contains("budget-clamped"),
            "nothing was clamped: {quiet}"
        );

        let wants: Vec<VtWant> = (0..3)
            .flat_map(|x| (0..2).map(move |y| VtWant::new(h, TileCoord::new(0, x, y))))
            .collect();
        r.apply_wants(&wants);
        let clamped = r.stats().summary();
        assert!(clamped.contains("4/4 slots resident"), "{clamped}");
        assert!(clamped.contains("6 wanted / 3 deferred"), "{clamped}");
        assert!(clamped.contains("[budget-clamped]"), "{clamped}");
    }

    /// **The floor may fill the pool exactly**, and then the pool is all floor.
    ///
    /// `register_texture` refuses a floor *past* the slot count, so a floor equal
    /// to it is admitted — the boundary, and the specified answer. What must not
    /// happen is a panic when the next want finds nothing evictable: every slot is
    /// pinned, so the want is deferred by the ordinary path and the number says so.
    #[test]
    fn a_pool_that_is_all_floor_defers_rather_than_panics() {
        let mut r = pool(2);
        let a = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .expect("one root");
        r.register_texture(full_pyramid(256, 256, 128, 4, true))
            .expect("the second root fills the pool exactly");
        assert_eq!(r.stats().roots, 2);
        assert_eq!(r.geometry().slot_count(), 2, "floor == budget, to the slot");
        assert_eq!(r.apply_wants(&[]).admits.len(), 2);

        let txn = r.apply_wants(&[VtWant::new(a, TileCoord::new(0, 0, 0))]);
        assert!(txn.admits.is_empty() && txn.evicts.is_empty());
        assert_eq!(txn.deferred, 1);
        assert!(r.stats().budget_clamped);
        // The law still holds with no cache at all: the address resolves, blurrily.
        assert_eq!(
            r.resolve(a, TileCoord::new(0, 0, 0)).map(|x| x.tile),
            Some(TileCoord::new(8, 0, 0))
        );
    }
}
