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
//! a seeded churn in `tests/residency_gate.rs` — the whole table, not a sample of
//! it.

use std::collections::{BTreeMap, BTreeSet};

use crate::address::{DescError, TileCoord, VtTextureDesc};
use crate::pool::{plan_pool, PageFormat, VtAdvisory, VtPoolConfig, VtPoolGeometry};
use crate::table::{pack_entry, unpack_entry, TableImage, TableTexture, MAX_SLOT_INDEX};

/// Source of residency stamps: **`inf_stream::next_stamp`**, the one domain
/// (P28.3, clause 1).
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
/// **It used to be a third counter of its own**, with a comment saying so and
/// naming P28.3 as where the three would merge. This is that merge, and what it
/// buys is not tidiness: three domains agree on the property each was built for
/// (never decreasing) and have no answer at all for the one cross-system
/// eviction asks — whether a cluster page was touched more recently than a
/// texture tile. See `inf_stream::stamp`.
use inf_stream::next_stamp;

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

/// How badly a tile is wanted — **the primary sort key**, lower first.
///
/// P26.2 shipped an unranked want set and said so: *"When feedback arrives,
/// priority becomes the primary key and payload order stays as the tie-break."*
/// This is that, and it is what makes the P26.4 floor a **floor**: floor wants
/// are [`VT_PRIORITY_FLOOR`] and feedback refinements are
/// [`VT_PRIORITY_FEEDBACK`], so a transaction seats every floor tile it can
/// before a refinement is offered a slot — and a refinement can never take a
/// slot from a floor tile the same transaction just admitted, because
/// [`VtResidency::apply_wants`] protects what it has touched.
///
/// **Since P28.3 it is `inf_stream::Lane`**, the one lane vocabulary, so this
/// crate's ranks and `inf-vsm`'s are the same numbers ordered by the same walk.
/// The rank now also outranks a *resident* weaker want, which it did not before
/// — see [`VtResidency::apply_wants`].
pub type VtPriority = inf_stream::Lane;

/// The analytic floor: residency may never drop below it while the budget
/// allows. `0` so [`VtWant::new`]'s existing callers keep their exact behaviour.
pub const VT_PRIORITY_FLOOR: VtPriority = inf_stream::LANE_FLOOR;

/// A refinement the GPU feedback asked for. Served after the whole floor.
pub const VT_PRIORITY_FEEDBACK: VtPriority = inf_stream::LANE_FEEDBACK;

/// **A tile the predictor speculated about** (P28.4) — the analytic floor
/// rule asked at a *predicted* camera rather than at the committed one.
///
/// `inf_stream::LANE_PREDICT`, so it is strictly below both producers above it:
/// a speculative want may never take a floor tile's slot nor a resident
/// refinement's, and every one of them may take a speculative tile's. That is
/// the ROADMAP's clause — *"speculative wants enter at strictly lower priority
/// than the analytic floor and feedback"* — spelled as a rank the one admission
/// walk already orders, rather than as a policy anything has to remember.
///
/// It is a *third* lane here because this consumer has two producers above it,
/// and it is the **same** lane `inf_vsm::VSM_PRIORITY_SPECULATIVE` uses even
/// though that consumer has one — which is the correction P28.4 made to its own
/// first reading. The invariant is one statement over all three consumers, so a
/// speculation ranked into the feedback lane next door would make *residency ⊇
/// floor ∪ feedback* mean something different there than it means here.
pub const VT_PRIORITY_PREDICT: VtPriority = inf_stream::LANE_PREDICT;

/// "This tile, please." The whole input language of residency — deliberately no
/// camera and no budget: see the crate docs. Since P26.4 it carries a
/// [`priority`](Self::priority), which is a *rank*, not a policy — the caller
/// still decides what to want and how badly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtWant {
    pub texture: VtTextureHandle,
    pub tile: TileCoord,
    /// Lower is served first. See [`VtPriority`].
    pub priority: VtPriority,
}

impl VtWant {
    /// A **floor** want ([`VT_PRIORITY_FLOOR`]) — the pre-P26.4 meaning of every
    /// existing call site, unchanged.
    pub const fn new(texture: VtTextureHandle, tile: TileCoord) -> Self {
        Self {
            texture,
            tile,
            priority: VT_PRIORITY_FLOOR,
        }
    }

    /// A **refinement** want ([`VT_PRIORITY_FEEDBACK`]) — what a decoded
    /// feedback mask produces.
    pub const fn refine(texture: VtTextureHandle, tile: TileCoord) -> Self {
        Self {
            texture,
            tile,
            priority: VT_PRIORITY_FEEDBACK,
        }
    }

    /// A **speculative** want ([`VT_PRIORITY_PREDICT`]) — what the dead-reckoning
    /// predictor produces (P28.4).
    pub const fn speculate(texture: VtTextureHandle, tile: TileCoord) -> Self {
        Self {
            texture,
            tile,
            priority: VT_PRIORITY_PREDICT,
        }
    }

    /// The same want at an explicit rank.
    pub const fn with_priority(self, priority: VtPriority) -> Self {
        Self { priority, ..self }
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
    /// Of [`deferred`](Self::deferred), how many the **per-frame upload budget**
    /// held back (IB-16) rather than the pool being full. These arrive on a later
    /// frame; the rest need a bigger pool.
    pub throttled: u32,
    /// Wants naming a tile outside their texture's grid — a caller bug, counted
    /// rather than panicked on, because a want set is computed from a camera.
    pub out_of_range: u32,
    /// Wants naming a texture handle this residency does not have (P26.4; the
    /// P26.3 ledger's *"a want naming an unknown texture handle is dropped
    /// silently rather than counted, unlike an out-of-range tile"*).
    ///
    /// It is a different defect from [`out_of_range`](Self::out_of_range) and
    /// deserves its own number: an out-of-range tile is a want set computed
    /// against the wrong extent, while an unknown handle is a want set computed
    /// against **a different registry**. Counted after dedup, exactly as
    /// `out_of_range` is.
    ///
    /// **It has no producer in this tree, and that is deliberate rather than
    /// hopeful** (P26.4 audit; the first write-up claimed a stale feedback mask
    /// after a level switch produced one, and none can). Every want-emitting path
    /// filters an unknown handle before this sees it:
    /// `VtFeedbackLayout::wants_at` resolves each handle through `res.desc()` and
    /// skips what it cannot, `inf_render::analytic_floor` does the same, and
    /// `want_floor` iterates the registry itself. So the number is a **tripwire
    /// for a future caller** — P28.3's unified streamer keeps want sets across a
    /// level boundary by design — and it reads zero today because nothing can
    /// make it read anything else, not because nothing has gone wrong.
    pub unknown_texture: u32,
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
        if self.unknown_texture > 0 {
            s.push_str(&format!("unk {}\n", self.unknown_texture));
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
    /// Whether the last transaction deferred a want at all — `deferred > 0`.
    ///
    /// **It stopped meaning "for want of a slot" when IB-16 landed** (corrected by
    /// the I4 audit): a want held back by the per-frame upload budget is deferred
    /// too, so this is now true in both regimes. [`throttled`](Self::throttled) is
    /// how the two are told apart, and `budget_clamped && throttled == deferred`
    /// is "nothing was short of a slot".
    pub budget_clamped: bool,
    /// Of [`deferred`](Self::deferred), how many the **per-frame upload budget**
    /// held back rather than the pool being full (IB-16).
    ///
    /// A pool with spare slots reporting only `deferred` is indistinguishable
    /// from a pool that is out of them, and the two want opposite fixes: a
    /// bigger pool against one more frame's patience.
    pub throttled: u32,
    /// Bytes of page data the last transaction admitted — what the upload budget
    /// is spent in, reported whether or not it bit.
    pub upload_bytes: u64,
    /// Consecutive frames the upload budget has held pages back. Past
    /// [`VT_SUSTAINED_THROTTLE_FRAMES`](crate::VT_SUSTAINED_THROTTLE_FRAMES) the
    /// residency raises [`VtAdvisory::UploadBudgetSustained`](crate::VtAdvisory).
    pub throttled_frames: u32,
    /// Wants naming an unknown texture handle since the residency was created
    /// (P26.4). Cumulative, unlike [`VtTransaction::unknown_texture`], because
    /// the interesting reading is "has this ever happened" — one stale mask is
    /// a bug however few frames it lasted.
    pub unknown_texture: u64,
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
        ) + &if self.unknown_texture > 0 {
            format!(" [{} unknown-handle wants]", self.unknown_texture)
        } else {
            String::new()
        }
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
    #[error(
        "this residency has {arms} page-pool arm(s) and the registration named arm {arm}; a \
         texture is registered into the arm whose format its container stores, so an arm that \
         does not exist means the caller planned its arms from different content than it is \
         registering"
    )]
    NoSuchArm { arm: usize, arms: usize },
}

/// One physical slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot {
    /// `(texture index, tile)` — `None` when free.
    occupant: Option<(u32, TileCoord)>,
    /// Last touched. **A stamp, not a measurement** (see
    /// [`inf_stream::next_stamp`], which `NEXT_VT_STAMP` merged into at P28.3).
    stamp: u64,
    /// A root: never evicted, for any reason, by anyone.
    pinned: bool,
}

/// One registered texture's residency.
#[derive(Debug, Clone)]
struct Texture {
    desc: VtTextureDesc,
    /// Exact-resident slot per virtual tile, indexed by `desc.entry_index`.
    /// The slot is an index into [`Arm::slots`] of [`Texture::arm`].
    resident: Vec<Option<u32>>,
    generation: u64,
    /// Whether this texture's root pages have been emitted in a transaction (and
    /// therefore actually exist in the atlas, not merely in the table).
    warm: bool,
    /// **Which arm holds this texture's pages** (wave IASSET2). Decided at
    /// registration from the stored format and never changed: a `.inf_tex`'s
    /// format is a property of the file.
    arm: u32,
}

/// **One physical page pool** — one atlas, one format, one slot size (wave
/// IASSET2).
///
/// A residency used to *be* this. It now holds one per stored page format, so a
/// level whose textures mix BC1 and BC5 pages both of them at their own size
/// instead of demoting the whole atlas to RGBA8. The rule that made the old
/// arrangement simple is untouched **inside** an arm: every slot is one fixed
/// size, so allocation is "take the lowest free index" and fragmentation cannot
/// exist. What changed is that there are several such arrays, and a texture is
/// in exactly one of them for its whole life.
#[derive(Debug, Clone)]
struct Arm {
    cfg: VtPoolConfig,
    geometry: VtPoolGeometry,
    page_bytes: u64,
    slots: Vec<Slot>,
    free: BTreeSet<u32>,
    /// Slots this arm has pinned as roots — the mandatory floor, per arm,
    /// because a floor that fits in total and not in its own arm is a floor that
    /// does not fit.
    roots: u32,
}

/// The residency of one virtual-texture pool — since wave IASSET2, of its
/// **arms**: one physical page pool per stored page format.
#[derive(Debug, Clone)]
pub struct VtResidency {
    arms: Vec<Arm>,
    /// Texels per side of a slot. Shared by every arm by construction — the
    /// container writes one [`crate::STORED_TILE_SIZE`] — so a texture is
    /// refused for the wrong tile geometry against one number rather than per
    /// arm.
    stored_tile_size: u32,
    trilinear: bool,
    textures: Vec<Texture>,
    table: TableImage,
    pending_roots: Vec<(VtTextureHandle, TileCoord)>,
    pending_evicts: Vec<VtEvict>,
    layout_generation: u64,
    layout_dirty: bool,
    stats: VtStats,
    /// Consecutive frames the per-frame upload budget has held pages back
    /// (IB-16). Reset the moment a frame throttles nothing, so it counts a
    /// *run* rather than a total — a burst that drains is not sustained demand.
    throttled_run: u32,
}

impl VtResidency {
    /// A **single-arm** pool sized from `cfg`, with the advisories planning it
    /// raised.
    ///
    /// The advisories are returned rather than logged, so a host decides whether
    /// they are a warning, a fatal, or a line in an import report — the
    /// `texture_import_advisories` shape.
    ///
    /// This is what every caller that has one stored format wants, and it is
    /// exactly the pre-IASSET2 residency: one arm, arm 0, and every existing
    /// transaction trace unchanged to the byte.
    pub fn new(cfg: VtPoolConfig) -> (Self, Vec<VtAdvisory>) {
        Self::new_multi(std::slice::from_ref(&cfg))
    }

    /// **A pool with one arm per stored page format** (wave IASSET2).
    ///
    /// `arms` is one config per format, in the order the arms are numbered — and
    /// that number reaches the shader (the table block's `pool` word) and the
    /// mirror (which atlas a page is written into), so it must be a function of
    /// the level's content rather than of a hash walk. `inf_render::build_vt_level`
    /// derives it from the registration order.
    ///
    /// Every arm shares `stored_tile_size`, `max_texture_dim` and `trilinear` —
    /// the first because the container writes one tile geometry, the second
    /// because there is one device, the third because it is a filtering rule and
    /// filtering a normal map differently from the albedo beside it is not a
    /// feature. The first arm's values win and the rest are ignored, which is
    /// safe because the one constructor that builds several
    /// ([`crate::split_pool_budget`]) copies them from a single base.
    ///
    /// An empty `arms` is an empty pool: legal, holds nothing, refuses every
    /// registration by the mandatory-floor rule. It is what a caller with no
    /// content at all would build and it is not a special case anywhere.
    pub fn new_multi(arms: &[VtPoolConfig]) -> (Self, Vec<VtAdvisory>) {
        let mut advisories = Vec::new();
        let base = arms.first().copied().unwrap_or_default();
        let mut built = Vec::with_capacity(arms.len());
        for cfg in arms {
            let cfg = VtPoolConfig {
                stored_tile_size: base.stored_tile_size,
                max_texture_dim: base.max_texture_dim,
                trilinear: base.trilinear,
                ..*cfg
            };
            let (geometry, mut a) = plan_pool(cfg);
            advisories.append(&mut a);
            let n = geometry.slot_count();
            debug_assert!(n <= MAX_SLOT_INDEX + 1, "a slot index must fit 16 bits");
            built.push(Arm {
                page_bytes: cfg.format.page_bytes(geometry.stored_tile_size),
                cfg,
                geometry,
                slots: vec![
                    Slot {
                        occupant: None,
                        stamp: 0,
                        pinned: false,
                    };
                    n as usize
                ],
                free: (0..n).collect(),
                roots: 0,
            });
        }
        let slots = built.iter().map(|a| a.geometry.slot_count()).sum();
        let stored_tile_size = built.first().map_or(base.stored_tile_size.max(4), |a| {
            a.geometry.stored_tile_size
        });
        let this = Self {
            table: TableImage::layout(&[], built.len() as u32, stored_tile_size, base.trilinear),
            arms: built,
            stored_tile_size,
            trilinear: base.trilinear,
            throttled_run: 0,
            textures: Vec::new(),
            pending_roots: Vec::new(),
            pending_evicts: Vec::new(),
            layout_generation: next_stamp(),
            layout_dirty: false,
            stats: VtStats {
                slots,
                ..Default::default()
            },
        };
        (this, advisories)
    }

    /// How many arms this residency has — atlases the mirror binds.
    #[inline]
    pub fn arm_count(&self) -> usize {
        self.arms.len()
    }

    /// Which arm holds `tex`'s pages, or `None` for an unknown handle.
    #[inline]
    pub fn arm_of(&self, tex: VtTextureHandle) -> Option<usize> {
        self.textures.get(tex.index()).map(|t| t.arm as usize)
    }

    /// The **first** arm's config — what a single-arm pool's caller means by
    /// "the config", and the arm every pre-IASSET2 constructor built.
    ///
    /// A multi-arm residency has one of these per arm; ask
    /// [`config_of`](Self::config_of).
    #[inline]
    pub fn config(&self) -> VtPoolConfig {
        self.config_of(0).unwrap_or_default()
    }
    /// Arm `arm`'s config.
    #[inline]
    pub fn config_of(&self, arm: usize) -> Option<VtPoolConfig> {
        self.arms.get(arm).map(|a| a.cfg)
    }
    /// The page format arm `arm` stores — what the mirror creates its atlas in.
    #[inline]
    pub fn arm_format(&self, arm: usize) -> Option<PageFormat> {
        self.arms.get(arm).map(|a| a.cfg.format)
    }
    /// The **first** arm's atlas rectangle. See [`config`](Self::config).
    #[inline]
    pub fn geometry(&self) -> VtPoolGeometry {
        self.geometry_of(0).unwrap_or(VtPoolGeometry {
            slots_x: 0,
            slots_y: 0,
            stored_tile_size: self.stored_tile_size,
        })
    }
    /// Arm `arm`'s atlas rectangle.
    #[inline]
    pub fn geometry_of(&self, arm: usize) -> Option<VtPoolGeometry> {
        self.arms.get(arm).map(|a| a.geometry)
    }
    /// Bytes one page of the **first** arm occupies. See [`config`](Self::config).
    #[inline]
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes_of(0).unwrap_or(0)
    }
    /// Bytes one page of arm `arm` occupies.
    #[inline]
    pub fn page_bytes_of(&self, arm: usize) -> Option<u64> {
        self.arms.get(arm).map(|a| a.page_bytes)
    }
    /// The VRAM every atlas costs together — `Σ slots × page_bytes`, and
    /// **never** more than the budget the arms were split from.
    #[inline]
    pub fn capacity_bytes(&self) -> u64 {
        self.arms
            .iter()
            .map(|a| u64::from(a.geometry.slot_count()) * a.page_bytes)
            .sum()
    }
    /// Bytes currently holding a page, across every arm.
    #[inline]
    pub fn resident_bytes(&self) -> u64 {
        self.arms
            .iter()
            .map(|a| a.slots.iter().filter(|s| s.occupant.is_some()).count() as u64 * a.page_bytes)
            .sum()
    }
    /// **Bytes the mandatory floor holds** — the always-resident root pages,
    /// priced in each arm's own page size.
    ///
    /// A residency-level answer rather than `roots × page_bytes()` at the call
    /// site, which is what the budget reporters did and which prices every arm's
    /// roots at the first arm's page size the moment there are two.
    #[inline]
    pub fn floor_bytes(&self) -> u64 {
        self.arms
            .iter()
            .map(|a| u64::from(a.roots) * a.page_bytes)
            .sum()
    }
    #[inline]
    /// **The sustained-demand advisory** (IB-16), or `None` while the budget is
    /// only absorbing a burst.
    ///
    /// Read once per frame by the host beside [`stats`](Self::stats). It fires on
    /// a *run* of throttled frames rather than on a total, because a burst that
    /// drains is the throttle working and a run that does not drain is content
    /// asking for more bandwidth than the budget grants — and telling an author
    /// the second thing when the first happened is the wrong-diagnosis hazard
    /// this tree has already paid for once (`AssetPayload::migrates_from`).
    pub fn upload_advisory(&self) -> Option<crate::VtAdvisory> {
        (self.throttled_run >= crate::VT_SUSTAINED_THROTTLE_FRAMES).then_some(
            crate::VtAdvisory::UploadBudgetSustained {
                // The arms' budgets summed — the number a host would raise, and
                // the one `split_pool_budget` divided in the first place.
                budget_bytes: self.arms.iter().map(|a| a.cfg.upload_budget_bytes).sum(),
                frames: self.throttled_run,
                pages: self.stats.throttled,
            },
        )
    }

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

    /// Whether this residency's registration of `tex` **has** the tile `at` —
    /// i.e. whether the address exists at all, as opposed to existing and not
    /// being seated yet.
    ///
    /// The distinction is the whole point and it is not cosmetic (P28.2 audit).
    /// [`is_resident`](Self::is_resident) answers `false` for both "not paged in"
    /// and "no such tile", and the two want opposite treatment from a caller
    /// holding an address it did not compute this frame: *not paged in* is a
    /// budget answer that a later transaction can change, while *no such tile* is
    /// a statement about a **different image** than the one the address was
    /// derived against, and no budget will ever change it. A consumer that
    /// retries the first for ever is right; a consumer that retries the second
    /// for ever never draws again.
    ///
    /// The producer is P28.2's cluster pairing, whose `.inf_vmesh` tiles section
    /// holds `(texture guid, mip, x, y)` addresses baked at cook against the
    /// `.inf_tex` of that cook. Nothing in either container ties the two, so a
    /// `.inf_vmesh` that meets a re-tiled image of its texture is asking for
    /// tiles that do not exist — measured, before this door existed: residency
    /// went to **zero pages and stayed there**, silently, because the root page's
    /// coarsest-mip tile is the first address a shrunken pyramid loses.
    pub fn can_address(&self, tex: VtTextureHandle, at: TileCoord) -> bool {
        self.textures
            .get(tex.index())
            .is_some_and(|t| t.desc.entry_index(at).is_some())
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
        self.register_texture_in(desc, 0)
    }

    /// [`register_texture`](Self::register_texture) into a named **arm** (wave
    /// IASSET2) — the door a caller with several stored formats uses.
    ///
    /// The arm is the caller's decision because the caller is the one holding
    /// the `.inf_tex` header: `inf_render::build_vt_level` reads
    /// [`crate::stored_page_format`] off every payload it resolved, plans one
    /// arm per distinct format, and registers each texture into its own. An arm
    /// index this residency does not have is refused by name rather than
    /// silently folded into arm 0, which would put a BC5 page in a BC1 atlas —
    /// the right length, the wrong texels, and no error anywhere.
    pub fn register_texture_in(
        &mut self,
        desc: VtTextureDesc,
        arm: usize,
    ) -> Result<VtTextureHandle, VtError> {
        desc.validate()?;
        if desc.stored_tile_size() != self.stored_tile_size {
            return Err(VtError::PoolGeometryMismatch {
                desc: desc.stored_tile_size(),
                pool: self.stored_tile_size,
            });
        }
        let Some(a) = self.arms.get(arm) else {
            return Err(VtError::NoSuchArm {
                arm,
                arms: self.arms.len(),
            });
        };
        let roots = desc.root_tiles();
        let floor = a.roots + roots.len() as u32;
        if floor > a.geometry.slot_count() {
            return Err(VtError::MandatoryFloorExceedsBudget {
                roots: floor,
                floor_bytes: u64::from(floor) * a.page_bytes,
                budget_bytes: a.cfg.budget_bytes,
                slots: a.geometry.slot_count(),
            });
        }

        let handle = VtTextureHandle(self.textures.len() as u32);
        let tiles = desc.tile_count() as usize;
        self.textures.push(Texture {
            desc,
            resident: vec![None; tiles],
            generation: next_stamp(),
            warm: false,
            arm: arm as u32,
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
                .acquire_slot(arm, &protected)
                .expect("the mandatory-floor check guarantees a slot for every root");
            if let Some(e) = evicted {
                self.pending_evicts.push(e);
            }
            self.seat(slot, handle.0, tile, true);
            protected.insert(slot);
            self.pending_roots.push((handle, tile));
        }
        self.stats.textures = self.textures.len();
        // Registration seats pages, so it moves the resident count. `apply_wants`
        // recomputes this too, and leaving it to the frame would leave a stats
        // read between a registration and the first frame under-counting a pool
        // that is already partly full.
        self.stats.resident = self.count_resident();
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
        //
        // **Per arm** since wave IASSET2: a slot index is arm-local, so one set
        // would have arm 1's slot 3 protecting arm 0's slot 3.
        let mut protected: Vec<BTreeSet<u32>> = vec![BTreeSet::new(); self.arms.len()];
        let mut upload_bytes = 0u64;
        for (handle, tile) in std::mem::take(&mut self.pending_roots) {
            let t = &self.textures[handle.index()];
            let arm = t.arm as usize;
            let Some(idx) = t.desc.entry_index(tile) else {
                continue;
            };
            let Some(slot) = t.resident[idx as usize] else {
                continue;
            };
            protected[arm].insert(slot);
            self.touch(arm, slot);
            txn.admits.push(VtAdmit {
                slot,
                texture: handle,
                tile,
            });
            self.stats.admits += 1;
            upload_bytes += self.arms[arm].page_bytes;
        }
        for t in &mut self.textures {
            t.warm = true;
        }

        // ── 2. normalize the want set ── (P28.3: `inf_stream::normalize`)
        //
        // **Lane first, payload order as the tie-break** (P26.4) — the exact
        // arrangement P26.2's crate docs promised for the day feedback arrived,
        // and since P28.3 it is one function shared with `inf-vsm` rather than
        // two copies of the same two passes. Payload order rather than
        // `TileCoord`'s derived `Ord`, which sorts (mip, x, y) while the tile
        // directory is (mip, y, x) and would walk the file sideways: the P26.1
        // audit's finding, used rather than merely recorded.
        let mut sorted: Vec<VtWant> = inf_stream::normalize(
            wants,
            |w| (w.texture.0, w.tile.payload_order()),
            |w| w.priority,
        );
        // An unknown handle is counted here rather than filtered in silence
        // (P26.4): see `VtTransaction::unknown_texture`. After the dedup, so one
        // stale tile asked for twice is one number.
        sorted.retain(|w| {
            let known = w.texture.index() < self.textures.len();
            if !known {
                txn.unknown_texture += 1;
            }
            known
        });
        self.stats.unknown_texture += u64::from(txn.unknown_texture);
        self.stats.wanted = sorted.len();

        // ── 3. drop the wants this registry cannot hold, and count them ──
        //
        // Before the walk rather than inside it, because an out-of-range tile is
        // a *want set computed against the wrong extent* and the arbiter's job
        // starts at addresses that exist. `retain` keeps the lane-major order.
        //
        // **Split by arm** since wave IASSET2, keeping the lane-major order
        // inside each: a BC5 page cannot go in a BC1 atlas, so the arbiter runs
        // once per arm over that arm's own wants. It is not merely a filter —
        // `admit_by_lane` stops offering slots the moment one acquisition fails
        // ("once acquisition fails nothing in this transaction frees a slot"),
        // and that reasoning is true *within* a pool of interchangeable slots
        // and false across two. One walk over both arms would let a full BC1
        // atlas defer every BC5 want in the same frame, with slots free.
        let mut lanes: Vec<Vec<(VtPriority, (u32, TileCoord))>> = vec![Vec::new(); self.arms.len()];
        for w in &sorted {
            let t = &self.textures[w.texture.index()];
            if t.desc.entry_index(w.tile).is_none() {
                txn.out_of_range += 1;
                continue;
            }
            lanes[t.arm as usize].push((w.priority, (w.texture.0, w.tile)));
        }

        // ── 4. THE ADMISSION WALK (P28.3) ──
        //
        // `inf_stream::admit_by_lane`, which protects and admits **one lane at a
        // time**. Before this batch the two steps were split the other way —
        // every resident want of any class protected, then every miss of any
        // class admitted — and the P28.2 audit named what that costs: *"a
        // refinement that got there first outranks a floor want that has
        // not"*, so a decoy feedback class could hold the pool while the
        // cluster pairing's floor tiles were deferred for ever. Now a floor
        // miss may take a resident refinement's slot, and a refinement may
        // never take a floor tile's.
        // **The per-frame upload budget** (IB-16). Bytes in, seats out, at this
        // pool's own page size — so the BC1 and RGBA8-transcode arms of the same
        // configuration get the same *bandwidth* rather than the same page count.
        // A want past it is deferred and re-offered next frame: late, never
        // never. `0` is unlimited, which is what every gate written before IB-16
        // configures and why none of their transactions move.
        //
        // **Arms are walked in index order** — the order the level's formats
        // were planned in — so a transaction is a pure function of the want set
        // and the arm plan, exactly as it was of the want set alone.
        let (mut deferred, mut throttled) = (0u32, 0u32);
        for (arm, arm_lanes) in lanes.iter().enumerate() {
            let a = &self.arms[arm];
            let budget =
                inf_stream::AdmitBudget::from_bytes(a.cfg.upload_budget_bytes, a.page_bytes);
            let page_bytes = a.page_bytes;
            let log = inf_stream::admit_by_lane(
                &mut ArmView(self, arm),
                arm_lanes,
                &mut protected[arm],
                budget,
            );
            for (slot, (t, tile)) in log.evicts {
                let texture = VtTextureHandle(t);
                dirty.insert(texture);
                txn.evicts.push(VtEvict {
                    slot,
                    texture,
                    tile,
                });
                self.stats.evicts += 1;
            }
            for (slot, (t, tile)) in log.admits {
                let texture = VtTextureHandle(t);
                dirty.insert(texture);
                txn.admits.push(VtAdmit {
                    slot,
                    texture,
                    tile,
                });
                self.stats.admits += 1;
                upload_bytes += page_bytes;
            }
            deferred += log.deferred;
            throttled += log.throttled;
        }

        txn.deferred = deferred;
        txn.throttled = throttled;
        txn.tables = dirty.into_iter().collect();
        self.stats.deferred = deferred;
        self.stats.budget_clamped = deferred > 0;
        self.stats.throttled = throttled;
        // Bytes, not `admits × page_bytes`: two arms' pages are two sizes, and
        // the number this reports is what the frame actually uploaded.
        self.stats.upload_bytes = upload_bytes;
        // A RUN, not a total: a burst that drains resets it, and only demand that
        // never drains crosses the advisory's threshold.
        self.throttled_run = match throttled {
            0 => 0,
            _ => self.throttled_run.saturating_add(1),
        };
        self.stats.throttled_frames = self.throttled_run;
        self.stats.resident = self.count_resident();
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

    /// What a slot of the **first** arm holds, if anything.
    pub fn slot_occupant(&self, slot: u32) -> Option<(VtTextureHandle, TileCoord)> {
        self.slot_occupant_in(0, slot)
    }

    /// What a slot of arm `arm` holds, if anything.
    pub fn slot_occupant_in(&self, arm: usize, slot: u32) -> Option<(VtTextureHandle, TileCoord)> {
        self.arms
            .get(arm)?
            .slots
            .get(slot as usize)?
            .occupant
            .map(|(t, c)| (VtTextureHandle(t), c))
    }

    /// Whether a slot of the **first** arm holds a pinned root.
    pub fn slot_is_root(&self, slot: u32) -> bool {
        self.slot_is_root_in(0, slot)
    }

    /// Whether a slot of arm `arm` holds a pinned root.
    pub fn slot_is_root_in(&self, arm: usize, slot: u32) -> bool {
        self.arms
            .get(arm)
            .and_then(|a| a.slots.get(slot as usize))
            .is_some_and(|s| s.pinned)
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

    /// One texture's block generation — moves when a page of it is seated or
    /// unseated. **A stamp, not a measurement.**
    ///
    /// It does **not** move on a relayout: a registration rewrites every block in
    /// the image and only [`layout_generation`](Self::layout_generation) records
    /// that, which is the pair a cache has to watch — the layout stamp says "your
    /// buffer is the wrong shape", this one says "this block's contents moved".
    ///
    /// **Ships before its caller.** The mirror writes exactly the blocks a
    /// transaction names, so it has no need of a generation today; the caller is
    /// P26.4, where several transactions may be folded before one upload and
    /// "which blocks actually moved" stops being the transaction's own list.
    /// Pinned by [`tests::a_block_generation_moves_only_when_its_pages_do`] until
    /// then.
    pub fn generation(&self, tex: VtTextureHandle) -> Option<u64> {
        self.textures.get(tex.index()).map(|t| t.generation)
    }

    // ── internals ───────────────────────────────────────────────────────────

    /// Slots holding a page, across every arm.
    fn count_resident(&self) -> u32 {
        self.arms
            .iter()
            .map(|a| a.slots.iter().filter(|s| s.occupant.is_some()).count() as u32)
            .sum()
    }

    fn relayout(&mut self) {
        let descs: Vec<VtTextureDesc> = self.textures.iter().map(|t| t.desc.clone()).collect();
        // Each texture's block carries the geometry of ITS arm — see
        // `crate::table`'s module docs for why that moved out of the pool header.
        let rows: Vec<TableTexture<'_>> = self
            .textures
            .iter()
            .zip(&descs)
            .map(|(t, desc)| TableTexture {
                desc,
                slots_x: self.arms[t.arm as usize].geometry.slots_x,
                pool: t.arm,
            })
            .collect();
        self.table = TableImage::layout(
            &rows,
            self.arms.len() as u32,
            self.stored_tile_size,
            self.trilinear,
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

    fn touch(&mut self, arm: usize, slot: u32) {
        if let Some(s) = self
            .arms
            .get_mut(arm)
            .and_then(|a| a.slots.get_mut(slot as usize))
        {
            s.stamp = next_stamp();
        }
    }

    /// Put `tile` of texture `t` in `slot` **of `t`'s own arm** and re-point the
    /// fallbacks beneath it.
    fn seat(&mut self, slot: u32, t: u32, tile: TileCoord, pinned: bool) {
        let ti = t as usize;
        let arm = self.textures[ti].arm as usize;
        self.arms[arm].slots[slot as usize] = Slot {
            occupant: Some((t, tile)),
            stamp: next_stamp(),
            pinned,
        };
        let idx = self.textures[ti]
            .desc
            .entry_index(tile)
            .expect("seating a tile that exists");
        self.textures[ti].resident[idx as usize] = Some(slot);
        self.propagate(ti, tile, pack_entry(slot, tile.mip));
        self.textures[ti].generation = next_stamp();
        if pinned {
            self.arms[arm].roots += 1;
            self.stats.roots += 1;
        }
    }

    /// The lowest free slot **of arm `arm`**, or its least-recently-stamped
    /// evictable one.
    ///
    /// Never returns a pinned root and never returns a `protected` slot (one this
    /// transaction has already admitted or touched), so **a transaction can never
    /// evict what it just brought in**. The search never leaves the arm: a slot
    /// of another arm is a page of another *size*, and handing one back would put
    /// a BC5 tile in a BC1 atlas.
    fn acquire_slot(
        &mut self,
        arm: usize,
        protected: &BTreeSet<u32>,
    ) -> Option<(u32, Option<VtEvict>)> {
        let a = self.arms.get_mut(arm)?;
        if let Some(&slot) = a.free.iter().next() {
            a.free.remove(&slot);
            return Some((slot, None));
        }
        let victim = lru_victim(a.slots.iter().enumerate().filter_map(|(i, s)| {
            let slot = i as u32;
            (s.occupant.is_some() && !s.pinned && !protected.contains(&slot))
                .then_some((s.stamp, slot))
        }))?;
        let evict = self.unseat(arm, victim);
        self.arms[arm].free.remove(&victim);
        Some((victim, Some(evict)))
    }

    /// Empty `slot` of arm `arm`, re-pointing its tile (and the subtree that fell
    /// back through it) at its parent's entry.
    fn unseat(&mut self, arm: usize, slot: u32) -> VtEvict {
        let (t, tile) = self.arms[arm].slots[slot as usize]
            .occupant
            .take()
            .expect("a victim is occupied");
        debug_assert!(
            !self.arms[arm].slots[slot as usize].pinned,
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
        self.arms[arm].free.insert(slot);
        VtEvict {
            slot,
            texture: VtTextureHandle(t),
            tile,
        }
    }
}

/// **The arbiter's view of this pool** (P28.3): the four operations
/// `inf_stream::admit_by_lane` needs, and nothing else.
///
/// A private newtype rather than `impl SlotPool for VtResidency`, deliberately.
/// The trait is public, so implementing it on the residency itself would make
/// `seat` and `acquire` **public API** — a caller could seat a tile outside a
/// transaction, which is exactly the one-door law (P21) the private `uploads`
/// field and the private `PageUpload` constructor exist to keep in `inf-vgeom`.
/// A private wrapper gives the walk everything it needs and gives an external
/// caller nothing.
/// One **arm** of it (wave IASSET2): the walk runs once per arm, over that arm's
/// own wants and that arm's own slot indices, so a full BC1 atlas cannot defer a
/// BC5 want.
struct ArmView<'a>(&'a mut VtResidency, usize);

impl inf_stream::SlotPool for ArmView<'_> {
    /// `(texture index, tile)` — the residency's own address, unpacked from
    /// [`VtWant`] before the walk so the arbiter never sees a handle it would
    /// have to validate.
    type Key = (u32, TileCoord);

    fn resident_slot(&self, key: &Self::Key) -> Option<u32> {
        let t = self.0.textures.get(key.0 as usize)?;
        // A key from another arm is not resident *here*, whatever slot it holds
        // there — the walk is only ever handed this arm's own wants, and this is
        // the arithmetic that says so rather than a comment claiming it.
        if t.arm as usize != self.1 {
            return None;
        }
        t.resident[t.desc.entry_index(key.1)? as usize]
    }

    fn touch(&mut self, slot: u32) {
        let arm = self.1;
        self.0.touch(arm, slot);
    }

    fn acquire(&mut self, protected: &BTreeSet<u32>) -> Option<inf_stream::Acquired<Self::Key>> {
        let arm = self.1;
        let (slot, evicted) = self.0.acquire_slot(arm, protected)?;
        Some(inf_stream::Acquired {
            slot,
            evicted: evicted.map(|e| (e.texture.0, e.tile)),
        })
    }

    fn seat(&mut self, slot: u32, key: Self::Key) {
        // Never pinned: a root is seated at registration and is not reachable
        // from a want, which is what makes `slot_is_root` a total answer.
        self.0.seat(slot, key.0, key.1, false);
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
            trilinear: false,
            // Unthrottled: these arms measure the residency RULE, and a throttle
            // would make every one of them a statement about flow control.
            upload_budget_bytes: 0,
        });
        r
    }

    /// **The measurement that made wave IASSET2 the shape it is**, run as an
    /// arm: a level of BC1 albedos and BC5 normals pages at BC rates in two
    /// atlases, where one atlas would have had to be RGBA8 and hold 8× fewer.
    ///
    /// `inf_material::ground`'s module doc wrote the number down when it chose
    /// to author every ground map — normals included — as BC1 rather than take
    /// the demotion: *"2 670 pages of camera-driven refinement against 289, a
    /// 9.2× cut in what can be resident at once"*. The arithmetic here is that
    /// sentence, held against the arms it now gets instead.
    #[test]
    fn a_mixed_level_pages_at_bc_rates_instead_of_demoting_to_rgba8() {
        const BUDGET: u64 = crate::DEFAULT_VT_BUDGET_BYTES;
        // What the level's content weighs: four BC1 albedos to one BC5 normal
        // set, the ground library's own ratio of tiles.
        let weights = [(PageFormat::Bc1, 4_000u64), (PageFormat::Bc5, 1_000)];
        let arms = crate::split_pool_budget(
            VtPoolConfig {
                budget_bytes: BUDGET,
                ..Default::default()
            },
            &weights,
        );
        let (r, advisories) = VtResidency::new_multi(&arms);
        assert!(advisories.is_empty(), "{advisories:?}");
        assert_eq!(r.arm_count(), 2);

        let split_slots =
            r.geometry_of(0).unwrap().slot_count() + r.geometry_of(1).unwrap().slot_count();
        // The one-pool answer this replaces: mixed formats fell back to RGBA8.
        let demoted = (BUDGET / PageFormat::Rgba8.page_bytes(136)) as u32;
        assert_eq!(demoted, 340, "the RGBA8 page count the demotion bought");
        assert!(
            split_slots >= demoted * 5,
            "two arms hold {split_slots} pages against the demotion's {demoted}"
        );
        // …and the VRAM is the SAME budget, not two of them. This is the half a
        // page-count comparison cannot see.
        assert!(
            r.capacity_bytes() <= BUDGET,
            "{} B of atlas for a {BUDGET} B budget",
            r.capacity_bytes()
        );
    }

    /// **An arm is a pool, and a full one does not defer another's wants.**
    ///
    /// `admit_by_lane` gives up on the first failed acquisition — correct over a
    /// flat array of interchangeable slots, and false across two arrays. So the
    /// walk runs once per arm; this is the arm that fails if it is ever folded
    /// back into one.
    #[test]
    fn a_full_arm_does_not_defer_another_arms_wants() {
        let page = PageFormat::Bc1.page_bytes(136);
        let base = VtPoolConfig {
            stored_tile_size: 136,
            max_texture_dim: 8192,
            trilinear: false,
            upload_budget_bytes: 0,
            ..Default::default()
        };
        // Arm 0: two slots (one root + one cache). Arm 1: eight.
        let (mut r, _) = VtResidency::new_multi(&[
            VtPoolConfig {
                format: PageFormat::Bc1,
                budget_bytes: page * 2,
                ..base
            },
            VtPoolConfig {
                format: PageFormat::Bc5,
                budget_bytes: PageFormat::Bc5.page_bytes(136) * 8,
                ..base
            },
        ]);
        let tight = r
            .register_texture_in(full_pyramid(1024, 1024, 128, 4, true), 0)
            .expect("one root fits two slots");
        let roomy = r
            .register_texture_in(full_pyramid(512, 512, 128, 4, false), 1)
            .expect("one root fits eight slots");
        assert_eq!(r.apply_wants(&[]).admits.len(), 2, "one root per arm");

        // Three tiles from each. Arm 0 can seat exactly one; arm 1 all three.
        let mut wants = Vec::new();
        for x in 0..3u32 {
            wants.push(VtWant::new(tight, TileCoord::new(0, x, 0)));
            wants.push(VtWant::new(roomy, TileCoord::new(0, x, 0)));
        }
        let txn = r.apply_wants(&wants);
        assert_eq!(txn.out_of_range, 0, "the fixture asks for real tiles");
        for x in 0..3u32 {
            assert!(
                r.is_resident(roomy, TileCoord::new(0, x, 0)),
                "arm 1 tile {x} was deferred by arm 0 running out: {}",
                txn.trace()
            );
        }
        assert_eq!(txn.deferred, 2, "arm 0's two extra tiles: {}", txn.trace());
        // ANTI-VACUITY: arm 0 really is out of slots, so the deferral above is a
        // measurement of the arms and not of a pool that fitted everything.
        assert_eq!(r.geometry_of(0).unwrap().slot_count(), 2);
    }

    /// A registration naming an arm this residency does not have is refused **by
    /// name** — never folded into arm 0, which would put a page of one format in
    /// an atlas of another: the right length, the wrong texels, no error.
    #[test]
    fn a_registration_into_an_absent_arm_is_refused_by_name() {
        let mut r = pool(16);
        assert_eq!(r.arm_count(), 1);
        let err = r
            .register_texture_in(full_pyramid(320, 192, 128, 4, true), 1)
            .expect_err("there is no arm 1");
        assert_eq!(err, VtError::NoSuchArm { arm: 1, arms: 1 });
        assert!(err.to_string().contains("arm 1"), "{err}");
        assert_eq!(r.texture_count(), 0, "the refusal is total");
    }

    /// `floor_bytes` prices every arm's roots in **its own** page size — the
    /// quantity a budget reporter wants, and the one `roots × page_bytes()` gets
    /// wrong the moment there are two arms.
    #[test]
    fn the_floor_is_priced_in_each_arms_own_pages() {
        let base = VtPoolConfig {
            upload_budget_bytes: 0,
            ..Default::default()
        };
        let (mut r, _) = VtResidency::new_multi(&[
            VtPoolConfig {
                format: PageFormat::Bc1,
                budget_bytes: PageFormat::Bc1.page_bytes(136) * 8,
                ..base
            },
            VtPoolConfig {
                format: PageFormat::Rgba8,
                budget_bytes: PageFormat::Rgba8.page_bytes(136) * 8,
                ..base
            },
        ]);
        r.register_texture_in(full_pyramid(512, 512, 128, 4, true), 0)
            .unwrap();
        r.register_texture_in(full_pyramid(512, 512, 128, 4, false), 1)
            .unwrap();
        assert_eq!(r.stats().roots, 2);
        assert_eq!(
            r.floor_bytes(),
            PageFormat::Bc1.page_bytes(136) + PageFormat::Rgba8.page_bytes(136)
        );
        // …and NOT the number the old expression gives, which is what makes this
        // an assertion about the arms rather than about two ways to spell one sum.
        assert_ne!(r.floor_bytes(), 2 * r.page_bytes());
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
            reconstruct_z: false,
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
            reconstruct_z: false,
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
        assert_eq!(txn.out_of_range, 2, "an unknown handle is its own counter");
        assert_eq!(
            txn.unknown_texture, 1,
            "a want naming a texture this residency does not have was dropped in \
             silence — the P26.3 remainder"
        );
        assert!(txn.admits.is_empty());
        // …and both reach the trace, which is the only place a caller watching a
        // transaction stream would ever see them.
        assert_eq!(txn.trace(), "oor 2\nunk 1\n");
        assert_eq!(r.stats().unknown_texture, 1, "cumulative, and it moved");
        assert!(
            r.stats().summary().contains("1 unknown-handle wants"),
            "{}",
            r.stats().summary()
        );

        // ANTI-VACUITY: the same want against a residency that HAS the handle is
        // not counted, so the number above measures the handle and not the call.
        let txn = r.apply_wants(&[VtWant::new(h, TileCoord::new(0, 0, 0))]);
        assert_eq!(txn.unknown_texture, 0);
        assert_eq!(txn.admits.len(), 1);
    }

    /// **Priority is the primary key and payload order the tie-break** — the
    /// arrangement P26.2's crate docs promised, and what makes the floor a floor.
    ///
    /// A pool of three cache slots is offered one floor want and three
    /// refinements, all of finer tiles that sort BEFORE the floor want in payload
    /// order (mip 0 precedes mip 1). Without the priority sort the floor tile is
    /// the one that gets deferred, which is precisely the regression this exists
    /// to catch: `want_floor` emits coarsest-first and `payload_order` is
    /// ascending in mip, so the two orders are *opposed*.
    #[test]
    fn the_floor_is_served_before_a_refinement_however_the_payload_sorts() {
        let mut r = pool(4);
        let h = r
            .register_texture(full_pyramid(512, 512, 128, 4, true))
            .expect("one root");
        // 4 slots: 1 pinned root + 3 cache slots.
        assert_eq!(r.apply_wants(&[]).admits.len(), 1);

        let floor = VtWant::new(h, TileCoord::new(1, 1, 1));
        let refines = [
            VtWant::refine(h, TileCoord::new(0, 0, 0)),
            VtWant::refine(h, TileCoord::new(0, 1, 0)),
            VtWant::refine(h, TileCoord::new(0, 2, 0)),
        ];
        let mut wants = refines.to_vec();
        wants.push(floor);
        let txn = r.apply_wants(&wants);

        assert_eq!(txn.deferred, 1, "four wants into three cache slots");
        assert_eq!(
            txn.admits[0].tile,
            floor.tile,
            "the floor tile was not admitted first: {}",
            txn.trace()
        );
        assert!(
            r.is_resident(h, floor.tile),
            "a refinement outranked the floor: {}",
            txn.trace()
        );
        // ANTI-VACUITY: the refinements really do sort before the floor tile in
        // payload order, so the assertion above is about the priority and not
        // about an order that happened to agree.
        assert!(
            refines[0].tile.payload_order() < floor.tile.payload_order(),
            "the fixture's refinements do not precede its floor want"
        );
    }

    /// **A floor want outranks a REFINEMENT THAT IS ALREADY RESIDENT** (P28.3,
    /// clause 3) — the priority-blind protection order the P28.2 audit named.
    ///
    /// The arm above proves a floor *miss* beats a refinement *miss*, which was
    /// true since P26.4. This is the case that was not: with every cache slot
    /// held by refinements the feedback keeps asking for, a floor tile that is
    /// not yet resident used to be deferred **for ever** — every slot was
    /// protected before any miss was offered one — and on the audit's fixture
    /// *"a decoy feedback class costs the pairing its finest page"*.
    ///
    /// It is the invariant's case, not a tuning one: a resident cluster page's
    /// tiles ride at [`VT_PRIORITY_FLOOR`], so before this fix a refinement
    /// could keep the pairing's tiles out of the atlas indefinitely and the
    /// page would be retracted on every frame.
    #[test]
    fn a_floor_want_outranks_a_refinement_that_is_already_resident() {
        let mut r = pool(4);
        let h = r
            .register_texture(full_pyramid(512, 512, 128, 4, true))
            .expect("one root");
        // 4 slots: 1 pinned root + 3 cache slots, all three filled by feedback.
        assert_eq!(r.apply_wants(&[]).admits.len(), 1);
        let refines = [
            TileCoord::new(0, 0, 0),
            TileCoord::new(0, 1, 0),
            TileCoord::new(0, 2, 0),
        ];
        let warm = r.apply_wants(&refines.map(|t| VtWant::refine(h, t)));
        assert_eq!(
            warm.admits.len(),
            3,
            "the cache is not full: {}",
            warm.trace()
        );

        // The floor now wants a tile that is not resident, and the feedback
        // asks for its same three — the steady state the defect hid in.
        let floor = TileCoord::new(1, 1, 1);
        let mut wants = vec![VtWant::new(h, floor)];
        wants.extend(refines.map(|t| VtWant::refine(h, t)));
        let txn = r.apply_wants(&wants);

        assert!(
            r.is_resident(h, floor),
            "a resident refinement outranked a floor want: {}",
            txn.trace()
        );
        assert_eq!(txn.admits.len(), 1, "{}", txn.trace());
        assert_eq!(
            txn.evicts.len(),
            1,
            "one refinement made room: {}",
            txn.trace()
        );
        assert_eq!(
            txn.evicts[0].tile,
            refines[0],
            "the least recently touched refinement is not the one that left: {}",
            txn.trace()
        );
        assert_eq!(txn.deferred, 1, "the displaced refinement is the deferral");
        // ANTI-VACUITY: the root is still pinned and still resident, so the
        // floor did not simply take the only free slot in the pool.
        assert!(r.slot_is_root(0) && r.stats().resident == 4);
    }

    /// One tile asked for by both the floor and the feedback is **one** want, at
    /// the floor's rank — otherwise a refinement's copy could be the one that
    /// survives dedup and the tile would be served late.
    #[test]
    fn a_tile_wanted_twice_keeps_the_stronger_rank() {
        let mut r = pool(8);
        let h = r
            .register_texture(full_pyramid(512, 512, 128, 4, true))
            .unwrap();
        r.apply_wants(&[]);
        let at = TileCoord::new(1, 0, 0);
        let txn = r.apply_wants(&[VtWant::refine(h, at), VtWant::new(h, at)]);
        assert_eq!(txn.admits.len(), 1, "one tile, one admit: {}", txn.trace());
        assert_eq!(r.stats().wanted, 1, "the duplicate survived normalization");
    }

    /// …and it is the FLOOR's copy that survives, which is a different claim and
    /// the one that matters (P26.4 audit).
    ///
    /// The arm above asserts the two wants become ONE want — true of *either*
    /// survivor, so it cannot see which. Measured: reversing the dedup's
    /// tie-break so the refinement's copy wins leaves it green, and every other
    /// arm in the workspace green, while the tile is served in the wrong class.
    ///
    /// The rank is only observable under budget pressure, so that is where this
    /// looks. The shared tile is at mip 1 and every competing refinement is at
    /// mip 0, and mip 0 sorts FIRST in payload order — so a shared tile demoted
    /// to the refinement class falls to the end of its class and is exactly the
    /// one that gets deferred.
    #[test]
    fn a_tile_wanted_twice_is_served_in_the_floors_class() {
        let mut r = pool(6);
        let h = r
            .register_texture(full_pyramid(1024, 1024, 128, 4, true))
            .expect("one root");
        // 6 slots: 1 pinned root + 5 cache slots.
        assert_eq!(r.apply_wants(&[]).admits.len(), 1);

        let shared = TileCoord::new(1, 0, 0);
        let mut wants = vec![VtWant::new(h, shared), VtWant::refine(h, shared)];
        let m0 = r.desc(h).expect("registered").mips[0];
        for y in 0..m0.tiles_y {
            for x in 0..m0.tiles_x {
                wants.push(VtWant::refine(h, TileCoord::new(0, x, y)));
            }
        }
        let txn = r.apply_wants(&wants);
        assert!(
            txn.deferred > 0,
            "the refinements fit the pool, so this arm cannot see a rank at all"
        );
        assert_eq!(
            txn.admits[0].tile,
            shared,
            "the shared tile was not admitted first: {}",
            txn.trace()
        );
        assert!(
            r.is_resident(h, shared),
            "a tile the floor asked for was kept at the feedback's rank and \
             deferred: {}",
            txn.trace()
        );
        // ANTI-VACUITY: the refinements really do precede it in payload order, so
        // the assertion above is about the rank and not about an order that
        // happened to agree.
        assert!(
            TileCoord::new(0, 0, 0).payload_order() < shared.payload_order(),
            "the fixture's refinements do not precede its shared tile"
        );
    }

    /// A block generation moves when a page of that texture is seated or
    /// unseated, and **not** when a frame merely touches it — otherwise a cache
    /// keyed on it re-uploads every block every frame and the stamp buys nothing.
    #[test]
    fn a_block_generation_moves_only_when_its_pages_do() {
        let mut r = pool(8);
        let a = r
            .register_texture(full_pyramid(320, 192, 128, 4, true))
            .unwrap();
        let b = r
            .register_texture(full_pyramid(256, 256, 128, 4, false))
            .unwrap();
        r.apply_wants(&[]);
        let (g0, h0) = (r.generation(a).unwrap(), r.generation(b).unwrap());
        assert_eq!(r.generation(VtTextureHandle(9)), None, "an unknown handle");

        // An admit into `a` moves a's block and leaves b's alone.
        let tile = TileCoord::new(0, 0, 0);
        let txn = r.apply_wants(&[VtWant::new(a, tile)]);
        assert_eq!(txn.admits.len(), 1);
        assert!(!txn.is_empty(), "a transaction that admits is not empty");
        assert_ne!(r.generation(a).unwrap(), g0);
        assert_eq!(r.generation(b).unwrap(), h0, "b's pages did not move");

        // Wanting what is already resident touches a slot and changes no block.
        let g1 = r.generation(a).unwrap();
        let txn = r.apply_wants(&[VtWant::new(a, tile)]);
        assert!(txn.is_empty(), "nothing moved: {}", txn.trace());
        assert_eq!(r.generation(a).unwrap(), g1, "a touch is not a change");
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
