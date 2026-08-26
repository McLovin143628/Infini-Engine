//! **Residency**: which virtual shadow pages are in the physical atlas right
//! now, which slot each one occupies, and what every *other* page falls back to.
//!
//! # The state, in three sentences
//!
//! An atlas is a flat array of interchangeable slots. Each registered light has
//! one indirection entry per virtual page, holding the `(slot, level)` of the
//! **finest resident ancestor** of that page — itself, when it is resident.
//! Nothing is pinned, so that walk may terminate in [`VSM_ENTRY_NONE`], which is
//! a value and not a failure (see [`crate::table`]).
//!
//! # Fallbacks are maintained, not searched
//!
//! Exactly `inf_vt::residency`'s construction and for exactly its reason: the
//! naive implementation resolves a fallback by walking up the tree at sample
//! time, which puts one dependent read per level in the *receiver* and makes the
//! walk's length depend on residency. Instead every entry is kept **already
//! resolved** — an admit writes its own address into itself and into the
//! non-resident subtree beneath it; an evict writes its parent's entry into the
//! same set. Each is bounded by the affected subtree rather than by the tree, and
//! the receiver does one read.
//!
//! Maintained state is state that can drift, so [`VsmResidency::resolve`] is
//! checked against an independent brute-force walk after **every** transaction of
//! a seeded churn in `tests/residency_gate.rs` — the whole table, not a sample.
//!
//! # A transaction is a pure function of `(state, wants)`
//!
//! [`VsmResidency::apply_wants`] sorts its wants into **entry order** (the
//! `inf_vt` payload-order law, met in an address space that has no payload: the
//! canonical order here is the one the table and the marking bitmask share, so a
//! mask scan hands `apply_wants` a list that is already sorted the way it wants
//! it), evicts by least-recently-stamped with a total tie-break, and allocates
//! the lowest free slot. Nothing consults a clock, a map iteration order, or a
//! thread, and there is no parallelism here at all — this is bookkeeping, not
//! compute (the P22.4 law). Two runs of one want sequence therefore produce
//! byte-identical [`VsmTransaction::trace`] output.
//!
//! # A stamp is not a measurement
//!
//! Slot stamps come from a **process-global** monotone counter, for the reason
//! `inf_vgeom::stream`, `inf_terrain` and `inf_vt` all give: a GPU mirror caches
//! "the block I last uploaded for light L was generation N", and a per-residency
//! counter restarting at 1 would let a freshly-built residency mint a generation
//! a stale cache already holds. The consequence is the one those crates carry —
//! **a stamp must never enter a trace, a golden or a comparison between runs**;
//! [`VsmTransaction`] contains none.

use std::collections::{BTreeMap, BTreeSet};

use crate::address::{VsmDescError, VsmLightDesc, VsmPage, VsmTreeKind};
use crate::atlas::{plan_atlas, VsmAdvisory, VsmAtlasConfig, VsmAtlasGeometry};
use crate::table::{pack_entry, unpack_entry, TableImage, MAX_SLOT_INDEX, VSM_ENTRY_NONE};

/// Source of residency stamps: **`inf_stream::next_stamp`**, the one domain
/// (P28.3, clause 1).
///
/// This file used to open with a counter of its own and a comment saying *"its
/// own domain, for now — P28.3 is where the four counters in this tree become
/// one"*. Three of the four merged (the terrain and voxel counters version
/// *content* rather than recency and deliberately did not — see
/// `inf_stream::stamp`), and the merge is what makes a shadow page's recency
/// comparable with a cluster page's.
use inf_stream::next_stamp;

/// A registered shadow-casting light. Index into the residency's light list, and
/// the id a receiver uses to find the light's block in the indirection table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VsmLightHandle(pub u32);

impl VsmLightHandle {
    #[inline]
    pub fn index(&self) -> usize {
        self.0 as usize
    }
}

/// How badly a page is wanted — **the primary sort key**, lower first.
///
/// **Since P28.3 it is `inf_stream::Lane`**, the one lane vocabulary this crate
/// and `inf-vt` now share, ordered by the one admission walk.
pub type VsmPriority = inf_stream::Lane;

/// **The depth buffer proved a visible pixel needs this page.** The only rank
/// with a producer in P27.1, and the one residency may never drop below while
/// the budget allows.
pub const VSM_PRIORITY_MARKED: VsmPriority = inf_stream::LANE_FLOOR;

/// A page nothing has proved is needed **yet** — the dead-reckoning predictor's
/// rank, whose whole contract is that a speculative want *"enters at strictly
/// lower priority than the analytic floor and feedback"*.
///
/// **P28.4 gives it its producer**: `inf_render::speculative_shadow_wants` takes
/// the pages last frame's depth buffer proved were needed and moves them to
/// where the predicted camera puts the clipmap window. It carried the
/// `inf_vt::VtTransaction::unknown_texture` disclosure ("no producer in this
/// tree, stated rather than implied") through P27.1 and P28.3; the disclosure is
/// discharged rather than deleted, because a constant that has waited two phases
/// for a producer should say which batch supplied it.
///
/// **P28.3 made it `inf_stream::LANE_FEEDBACK` and P28.4 makes it
/// `LANE_PREDICT`.** P28.3's argument was that a shadow page has no feedback
/// refinement class, so the rank immediately below marked is the one a
/// speculation takes. That is still true of this consumer read alone, and it is
/// the wrong reading now that the lane has a producer: the invariant P28.4 has
/// to assert is one statement over all three consumers — *residency ⊇ floor ∪
/// feedback under full speculative pressure* — and a speculation sitting in the
/// feedback lane makes that statement mean something different here than it
/// means next door. One producer, one lane, one proof. Nothing observable moves:
/// no producer of this crate's has ever emitted `LANE_FEEDBACK`, so the walk
/// sees the same two ranks in the same order and no committed trace changes.
pub const VSM_PRIORITY_SPECULATIVE: VsmPriority = inf_stream::LANE_PREDICT;

/// The most pages one light's virtual space may hold.
///
/// A 16 k clipmap is 128 pages a side; twelve levels of it is 196 608 pages, a
/// 768 KiB table block and a 24 KiB mask. One million is generous headroom above
/// anything a light configuration reaches, and it is a **ceiling** rather than a
/// comfort: the table block and the `resident` vector are both linear in it, so
/// an unbounded descriptor is an unbounded allocation at registration.
pub const MAX_VSM_PAGES_PER_LIGHT: u32 = 1 << 20;

/// "This page, please." The whole input language of residency — deliberately no
/// camera, no light matrix and no budget policy: the caller computes the wants
/// (P27.1's marking pass does, off the depth buffer), this layer executes them
/// against a budget. The `inf_terrain::residency` split, kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmWant {
    pub light: VsmLightHandle,
    pub page: VsmPage,
    /// Lower is served first. See [`VsmPriority`].
    pub priority: VsmPriority,
}

impl VsmWant {
    /// A **marked** want ([`VSM_PRIORITY_MARKED`]) — what a decoded mask produces.
    pub const fn new(light: VsmLightHandle, page: VsmPage) -> Self {
        Self {
            light,
            page,
            priority: VSM_PRIORITY_MARKED,
        }
    }

    /// A **speculative** want ([`VSM_PRIORITY_SPECULATIVE`]).
    pub const fn speculate(light: VsmLightHandle, page: VsmPage) -> Self {
        Self {
            light,
            page,
            priority: VSM_PRIORITY_SPECULATIVE,
        }
    }

    /// The same want at an explicit rank.
    pub const fn with_priority(self, priority: VsmPriority) -> Self {
        Self { priority, ..self }
    }
}

/// A page entering the atlas: **this slot is yours, rasterize into it.**
///
/// Unlike `inf_vt::VtAdmit` there are no bytes to fetch — a shadow page's
/// content is produced by a raster pass (P27.2), not read from a container. The
/// admit is therefore the *allocation*, and the mirror's job is to publish the
/// slot rather than to upload anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmAdmit {
    pub slot: u32,
    pub light: VsmLightHandle,
    pub page: VsmPage,
}

/// A page leaving the atlas.
///
/// **There is nothing to do on the GPU for one.** The slot's texels simply
/// become unreachable — no live indirection entry names the slot any more — and
/// they are overwritten by the next admit that takes it. This record exists so a
/// mirror can account for the slot and a debug view can colour it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmEvict {
    pub slot: u32,
    pub light: VsmLightHandle,
    pub page: VsmPage,
}

/// Where a virtual page actually lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmResolved {
    pub slot: u32,
    /// The resident page whose texels serve this address — the finest resident
    /// ancestor. Equal to the queried address when that page is itself resident.
    pub page: VsmPage,
}

/// The complete difference between two residency states.
///
/// **Deterministic and stamp-free**, so two runs of one want sequence produce
/// byte-identical [`trace`](Self::trace) output.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct VsmTransaction {
    /// Pages that gained a slot, in entry order.
    pub admits: Vec<VsmAdmit>,
    /// Pages whose slots became unreachable, in the order they were taken.
    pub evicts: Vec<VsmEvict>,
    /// Lights whose indirection block changed, ascending.
    pub tables: Vec<VsmLightHandle>,
    /// Whether the table's *layout* moved (a light was registered), so the
    /// mirror must re-create and re-upload the whole buffer rather than patch it.
    pub layout_rebuilt: bool,
    /// Wants that could not be seated because every slot was taken by this
    /// transaction. Never a silent cap.
    pub deferred: u32,
    /// Wants naming a page outside their light's tree — counted rather than
    /// panicked on, because a want set is computed from a depth buffer.
    pub out_of_range: u32,
    /// Wants naming a light handle this residency does not have. A different
    /// defect from [`out_of_range`](Self::out_of_range): an out-of-range page is
    /// a mask decoded against the wrong extent, an unknown handle is a mask
    /// decoded against **a different registry** — which is what a mask read out
    /// of the ring after a light was removed is.
    pub unknown_light: u32,
}

impl VsmTransaction {
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
                "admit slot={} light={} face={} level={} x={} y={}\n",
                a.slot, a.light.0, a.page.face, a.page.level, a.page.x, a.page.y
            ));
        }
        for e in &self.evicts {
            s.push_str(&format!(
                "evict slot={} light={} face={} level={} x={} y={}\n",
                e.slot, e.light.0, e.page.face, e.page.level, e.page.x, e.page.y
            ));
        }
        for t in &self.tables {
            s.push_str(&format!("table light={}\n", t.0));
        }
        if self.deferred > 0 {
            s.push_str(&format!("defer {}\n", self.deferred));
        }
        if self.out_of_range > 0 {
            s.push_str(&format!("oor {}\n", self.out_of_range));
        }
        if self.unknown_light > 0 {
            s.push_str(&format!("unk {}\n", self.unknown_light));
        }
        s
    }
}

/// **What a clipmap scroll did to residency** (island wave VSM2) — the return of
/// [`VsmResidency::set_clip_origins`].
///
/// A clipmap level's page grid is a *window* on a fixed world lattice, so when
/// the window slides the world cell that was page `(x, y)` becomes page
/// `(x − Δx, y − Δy)`. Until this wave the pages kept their **labels** and
/// therefore changed cell: every slot in the atlas suddenly described somewhere
/// else, and the raster re-drew depth it already held. They keep their **cell**
/// now — the slot moves with the world, and only the row and column the window
/// newly exposes are missing.
///
/// The two counts are the engagement counters that make that falsifiable. A
/// re-seat that did nothing and a scroll that moved no page produce the same
/// residency; only [`carried`](Self::carried) tells them apart, and only
/// [`evicted`](Self::evicted) says the trailing band really left.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VsmScroll {
    /// Whether the origins moved at all. A no-op call recomputes nothing, so a
    /// host may call [`VsmResidency::set_clip_origins`] every frame.
    pub moved: bool,
    /// Resident pages re-seated onto their new label, **keeping their slot** and
    /// therefore keeping their texels.
    pub carried: u32,
    /// Resident pages whose world cell left the window: their slots are freed,
    /// which is what makes room for the row and column entering the other side.
    pub evicted: u32,
}

/// Counters for the debug/stats path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VsmStats {
    pub lights: usize,
    /// Slots the atlas has.
    pub slots: u32,
    /// Slots holding a page.
    pub resident: u32,
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
    /// Wants naming an unknown light handle since the residency was created.
    /// Cumulative, because the interesting reading is "has this ever happened".
    pub unknown_light: u64,
    /// Pages carried across a clipmap scroll — see [`VsmScroll::carried`].
    /// Cumulative.
    pub scroll_carried: u64,
    /// Pages a clipmap scroll dropped off the trailing edge — see
    /// [`VsmScroll::evicted`]. Cumulative.
    pub scroll_evicted: u64,
}

impl VsmStats {
    /// A one-line human summary — the shape `VtStats::summary` and
    /// `TerrainStreamStats::summary` already ship, so the streamers read alike
    /// in one log.
    pub fn summary(&self) -> String {
        format!(
            "vsm residency: {} lights, {}/{} pages resident, {} wanted / {} deferred, \
             {} admits / {} evicts{}",
            self.lights,
            self.resident,
            self.slots,
            self.wanted,
            self.deferred,
            self.admits,
            self.evicts,
            if self.budget_clamped {
                " [budget-clamped]"
            } else {
                ""
            }
        ) + &if self.unknown_light > 0 {
            format!(" [{} unknown-handle wants]", self.unknown_light)
        } else {
            String::new()
        }
    }
}

/// A registration this atlas cannot honour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VsmError {
    #[error("the descriptor is not a virtual page tree: {0}")]
    Desc(#[from] VsmDescError),
    #[error(
        "this light's page space is {pages} pages and the ceiling is {max}; the table block \
         and the residency vector are both linear in it, so an unbounded tree is an unbounded \
         allocation at registration. Use fewer levels or a smaller page grid"
    )]
    PageSpaceTooLarge { pages: u32, max: u32 },
}

/// One physical slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Slot {
    /// `(light index, page)` — `None` when free.
    occupant: Option<(u32, VsmPage)>,
    /// Last touched. **A stamp, not a measurement** (see
    /// [`inf_stream::next_stamp`], which `NEXT_VSM_STAMP` merged into at P28.3).
    stamp: u64,
}

/// One registered light's residency.
#[derive(Debug, Clone)]
struct Light {
    desc: VsmLightDesc,
    /// Exact-resident slot per virtual page, indexed by `desc.entry_index`.
    resident: Vec<Option<u32>>,
    /// Where each clipmap level's grid sits on the world lattice — see
    /// [`VsmLightDesc::clip_origin`]. Empty until a caller sets it, which is the
    /// exactly-concentric arrangement P27.1 shipped.
    clip_origins: Vec<(i64, i64)>,
    generation: u64,
}

/// The residency of one physical shadow-page atlas.
#[derive(Debug, Clone)]
pub struct VsmResidency {
    cfg: VsmAtlasConfig,
    geometry: VsmAtlasGeometry,
    page_bytes: u64,
    slots: Vec<Slot>,
    free: BTreeSet<u32>,
    lights: Vec<Light>,
    table: TableImage,
    layout_generation: u64,
    layout_dirty: bool,
    stats: VsmStats,
}

impl VsmResidency {
    /// An atlas sized from `cfg`, with the advisories planning it raised.
    ///
    /// The advisories are returned rather than logged, so a host decides whether
    /// they are a warning, a fatal, or a line in a report — `inf-vt`'s shape.
    pub fn new(cfg: VsmAtlasConfig) -> (Self, Vec<VsmAdvisory>) {
        let (geometry, advisories) = plan_atlas(cfg);
        let page_bytes = cfg.page_bytes();
        let n = geometry.slot_count();
        debug_assert!(n <= MAX_SLOT_INDEX + 1, "a slot index must fit 16 bits");
        let table = TableImage::layout(
            &[],
            cfg.page_size,
            cfg.border,
            geometry.slots_x,
            geometry.stored_page_size,
        );
        let this = Self {
            cfg,
            geometry,
            page_bytes,
            slots: vec![
                Slot {
                    occupant: None,
                    stamp: 0,
                };
                n as usize
            ],
            free: (0..n).collect(),
            lights: Vec::new(),
            table,
            layout_generation: next_stamp(),
            layout_dirty: false,
            stats: VsmStats {
                slots: n,
                ..Default::default()
            },
        };
        (this, advisories)
    }

    #[inline]
    pub fn config(&self) -> VsmAtlasConfig {
        self.cfg
    }
    #[inline]
    pub fn geometry(&self) -> VsmAtlasGeometry {
        self.geometry
    }
    /// Bytes one page occupies.
    #[inline]
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }
    /// The VRAM the atlas costs — `slots × page_bytes`, and **never** more than
    /// [`VsmAtlasConfig::budget_bytes`].
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
    pub fn stats(&self) -> &VsmStats {
        &self.stats
    }
    #[inline]
    pub fn light_count(&self) -> usize {
        self.lights.len()
    }
    /// One light's page space.
    pub fn desc(&self, light: VsmLightHandle) -> Option<&VsmLightDesc> {
        self.lights.get(light.index()).map(|l| &l.desc)
    }

    /// **Where each of this light's clipmap levels sits on the world lattice** —
    /// see [`VsmLightDesc::clip_origin`]. Empty means exactly concentric.
    pub fn clip_origins(&self, light: VsmLightHandle) -> &[(i64, i64)] {
        self.lights
            .get(light.index())
            .map_or(&[], |l| &l.clip_origins)
    }

    /// **Re-point one light's per-level clipmap grids, carrying the pages with
    /// the world** (P27.3, rewritten by island wave VSM2).
    ///
    /// P27.1 shipped one snapped centre for every level, which made the
    /// concentric `N/4 + x/2` parent rule exact and made a *coarse* level's grid
    /// shift on any camera motion — so nothing coarse could ever cache. Snapping
    /// each level to its own page stride fixes that and costs exactly this: the
    /// levels stop being concentric, so the fallback chain has to be recomputed
    /// against the new offsets ([`VsmLightDesc::parent_at`]).
    ///
    /// # THE SLOT BELONGS TO THE WORLD CELL, NOT TO THE LABEL
    ///
    /// P27.3 shipped this door with the sentence *"the pages keep their slots. A
    /// page whose world cell moved holds stale **content**, not a stale
    /// allocation"*, and §5 of `docs/memos/p27-3-page-cache.md` carried the price
    /// as an honest bound: *"there is no clipmap scroll … every resident page of
    /// that level names a different world cell and is re-rasterized"*. Island
    /// wave I7b put the number on it — on the lit island **532 of 933 dirty pages
    /// a frame were "moved"**, about 98 % of them holding depth the atlas already
    /// had, in a slot that now answered to a different label — and routed the fix
    /// by name. This is it.
    ///
    /// A window that slides by `Δ` re-labels page `(x, y)` to `(x − Δx, y − Δy)`:
    /// a **translation in page-index space and nothing else** — no matrix, no
    /// projection, no basis, and the same arithmetic
    /// `inf_render::scrolled_page` already predicts a want with. So the whole
    /// re-seat is a permutation of `resident` and of the slots' own occupant
    /// records, `O(resident slots)` rather than `O(pages)`, and the pages that
    /// fall off the trailing edge are the only ones that lose their allocation.
    ///
    /// The translation is a **bijection** on the surviving set, so the old entries
    /// are all cleared before any new one is written: a one-pass walk would
    /// otherwise clear a cell a page it had already carried was sitting in.
    ///
    /// # Only the levels that moved are recomputed
    ///
    /// A level's entries are a function of its own residency, its own origin, its
    /// parent's origin and its parent's already-final words. A level above the
    /// deepest one whose origin moved has all four unchanged, so its words are
    /// unchanged — and the fine levels, which are the ones that shift, are a
    /// small suffix of the walk. On the shipped eight-level ladder a camera at
    /// 0.9 m a frame moves level 0's window nine frames in ten and level 7's one
    /// in a hundred and forty.
    ///
    /// Returns the counts; a no-op call recomputes nothing, re-seats nothing and
    /// does not move the light's generation, so a host may call it every frame.
    pub fn set_clip_origins(&mut self, light: VsmLightHandle, origins: &[(i64, i64)]) -> VsmScroll {
        let Some(l) = self.lights.get_mut(light.index()) else {
            return VsmScroll::default();
        };
        if l.clip_origins == origins {
            return VsmScroll::default();
        }
        let was = std::mem::replace(&mut l.clip_origins, origins.to_vec());
        let li = light.index();
        let (carried, evicted) = self.reseat_world_cells(li, &was);
        let deepest = self.deepest_moved_level(li, &was);
        self.recompute_entries_to(li, deepest);
        self.lights[li].generation = next_stamp();
        self.stats.scroll_carried += u64::from(carried);
        self.stats.scroll_evicted += u64::from(evicted);
        self.stats.evicts += u64::from(evicted);
        self.stats.resident = self.stats.resident.saturating_sub(evicted);
        VsmScroll {
            moved: true,
            carried,
            evicted,
        }
    }

    /// **Register a shadow-casting light's page tree.**
    ///
    /// Unlike `inf_vt::VtResidency::register_texture` this seats **nothing** and
    /// can never be refused for want of budget: there is no mandatory floor (see
    /// [`crate::address`]), so a registration is a table layout and an empty
    /// residency. Every entry resolves to [`VSM_ENTRY_NONE`] until a marked page
    /// is admitted, and `NONE` is a value a receiver reads as *lit*.
    pub fn register_light(&mut self, desc: VsmLightDesc) -> Result<VsmLightHandle, VsmError> {
        desc.validate()?;
        let pages = desc.page_count();
        if pages > MAX_VSM_PAGES_PER_LIGHT {
            return Err(VsmError::PageSpaceTooLarge {
                pages,
                max: MAX_VSM_PAGES_PER_LIGHT,
            });
        }
        let handle = VsmLightHandle(self.lights.len() as u32);
        self.lights.push(Light {
            desc,
            resident: vec![None; pages as usize],
            clip_origins: Vec::new(),
            generation: next_stamp(),
        });
        // The directory sits before the blocks, so a new light moves every block
        // after it: the image is laid out again and refilled from the live
        // residency (a coarse-to-fine sweep, the only place entries are computed
        // from scratch).
        self.relayout();
        self.stats.lights = self.lights.len();
        Ok(handle)
    }

    /// **The sync point.** Advance residency one deterministic step toward what
    /// `wants` asks for, and hand back the work.
    ///
    /// Call once per frame. The result is a pure function of `(state, wants)`.
    pub fn apply_wants(&mut self, wants: &[VsmWant]) -> VsmTransaction {
        let mut txn = VsmTransaction {
            layout_rebuilt: std::mem::take(&mut self.layout_dirty),
            ..Default::default()
        };
        let mut dirty: BTreeSet<VsmLightHandle> = BTreeSet::new();

        // ── 1. normalize the want set ── (P28.3: `inf_stream::normalize`)
        //
        // **Lane first, entry order as the tie-break** — one function shared
        // with `inf-vt` since P28.3 rather than two copies of the same two
        // passes. Entry order is this address space's payload order: the order
        // the indirection table and the marking bitmask share, so a mask scan
        // hands the walk a list already sorted the way it wants it.
        let mut sorted: Vec<VsmWant> =
            inf_stream::normalize(wants, |w| (w.light.0, w.page.entry_order()), |w| w.priority);
        sorted.retain(|w| {
            let known = w.light.index() < self.lights.len();
            if !known {
                txn.unknown_light += 1;
            }
            known
        });
        self.stats.unknown_light += u64::from(txn.unknown_light);
        self.stats.wanted = sorted.len();

        // ── 2. drop the wants this registry cannot hold, and count them ──
        let mut lanes: Vec<(VsmPriority, (u32, VsmPage))> = Vec::with_capacity(sorted.len());
        for w in &sorted {
            let l = &self.lights[w.light.index()];
            if l.desc.entry_index(w.page).is_none() {
                txn.out_of_range += 1;
                continue;
            }
            lanes.push((w.priority, (w.light.0, w.page)));
        }

        // ── 3. THE ADMISSION WALK (P28.3) ──
        //
        // `inf_stream::admit_by_lane`, which protects and admits **one lane at
        // a time** — so a marked page that is not resident outranks a
        // speculation that is, which the previous protect-everything-then-admit
        // split could not express (the P28.2 audit's priority-blind protection
        // order, fixed for both consumers by one walk).
        let mut protected: BTreeSet<u32> = BTreeSet::new();
        let log = inf_stream::admit_by_lane(
            &mut PoolView(self),
            &lanes,
            &mut protected,
            // **VSM is not throttled** (IB-16). A shadow page the receiver is
            // sampling THIS frame that has not arrived is a lit surface with no
            // shadow on it — a visible wrong answer, not a blurry one. The VT
            // path can defer because a missing tile resolves through its own
            // coarser ancestor; a missing shadow page has no ancestor to fall
            // back to.
            inf_stream::AdmitBudget::UNLIMITED,
        );
        for (slot, (l, page)) in log.evicts {
            let light = VsmLightHandle(l);
            dirty.insert(light);
            txn.evicts.push(VsmEvict { slot, light, page });
            self.stats.evicts += 1;
        }
        for (slot, (l, page)) in log.admits {
            let light = VsmLightHandle(l);
            dirty.insert(light);
            txn.admits.push(VsmAdmit { slot, light, page });
            self.stats.admits += 1;
        }

        let deferred = log.deferred;
        txn.deferred = deferred;
        txn.tables = dirty.into_iter().collect();
        self.stats.deferred = deferred;
        self.stats.budget_clamped = deferred > 0;
        self.stats.resident = self.slots.iter().filter(|s| s.occupant.is_some()).count() as u32;
        txn
    }

    /// Where a virtual page actually lands — the finest resident ancestor, or
    /// `None` when nothing in its chain is resident.
    ///
    /// Reads the maintained table, exactly as the receiver will, and then
    /// re-derives the resolved page's address by walking the same chain.
    pub fn resolve(&self, light: VsmLightHandle, at: VsmPage) -> Option<VsmResolved> {
        let l = self.lights.get(light.index())?;
        let word = *self
            .table
            .words
            .get(self.table.entry_word(light.index(), &l.desc, at)?)?;
        let e = unpack_entry(word)?;
        Some(VsmResolved {
            slot: e.slot,
            page: l.desc.ancestor_at(&l.clip_origins, at, e.level)?,
        })
    }

    /// Whether `at` is itself resident (as opposed to resolving to an ancestor).
    pub fn is_resident(&self, light: VsmLightHandle, at: VsmPage) -> bool {
        self.lights.get(light.index()).is_some_and(|l| {
            l.desc
                .entry_index(at)
                .is_some_and(|i| l.resident[i as usize].is_some())
        })
    }

    /// What a slot holds, if anything.
    pub fn slot_occupant(&self, slot: u32) -> Option<(VsmLightHandle, VsmPage)> {
        self.slots
            .get(slot as usize)?
            .occupant
            .map(|(l, p)| (VsmLightHandle(l), p))
    }

    // ── the table, for the mirror ───────────────────────────────────────────

    /// The whole indirection image.
    #[inline]
    pub fn table_words(&self) -> &[u32] {
        &self.table.words
    }

    /// One light's block: `(word offset, words)`.
    pub fn table_block(&self, light: VsmLightHandle) -> Option<(usize, &[u32])> {
        let b = self.table.blocks.get(light.index())?;
        Some((b.base, &self.table.words[b.base..b.base + b.len]))
    }

    /// The layout generation — moves when a light is registered, so a mirror
    /// knows its buffer is the wrong shape. **A stamp, not a measurement.**
    #[inline]
    pub fn layout_generation(&self) -> u64 {
        self.layout_generation
    }

    /// One light's block generation — moves when a page of it is seated or
    /// unseated, and **not** on a relayout (that is what
    /// [`layout_generation`](Self::layout_generation) records) and **not** when a
    /// frame merely touches it. **A stamp, not a measurement.**
    pub fn generation(&self, light: VsmLightHandle) -> Option<u64> {
        self.lights.get(light.index()).map(|l| l.generation)
    }

    // ── internals ───────────────────────────────────────────────────────────

    fn relayout(&mut self) {
        let descs: Vec<VsmLightDesc> = self.lights.iter().map(|l| l.desc.clone()).collect();
        self.table = TableImage::layout(
            &descs,
            self.cfg.page_size,
            self.cfg.border,
            self.geometry.slots_x,
            self.geometry.stored_page_size,
        );
        for l in 0..self.lights.len() {
            self.recompute_entries(l);
        }
        self.layout_generation = next_stamp();
        self.layout_dirty = true;
    }

    /// **The deepest level whose window moved between `was` and the light's
    /// current origins** (island wave VSM2), or `None` when none did.
    ///
    /// Read through [`VsmLightDesc::clip_origin`] on both sides rather than off
    /// the two slices, so a table that is *shorter* than the ladder — the empty
    /// one a freshly registered light carries, which means "exactly concentric" —
    /// compares against the value it stands for instead of against a missing
    /// entry.
    fn deepest_moved_level(&self, l: usize, was: &[(i64, i64)]) -> Option<u32> {
        let light = &self.lights[l];
        (0..light.desc.level_count())
            .filter(|&lv| {
                light.desc.clip_origin(was, lv) != light.desc.clip_origin(&light.clip_origins, lv)
            })
            .next_back()
    }

    /// **Carry every resident clipmap page onto the label its world cell now
    /// wears** (island wave VSM2). Returns `(carried, evicted)`.
    ///
    /// See [`set_clip_origins`](Self::set_clip_origins) for the ruling. Only a
    /// clipmap has a window on the world: a quadtree's and a cube's levels are
    /// nested inside the light's own frustum, their origins mean nothing
    /// ([`VsmLightDesc::clip_origin`] says so), and a translation applied to them
    /// would move pages that had not moved.
    fn reseat_world_cells(&mut self, l: usize, was: &[(i64, i64)]) -> (u32, u32) {
        let desc = self.lights[l].desc.clone();
        if !matches!(desc.kind, VsmTreeKind::Clipmap) {
            return (0, 0);
        }
        let now = self.lights[l].clip_origins.clone();
        let deltas: Vec<(i64, i64)> = (0..desc.level_count())
            .map(|lv| {
                let a = desc.clip_origin(was, lv);
                let b = desc.clip_origin(&now, lv);
                (a.0 - b.0, a.1 - b.1)
            })
            .collect();

        // One pass over the SLOTS. An atlas has thousands of them and a clipmap
        // ladder has millions of addresses, which is `collect_pages`' argument for
        // walking slots and it is the same argument here.
        let mut moves: Vec<(u32, VsmPage, Option<VsmPage>)> =
            Vec::with_capacity(self.stats.resident as usize);
        for (slot, s) in self.slots.iter().enumerate() {
            let Some((owner, page)) = s.occupant else {
                continue;
            };
            if owner as usize != l {
                continue;
            }
            let (dx, dy) = deltas[page.level as usize];
            let g = desc.levels[page.level as usize];
            let (x, y) = (i64::from(page.x) + dx, i64::from(page.y) + dy);
            let inside =
                (0..i64::from(g.pages_x)).contains(&x) && (0..i64::from(g.pages_y)).contains(&y);
            let to = inside.then(|| VsmPage::new(page.face, page.level, x as u32, y as u32));
            moves.push((slot as u32, page, to));
        }

        // Clear first, seat second: the translation is a bijection on the pages
        // that survive it, but a page's destination may still hold a page that
        // has not been carried yet — and a one-pass walk would clear the entry it
        // had already written.
        for &(_, from, _) in &moves {
            let idx = desc.entry_index(from).expect("a seated page exists");
            self.lights[l].resident[idx as usize] = None;
        }
        let (mut carried, mut evicted) = (0u32, 0u32);
        for &(slot, _, to) in &moves {
            match to {
                Some(to) => {
                    let idx = desc.entry_index(to).expect("checked against the grid");
                    debug_assert!(
                        self.lights[l].resident[idx as usize].is_none(),
                        "two slots carried onto page {to:?} — the scroll is not a bijection"
                    );
                    self.lights[l].resident[idx as usize] = Some(slot);
                    self.slots[slot as usize].occupant = Some((l as u32, to));
                    carried += 1;
                }
                None => {
                    self.slots[slot as usize].occupant = None;
                    self.free.insert(slot);
                    evicted += 1;
                }
            }
        }
        (carried, evicted)
    }

    /// Fill one light's entries from scratch, coarsest level first, so a level's
    /// non-resident pages can simply copy their parent's already-final entry.
    ///
    /// The only from-scratch computation in the crate; everything else patches.
    fn recompute_entries(&mut self, l: usize) {
        self.recompute_entries_to(l, Some(self.lights[l].desc.coarsest_level()));
    }

    /// [`recompute_entries`](Self::recompute_entries) over levels `0..=deepest`
    /// only (island wave VSM2) — see [`set_clip_origins`](Self::set_clip_origins)
    /// for why that suffix is the whole of what a scroll can change. `None`
    /// recomputes nothing.
    fn recompute_entries_to(&mut self, l: usize, deepest: Option<u32>) {
        let Some(deepest) = deepest else {
            return;
        };
        let desc = self.lights[l].desc.clone();
        let origins = self.lights[l].clip_origins.clone();
        let deepest = deepest.min(desc.coarsest_level());
        // **The index arithmetic is hoisted, not re-derived per page** (island
        // wave VSM2). `entry_index` sums every coarser level's page count and
        // `entry_word` calls it a second time, so the shipped loop paid two
        // `O(levels)` walks for each of a ladder's pages — 32 768 of them on the
        // shipped 8 × 64² clipmap, on the frame budget of a pass that runs
        // whenever the camera crosses a metre. Within one `(face, level)` the
        // entry index is `first + y·pages_x + x`, which is exactly what the
        // table's own level record hands the receiver.
        let entries = self.table.blocks[l].entries;
        for face in 0..desc.faces() {
            for level in (0..=deepest).rev() {
                let g = desc.levels[level as usize];
                let first = desc.level_first_entry(face, level).expect("in tree") as usize;
                for y in 0..g.pages_y {
                    let row = first + (y * g.pages_x) as usize;
                    for x in 0..g.pages_x {
                        let idx = row + x as usize;
                        let word = match self.lights[l].resident[idx] {
                            Some(slot) => pack_entry(slot, level),
                            // A parent the offsets put outside the coarser grid
                            // has no entry to inherit, and `NONE` is the right
                            // answer rather than a panic: it reads as *lit*.
                            None => match desc
                                .parent_at(&origins, VsmPage::new(face, level, x, y))
                                .and_then(|p| desc.entry_index(p))
                            {
                                Some(p) => self.table.words[entries + p as usize],
                                None => VSM_ENTRY_NONE,
                            },
                        };
                        self.table.words[entries + idx] = word;
                    }
                }
            }
        }
    }

    /// Write `word` into `from`'s entry and into every **non-resident**
    /// descendant whose fallback ran through it. A resident descendant stops the
    /// walk: its own subtree resolves to it, not through `from`.
    fn propagate(&mut self, l: usize, from: VsmPage, word: u32) {
        let desc = self.lights[l].desc.clone();
        let origins = self.lights[l].clip_origins.clone();
        let mut stack = vec![from];
        while let Some(v) = stack.pop() {
            let w = self.table.entry_word(l, &desc, v).expect("in tree");
            self.table.words[w] = word;
            if v.level == 0 {
                continue;
            }
            for cy in desc.child_y_range_at(&origins, v.level, v.y) {
                for cx in desc.child_x_range_at(&origins, v.level, v.x) {
                    let c = VsmPage::new(v.face, v.level - 1, cx, cy);
                    let idx = desc.entry_index(c).expect("child in tree");
                    if self.lights[l].resident[idx as usize].is_none() {
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

    /// Put `page` of light `l` in `slot` and re-point the fallbacks beneath it.
    fn seat(&mut self, slot: u32, l: u32, page: VsmPage) {
        self.slots[slot as usize] = Slot {
            occupant: Some((l, page)),
            stamp: next_stamp(),
        };
        let li = l as usize;
        let idx = self.lights[li]
            .desc
            .entry_index(page)
            .expect("seating a page that exists");
        self.lights[li].resident[idx as usize] = Some(slot);
        self.propagate(li, page, pack_entry(slot, page.level));
        self.lights[li].generation = next_stamp();
    }

    /// The lowest free slot, or the least-recently-stamped evictable one.
    ///
    /// Never returns a `protected` slot (one this transaction has already
    /// admitted or touched), so **a transaction can never evict what it just
    /// brought in** — which is what makes the marked class a floor against the
    /// speculative one.
    fn acquire_slot(&mut self, protected: &BTreeSet<u32>) -> Option<(u32, Option<VsmEvict>)> {
        if let Some(&slot) = self.free.iter().next() {
            self.free.remove(&slot);
            return Some((slot, None));
        }
        let victim = lru_victim(self.slots.iter().enumerate().filter_map(|(i, s)| {
            let slot = i as u32;
            (s.occupant.is_some() && !protected.contains(&slot)).then_some((s.stamp, slot))
        }))?;
        let evict = self.unseat(victim);
        self.free.remove(&victim);
        Some((victim, Some(evict)))
    }

    /// Empty `slot`, re-pointing its page (and the subtree that fell back
    /// through it) at its parent's entry — which may itself be
    /// [`VSM_ENTRY_NONE`], and that is the correct answer rather than a failure.
    fn unseat(&mut self, slot: u32) -> VsmEvict {
        let (l, page) = self.slots[slot as usize]
            .occupant
            .take()
            .expect("a victim is occupied");
        let li = l as usize;
        let desc = self.lights[li].desc.clone();
        let origins = self.lights[li].clip_origins.clone();
        let idx = desc.entry_index(page).expect("seated page exists");
        self.lights[li].resident[idx as usize] = None;
        let word = match desc
            .parent_at(&origins, page)
            .and_then(|p| self.table.entry_word(li, &desc, p))
        {
            Some(w) => self.table.words[w],
            None => VSM_ENTRY_NONE,
        };
        self.propagate(li, page, word);
        self.lights[li].generation = next_stamp();
        self.free.insert(slot);
        VsmEvict {
            slot,
            light: VsmLightHandle(l),
            page,
        }
    }
}

/// **The arbiter's view of this atlas** (P28.3): the four operations
/// `inf_stream::admit_by_lane` needs, and nothing else.
///
/// A private newtype rather than `impl SlotPool for VsmResidency`, for
/// `inf-vt`'s reason: the trait is public, so implementing it on the residency
/// would make `seat` and `acquire` public API and let a caller seat a page
/// outside a transaction.
struct PoolView<'a>(&'a mut VsmResidency);

impl inf_stream::SlotPool for PoolView<'_> {
    /// `(light index, page)` — the residency's own address, unpacked from
    /// [`VsmWant`] before the walk.
    type Key = (u32, VsmPage);

    fn resident_slot(&self, key: &Self::Key) -> Option<u32> {
        let l = self.0.lights.get(key.0 as usize)?;
        l.resident[l.desc.entry_index(key.1)? as usize]
    }

    fn touch(&mut self, slot: u32) {
        self.0.touch(slot);
    }

    fn acquire(&mut self, protected: &BTreeSet<u32>) -> Option<inf_stream::Acquired<Self::Key>> {
        let (slot, evicted) = self.0.acquire_slot(protected)?;
        Some(inf_stream::Acquired {
            slot,
            evicted: evicted.map(|e| (e.light.0, e.page)),
        })
    }

    fn seat(&mut self, slot: u32, key: Self::Key) {
        self.0.seat(slot, key.0, key.1);
    }
}

/// The eviction rule: the smallest `(stamp, slot)`.
///
/// A free function so the tie-break can be exercised directly. Stamps come from
/// a global monotone counter, so two live slots cannot in fact hold the same one
/// — which is exactly why the tie-break must not be left implicit: ordering that
/// depended on the counter's uniqueness would stop being total the day the
/// counter is shared, wrapped or replaced (P28.3 proposes all three).
pub(crate) fn lru_victim(candidates: impl Iterator<Item = (u64, u32)>) -> Option<u32> {
    candidates
        .min_by_key(|&(stamp, slot)| (stamp, slot))
        .map(|(_, s)| s)
}

/// A `BTreeMap` of every light's resolved table, for tests and debug views.
///
/// Ships (rather than living in a test) because the mirror's readback arm
/// compares GPU bytes against exactly this.
pub fn resolved_table(res: &VsmResidency) -> BTreeMap<(u32, VsmPage), Option<VsmResolved>> {
    let mut out = BTreeMap::new();
    for l in 0..res.light_count() {
        let handle = VsmLightHandle(l as u32);
        let Some(desc) = res.desc(handle) else {
            continue;
        };
        for face in 0..desc.faces() {
            for level in 0..desc.level_count() {
                let g = desc.levels[level as usize];
                for y in 0..g.pages_y {
                    for x in 0..g.pages_x {
                        let at = VsmPage::new(face, level, x, y);
                        out.insert((l as u32, at), res.resolve(handle, at));
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
    use crate::atlas::VsmAtlasConfig;

    fn atlas(pages: u64) -> VsmResidency {
        let cfg = VsmAtlasConfig {
            budget_bytes: VsmAtlasConfig::default().page_bytes() * pages,
            ..Default::default()
        };
        let (r, _) = VsmResidency::new(cfg);
        r
    }

    #[test]
    fn the_lru_tie_break_is_the_slot_index() {
        // Equal stamps cannot arise from the global counter — which is why this
        // is exercised on the rule directly rather than through the atlas.
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

    /// **Registration seats nothing and every page reads as NONE** — the
    /// deliberate difference from `inf-vt`, where registration pins a floor.
    #[test]
    fn a_registered_light_starts_with_no_shadow_data_at_all() {
        let mut r = atlas(16);
        let h = r
            .register_light(VsmLightDesc::clipmap(4, 8))
            .expect("no floor to fit");
        assert_eq!(r.stats().resident, 0, "registration seated a page");
        assert_eq!(r.light_count(), 1);
        assert_eq!(r.resolve(h, VsmPage::flat(0, 3, 3)), None);
        assert_eq!(r.resolve(h, VsmPage::flat(3, 0, 0)), None);
        // …and the first transaction admits nothing on its own, unlike VT's,
        // which emits the root pages.
        let txn = r.apply_wants(&[]);
        assert!(txn.layout_rebuilt, "the layout moved at registration");
        assert!(txn.admits.is_empty());
        assert!(r.apply_wants(&[]).is_empty(), "and it settles");
    }

    /// A marked page is admitted, resolves to itself, and its whole subtree
    /// resolves **through** it — the maintained-fallback invariant, on the
    /// clipmap's concentric parent rule.
    #[test]
    fn an_admitted_page_serves_its_own_subtree_and_an_evict_hands_it_back() {
        let mut r = atlas(8);
        let h = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        r.apply_wants(&[]);

        let coarse = VsmPage::flat(1, 3, 3);
        let txn = r.apply_wants(&[VsmWant::new(h, coarse)]);
        assert_eq!(txn.admits.len(), 1);
        let slot = txn.admits[0].slot;
        assert_eq!(
            r.resolve(h, coarse),
            Some(VsmResolved { slot, page: coarse })
        );
        // Level 0's pages (2,2) and (3,3) hang under level 1's (3,3): the
        // concentric rule, `2/4 + x/2` = 2 + 1 = 3.
        for child in [VsmPage::flat(0, 2, 2), VsmPage::flat(0, 3, 3)] {
            assert_eq!(
                r.resolve(h, child),
                Some(VsmResolved { slot, page: coarse }),
                "{child:?} does not fall back through its parent"
            );
            assert!(!r.is_resident(h, child));
        }
        // A page under a DIFFERENT parent still has nothing.
        assert_eq!(r.resolve(h, VsmPage::flat(0, 0, 0)), None);

        // Admitting the child re-points its own subtree at itself…
        let txn = r.apply_wants(&[VsmWant::new(h, VsmPage::flat(0, 2, 2))]);
        let child_slot = txn.admits[0].slot;
        assert_ne!(child_slot, slot);
        assert_eq!(
            r.resolve(h, VsmPage::flat(0, 2, 2)).map(|x| x.slot),
            Some(child_slot)
        );
        assert_eq!(
            r.resolve(h, VsmPage::flat(0, 3, 3)).map(|x| x.slot),
            Some(slot),
            "a sibling was re-pointed at the child"
        );
    }

    /// Eviction hands a subtree back to the coarser page, and to **NONE** when
    /// there is no coarser page — the case `inf-vt` cannot reach.
    #[test]
    fn evicting_the_last_ancestor_returns_the_subtree_to_none() {
        let mut r = atlas(2);
        let h = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        r.apply_wants(&[]);
        let coarsest = VsmPage::flat(2, 3, 3);
        r.apply_wants(&[VsmWant::new(h, coarsest)]);
        assert!(r.resolve(h, VsmPage::flat(0, 2, 2)).is_some());

        // Two more pages into a two-slot atlas: the coarsest is the LRU victim.
        let txn = r.apply_wants(&[
            VsmWant::new(h, VsmPage::flat(0, 0, 0)),
            VsmWant::new(h, VsmPage::flat(0, 1, 0)),
        ]);
        assert_eq!(txn.evicts.len(), 1, "{}", txn.trace());
        assert_eq!(txn.evicts[0].page, coarsest);
        assert_eq!(
            r.resolve(h, VsmPage::flat(0, 2, 2)),
            None,
            "a page whose only ancestor was evicted must read as NO DATA, and a \
             receiver reads that as lit"
        );
        // …and the two that took its place resolve to themselves.
        assert!(r.is_resident(h, VsmPage::flat(0, 0, 0)));
        assert!(r.is_resident(h, VsmPage::flat(0, 1, 0)));
    }

    /// **A speculative want never takes a marked page's slot, however the entry
    /// order sorts.** The rank mechanism's only test, and the reason it ships
    /// before its producer.
    #[test]
    fn a_speculative_want_never_takes_a_marked_pages_slot() {
        let mut r = atlas(3);
        let h = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        r.apply_wants(&[]);

        // The marked want is at a COARSER level, so it sorts AFTER every
        // speculative one in entry order — which is precisely the regression
        // this exists to catch.
        let marked = VsmWant::new(h, VsmPage::flat(1, 4, 4));
        let speculative = [
            VsmWant::speculate(h, VsmPage::flat(0, 0, 0)),
            VsmWant::speculate(h, VsmPage::flat(0, 1, 0)),
            VsmWant::speculate(h, VsmPage::flat(0, 2, 0)),
        ];
        let mut wants = speculative.to_vec();
        wants.push(marked);
        let txn = r.apply_wants(&wants);

        assert_eq!(txn.deferred, 1, "four wants into three slots");
        assert_eq!(
            txn.admits[0].page,
            marked.page,
            "the marked page was not admitted first: {}",
            txn.trace()
        );
        assert!(
            r.is_resident(h, marked.page),
            "a speculation outranked the depth buffer: {}",
            txn.trace()
        );
        // ANTI-VACUITY: the speculations really do precede it in entry order, so
        // the assertion above is about the rank and not about an order that
        // happened to agree.
        assert!(
            speculative[0].page.entry_order() < marked.page.entry_order(),
            "the fixture's speculations do not precede its marked want"
        );
    }

    /// **A marked want outranks a speculation THAT IS ALREADY RESIDENT**
    /// (P28.3, clause 3).
    ///
    /// The arm above proves a marked *miss* beats a speculative *miss*, which
    /// held from P27.1. This is the case that did not: with every slot held by
    /// speculations the predictor keeps asking for, a page the depth buffer
    /// proved a visible pixel needs used to be deferred — every resident want
    /// of any class was protected before any miss of any class was offered a
    /// slot — and a deferred marked page reads as `VSM_ENTRY_NONE`, i.e. **lit**.
    /// A predictor that guessed wrong would have leaked light through the pages
    /// the depth buffer was certain about.
    ///
    /// It ships before its producer for the same reason the rank itself does.
    #[test]
    fn a_marked_want_outranks_a_speculation_that_is_already_resident() {
        let mut r = atlas(3);
        let h = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        r.apply_wants(&[]);
        let speculative = [
            VsmPage::flat(0, 0, 0),
            VsmPage::flat(0, 1, 0),
            VsmPage::flat(0, 2, 0),
        ];
        let warm = r.apply_wants(&speculative.map(|p| VsmWant::speculate(h, p)));
        assert_eq!(
            warm.admits.len(),
            3,
            "the atlas is not full: {}",
            warm.trace()
        );

        let marked = VsmPage::flat(1, 4, 4);
        let mut wants = vec![VsmWant::new(h, marked)];
        wants.extend(speculative.map(|p| VsmWant::speculate(h, p)));
        let txn = r.apply_wants(&wants);

        assert!(
            r.is_resident(h, marked),
            "a resident speculation outranked the depth buffer: {}",
            txn.trace()
        );
        assert_eq!(txn.admits.len(), 1, "{}", txn.trace());
        assert_eq!(txn.evicts.len(), 1, "{}", txn.trace());
        assert_eq!(
            txn.evicts[0].page,
            speculative[0],
            "the least recently touched speculation is not the one that left: {}",
            txn.trace()
        );
        assert_eq!(txn.deferred, 1);
    }

    /// One page wanted at two ranks is **one** want at the stronger rank.
    #[test]
    fn a_page_wanted_twice_keeps_the_stronger_rank() {
        let mut r = atlas(8);
        let h = r.register_light(VsmLightDesc::quadtree(4)).unwrap();
        r.apply_wants(&[]);
        let at = VsmPage::flat(1, 0, 0);
        let txn = r.apply_wants(&[VsmWant::speculate(h, at), VsmWant::new(h, at)]);
        assert_eq!(txn.admits.len(), 1, "one page, one admit: {}", txn.trace());
        assert_eq!(r.stats().wanted, 1, "the duplicate survived normalization");
    }

    #[test]
    fn a_want_outside_the_tree_is_counted_not_admitted() {
        let mut r = atlas(16);
        let h = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        r.apply_wants(&[]);
        let txn = r.apply_wants(&[
            VsmWant::new(h, VsmPage::flat(0, 99, 0)),
            VsmWant::new(h, VsmPage::flat(99, 0, 0)),
            VsmWant::new(h, VsmPage::new(3, 0, 0, 0)),
            VsmWant::new(VsmLightHandle(7), VsmPage::flat(0, 0, 0)),
        ]);
        assert_eq!(txn.out_of_range, 3, "an unknown handle is its own counter");
        assert_eq!(txn.unknown_light, 1);
        assert!(txn.admits.is_empty());
        assert_eq!(txn.trace(), "oor 3\nunk 1\n");
        assert_eq!(r.stats().unknown_light, 1, "cumulative, and it moved");
        assert!(
            r.stats().summary().contains("1 unknown-handle wants"),
            "{}",
            r.stats().summary()
        );

        // ANTI-VACUITY: the same want against a residency that HAS the handle is
        // not counted, so the number above measures the handle and not the call.
        let txn = r.apply_wants(&[VsmWant::new(h, VsmPage::flat(0, 0, 0))]);
        assert_eq!(txn.unknown_light, 0);
        assert_eq!(txn.admits.len(), 1);
    }

    /// A block generation moves when a page of that light is seated or unseated,
    /// and **not** when a frame merely touches it — otherwise a cache keyed on it
    /// re-uploads every block every frame and the stamp buys nothing.
    #[test]
    fn a_block_generation_moves_only_when_its_pages_do() {
        let mut r = atlas(8);
        let a = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        let b = r.register_light(VsmLightDesc::quadtree(3)).unwrap();
        r.apply_wants(&[]);
        let (g0, h0) = (r.generation(a).unwrap(), r.generation(b).unwrap());
        assert_eq!(r.generation(VsmLightHandle(9)), None, "an unknown handle");

        let page = VsmPage::flat(0, 0, 0);
        let txn = r.apply_wants(&[VsmWant::new(a, page)]);
        assert_eq!(txn.admits.len(), 1);
        assert!(!txn.is_empty(), "a transaction that admits is not empty");
        assert_ne!(r.generation(a).unwrap(), g0);
        assert_eq!(r.generation(b).unwrap(), h0, "b's pages did not move");

        let g1 = r.generation(a).unwrap();
        let txn = r.apply_wants(&[VsmWant::new(a, page)]);
        assert!(txn.is_empty(), "nothing moved: {}", txn.trace());
        assert_eq!(r.generation(a).unwrap(), g1, "a touch is not a change");
    }

    /// The stats line names every number it carries, and the clamp flag only
    /// when it is set.
    #[test]
    fn the_stats_line_says_what_it_counts() {
        let mut r = atlas(4);
        let h = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        r.apply_wants(&[]);
        let quiet = r.stats().summary();
        assert!(quiet.contains("vsm residency"), "{quiet}");
        assert!(quiet.contains("1 lights"), "{quiet}");
        assert!(quiet.contains("0/4 pages resident"), "{quiet}");
        assert!(!quiet.contains("budget-clamped"), "{quiet}");

        let wants: Vec<VsmWant> = (0..6)
            .map(|x| VsmWant::new(h, VsmPage::flat(0, x, 0)))
            .collect();
        r.apply_wants(&wants);
        let clamped = r.stats().summary();
        assert!(clamped.contains("4/4 pages resident"), "{clamped}");
        assert!(clamped.contains("6 wanted / 2 deferred"), "{clamped}");
        assert!(clamped.contains("[budget-clamped]"), "{clamped}");
        assert_eq!(r.capacity_bytes(), 4 * 64 * 1024);
        assert_eq!(r.resident_bytes(), 4 * 64 * 1024);
    }

    /// An atlas with **no slots at all** defers rather than panicking — the
    /// boundary a budget below one page produces, and the one a residency with no
    /// mandatory floor can actually reach at runtime.
    #[test]
    fn an_atlas_with_no_slots_defers_rather_than_panicking() {
        let (mut r, advisories) = VsmResidency::new(VsmAtlasConfig {
            budget_bytes: 1_024,
            ..Default::default()
        });
        assert_eq!(advisories.len(), 1, "the empty atlas is named");
        let h = r
            .register_light(VsmLightDesc::clipmap(3, 8))
            .expect("registration does not need a slot");
        let txn = r.apply_wants(&[VsmWant::new(h, VsmPage::flat(0, 0, 0))]);
        assert_eq!(txn.deferred, 1);
        assert!(txn.admits.is_empty());
        assert!(r.stats().budget_clamped);
        // …and every page still resolves — to nothing, which is the law.
        assert_eq!(r.resolve(h, VsmPage::flat(0, 0, 0)), None);
    }

    #[test]
    fn a_light_whose_page_space_is_absurd_is_refused_by_name() {
        let mut r = atlas(16);
        // 16 levels of a 1 024-page-a-side clipmap: 16.7 M pages.
        let err = r
            .register_light(VsmLightDesc::clipmap(16, 1_024))
            .expect_err("past the ceiling");
        assert_eq!(
            err,
            VsmError::PageSpaceTooLarge {
                pages: 16 * 1_024 * 1_024,
                max: MAX_VSM_PAGES_PER_LIGHT,
            }
        );
        assert_eq!(r.light_count(), 0, "the refusal is total");
        // …and a 16 k clipmap — the configuration the phase's "Done when" names —
        // fits with room, which is what makes the ceiling a bound and not a wall.
        r.register_light(VsmLightDesc::clipmap(12, 128))
            .expect("a 16k clipmap fits");
        const { assert!(12 * 128 * 128 < MAX_VSM_PAGES_PER_LIGHT) };

        let err = r
            .register_light(VsmLightDesc {
                kind: crate::VsmTreeKind::Clipmap,
                levels: vec![crate::VsmLevelDesc::square(6)],
            })
            .expect_err("6 is not a multiple of 4");
        assert!(matches!(err, VsmError::Desc(_)), "{err}");
    }

    /// **And the refusal survives an absurd PAGE GRID**, which is where the
    /// P27.1 audit found it did not.
    ///
    /// `clipmap_pages_per_side` is a host setting with no clamp of its own — the
    /// tier only ever `min`s it down — so it reaches `VsmLevelDesc::square`
    /// unbounded. Before the fix, `65 536²` wrapped to **zero** pages in release:
    /// `validate` passed, this ceiling compared 0 against a million and let the
    /// light in, and `relayout` then walked `65 536 × 65 536 × levels` addresses
    /// into a table image with no entry words at all. In debug it was an
    /// arithmetic panic. The refusal is what has to happen, on both profiles.
    #[test]
    fn an_absurd_page_grid_is_refused_rather_than_registered() {
        for grid in [65_536u32, 100_000, u32::MAX - 3] {
            let mut r = atlas(16);
            let err = r
                .register_light(VsmLightDesc::clipmap(8, grid))
                .expect_err("a grid that cannot be addressed must be refused");
            assert_eq!(
                err,
                VsmError::PageSpaceTooLarge {
                    pages: u32::MAX,
                    max: MAX_VSM_PAGES_PER_LIGHT,
                },
                "{grid} pages a side"
            );
            assert_eq!(r.light_count(), 0, "{grid}: the refusal is total");
            // …and the residency is still usable afterwards, which is the whole
            // point of a refusal rather than an abort.
            let h = r
                .register_light(VsmLightDesc::clipmap(4, 8))
                .expect("a legal light after a refused one");
            assert_eq!(
                r.apply_wants(&[VsmWant::new(h, VsmPage::flat(0, 0, 0))])
                    .admits
                    .len(),
                1
            );
        }
    }

    /// **Registering a light re-derives every other light's table from scratch,
    /// and that pass must run COARSEST-FIRST.**
    ///
    /// `recompute_entries` is the only from-scratch computation in the crate: a
    /// non-resident page copies its parent's already-final entry, so the sweep
    /// only works if the parent was computed first. Every other arm registers
    /// its lights before anything is resident, where a fresh image is all
    /// `NONE` and either order agrees — measured in the P27.1 audit, where
    /// reversing the sweep survived the whole crate.
    ///
    /// A host adds a shadow-casting light mid-session, so this is the product's
    /// path and not only a mutation's.
    #[test]
    fn a_light_registered_later_does_not_flatten_the_others_fallbacks() {
        let mut r = atlas(8);
        let a = r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        // Seat a page at the COARSEST level, so a whole subtree two levels deep
        // resolves through it — the depth a one-level-only sweep would lose.
        let coarse = VsmPage::flat(2, 3, 3);
        assert_eq!(r.apply_wants(&[VsmWant::new(a, coarse)]).admits.len(), 1);
        let deep = VsmPage::flat(0, 0, 0);
        let before = r.resolve(a, deep).expect("the fixture seated nothing");
        assert_eq!(
            before.page, coarse,
            "the fixture's page is not the ancestor"
        );

        let b = r.register_light(VsmLightDesc::quadtree(3)).unwrap();
        assert_eq!(
            r.resolve(a, deep),
            Some(before),
            "registering a second light flattened the first light's fallbacks — \
             the relayout's from-scratch sweep ran fine-to-coarse"
        );
        assert_eq!(r.resolve(a, coarse), Some(before), "the seated page moved");
        assert!(r.is_resident(a, coarse), "the residency itself moved");
        // …and the new light starts where every light starts: at nothing.
        assert_eq!(r.resolve(b, VsmPage::flat(0, 0, 0)), None);
        assert_eq!(r.stats().resident, 1, "the relayout seated something");
    }

    /// **Two runs of one want sequence produce one trace**, and the trace carries
    /// no stamp — the determinism claim, made executable at Ring 0.
    #[test]
    fn one_want_sequence_produces_one_trace() {
        let steps: Vec<Vec<VsmWant>> = (0..6)
            .map(|i| {
                let h = VsmLightHandle(0);
                (0..5)
                    .map(|k| VsmWant::new(h, VsmPage::flat(i % 3, (i + k) % 8, k % 8)))
                    .collect()
            })
            .collect();
        let run = || {
            let mut r = atlas(6);
            r.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
            steps
                .iter()
                .map(|w| r.apply_wants(w).trace())
                .collect::<Vec<_>>()
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "two runs of one want sequence diverged");
        // ANTI-VACUITY: the sequence actually did something, and it moved.
        assert!(a.iter().any(|t| t.contains("admit")));
        assert!(a.iter().any(|t| t.contains("evict")), "no pressure: {a:?}");
        assert!(
            a.iter().collect::<BTreeSet<_>>().len() > 1,
            "every step produced the same trace"
        );
    }
}
