//! The **meshlet streamer** (P18.2): which `.inf_vmesh` pages are in VRAM right
//! now, where in the shared pools they live, and how the GPU cut is clamped to
//! them.
//!
//! This is the GPU-free half — the residency state machine, the suballocator and
//! the upload staging — deliberately in Ring 0 beside the format it pages, exactly
//! as `inf_terrain::stream` sits beside `.inf_terrain`. `inf-render` owns the four
//! `wgpu` buffers and does nothing but execute a [`VgeomStreamPlan`]. Everything
//! interesting is therefore unit-testable **with no adapter**, on every CI leg.
//!
//! # Residency is a prefix, and that is the whole safety argument
//!
//! An asset's pages are ordered coarsest-first with **page 0 = every root**
//! (see [`crate::asset`]). Residency is always a prefix of that order:
//!
//! * page 0 is granted first and never evicted, so **every root-to-leaf path
//!   always has a resident meshlet** — a hole is not reachable, only softer
//!   detail;
//! * a non-root at level `L` has its parent at `L+1`, which is in an earlier page,
//!   so ancestor closure is structural rather than checked.
//!
//! The finest resident page's [`floor_lod`](crate::asset::VgeomPageEntry::floor_lod)
//! is the single scalar the GPU cut needs:
//! [`VgeomMesh::select_with_residency`](crate::VgeomMesh::select_with_residency)
//! is the CPU twin of the rule `vgeom_cull.wgsl` applies, and at full residency
//! (`floor_lod == 0`) both degenerate to the pre-P18.2 cut **exactly** — which is
//! what makes the goldens byte-identical.
//!
//! # Determinism: a bounded batch that is a pure function of the wants
//!
//! [`VgeomStreamer::plan`] is called at **one** point in the frame (the render
//! sync point, before culling) and is a pure function of
//! `(wants, current residency, budget)`. Nothing about it depends on IO timing:
//!
//! * the want per asset comes from the same per-instance screen-error scalar `t`
//!   the LOD cut uses, through [`ideal_page_count`] — no GPU readback, no frame
//!   history (see "why not cull feedback" below);
//! * grants are auctioned worst-error-first through a **total** order that breaks
//!   ties on the asset id, so the granted set is a function of the want set alone;
//! * fetches are `read_ref` slices of an mmap — synchronous by nature — and the
//!   *staging* (the only real work) runs through `inf_core::parallel_map_ref`, the
//!   deterministic in-order pure map, exactly as terrain's B2 load batch does;
//! * [`VgeomStreamBudget::max_loads_per_sync`] bounds the batch, and because the
//!   grant order is per-asset ascending, truncating it still leaves every asset
//!   holding a *prefix*.
//!
//! So a scripted flythrough reproduces a byte-identical resident-set trace — and
//! therefore a byte-identical pixel trace — run to run. What may *not* be assumed
//! is that a given frame is fully converged: an unconverged frame draws coarser
//! meshlets, never fewer.
//!
//! ## Why the wants are CPU-side and not cull feedback
//!
//! The cull compute already knows which pairs wanted finer detail than was
//! resident, and feeding that back would be the natural "requested-but-missing"
//! signal. It is deliberately **not** used to drive residency: a GPU readback is
//! one frame latent, so the want set would depend on the frame *history*, and two
//! runs differing by a single dropped frame would diverge — precisely what the
//! render-trace gates pin. The CPU want is derived from the same `t` the GPU cut
//! uses, so it is the identical quantity without the latency. The GPU signal
//! survives as an **audit counter only** (`VgeomAudit::clamped`, off by default),
//! so the instrument exists with no path from it to what gets loaded.

use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::asset::{
    MeshletRec, VgeomPageEntry, VgeomPageSections, VgeomSource, MESHLET_REC_LEN, VERTEX_REC_LEN,
};

/// Sentinel in a remap table: this meshlet's page is not resident.
pub const NOT_RESIDENT: u32 = u32::MAX;

/// Source of residency stamps, **process-global and monotone** — the same
/// construction, and the same reason, as terrain's per-tile version counter
/// (P16.3).
///
/// A renderer caches "the remap table I last uploaded for asset A was stamp N"
/// and re-uploads only when the stamp moves. A per-residency counter restarting
/// at 1 would let a *freshly created* residency mint a stamp a stale cache
/// already holds — after a budget change drops every asset, or after an asset is
/// re-registered against a different source — and the GPU would then read a remap
/// table describing pool slots that no longer exist. A global counter never
/// decreases, so a new residency's stamp is strictly greater than any stamp any
/// cache can be holding.
static NEXT_RESIDENCY_STAMP: AtomicU64 = AtomicU64::new(1);

/// The next residency stamp. Never 0, so 0 is safe as "nothing uploaded yet".
fn next_stamp() -> u64 {
    NEXT_RESIDENCY_STAMP.fetch_add(1, Ordering::Relaxed)
}

// ── the suballocator ────────────────────────────────────────────────────────

/// A half-open range of pool **units** (not bytes — a pool's unit is its element:
/// a vertex record, a meshlet record, a `u32`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct PoolBlock {
    pub offset: u64,
    pub len: u64,
}

impl PoolBlock {
    #[inline]
    pub fn end(&self) -> u64 {
        self.offset + self.len
    }

    /// Whether two blocks share a unit — the property the allocator's tests assert
    /// never holds between live blocks.
    #[inline]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.len > 0 && other.len > 0 && self.offset < other.end() && other.offset < self.end()
    }
}

/// A first-fit free-list suballocator over one growable pool.
///
/// **Deterministic by construction**: the free list is kept sorted by offset and
/// fully coalesced, first-fit picks the lowest-addressed hole that fits, and
/// growth doubles toward a fixed limit. Two runs that make the same alloc/free
/// calls produce the same offsets — which is what lets a resident-set trace be
/// compared byte for byte.
///
/// Growth only ever **appends**, so a caller that reallocates the backing GPU
/// buffer can copy the old contents to offset 0 and every live block keeps its
/// address.
#[derive(Debug, Clone)]
pub struct PoolAllocator {
    unit_bytes: u64,
    capacity: u64,
    used: u64,
    /// Sorted by offset, disjoint, coalesced.
    free: Vec<PoolBlock>,
}

impl PoolAllocator {
    /// The first growth step, in units (kept small so an idle renderer holds no
    /// meaningful VRAM; the pool doubles from here).
    const INITIAL_UNITS: u64 = 1024;

    /// A pool whose element is `unit_bytes` wide.
    ///
    /// It carries **no ceiling of its own**: growth is granted by [`VgeomPools`],
    /// which holds the one budget all four share. See there for why a per-pool
    /// share was wrong.
    pub fn new(unit_bytes: u64) -> Self {
        let unit_bytes = unit_bytes.max(1);
        Self {
            unit_bytes,
            capacity: 0,
            used: 0,
            free: Vec::new(),
        }
    }

    #[inline]
    pub fn unit_bytes(&self) -> u64 {
        self.unit_bytes
    }
    #[inline]
    pub fn capacity_units(&self) -> u64 {
        self.capacity
    }
    #[inline]
    pub fn capacity_bytes(&self) -> u64 {
        self.capacity * self.unit_bytes
    }
    #[inline]
    pub fn used_units(&self) -> u64 {
        self.used
    }
    #[inline]
    pub fn used_bytes(&self) -> u64 {
        self.used * self.unit_bytes
    }
    /// Free runs outstanding — the fragmentation the pool is carrying.
    #[inline]
    pub fn free_runs(&self) -> usize {
        self.free.len()
    }

    /// Allocate `units` from the pool's **existing** capacity, growing it only
    /// within `headroom_bytes` of extra VRAM. `None` means neither a hole nor the
    /// headroom could serve it — the caller degrades, never waits.
    ///
    /// `speculate` allows the growth step to overshoot the request (see
    /// [`grow`](Self::grow)). [`VgeomPools::alloc_page`] turns it **off** on its
    /// retry, which is what makes the floor guarantee exact.
    pub fn alloc(&mut self, units: u64, headroom_bytes: u64, speculate: bool) -> Option<PoolBlock> {
        if units == 0 {
            // A zero-length section still needs a well-defined address so callers
            // can treat every section uniformly.
            return Some(PoolBlock::default());
        }
        if let Some(b) = self.take_first_fit(units) {
            return Some(b);
        }
        if !self.grow(units, headroom_bytes / self.unit_bytes, speculate) {
            return None;
        }
        self.take_first_fit(units)
    }

    /// Return the pool's trailing free run to the budget, shrinking its capacity.
    /// Returns the units given back.
    ///
    /// Growth only ever **appends**, so whatever sits unallocated at the tail is
    /// exactly the speculation nothing has claimed: reclaiming it can never move
    /// or invalidate a live block, and it cannot touch interior fragmentation
    /// (which is real usage history, not slack).
    pub fn trim_tail(&mut self) -> u64 {
        let Some(last) = self.free.last().copied() else {
            return 0;
        };
        if last.end() != self.capacity {
            return 0;
        }
        self.free.pop();
        self.capacity -= last.len;
        last.len
    }

    fn take_first_fit(&mut self, units: u64) -> Option<PoolBlock> {
        let i = self.free.iter().position(|b| b.len >= units)?;
        let hole = self.free[i];
        if hole.len == units {
            self.free.remove(i);
        } else {
            self.free[i] = PoolBlock {
                offset: hole.offset + units,
                len: hole.len - units,
            };
        }
        self.used += units;
        Some(PoolBlock {
            offset: hole.offset,
            len: units,
        })
    }

    /// Extend the pool's tail so at least `need` more units are contiguous, taking
    /// at most `headroom` units of new capacity. Returns whether it moved.
    ///
    /// When `speculate`, growth **doubles** (floored at [`INITIAL_UNITS`], and only
    /// while that step fits in half the headroom), which amortizes the
    /// reallocate-and-restage a growth costs across many page loads. Otherwise it
    /// takes precisely `need`.
    ///
    /// The speculative step is genuine over-allocation charged to the **shared**
    /// budget: a first growth rounds a 14-unit request up to 1024, and those 1010
    /// units are a page the other three pools then cannot have. That is a cost
    /// question while there is room, and a correctness question when there is not —
    /// so [`VgeomPools::alloc_page`] never lets it be the reason a page fails: it
    /// reclaims every pool's untaken slack ([`trim_tail`](Self::trim_tail)) and
    /// retries with `speculate = false`.
    ///
    /// [`INITIAL_UNITS`]: Self::INITIAL_UNITS
    fn grow(&mut self, need: u64, headroom: u64, speculate: bool) -> bool {
        if headroom == 0 {
            return false;
        }
        // The tail hole (if any) counts toward the contiguous run we are after.
        let tail = self
            .free
            .last()
            .filter(|b| b.end() == self.capacity)
            .map(|b| b.len)
            .unwrap_or(0);
        let need_new = need.saturating_sub(tail);
        if need_new > headroom {
            return false;
        }
        let doubling = self.capacity.max(Self::INITIAL_UNITS);
        let step = if speculate && doubling <= headroom / 2 {
            doubling.max(need_new)
        } else {
            need_new
        };
        if step == 0 || step > headroom {
            return false;
        }
        let added = PoolBlock {
            offset: self.capacity,
            len: step,
        };
        self.capacity += step;
        self.insert_free(added);
        true
    }

    /// Return a block to the pool, coalescing with its neighbours.
    pub fn free(&mut self, block: PoolBlock) {
        if block.len == 0 {
            return;
        }
        debug_assert!(self.used >= block.len, "freeing more than was allocated");
        self.used = self.used.saturating_sub(block.len);
        self.insert_free(block);
    }

    fn insert_free(&mut self, block: PoolBlock) {
        let i = self.free.partition_point(|b| b.offset < block.offset);
        debug_assert!(
            self.free.get(i).is_none_or(|n| block.end() <= n.offset)
                && (i == 0 || self.free[i - 1].end() <= block.offset),
            "freed block overlaps a free run"
        );
        self.free.insert(i, block);
        // Coalesce with the next, then the previous.
        if i + 1 < self.free.len() && self.free[i].end() == self.free[i + 1].offset {
            self.free[i].len += self.free[i + 1].len;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].end() == self.free[i].offset {
            self.free[i - 1].len += self.free[i].len;
            self.free.remove(i);
        }
    }
}

/// The four suballocated GPU pools meshlet data lives in.
///
/// The **four pools share one budget**, and that budget is the VRAM ceiling:
/// total capacity can never exceed `budget_bytes` because a pool may only grow
/// into the headroom the other three are not using. There is no second ceiling to
/// drift from it, and no per-pool share to be wrong about.
///
/// **A fixed per-pool split was the first design and it was wrong.** The shape of
/// the format suggests one (~64 vertices and ~124 triangles per meshlet: 32 B per
/// vertex dominates, then the 3 B/triangle micro indices, then the 4 B/entry
/// vertex-index lists, and the 64 B descriptor is noise) — but the ratio is not
/// constant across an asset. A coarse page carries a handful of meshlets with very
/// few vertices each, so its descriptor share climbs past 7 % where a fine page's
/// is ~3 %. Any fixed split therefore makes *some* pool, rather than the byte
/// budget, the binding constraint — and the pages it binds hardest on are the
/// coarse ones, which is exactly backwards: **page 0 is the always-resident floor
/// that makes "never a hole" true, and it is the page a share-based split is most
/// likely to starve.** Sharing one budget removes the failure mode rather than
/// tuning around it: if the budget can hold the roots, the roots fit, whatever
/// their mix.
///
/// A pool that does run out of *budget* is not a failure: the allocation fails,
/// the page is not paged in, `budget_clamped` is reported, and the cut clamps to
/// what is resident — softer detail, never a hole.
#[derive(Debug, Clone)]
pub struct VgeomPools {
    budget_bytes: u64,
    pub vertices: PoolAllocator,
    pub meshlets: PoolAllocator,
    pub mlverts: PoolAllocator,
    pub mltris: PoolAllocator,
}

/// Which of the four pools an allocation is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    Vertices,
    Meshlets,
    Mlverts,
    Mltris,
}

impl VgeomPools {
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            vertices: PoolAllocator::new(VERTEX_REC_LEN as u64),
            meshlets: PoolAllocator::new(MESHLET_REC_LEN as u64),
            mlverts: PoolAllocator::new(4),
            mltris: PoolAllocator::new(4),
        }
    }

    /// Every pool, in the order [`alloc_page`](Self::alloc_page) reserves them:
    /// vertices first, because they are the largest claim by an order of magnitude
    /// and a page that cannot fit should fail having reserved the least.
    const ORDER: [PoolKind; 4] = [
        PoolKind::Vertices,
        PoolKind::Meshlets,
        PoolKind::Mlverts,
        PoolKind::Mltris,
    ];

    fn pool_mut(&mut self, kind: PoolKind) -> &mut PoolAllocator {
        match kind {
            PoolKind::Vertices => &mut self.vertices,
            PoolKind::Meshlets => &mut self.meshlets,
            PoolKind::Mlverts => &mut self.mlverts,
            PoolKind::Mltris => &mut self.mltris,
        }
    }

    /// Allocate `units` from one pool, letting it grow (speculatively) only into
    /// the budget the other three have not claimed.
    ///
    /// The single-section door. A page must go through
    /// [`alloc_page`](Self::alloc_page), which is transactional and carries the
    /// floor guarantee.
    pub fn alloc(&mut self, kind: PoolKind, units: u64) -> Option<PoolBlock> {
        let headroom = self.budget_bytes.saturating_sub(self.capacity_bytes());
        self.pool_mut(kind).alloc(units, headroom, true)
    }

    /// Return a block to one pool.
    pub fn free(&mut self, kind: PoolKind, block: PoolBlock) {
        self.pool_mut(kind).free(block);
    }

    /// Reserve all four sections of one page **atomically**, in [`ORDER`], or
    /// nothing.
    ///
    /// Two attempts. The first lets the pools grow speculatively, so the
    /// reallocate-and-restage a growth costs is amortized across many loads. If any
    /// of the four cannot be served, the partial reservation is rolled back, every
    /// pool's untaken tail slack is returned to the budget, and the page is retried
    /// with growth that takes **precisely** what each section needs.
    ///
    /// That retry is what makes the floor guarantee exact, and it is not a nicety:
    /// without it, a first growth rounds each section up to
    /// [`PoolAllocator::INITIAL_UNITS`] and charges the slack to the shared budget,
    /// so an asset whose root page is smaller than that overshoot ends with **zero**
    /// resident pages — and `VgeomNode` skips an asset with no residency, so the
    /// object silently disappears from the frame rather than degrading to coarser
    /// detail. Speculation may now cost efficiency; it can no longer cost an object.
    ///
    /// [`ORDER`]: Self::ORDER
    pub fn alloc_page(&mut self, needs: [u64; 4]) -> Option<[PoolBlock; 4]> {
        if let Some(blocks) = self.try_page(needs, true) {
            return Some(blocks);
        }
        for kind in Self::ORDER {
            self.pool_mut(kind).trim_tail();
        }
        self.try_page(needs, false)
    }

    fn try_page(&mut self, needs: [u64; 4], speculate: bool) -> Option<[PoolBlock; 4]> {
        let mut out = [PoolBlock::default(); 4];
        for (i, kind) in Self::ORDER.into_iter().enumerate() {
            let headroom = self.budget_bytes.saturating_sub(self.capacity_bytes());
            match self.pool_mut(kind).alloc(needs[i], headroom, speculate) {
                Some(b) => out[i] = b,
                None => {
                    for (j, done) in Self::ORDER.into_iter().enumerate().take(i) {
                        self.pool_mut(done).free(out[j]);
                    }
                    return None;
                }
            }
        }
        Some(out)
    }

    /// Bytes currently backed by GPU buffers (what the pools actually cost).
    pub fn capacity_bytes(&self) -> u64 {
        self.vertices.capacity_bytes()
            + self.meshlets.capacity_bytes()
            + self.mlverts.capacity_bytes()
            + self.mltris.capacity_bytes()
    }

    /// Bytes handed out to resident pages.
    pub fn used_bytes(&self) -> u64 {
        self.vertices.used_bytes()
            + self.meshlets.used_bytes()
            + self.mlverts.used_bytes()
            + self.mltris.used_bytes()
    }

    /// The hard ceiling: total pool capacity can never exceed this, because a
    /// pool only ever grows into the headroom the others leave.
    pub fn limit_bytes(&self) -> u64 {
        self.budget_bytes
    }
}

/// Where one resident page's four sections live in the pools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PageBlocks {
    /// Meshlet records (units of one [`MeshletRec`]).
    pub meshlets: PoolBlock,
    /// Micro vertex-index entries (units of `u32`).
    pub mlverts: PoolBlock,
    /// Micro triangle-index bytes (units of `u32` words).
    pub mltris: PoolBlock,
    /// Vertex records (units of one [`VgeomVertex`](crate::VgeomVertex)).
    pub vertices: PoolBlock,
}

// ── budget + stats ──────────────────────────────────────────────────────────

/// 256 MiB of meshlet pools — comfortably above every shipping sample, so the
/// default path is "stream everything the camera wants" and the budget only bites
/// on content that genuinely does not fit.
pub const DEFAULT_VGEOM_BUDGET_BYTES: u64 = 256 * 1024 * 1024;

/// The residency budget a [`VgeomStreamer`] respects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VgeomStreamBudget {
    /// VRAM ceiling for the four meshlet pools, in bytes. The pools are *sized*
    /// from this and grow lazily toward it, so an idle renderer pays nothing and
    /// the ceiling can never be exceeded (an allocation past a pool's limit fails
    /// and the page simply is not paged in).
    pub budget_bytes: u64,
    /// Ceiling on how many not-yet-resident pages one [`VgeomStreamer::plan`] may
    /// bring in (`0` = unlimited, the default).
    ///
    /// **Why unlimited by default**: a golden renders one frame from cold, so a
    /// per-frame load cap would show a partially converged mesh and change every
    /// committed image. Unlimited makes a cold frame fully resident within budget,
    /// which is exactly the equivalence the goldens gate. The cap exists (and is
    /// tested) for a host that would rather spread the hitch.
    pub max_loads_per_sync: usize,
    /// Eviction hysteresis: a page is kept while the threshold stays below
    /// `hysteresis ×` the value that would acquire it. `1.0` disables it.
    pub hysteresis: f32,
}

impl Default for VgeomStreamBudget {
    fn default() -> Self {
        Self {
            budget_bytes: DEFAULT_VGEOM_BUDGET_BYTES,
            max_loads_per_sync: 0,
            hysteresis: 1.5,
        }
    }
}

/// Streaming counters for the debug/stats path (Ring 0, so the player and the
/// editor viewport can both name one type).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct VgeomStreamStats {
    /// Assets with any residency.
    pub assets: usize,
    /// Resident pages across every asset.
    pub resident_pages: usize,
    /// Pages the camera asked for.
    pub wanted_pages: usize,
    /// Wanted pages not resident — the convergence backlog.
    pub pending_pages: usize,
    /// Assets whose cut is coarser than the camera asked for (the
    /// "requested-but-missing" signal, CPU-side and deterministic).
    pub clamped_assets: usize,
    /// Bytes handed out to resident pages.
    pub resident_bytes: u64,
    /// Bytes actually backed by GPU buffers (pool capacity).
    pub pool_bytes: u64,
    /// High-water mark of `resident_bytes`.
    pub peak_resident_bytes: u64,
    /// Pages loaded since the streamer was created.
    pub loads: u64,
    /// Pages evicted since the streamer was created.
    pub evictions: u64,
    /// Pages whose bytes could not be read (never silently becomes geometry).
    pub failed: u64,
    /// Whether the last plan had to leave a wanted page out for budget.
    pub budget_clamped: bool,
}

impl VgeomStreamStats {
    /// A one-line human summary for the Output Log / a `stats` dump.
    pub fn summary(&self) -> String {
        format!(
            "vgeom streaming: {} assets, {} pages resident ({} wanted, {} pending, {} clamped), \
             {:.1} MiB resident / {:.1} MiB pools, {} loads / {} evictions / {} failed{}",
            self.assets,
            self.resident_pages,
            self.wanted_pages,
            self.pending_pages,
            self.clamped_assets,
            self.resident_bytes as f64 / (1024.0 * 1024.0),
            self.pool_bytes as f64 / (1024.0 * 1024.0),
            self.loads,
            self.evictions,
            self.failed,
            if self.budget_clamped {
                " [budget-clamped]"
            } else {
                ""
            }
        )
    }
}

// ── wants ───────────────────────────────────────────────────────────────────

/// One asset's want for this frame: the **smallest** per-instance LOD threshold
/// `t` any of its placed instances asked for (the closest / largest instance
/// drives the detail the asset needs).
#[derive(Clone, Copy)]
pub struct VgeomWant<'a> {
    pub asset: u128,
    pub source: &'a VgeomSource,
    /// The screen-space-error scalar from `passes::vgeom::lod_threshold` — the
    /// same number the GPU cut compares meshlet errors against.
    pub threshold: f32,
}

/// How many pages (a prefix of the coarse-first page order) a cut at `threshold`
/// can actually use.
///
/// A page contributes to the cut only while some meshlet in it has
/// `parent_error > threshold`, and
/// [`max_parent_error`](VgeomPageEntry::max_parent_error) descends as the pages
/// get finer — so the useful pages are exactly a leading run. Page 0 is `+∞` (it
/// holds the roots), so the answer is never 0 for a non-empty asset: the floor is
/// always wanted.
///
/// # Why this exact bound: streaming costs VRAM, never detail
///
/// Grant this many pages and the residency-clamped cut is **identical** to the
/// unclamped one — the pages left unloaded are precisely the ones the cut could
/// not have selected anyway:
///
/// * a page past the bound has `max_parent_error <= t`, so every meshlet in it
///   fails `t < parent_error`. Declining to load it removes nothing.
/// * the clamp zeroes `error` at and below the floor, which could in principle
///   *add* a meshlet the unclamped cut rejected for `error > t`. It cannot here:
///   a meshlet at the floor level `L` was produced by a group whose inputs are the
///   non-roots at `L-1`, so its `error` **is** their `parent_error`, and their page
///   is past the bound ⇒ `error <= t`. (A root's error is the same quantity; a
///   LOD-0 root's is 0.) So the zeroing is a no-op on the drawn set.
///
/// That is what lets every committed golden stay byte-identical with streaming on
/// while the streamer still declines to page in most of a large asset — it is
/// asserted, not assumed, by `golden.rs::vgeom_cpu_gpu_cut_parity`.
///
/// Under a **budget clamp** the granted prefix is shorter than this, and then the
/// cut really is coarser. That is the intended degradation: softer detail, never
/// a hole.
pub fn ideal_page_count(pages: &[VgeomPageEntry], threshold: f32) -> usize {
    if pages.is_empty() {
        return 0;
    }
    pages
        .iter()
        .take_while(|p| p.max_parent_error > threshold)
        .count()
        .clamp(1, pages.len())
}

// ── residency ───────────────────────────────────────────────────────────────

/// One asset's residency: which prefix of its pages is in the pools, where, and
/// the meshlet→pool remap the cull shader reads.
#[derive(Debug, Clone)]
pub struct AssetResidency {
    pages: Vec<VgeomPageEntry>,
    meshlet_count: u32,
    max_lod: u32,
    /// Resident prefix length.
    resident: usize,
    /// Pool blocks, index-aligned with the resident prefix.
    blocks: Vec<PageBlocks>,
    /// Each resident page's global meshlet indices, so an eviction can clear
    /// exactly its remap entries without re-reading the payload.
    indices: Vec<Vec<u32>>,
    /// Meshlet index → pool meshlet slot, or [`NOT_RESIDENT`].
    remap: Vec<u32>,
    /// Stamped from the process-global monotone counter whenever the residency
    /// (and therefore `remap`) changes, so the render side re-uploads the table
    /// only when it moved — and can never mistake a fresh residency's stamp for
    /// one it already holds. See [`NEXT_RESIDENCY_STAMP`].
    generation: u64,
    /// The finest page that could be read; pages at or past it are never
    /// requested again (the terrain blocked-set precedent, in prefix form).
    blocked_from: Option<usize>,
    bytes: u64,
}

impl AssetResidency {
    fn new(source: &VgeomSource) -> Self {
        let meshlet_count = source.meshlet_count();
        Self {
            pages: source.pages().to_vec(),
            meshlet_count,
            max_lod: source.header().max_lod,
            resident: 0,
            blocks: Vec::new(),
            indices: Vec::new(),
            remap: vec![NOT_RESIDENT; meshlet_count as usize],
            generation: next_stamp(),
            blocked_from: None,
            bytes: 0,
        }
    }

    /// Whether this residency still describes `source` (a re-cooked or swapped
    /// asset must reset rather than be patched).
    ///
    /// Compares the **whole page directory**, not just its length. In a cooked
    /// pack an asset id names immutable bytes, so counts were enough; in the
    /// editor's content root (P18.3) a re-import can hand the same id a different
    /// payload, and an edited mesh with unchanged topology produces a DAG that
    /// agrees on meshlet and page counts while disagreeing on every error in the
    /// directory. Patching residency across that would draw the *old* geometry out
    /// of pool blocks staged for it. The directory is a handful of entries, so this
    /// is a few dozen bytes of comparison once per asset per frame.
    ///
    /// **It is a cheap discriminator, not a content hash, and that distinction is
    /// deliberate.** A payload whose directory is byte-identical — a mesh whose
    /// vertices were merely translated, say — still passes. Making this exact needs
    /// a content stamp in the header, i.e. a format change. It is not the editor's
    /// guarantee either way: `inf_editor_core::render_assets` keys a vgeom asset by
    /// its payload's content hash precisely so that *changed bytes are a different
    /// asset* and this check is never asked a question it cannot answer.
    fn matches(&self, source: &VgeomSource) -> bool {
        self.meshlet_count == source.meshlet_count() && self.pages == source.pages()
    }

    /// Resident prefix length.
    #[inline]
    pub fn resident_pages(&self) -> usize {
        self.resident
    }

    /// Pages the asset has in total.
    #[inline]
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// The asset's page directory.
    #[inline]
    pub fn pages(&self) -> &[VgeomPageEntry] {
        &self.pages
    }

    /// The finest resident LOD level — the **only** parameter the clamped cut
    /// needs, on the CPU
    /// ([`select_with_residency`](crate::VgeomMesh::select_with_residency)) and on
    /// the GPU alike.
    #[inline]
    pub fn floor_lod(&self) -> u32 {
        match self.resident.checked_sub(1).and_then(|i| self.pages.get(i)) {
            Some(p) => p.floor_lod,
            None => self.max_lod,
        }
    }

    /// Meshlet index → pool meshlet slot ([`NOT_RESIDENT`] where the page is out).
    #[inline]
    pub fn remap(&self) -> &[u32] {
        &self.remap
    }

    /// The residency stamp: strictly increasing, process-globally unique, and
    /// bumped whenever [`remap`](Self::remap) changes. A cache keyed on it can
    /// never serve a stale table (see [`NEXT_RESIDENCY_STAMP`]).
    ///
    /// **Never put this in a trace, a golden, or any comparison between runs.** It
    /// is drawn from a process-global counter, so two runs of the same frame
    /// sequence legitimately produce different stamps — it identifies *a* residency,
    /// it does not describe one. Compare [`resident_pages`](Self::resident_pages),
    /// [`floor_lod`](Self::floor_lod), [`remap`](Self::remap) or
    /// [`resident_bytes`](Self::resident_bytes) instead; those are the content, and
    /// they are what the determinism gates pin.
    ///
    /// **In one line: it is a stamp, not a measurement — it must never enter a
    /// trace or a golden, and the only legal operation on it is `!=` against a
    /// previously-observed stamp** (the "has this moved since I last looked?"
    /// question a cache asks). Ordering it, subtracting it, or printing it are all
    /// ways of pretending it describes something.
    #[inline]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bytes this asset holds in the pools.
    #[inline]
    pub fn resident_bytes(&self) -> u64 {
        self.bytes
    }

    /// The page this asset is blocked from (a page whose bytes could not be read).
    #[inline]
    pub fn blocked_from(&self) -> Option<usize> {
        self.blocked_from
    }

    /// Pool blocks of resident page `page`.
    #[inline]
    pub fn blocks(&self, page: usize) -> Option<&PageBlocks> {
        self.blocks.get(page)
    }

    /// Map an **image** vertex index to its pool slot, or `None` when the vertex
    /// belongs to a page that is not resident.
    ///
    /// Vertex blocks tile `[0, prefix)` in page order (the container guarantees
    /// it), so this is a binary search over the resident prefix.
    pub fn pool_vertex(&self, image_index: u32) -> Option<u32> {
        let i = self.pages[..self.resident]
            .partition_point(|p| p.vertex_start <= image_index)
            .checked_sub(1)?;
        let p = &self.pages[i];
        if image_index >= p.vertex_start + p.vertex_count {
            return None;
        }
        let b = self.blocks.get(i)?;
        Some((b.vertices.offset + u64::from(image_index - p.vertex_start)) as u32)
    }
}

/// Free page `page`'s blocks, shrink the prefix and clear its remap entries.
/// Only the **finest** resident page may be released — residency is a prefix.
fn release(pools: &mut VgeomPools, res: &mut AssetResidency, page: usize) {
    if page + 1 != res.resident || res.blocks.len() != res.resident {
        debug_assert!(false, "residency must be released from the fine end");
        return;
    }
    let b = res.blocks.pop().expect("resident page has blocks");
    pools.free(PoolKind::Meshlets, b.meshlets);
    pools.free(PoolKind::Mlverts, b.mlverts);
    pools.free(PoolKind::Mltris, b.mltris);
    pools.free(PoolKind::Vertices, b.vertices);
    for g in res.indices.pop().unwrap_or_default() {
        if let Some(slot) = res.remap.get_mut(g as usize) {
            *slot = NOT_RESIDENT;
        }
    }
    res.bytes = res.bytes.saturating_sub(res.pages[page].resident_bytes());
    res.resident = page;
    res.generation = next_stamp();
}

/// Allocate page `page`'s four blocks and extend the resident prefix. Rolls every
/// partial allocation back on failure.
fn reserve(pools: &mut VgeomPools, res: &mut AssetResidency, page: usize) -> bool {
    let e = res.pages[page];
    let mltri_words = u64::from(e.mltri_count).div_ceil(4);
    // Atomic, and guaranteed to succeed whenever the budget can hold the page —
    // which is what keeps the always-resident root page resident.
    let Some([vertices, meshlets, mlverts, mltris]) = pools.alloc_page([
        u64::from(e.vertex_count),
        u64::from(e.meshlet_count),
        u64::from(e.mlvert_count),
        mltri_words,
    ]) else {
        return false;
    };
    res.blocks.push(PageBlocks {
        meshlets,
        mlverts,
        mltris,
        vertices,
    });
    res.indices.push(Vec::new());
    res.resident = page + 1;
    res.bytes += e.resident_bytes();
    res.generation = next_stamp();
    true
}

// ── the plan ────────────────────────────────────────────────────────────────

/// What one [`VgeomStreamer::plan`] decided: the pages to upload (already staged
/// into pool-absolute bytes) and the ones that left.
#[derive(Debug, Default)]
pub struct VgeomStreamPlan {
    /// Pages to write into the pools, in deterministic order.
    pub uploads: Vec<PageUpload>,
    /// `(asset, page)` pairs evicted this sync.
    pub evicted: Vec<(u128, usize)>,
    /// Assets that left the frame entirely and were fully freed.
    pub dropped: Vec<u128>,
    /// Pages whose bytes could not be read; each blocks its asset from there on.
    pub failed: Vec<(u128, usize, String)>,
    /// Whether a pool's capacity moved, so the render side must grow (and copy)
    /// its backing buffers before writing the uploads.
    pub pools_grew: bool,
}

/// One page staged for upload: pool-absolute bytes, ready for `write_buffer`.
#[derive(Debug)]
pub struct PageUpload {
    pub asset: u128,
    pub page: usize,
    pub blocks: PageBlocks,
    /// [`MeshletRec`]s with `vertex_offset` / `triangle_offset` rebased to the
    /// pools.
    pub meshlets: Vec<u8>,
    /// Micro vertex-index entries rewritten to pool vertex slots.
    pub mlverts: Vec<u8>,
    /// Micro triangle-index bytes, verbatim, zero-padded to a 4-byte multiple
    /// (`write_buffer` demands it, and the pool is a `u32` array).
    pub mltris: Vec<u8>,
    /// Vertex records, verbatim.
    pub vertices: Vec<u8>,
    /// The page's global meshlet indices (the remap table's keys).
    indices: Vec<u32>,
}

// ── the streamer ────────────────────────────────────────────────────────────

/// Drives meshlet residency for every vmesh asset in a frame.
pub struct VgeomStreamer {
    budget: VgeomStreamBudget,
    pools: VgeomPools,
    assets: BTreeMap<u128, AssetResidency>,
    stats: VgeomStreamStats,
}

impl VgeomStreamer {
    pub fn new(budget: VgeomStreamBudget) -> Self {
        Self {
            pools: VgeomPools::new(budget.budget_bytes),
            budget,
            assets: BTreeMap::new(),
            stats: VgeomStreamStats::default(),
        }
    }

    #[inline]
    pub fn budget(&self) -> VgeomStreamBudget {
        self.budget
    }

    /// Re-budget. Changing the **byte** ceiling resizes the pools, which means
    /// every offset in them is void, so residency is dropped and re-paged; the
    /// other knobs take effect on the next plan.
    pub fn set_budget(&mut self, budget: VgeomStreamBudget) {
        if budget.budget_bytes != self.budget.budget_bytes {
            self.pools = VgeomPools::new(budget.budget_bytes);
            self.assets.clear();
            self.stats.resident_pages = 0;
            self.stats.resident_bytes = 0;
        }
        self.budget = budget;
    }

    #[inline]
    pub fn pools(&self) -> &VgeomPools {
        &self.pools
    }

    #[inline]
    pub fn stats(&self) -> &VgeomStreamStats {
        &self.stats
    }

    /// One asset's residency, for the render side's binds and the parity tests.
    #[inline]
    pub fn residency(&self, asset: u128) -> Option<&AssetResidency> {
        self.assets.get(&asset)
    }

    /// Every asset with residency, in id order.
    pub fn assets(&self) -> impl Iterator<Item = (u128, &AssetResidency)> {
        self.assets.iter().map(|(k, v)| (*k, v))
    }

    /// **The sync point.** Advance residency one bounded, deterministic step
    /// toward what `wants` asks for and hand back the work to upload.
    ///
    /// Call once per frame, before culling. See the module docs for why the result
    /// is a pure function of `(wants, current residency, budget)`.
    pub fn plan(&mut self, wants: &[VgeomWant<'_>]) -> VgeomStreamPlan {
        let mut plan = VgeomStreamPlan::default();
        let capacity_before = self.pools.capacity_bytes();

        // Deterministic regardless of how the caller assembled the frame — which
        // takes more than sorting by id. Two wants for one asset are a legitimate
        // input (a host that appends per instance rather than per asset), and
        // `dedup` keeps the FIRST of each run, so sorting by id alone would let
        // caller order pick the surviving threshold. Sorting by `(asset, threshold)`
        // makes the survivor the SMALLEST threshold — the finest detail any
        // instance asked for, which is exactly the merge rule the caller would have
        // applied itself.
        let mut wants: Vec<VgeomWant<'_>> = wants.to_vec();
        wants.sort_by(|a, b| {
            a.asset
                .cmp(&b.asset)
                .then(a.threshold.total_cmp(&b.threshold))
        });
        wants.dedup_by_key(|w| w.asset);

        // ── 1. assets that left the frame are freed entirely ──
        let live: BTreeSet<u128> = wants.iter().map(|w| w.asset).collect();
        let leaving: Vec<u128> = self
            .assets
            .keys()
            .copied()
            .filter(|a| !live.contains(a))
            .collect();
        for a in leaving {
            let mut res = self.assets.remove(&a).expect("key came from the map");
            while res.resident > 0 {
                let i = res.resident - 1;
                release(&mut self.pools, &mut res, i);
                plan.evicted.push((a, i));
                self.stats.evictions += 1;
            }
            plan.dropped.push(a);
        }

        // ── 2. register / refresh ──
        for w in &wants {
            let stale = self
                .assets
                .get(&w.asset)
                .is_none_or(|res| !res.matches(w.source));
            if stale {
                if let Some(mut old) = self.assets.remove(&w.asset) {
                    while old.resident > 0 {
                        let i = old.resident - 1;
                        release(&mut self.pools, &mut old, i);
                        self.stats.evictions += 1;
                    }
                }
                self.assets.insert(w.asset, AssetResidency::new(w.source));
            }
        }

        // ── 3. per-asset target prefix (want + hysteresis + blocked clamp) ──
        let h = self.budget.hysteresis.max(1.0);
        let mut targets: BTreeMap<u128, Target> = BTreeMap::new();
        for w in &wants {
            let res = &self.assets[&w.asset];
            if res.pages.is_empty() {
                continue;
            }
            let want = ideal_page_count(&res.pages, w.threshold);
            let keep = ideal_page_count(&res.pages, w.threshold / h);
            let cur = res.resident;
            let mut target = if cur > keep {
                keep
            } else if cur < want {
                want
            } else {
                cur
            };
            if let Some(b) = res.blocked_from {
                target = target.min(b);
            }
            targets.insert(
                w.asset,
                Target {
                    target: target.clamp(1, res.pages.len()),
                    want,
                    threshold: w.threshold,
                },
            );
        }

        // ── 4. grant against the byte budget, worst-error-first ──
        //
        // Every asset's page 0 (its roots) is granted unconditionally: it is the
        // floor that makes "never a hole" true, and it is small by construction.
        // Refinement past it is auctioned by how much object-space error the asset
        // is still suffering relative to what its closest instance asked for, so
        // the geometry the camera is looking at refines first.
        let mut spent: u64 = 0;
        let mut order: Vec<(u128, usize)> = Vec::new();
        let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut budget_clamped = false;
        for (asset, t) in &targets {
            let res = &self.assets[asset];
            spent += res.pages[0].resident_bytes();
            order.push((*asset, 0));
            if t.target > 1 {
                heap.push(Candidate {
                    deficit: deficit_of(&res.pages[1], t.threshold),
                    asset: *asset,
                    page: 1,
                    threshold: t.threshold,
                });
            }
        }
        let ceiling = self.budget.budget_bytes;
        let mut granted: BTreeMap<u128, usize> = targets.keys().map(|a| (*a, 1)).collect();
        while let Some(c) = heap.pop() {
            let res = &self.assets[&c.asset];
            let cost = res.pages[c.page].resident_bytes();
            if spent + cost > ceiling {
                // This page cannot fit; a smaller one from another asset still
                // might, so only this asset stops refining.
                budget_clamped = true;
                continue;
            }
            spent += cost;
            granted.insert(c.asset, c.page + 1);
            order.push((c.asset, c.page));
            if c.page + 1 < targets[&c.asset].target {
                heap.push(Candidate {
                    deficit: deficit_of(&res.pages[c.page + 1], c.threshold),
                    asset: c.asset,
                    page: c.page + 1,
                    threshold: c.threshold,
                });
            }
        }

        // ── 5. evict everything past the grant ──
        for (&asset, &want) in &granted {
            let Some(res) = self.assets.get_mut(&asset) else {
                continue;
            };
            while res.resident > want {
                let i = res.resident - 1;
                release(&mut self.pools, res, i);
                plan.evicted.push((asset, i));
                self.stats.evictions += 1;
            }
        }

        // ── 6. reserve the granted pages that are not resident, bounded ──
        let cap = if self.budget.max_loads_per_sync == 0 {
            usize::MAX
        } else {
            self.budget.max_loads_per_sync
        };
        let mut jobs: Vec<(u128, usize)> = Vec::new();
        // An asset whose reservation failed must not have its FINER grants tried:
        // residency is a prefix, so page p+1 without page p is not a state that
        // exists. Stalling the asset here is the graceful degradation — it keeps
        // what it has and the cut clamps to it.
        let mut stalled: BTreeSet<u128> = BTreeSet::new();
        // `order` is per-asset ascending, so truncating it still leaves a prefix.
        for (asset, page) in order {
            if jobs.len() >= cap {
                break;
            }
            if stalled.contains(&asset) {
                continue;
            }
            let Some(res) = self.assets.get_mut(&asset) else {
                continue;
            };
            if page < res.resident {
                continue; // already there
            }
            debug_assert_eq!(page, res.resident, "grants must extend the prefix");
            if !reserve(&mut self.pools, res, page) {
                budget_clamped = true;
                stalled.insert(asset);
                continue;
            }
            jobs.push((asset, page));
        }

        // ── 7. stage as ONE deterministic batch ──
        //
        // A pool that GREW means its backing GPU buffer is reallocated, so every
        // page already in it must be written again. Growth only ever appends, so
        // no live block moved — but the new buffer's contents start undefined, and
        // re-staging is both simpler and safer than a same-submit buffer-to-buffer
        // copy (`queue.write_buffer` is ordered *before* the encoder's commands in
        // a submit, so a copy there would clobber this frame's uploads). Growth is
        // O(log budget) times per session, so paying a full re-upload for it is
        // cheap and keeps the ordering trivially correct.
        // Strictly GREW: `alloc_page`'s retry can also *shrink* capacity by
        // reclaiming slack, and a shrink recreates no buffer, so it must not drag
        // a full re-stage behind it.
        plan.pools_grew = self.pools.capacity_bytes() > capacity_before;
        let newly_reserved: BTreeSet<(u128, usize)> = jobs.iter().copied().collect();
        if plan.pools_grew {
            let already = &newly_reserved;
            for (asset, res) in &self.assets {
                for page in 0..res.resident {
                    if !already.contains(&(*asset, page)) {
                        jobs.push((*asset, page));
                    }
                }
            }
            jobs.sort_unstable();
        }
        let sources: BTreeMap<u128, &VgeomSource> =
            wants.iter().map(|w| (w.asset, w.source)).collect();
        let staged = stage_batch(&self.assets, &sources, &jobs);

        // The finest page each asset failed on truncates it: a page that could not
        // be read blocks everything below it, and any page of the same asset staged
        // *after* it in this batch rested on its vertices.
        let mut cut_at: BTreeMap<u128, usize> = BTreeMap::new();
        for ((asset, page), out) in jobs.iter().copied().zip(&staged) {
            if let Err(e) = out {
                let slot = cut_at.entry(asset).or_insert(page);
                *slot = (*slot).min(page);
                plan.failed.push((asset, page, e.clone()));
                self.stats.failed += 1;
            }
        }
        for ((asset, page), out) in jobs.into_iter().zip(staged) {
            let truncated = cut_at.get(&asset).is_some_and(|c| page >= *c);
            match out {
                Ok(mut upload) if !truncated => {
                    upload.asset = asset;
                    upload.page = page;
                    if let Some(res) = self.assets.get_mut(&asset) {
                        res.apply_remap(page, &upload.indices, upload.blocks.meshlets.offset);
                    }
                    if newly_reserved.contains(&(asset, page)) {
                        self.stats.loads += 1;
                    }
                    plan.uploads.push(upload);
                }
                _ => {}
            }
        }
        for (asset, from) in cut_at {
            let Some(res) = self.assets.get_mut(&asset) else {
                continue;
            };
            while res.resident > from {
                let i = res.resident - 1;
                release(&mut self.pools, res, i);
                plan.evicted.push((asset, i));
            }
            res.blocked_from = Some(res.blocked_from.map_or(from, |b| b.min(from)));
            tracing::warn!(
                "inf-vgeom: vmesh {asset:#034x} page {from} failed to page in — excluded from \
                 streaming until the asset is reopened. The cut clamps to the coarser pages that \
                 did load: softer detail, never a hole."
            );
        }
        // Uploads for pages that were rolled back must not be written.
        plan.uploads.retain(|u| {
            self.assets
                .get(&u.asset)
                .is_some_and(|r| u.page < r.resident)
        });

        // ── 8. stats ──
        self.stats.assets = self.assets.len();
        self.stats.resident_pages = self.assets.values().map(|r| r.resident).sum();
        self.stats.resident_bytes = self.assets.values().map(|r| r.bytes).sum();
        self.stats.peak_resident_bytes = self
            .stats
            .peak_resident_bytes
            .max(self.stats.resident_bytes);
        self.stats.pool_bytes = self.pools.capacity_bytes();
        self.stats.wanted_pages = targets.values().map(|t| t.want).sum();
        self.stats.pending_pages = targets
            .iter()
            .map(|(a, t)| {
                t.want
                    .saturating_sub(self.assets.get(a).map_or(0, |r| r.resident))
            })
            .sum();
        self.stats.clamped_assets = targets
            .iter()
            .filter(|(a, t)| self.assets.get(a).is_none_or(|r| r.resident < t.want))
            .count();
        self.stats.budget_clamped = budget_clamped;
        plan
    }
}

impl AssetResidency {
    /// Write `page`'s meshlets into the remap table at their pool slots.
    fn apply_remap(&mut self, page: usize, indices: &[u32], base: u64) {
        for (k, &g) in indices.iter().enumerate() {
            if let Some(slot) = self.remap.get_mut(g as usize) {
                *slot = (base + k as u64) as u32;
            }
        }
        if let Some(slot) = self.indices.get_mut(page) {
            *slot = indices.to_vec();
        }
        self.generation = next_stamp();
    }
}

/// A per-asset target prefix for one plan.
struct Target {
    target: usize,
    want: usize,
    threshold: f32,
}

/// A refinement candidate in the grant auction.
struct Candidate {
    deficit: f32,
    asset: u128,
    page: usize,
    threshold: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    /// Largest deficit first; ties broken on `(asset, page)` **descending**,
    /// because `BinaryHeap` is a max-heap and the tie-break only has to be a total
    /// order that does not depend on insertion order.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.deficit
            .total_cmp(&other.deficit)
            .then_with(|| other.asset.cmp(&self.asset))
            .then_with(|| other.page.cmp(&self.page))
    }
}

/// How much object-space error paging `candidate` in would retire, in units of
/// the threshold the asset's closest instance asked for. Larger ⇒ grant sooner.
///
/// [`max_parent_error`](VgeomPageEntry::max_parent_error) is the coarsest cut
/// boundary the page covers, so it descends as pages get finer — a page's value
/// therefore falls the deeper into an asset the auction goes, and rises the closer
/// the asset's camera is. Dividing by `t` is what makes two different assets
/// comparable at all: it is the error measured in *pixels of tolerance*, which is
/// the only currency both of them are denominated in.
///
/// Page 0 is never a candidate (it is the unconditional floor), which is what
/// keeps this finite — the root page's value is `+∞`.
fn deficit_of(candidate: &VgeomPageEntry, threshold: f32) -> f32 {
    let t = threshold.abs().max(1e-6);
    let e = candidate.max_parent_error;
    if e.is_finite() {
        e / t
    } else {
        f32::MAX
    }
}

/// Stage every job as one deterministic batch.
///
/// `inf_core::parallel_map_ref` is a pure in-order map, so the output is a
/// function of `jobs` alone — parallel without becoming timing-dependent. Small
/// batches (and wasm, which has no pool) take the serial path and produce the
/// identical result.
fn stage_batch(
    assets: &BTreeMap<u128, AssetResidency>,
    sources: &BTreeMap<u128, &VgeomSource>,
    jobs: &[(u128, usize)],
) -> Vec<Result<PageUpload, String>> {
    let one = |&(asset, page): &(u128, usize)| -> Result<PageUpload, String> {
        let res = assets
            .get(&asset)
            .ok_or_else(|| "asset residency vanished".to_string())?;
        let src = *sources
            .get(&asset)
            .ok_or_else(|| "asset source vanished".to_string())?;
        let blocks = *res
            .blocks(page)
            .ok_or_else(|| format!("page {page} has no reservation"))?;
        src.with_page_sections(page, |s| stage_page(res, blocks, s))
            .ok_or_else(|| format!("page {page} sections unavailable"))?
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        /// Below this many pages the pool's dispatch costs more than the staging.
        const PARALLEL_THRESHOLD: usize = 4;
        if jobs.len() >= PARALLEL_THRESHOLD {
            return inf_core::parallel_map_ref(jobs, one);
        }
    }
    jobs.iter().map(one).collect()
}

/// Rewrite one page's sections into pool-absolute bytes.
///
/// Two `u32` adds per meshlet record and one lookup per micro vertex index; the
/// triangle bytes and the vertex records are verbatim. This is the whole "decode"
/// a meshlet page costs — no parsing, no allocation beyond the staging buffers.
fn stage_page(
    res: &AssetResidency,
    blocks: PageBlocks,
    s: VgeomPageSections<'_>,
) -> Result<PageUpload, String> {
    let mut meshlets = s.meshlets.to_vec();
    let mlvert_base = blocks.mlverts.offset as u32;
    let mltri_byte_base = (blocks.mltris.offset * 4) as u32;
    for r in bytemuck::cast_slice_mut::<u8, MeshletRec>(&mut meshlets) {
        r.vertex_offset += mlvert_base;
        r.triangle_offset += mltri_byte_base;
    }

    let src_mlverts = bytemuck::cast_slice::<u8, u32>(s.mlverts);
    let mut mlverts: Vec<u32> = Vec::with_capacity(src_mlverts.len());
    for &v in src_mlverts {
        mlverts.push(
            res.pool_vertex(v)
                .ok_or_else(|| format!("micro index references non-resident vertex {v}"))?,
        );
    }

    let mut mltris = s.mltris.to_vec();
    mltris.resize(mltris.len().next_multiple_of(4), 0);

    Ok(PageUpload {
        asset: 0,
        page: 0,
        blocks,
        meshlets,
        mlverts: bytemuck::cast_slice(&mlverts).to_vec(),
        mltris,
        vertices: s.vertices.to_vec(),
        indices: bytemuck::cast_slice::<u8, u32>(s.indices).to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::dense_mesh;
    use crate::VgeomMesh;

    // ── allocator ───────────────────────────────────────────────────────────

    /// Headroom generous enough that growth is never the thing under test.
    const ROOMY: u64 = 1 << 24;

    #[test]
    fn alloc_free_never_overlaps_and_reuses_holes() {
        let mut p = PoolAllocator::new(4);
        let a = p.alloc(100, ROOMY, true).unwrap();
        let b = p.alloc(50, ROOMY, true).unwrap();
        let c = p.alloc(25, ROOMY, true).unwrap();
        for (x, y) in [(a, b), (a, c), (b, c)] {
            assert!(!x.overlaps(&y), "{x:?} overlaps {y:?}");
        }
        assert_eq!(p.used_units(), 175);
        p.free(b);
        assert_eq!(p.used_units(), 125);
        let d = p.alloc(50, ROOMY, true).unwrap();
        assert_eq!(d, b, "first fit reuses the freed hole exactly");
        p.free(a);
        p.free(c);
        p.free(d);
        assert_eq!(p.used_units(), 0);
        assert_eq!(p.free_runs(), 1, "adjacent frees coalesce");
    }

    /// Property: over a long randomised (but fixed-seed) alloc/free sequence, no
    /// two live blocks ever overlap and nothing escapes the capacity.
    #[test]
    fn allocator_property_no_overlap_ever() {
        let mut p = PoolAllocator::new(8);
        let mut live: Vec<PoolBlock> = Vec::new();
        let mut rng = 0x1234_5678u64;
        let mut next = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            rng >> 33
        };
        for step in 0..2000u64 {
            if step % 3 == 2 && !live.is_empty() {
                let i = (next() as usize) % live.len();
                p.free(live.swap_remove(i));
                continue;
            }
            let n = 1 + next() % 97;
            let Some(b) = p.alloc(n, ROOMY, true) else {
                continue;
            };
            assert!(b.end() <= p.capacity_units(), "{b:?} past the capacity");
            for other in &live {
                assert!(!b.overlaps(other), "{b:?} overlaps live {other:?}");
            }
            live.push(b);
        }
        let held: u64 = live.iter().map(|b| b.len).sum();
        assert_eq!(p.used_units(), held, "used == the sum of live blocks");
    }

    #[test]
    fn alloc_is_deterministic_across_runs() {
        let run = || {
            let mut p = PoolAllocator::new(4);
            let mut out = Vec::new();
            let mut live = Vec::new();
            for i in 0..64u64 {
                let b = p.alloc(1 + i % 7, ROOMY, true).unwrap();
                out.push(b);
                live.push(b);
                if i % 3 == 0 && live.len() > 2 {
                    p.free(live.remove(0));
                }
            }
            out
        };
        assert_eq!(run(), run(), "two identical call sequences agree exactly");
    }

    /// A pool grows only into the headroom it is handed, and a growth that would
    /// need more than that fails cleanly instead of overrunning the budget.
    #[test]
    fn growth_never_exceeds_the_headroom() {
        let mut p = PoolAllocator::new(16);
        let mut budget_units = 1000u64;
        let mut total = 0u64;
        loop {
            let headroom = (budget_units - p.capacity_units()) * 16;
            let Some(b) = p.alloc(64, headroom, true) else {
                break;
            };
            total += b.len;
            assert!(b.end() <= budget_units, "block {b:?} past the budget");
        }
        assert!(total <= 1000);
        assert!(p.capacity_units() <= 1000);
        // With zero headroom the pool can still serve out of what it already
        // holds — it just cannot get any bigger. A request past the largest
        // remaining hole therefore fails rather than overrunning the budget.
        budget_units = p.capacity_units();
        assert!(
            p.alloc(budget_units + 1, 0, true).is_none(),
            "no headroom, no growth"
        );
    }

    /// Growth doubles — so the reallocate-and-restage it costs is amortized — but
    /// never takes more than half the headroom unless the request itself needs
    /// more. Four pools share one budget, and a doubling pool that swallowed the
    /// remainder would starve the other three of a page they are paging in
    /// together.
    #[test]
    fn growth_leaves_room_for_the_other_pools() {
        let mut p = PoolAllocator::new(4);
        let headroom_units = 100_000u64;
        p.alloc(8, headroom_units * 4, true).unwrap();
        assert!(
            p.capacity_units() <= headroom_units / 2,
            "one growth took {} of {headroom_units} units",
            p.capacity_units()
        );
        // A request that genuinely needs more than half still gets it.
        let mut q = PoolAllocator::new(4);
        let big = q.alloc(80_000, headroom_units * 4, true).unwrap();
        assert_eq!(big.len, 80_000);
    }

    #[test]
    fn growth_only_appends_so_live_blocks_keep_their_addresses() {
        let mut p = PoolAllocator::new(4);
        let a = p.alloc(10, ROOMY, true).unwrap();
        let cap0 = p.capacity_units();
        let big = p.alloc(cap0 * 8, ROOMY, true).unwrap();
        assert!(p.capacity_units() > cap0);
        assert_eq!(a.offset, 0, "the first block never moves");
        assert!(big.offset >= a.end());
    }

    #[test]
    fn zero_length_sections_allocate_trivially() {
        let mut p = PoolAllocator::new(4);
        let z = p.alloc(0, ROOMY, true).unwrap();
        assert_eq!(z.len, 0);
        assert_eq!(p.used_units(), 0);
        p.free(z); // a no-op, not an underflow
        assert_eq!(p.used_units(), 0);
    }

    /// **The floor guarantee, swept exhaustively.** For any page shape, a budget
    /// equal to that page's `resident_bytes()` reserves it.
    ///
    /// This is the regression test for the defect the P18.2 audit found. Growth
    /// used to overshoot unconditionally — a first growth rounded each section up
    /// to `INITIAL_UNITS` (1024) and charged the slack to the shared budget — so a
    /// page whose sections were smaller than that overshoot could not be reserved
    /// even though the ledger said it fit. For the always-resident root page that
    /// is not a detail: `VgeomNode` skips an asset with zero resident pages, so the
    /// object *silently vanished* from the frame instead of degrading to coarser
    /// detail, and it did so across a wide band of ordinary meshlet counts.
    ///
    /// The sweep therefore walks the whole 1..=2048 band — under, across and over
    /// the 1024-unit overshoot boundary — rather than sampling one shape.
    #[test]
    fn a_page_fits_whenever_the_budget_equals_its_bytes() {
        // (vertices, meshlets, mlverts, mltri words) — the four section sizes.
        let shapes = |n: u64| {
            [
                [n, n, n, n],
                // Realistic ratios: ~64 verts and ~124 tris per meshlet.
                [n * 62, n, n * 64, n * 93],
                // Degenerate mixes that a coarse page really does produce.
                [n, 1, 1, 1],
                [1, n, 1, 1],
                [0, n, 0, 0],
            ]
        };
        let bytes = |s: &[u64; 4]| s[0] * 32 + s[1] * 64 + s[2] * 4 + s[3] * 4;
        // The band the audit named, plus the boundary and well past it.
        let counts: Vec<u64> = (1..=48)
            .chain([
                63, 64, 65, 100, 511, 512, 513, 700, 1000, 1023, 1024, 1025, 1500, 2048,
            ])
            .collect();
        for n in counts {
            for shape in shapes(n) {
                let budget = bytes(&shape);
                if budget == 0 {
                    continue;
                }
                let mut pools = VgeomPools::new(budget);
                let got = pools.alloc_page(shape);
                assert!(
                    got.is_some(),
                    "a {budget}-byte budget must hold a {shape:?} page (n = {n})"
                );
                assert!(
                    pools.capacity_bytes() <= budget,
                    "capacity {} passed the {budget}-byte budget for {shape:?}",
                    pools.capacity_bytes()
                );
                // The blocks are real and disjoint within each pool.
                let blocks = got.unwrap();
                for (i, b) in blocks.iter().enumerate() {
                    assert_eq!(b.len, shape[i], "section {i} short-changed");
                }
            }
        }
    }

    /// A page one byte too big for the budget fails — cleanly, leaving the pools
    /// exactly as they were, so the streamer's next attempt is not poisoned by a
    /// half-reserved page.
    #[test]
    fn a_page_that_does_not_fit_rolls_back_completely() {
        let shape = [62u64, 1, 64, 93];
        let bytes = shape[0] * 32 + shape[1] * 64 + shape[2] * 4 + shape[3] * 4;
        let mut pools = VgeomPools::new(bytes - 4);
        assert!(pools.alloc_page(shape).is_none(), "must not fit");
        assert_eq!(pools.used_bytes(), 0, "a failed page leaves nothing behind");
        // And with one more byte of budget it does fit.
        let mut pools = VgeomPools::new(bytes);
        assert!(pools.alloc_page(shape).is_some());
    }

    /// Speculation still happens when there is room for it — the retry is a safety
    /// net, not a replacement for amortized growth. A roomy budget must leave the
    /// pools with slack to absorb the next page without another reallocation.
    #[test]
    fn a_roomy_budget_still_grows_speculatively() {
        let shape = [62u64, 1, 64, 93];
        let bytes = shape[0] * 32 + shape[1] * 64 + shape[2] * 4 + shape[3] * 4;
        let mut pools = VgeomPools::new(bytes * 4096);
        pools.alloc_page(shape).expect("fits with room to spare");
        assert!(
            pools.capacity_bytes() > pools.used_bytes(),
            "a roomy budget should over-allocate so the next page needs no growth"
        );
        let capacity = pools.capacity_bytes();
        pools.alloc_page(shape).expect("second page fits");
        assert_eq!(
            pools.capacity_bytes(),
            capacity,
            "the second page must come out of the slack, not a new growth"
        );
    }

    /// The four pools share ONE budget, and total capacity can never pass it —
    /// whichever mix of sections the content happens to need. This is what
    /// replaced a fixed per-pool split, which made some pool rather than the
    /// budget the binding constraint (see [`VgeomPools`]).
    #[test]
    fn the_four_pools_share_one_budget() {
        const BUDGET: u64 = 1 << 16;
        let mut pools = VgeomPools::new(BUDGET);
        assert_eq!(pools.limit_bytes(), BUDGET);
        assert_eq!(pools.capacity_bytes(), 0, "pools start empty");
        // Hammer every pool in turn until nothing more can be had.
        let kinds = [
            PoolKind::Vertices,
            PoolKind::Meshlets,
            PoolKind::Mlverts,
            PoolKind::Mltris,
        ];
        let mut granted = 0u64;
        for round in 0..64 {
            for k in kinds {
                if let Some(b) = pools.alloc(k, 8 + round) {
                    granted += b.len;
                }
                assert!(
                    pools.capacity_bytes() <= BUDGET,
                    "capacity {} passed the {BUDGET}-byte budget",
                    pools.capacity_bytes()
                );
            }
        }
        assert!(granted > 0, "a 64 KiB budget must serve something");
        assert!(pools.used_bytes() <= pools.capacity_bytes());
    }

    // ── wants ───────────────────────────────────────────────────────────────

    #[test]
    fn ideal_page_count_is_monotone_and_never_zero() {
        let m = dense_mesh(24);
        let src = VgeomSource::from_mesh(&m).unwrap();
        let pages = src.pages();
        assert_eq!(
            ideal_page_count(pages, 0.0),
            pages.len(),
            "a zero threshold wants every page"
        );
        assert_eq!(
            ideal_page_count(pages, f32::MAX),
            1,
            "a distant instance wants only the roots"
        );
        let mut prev = pages.len();
        let mut t = 0.0f32;
        while t < 10.0 {
            let n = ideal_page_count(pages, t);
            assert!(n <= prev, "want must not grow with the threshold");
            assert!(n >= 1);
            prev = n;
            t += 0.001;
        }
        assert_eq!(ideal_page_count(&[], 1.0), 0);
    }

    // ── the streamer ────────────────────────────────────────────────────────

    fn fixture() -> (VgeomMesh, VgeomSource) {
        let m = dense_mesh(28);
        let s = VgeomSource::from_mesh(&m).unwrap();
        (m, s)
    }

    fn want<'a>(src: &'a VgeomSource, t: f32) -> VgeomWant<'a> {
        VgeomWant {
            asset: 1,
            source: src,
            threshold: t,
        }
    }

    /// A displaced grid with a tunable amplitude — the same topology, so the
    /// clusterization (and therefore the meshlet and page **counts**) is
    /// unchanged, while every simplification error differs. That is what a
    /// re-import of an edited mesh looks like, and it is precisely the case the
    /// pre-P18.3 count-only staleness check could not see.
    fn displaced_mesh(n: usize, amp: f32) -> VgeomMesh {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        for j in 0..=n {
            for i in 0..=n {
                let u = i as f32 / n as f32;
                let v = j as f32 / n as f32;
                let x = (u - 0.5) * 2.0;
                let z = (v - 0.5) * 2.0;
                positions.push([x, amp * (x * 3.0).sin() * (z * 3.0).cos(), z]);
                normals.push([0.0, 1.0, 0.0]);
                uvs.push([u, v]);
            }
        }
        let stride = (n + 1) as u32;
        let mut indices = Vec::new();
        for j in 0..n as u32 {
            for i in 0..n as u32 {
                let a = j * stride + i;
                indices.extend_from_slice(&[
                    a,
                    a + stride,
                    a + 1,
                    a + 1,
                    a + stride,
                    a + stride + 1,
                ]);
            }
        }
        crate::build::build_vgeom(
            &positions,
            &normals,
            &uvs,
            &indices,
            crate::build::BuildParams::default(),
        )
    }

    /// **Re-imported content under a stable id must reset residency** (P18.3).
    ///
    /// In a cooked pack an asset id names immutable bytes, so the old staleness
    /// check — meshlet count + page *count* — was enough. The editor's content
    /// root is not immutable: a re-import hands the same id a different payload,
    /// and an edited mesh with unchanged topology produces a DAG that agrees on
    /// both counts while disagreeing on every error in the directory. Patching
    /// residency across that draws the OLD geometry out of pool blocks staged for
    /// it, under the NEW source's cut — a stale render no other test would catch.
    ///
    /// The fixture is asserted count-identical, so this can never quietly become
    /// vacuous by drifting into a pair the weak check would already separate.
    #[test]
    fn reimported_content_under_one_id_resets_residency() {
        let a = VgeomSource::from_mesh(&displaced_mesh(28, 0.30)).unwrap();
        let b = VgeomSource::from_mesh(&displaced_mesh(28, 0.75)).unwrap();
        assert_eq!(
            (a.meshlet_count(), a.pages().len()),
            (b.meshlet_count(), b.pages().len()),
            "fixture must defeat the count-only check, or this proves nothing"
        );
        assert_ne!(a.pages(), b.pages(), "…while differing in the directory");

        let mut s = VgeomStreamer::new(VgeomStreamBudget::default());
        s.plan(&[want(&a, 0.0)]);
        assert!(s.residency(1).unwrap().resident_pages() > 0);

        // The same id, different bytes.
        let plan = s.plan(&[want(&b, 0.0)]);
        let after = s.residency(1).unwrap();
        assert_eq!(
            after.pages(),
            b.pages(),
            "residency must describe the NEW source, not be patched onto the old"
        );
        assert!(
            !plan.uploads.is_empty(),
            "the new payload must actually be staged"
        );
        // Every resident meshlet maps into a slot allocated for the new source.
        let mut seen = BTreeSet::new();
        for &slot in after.remap() {
            if slot != NOT_RESIDENT {
                assert!(
                    seen.insert(slot),
                    "pool slot {slot} used twice after a swap"
                );
            }
        }
    }

    /// An unlimited budget converges to full residency in ONE plan — the property
    /// the goldens' byte-identity rests on (a cold frame is fully paged in).
    #[test]
    fn unlimited_budget_converges_in_one_plan() {
        let (_, src) = fixture();
        let mut s = VgeomStreamer::new(VgeomStreamBudget::default());
        let plan = s.plan(&[want(&src, 0.0)]);
        let res = s.residency(1).unwrap();
        assert_eq!(
            res.resident_pages(),
            src.pages().len(),
            "all pages resident"
        );
        assert_eq!(res.floor_lod(), 0, "the floor is the finest level");
        assert_eq!(plan.uploads.len(), src.pages().len());
        assert!(plan.failed.is_empty());
        assert!(!s.stats().budget_clamped);
        assert_eq!(s.stats().pending_pages, 0);
        assert_eq!(s.stats().clamped_assets, 0);
        // Every meshlet has a pool slot, and the slots are distinct.
        let mut seen = BTreeSet::new();
        for (i, &slot) in res.remap().iter().enumerate() {
            assert_ne!(slot, NOT_RESIDENT, "meshlet {i} unmapped at full residency");
            assert!(seen.insert(slot), "pool slot {slot} used twice");
        }
    }

    /// A second plan with the same wants does nothing — no churn, no reallocation.
    #[test]
    fn a_converged_plan_is_idle() {
        let (_, src) = fixture();
        let mut s = VgeomStreamer::new(VgeomStreamBudget::default());
        s.plan(&[want(&src, 0.0)]);
        let gen = s.residency(1).unwrap().generation();
        let plan = s.plan(&[want(&src, 0.0)]);
        assert!(plan.uploads.is_empty() && plan.evicted.is_empty());
        assert!(!plan.pools_grew);
        assert_eq!(s.residency(1).unwrap().generation(), gen, "remap untouched");
    }

    /// A distant instance keeps only the roots — and page 0 is never evicted, so
    /// the floor holds.
    #[test]
    fn a_distant_instance_keeps_only_the_root_floor() {
        let (m, src) = fixture();
        let mut s = VgeomStreamer::new(VgeomStreamBudget::default());
        s.plan(&[want(&src, f32::MAX / 4.0)]);
        let res = s.residency(1).unwrap();
        assert_eq!(res.resident_pages(), 1, "only the root page");
        assert_eq!(res.floor_lod(), src.header().max_lod);
        // Every root is mapped; nothing else is.
        for (i, ml) in m.meshlets.iter().enumerate() {
            assert_eq!(
                res.remap()[i] != NOT_RESIDENT,
                ml.is_root(),
                "meshlet {i} residency must be exactly the root set"
            );
        }
    }

    /// The byte budget is a **hard** bound, and the floor still gets in.
    #[test]
    fn byte_budget_is_a_hard_bound_with_a_coarse_floor() {
        let (_, src) = fixture();
        let floor = src.pages()[0].resident_bytes();
        // Just enough for the floor plus a sliver.
        let budget = floor + floor / 2;
        let mut s = VgeomStreamer::new(VgeomStreamBudget {
            budget_bytes: budget,
            ..Default::default()
        });
        s.plan(&[want(&src, 0.0)]);
        let res = s.residency(1).unwrap();
        assert!(res.resident_pages() >= 1, "the floor is always granted");
        assert!(
            res.resident_pages() < src.pages().len(),
            "a tight budget must force partial residency"
        );
        assert!(res.resident_bytes() <= budget, "budget exceeded");
        assert!(s.pools().used_bytes() <= s.pools().limit_bytes());
        assert!(s.stats().budget_clamped);
        assert!(s.stats().clamped_assets > 0, "the clamp is reported");
    }

    /// Hysteresis: pulling back slightly does not immediately drop a page, but
    /// pulling back far does.
    #[test]
    fn hysteresis_holds_pages_through_small_moves() {
        let (_, src) = fixture();
        let mut s = VgeomStreamer::new(VgeomStreamBudget {
            hysteresis: 4.0,
            ..Default::default()
        });
        // Find a threshold where the want is strictly between the extremes.
        let pages = src.pages();
        let mid = pages[pages.len() / 2].max_parent_error;
        s.plan(&[want(&src, 0.0)]);
        let full = s.residency(1).unwrap().resident_pages();
        // A threshold whose plain want is coarser, but inside the hysteresis band
        // of the finer one, keeps what it had.
        s.plan(&[want(&src, mid / 8.0)]);
        assert_eq!(
            s.residency(1).unwrap().resident_pages(),
            full,
            "a small pull-back must not evict"
        );
        // A big pull-back does evict.
        s.plan(&[want(&src, f32::MAX / 4.0)]);
        assert!(s.residency(1).unwrap().resident_pages() < full);
    }

    /// `max_loads_per_sync` bounds the batch, and every intermediate state is
    /// still a valid prefix — the property that makes a partial frame legal.
    #[test]
    fn bounded_loads_converge_through_valid_prefixes() {
        let (_, src) = fixture();
        let mut s = VgeomStreamer::new(VgeomStreamBudget {
            max_loads_per_sync: 1,
            ..Default::default()
        });
        let total = src.pages().len();
        for step in 1..=total {
            s.plan(&[want(&src, 0.0)]);
            let res = s.residency(1).unwrap();
            assert_eq!(res.resident_pages(), step.min(total));
            // A prefix, always: blocks exist for exactly the resident pages.
            for p in 0..res.resident_pages() {
                assert!(res.blocks(p).is_some());
            }
            assert!(res.blocks(res.resident_pages()).is_none());
        }
        assert_eq!(s.residency(1).unwrap().resident_pages(), total);
    }

    /// **The floor is not negotiable, end to end.** Whenever the budget can hold
    /// the root page at all, the root page is resident — so "softer detail, never a
    /// hole" has no unstated exception hiding under a tight budget.
    ///
    /// The fixture is deliberately **wide**: 160 x 160 quads is ~1 300 meshlets over
    /// nine pages, so the per-page reservations straddle the allocator's 1 024-unit
    /// speculative-growth boundary in both directions. The narrow fixture the rest
    /// of this module uses never reaches it, which is how the defect this pins
    /// survived the first round of tests: growth over-allocated each section up to
    /// 1 024 units and charged the slack to the shared budget, so an asset whose
    /// root page was smaller than the overshoot ended with **zero** resident pages —
    /// and `VgeomNode` skips an asset with no residency, so the object silently
    /// vanished rather than degrading. (`a_page_fits_whenever_the_budget_equals_its_bytes`
    /// sweeps the same property at the allocator, exhaustively and in microseconds;
    /// this one proves it survives the streamer, the want rules and the budget
    /// auction on top.)
    #[test]
    fn the_root_page_survives_every_budget_that_can_hold_it() {
        let mesh = dense_mesh(160);
        let src = VgeomSource::from_mesh(&mesh).unwrap();
        assert!(
            src.meshlet_count() > 1024,
            "the fixture must straddle the speculative-growth boundary, got {}",
            src.meshlet_count()
        );
        assert!(src.pages().len() >= 8, "and be deep enough to page");
        let floor_bytes = src.pages()[0].resident_bytes();
        let total = src.total_resident_bytes();

        // Every budget from "exactly the floor" up to the whole asset, including
        // the awkward sizes either side of the floor and a fine sweep of the band
        // the allocator's overshoot used to break.
        let mut budgets: Vec<u64> = vec![floor_bytes, floor_bytes + 1, floor_bytes + 7];
        for d in 1..=32u64 {
            budgets.push((total / d).max(floor_bytes));
        }
        for k in 1..=64u64 {
            budgets.push(floor_bytes + k * 1024);
        }
        budgets.sort_unstable();
        budgets.dedup();

        for b in budgets {
            let mut s = VgeomStreamer::new(VgeomStreamBudget {
                budget_bytes: b,
                ..Default::default()
            });
            s.plan(&[VgeomWant {
                asset: 1,
                source: &src,
                threshold: 0.0,
            }]);
            let res = s.residency(1).expect("the asset registered");
            assert!(
                res.resident_pages() >= 1,
                "a {b}-byte budget must still hold the {floor_bytes}-byte root page"
            );
            assert!(
                res.resident_bytes() <= b,
                "resident {} exceeds the {b}-byte budget",
                res.resident_bytes()
            );
            assert!(s.pools().capacity_bytes() <= s.pools().limit_bytes());
            // The remap really maps every root, so the cut has something to select
            // on every root-to-leaf path.
            let mapped = res.remap().iter().filter(|&&x| x != NOT_RESIDENT).count();
            assert!(
                mapped >= src.pages()[0].meshlet_count as usize,
                "a {b}-byte budget mapped {mapped} meshlets, fewer than the {} roots",
                src.pages()[0].meshlet_count
            );
        }

        // Below the floor the asset is honestly empty — and reports it as zero
        // resident pages rather than pretending the roots are up.
        let mut s = VgeomStreamer::new(VgeomStreamBudget {
            budget_bytes: floor_bytes / 4,
            ..Default::default()
        });
        s.plan(&[VgeomWant {
            asset: 1,
            source: &src,
            threshold: 0.0,
        }]);
        assert_eq!(s.residency(1).unwrap().resident_pages(), 0);
        assert!(s.stats().budget_clamped);
    }

    /// Two identical want traces produce identical resident-set traces — the
    /// determinism gate, in miniature and with no GPU.
    #[test]
    fn identical_want_traces_produce_identical_residency_traces() {
        let (_, src) = fixture();
        let thresholds: Vec<f32> = (0..40)
            .map(|i| (i as f32 * 0.37).sin().abs() * 0.05 + 0.0005)
            .collect();
        let trace = || {
            let mut s = VgeomStreamer::new(VgeomStreamBudget {
                budget_bytes: src.total_resident_bytes() / 2,
                max_loads_per_sync: 2,
                ..Default::default()
            });
            let mut out = Vec::new();
            for &t in &thresholds {
                let plan = s.plan(&[want(&src, t)]);
                let res = s.residency(1).unwrap();
                out.push((
                    res.resident_pages(),
                    res.floor_lod(),
                    res.resident_bytes(),
                    plan.uploads.iter().map(|u| u.page).collect::<Vec<_>>(),
                    plan.evicted.clone(),
                    res.remap().to_vec(),
                ));
            }
            out
        };
        assert_eq!(trace(), trace(), "the resident-set trace is reproducible");
    }

    /// An asset that leaves the frame is fully freed and the pools take the space
    /// back — no leak across level changes.
    #[test]
    fn leaving_the_frame_frees_everything() {
        let (_, src) = fixture();
        let mut s = VgeomStreamer::new(VgeomStreamBudget::default());
        s.plan(&[want(&src, 0.0)]);
        assert!(s.pools().used_bytes() > 0);
        let plan = s.plan(&[]);
        assert_eq!(plan.dropped, vec![1]);
        assert!(s.residency(1).is_none());
        assert_eq!(s.pools().used_bytes(), 0, "every block returned");
        // And it can come back.
        s.plan(&[want(&src, 0.0)]);
        assert_eq!(
            s.residency(1).unwrap().resident_pages(),
            src.pages().len(),
            "a returning asset re-pages cleanly"
        );
    }

    /// Two assets share one budget: the closer one (smaller threshold) refines
    /// first, and the auction is stable.
    #[test]
    fn the_closer_asset_wins_the_budget() {
        let (_, src) = fixture();
        let budget = src.pages()[0].resident_bytes() * 2 + src.total_resident_bytes() / 3;
        let mut s = VgeomStreamer::new(VgeomStreamBudget {
            budget_bytes: budget,
            ..Default::default()
        });
        // Both want *every* page (their thresholds sit below the finest page's
        // boundary), so the budget — not the want — is what separates them.
        let finest = src.pages().last().unwrap().max_parent_error;
        let near = VgeomWant {
            asset: 1,
            source: &src,
            threshold: finest / 100.0,
        };
        let far = VgeomWant {
            asset: 2,
            source: &src,
            threshold: finest / 2.0,
        };
        assert_eq!(
            ideal_page_count(src.pages(), near.threshold),
            src.pages().len()
        );
        assert_eq!(
            ideal_page_count(src.pages(), far.threshold),
            src.pages().len()
        );
        s.plan(&[near, far]);
        let a = s.residency(1).unwrap().resident_pages();
        let b = s.residency(2).unwrap().resident_pages();
        assert!(a >= 1 && b >= 1, "both keep their floor");
        assert!(a > b, "the closer asset refines further ({a} vs {b})");
    }
}
