//! **Soil displacement** (P21.3): excavated voxels do not vanish — they become a
//! spoil pile somewhere else, and the count balances **exactly**.
//!
//! # The conservation law, stated once
//!
//! `spoiled[m] == removed[m]` for every material `m`, as **integers**. Not
//! approximately, not within a tolerance, and with **no bulking factor** — real
//! soil expands when you dig it and this model deliberately does not, because a
//! 1:1 identity is a gate and a 1.25× fudge is a number nobody can test. (Bulking
//! is a documented non-goal; when it arrives it will be a declared multiplier
//! applied to a still-exact count, not a rounding.)
//!
//! [`VoxelData::apply_op`](crate::VoxelData::apply_op) already returns exact
//! per-material transition counts — that is the whole reason [`OpReport`] counts
//! rather than estimates. This module is the other half: given
//! `counts[m]` voxels of material `m` and a place to put them, it makes
//! **exactly** `counts[m]` samples of material `m` go empty → solid.
//!
//! The erosion mass gates (`inf_terrain::erosion`) are the model this follows:
//! the identity is the test, the test is over random inputs, and an
//! implementation that can only *nearly* balance is a wrong implementation.
//!
//! # The pile, and why it is a rank order rather than a shape
//!
//! An analytic mound cannot hold an arbitrary integer count — a cone of radius
//! `R` contains whatever number of lattice points it contains, and the next
//! radius up contains a different one. So the pile is defined as an **order**,
//! not as a solid:
//!
//! ```text
//! rank(cell) = horizontal_distance(cell, base) · tan(θ) + height_above(base)
//! ```
//!
//! with `θ` the **angle of repose** ([`REPOSE_DEG`], 35° — loose earth). `rank`
//! is the apex height of the smallest repose cone that contains the cell, so
//! taking cells in ascending rank grows a cone monotonically: the pile after `k`
//! placements is the smallest cone holding `k` cells, plus a partial outermost
//! shell. Ties break on `(height, x, z)`, which is a **total** order on distinct
//! lattice cells — so the pile is a pure function of `(base, count)` and two
//! sessions build the same mound.
//!
//! Three rules make the identity exact rather than approximate:
//!
//! * **A cell that is already solid is not a placement.** Piling onto rock adds
//!   nothing to the solid set, so those cells are skipped and the pile grows
//!   around them. This is what makes a spoil site dropped on a hillside conserve
//!   as exactly as one dropped on a plain.
//! * **The candidate region grows until it can hold the count.** There is no
//!   "closest fit" and no rounding: the search radius is enlarged and retried
//!   until at least `total` placeable cells exist, and then exactly `total` of
//!   them are used. The truncation lands mid-shell, which is what a real dumped
//!   pile looks like.
//! * **Materials are laid down in ascending index order**, each taking a
//!   contiguous run of the rank order. A dig through two strata therefore builds
//!   a visibly layered mound, bottom-up, and the layering is deterministic.
//!
//! # Units and determinism
//!
//! World **metres**, `f64`, `y` up. No `std` trig and no `f64::cbrt`: `tan(θ)`
//! and `cos(θ)` are frozen literals and the cube root is [`cbrt_det`], a
//! fixed-iteration Newton solve in pure IEEE arithmetic. libm is not bit-portable
//! (the P14 law that put `psin64`/`pcos64` in `inf-math`), and these numbers
//! decide where a **committed** pile stands.

use glam::DVec3;

use crate::chunk::{clamp_sdf, ChunkKey, CHUNK_DIM, MATERIAL_COUNT, SDF_BAND};
use crate::data::{local_of, VoxelData};
use crate::ops::{integer_box, VoxelDeltaBuilder};
use crate::residency::ChunkStore;

/// The angle of repose used for spoil, **degrees** — loose excavated earth.
///
/// Documentary: nothing computes a trig function from it (see [`REPOSE_TAN`]).
/// It is here so the two frozen literals below can be checked against a table
/// by a human, and by [`the_repose_constants_are_the_stated_angle`].
pub const REPOSE_DEG: f64 = 35.0;

/// `tan(35°)`, frozen. The pile's slope: one metre up per `1/REPOSE_TAN` metres
/// out.
pub const REPOSE_TAN: f64 = 0.700_207_538_209_709_7;

/// `cos(35°)`, frozen — the factor that turns the rank difference (a *vertical*
/// measure) into a distance perpendicular to the cone's slanted face.
pub const REPOSE_COS: f64 = 0.819_152_044_288_991_8;

/// Clear air left between the cut's east face and the pile's own footprint in
/// the [`default_spoil_site`] rule, metres. A pile that touches the rim slumps
/// back into the hole.
pub const SPOIL_GAP_M: f64 = 1.0;

/// How far a placed sample sits inside the pile's surface, **voxels**.
///
/// The outermost shell of the pile would otherwise land at `sdf == 0`, and `0`
/// is not solid (the crate-wide convention) — so the field would carry fewer
/// solid samples than the report claims and the conservation identity would be
/// off by the size of the rim. A quarter voxel is enough to be unambiguous and
/// small enough that the meshed surface still lands where the cone says.
const PLACE_DEPTH_VOXELS: f64 = 0.25;

/// How many times the candidate region may be enlarged before giving up.
///
/// Each step multiplies the apex height by [`GROWTH`], so the reachable radius
/// grows by `GROWTH^32` — past the `i32` sample lattice long before the counter
/// runs out. Reaching the cap means the volume is solid for kilometres in every
/// direction, which [`SpoilReport::shortfall`] reports rather than panicking.
const MAX_GROWTH_STEPS: u32 = 32;

/// Multiplier applied to the search apex on each growth step.
const GROWTH: f64 = 1.4;

/// What to displace, and where.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpoilPlan {
    /// Voxels to place, per material — normally an
    /// [`OpReport::carved`](crate::OpReport::carved), or the sum of a stroke's.
    pub counts: [u64; MATERIAL_COUNT],
    /// Where the pile's apex axis meets its base plane, world metres. The pile
    /// grows **upward** from `base.y`; nothing is written below it.
    pub base: DVec3,
    /// A world AABB the pile may **never** place into — the cut that produced
    /// it (P21.3 audit).
    ///
    /// # Why this is a hard exclusion and not a clearance
    ///
    /// [`default_spoil_site`] stands the heap clear of the cut by the pile's
    /// *analytic* radius, and the analytic radius is an estimate: the real
    /// footprint is `apex_actual / tan θ`, which the lattice, the truncated
    /// outermost shell and every already-solid cell the search grew around make
    /// **wider** — measured at 81 % wider on a real dig, which put 59 of 729
    /// excavated samples straight back into the hole they came out of. The
    /// conservation identity did not notice, because refilling the pit is a
    /// perfectly balanced place to put the soil.
    ///
    /// So the guarantee is structural rather than arithmetic: cells inside this
    /// box are not candidates, exactly as already-solid cells are not
    /// candidates, and the pile grows around it. `None` for a caller with no
    /// hole to protect (a scripted heap, a test).
    pub exclude: Option<(DVec3, DVec3)>,
}

impl SpoilPlan {
    /// A plan that displaces `counts` to `base`, with nothing excluded.
    pub fn new(counts: [u64; MATERIAL_COUNT], base: DVec3) -> Self {
        Self {
            counts,
            base,
            exclude: None,
        }
    }

    /// …and one that must not place inside `lo..hi` — the cut's own bounds.
    pub fn excluding(mut self, lo: DVec3, hi: DVec3) -> Self {
        self.exclude = (lo.is_finite() && hi.is_finite()).then_some((lo, hi));
        self
    }

    /// Total voxels to place.
    pub fn total(&self) -> u64 {
        self.counts.iter().sum()
    }

    /// `true` when there is nothing to displace.
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }
}

/// What a placement pass actually did.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SpoilReport {
    /// Samples made solid, per material. **Equal to the plan's counts** on every
    /// path that is not a shortfall.
    pub placed: [u64; MATERIAL_COUNT],
    /// Voxels the pile could not find room for, per material.
    ///
    /// **Always zero in practice** — the candidate region grows until it fits —
    /// and reported rather than asserted so a pathological volume degrades into
    /// a refusal the tools can quote instead of a panic mid-transaction.
    pub shortfall: [u64; MATERIAL_COUNT],
    /// Height of the finished pile above its base, metres.
    pub apex_height_m: f64,
    /// Radius of the pile's footprint, metres (`apex / tan θ`).
    pub base_radius_m: f64,
    /// Chunks the pile materialized from nothing.
    pub created_chunks: u64,
}

impl SpoilReport {
    /// Total samples made solid.
    pub fn total_placed(&self) -> u64 {
        self.placed.iter().sum()
    }

    /// `true` when every voxel in the plan found a home — **the conservation
    /// gate's own predicate**.
    pub fn is_exact(&self) -> bool {
        self.shortfall.iter().all(|&n| n == 0)
    }
}

/// The radius a repose cone needs to hold `voxels` cells at `voxel_size_m`.
///
/// From `V = (π/3)·R²·H` with `H = R·tan θ`, so `R = ∛(3V / (π·tan θ))`. Used
/// for the search's starting size and for the [`default_spoil_site`] clearance —
/// both tolerant of the analytic-vs-lattice discrepancy, and both required to be
/// **deterministic**, which is why the cube root is [`cbrt_det`].
pub fn pile_base_radius_m(voxels: u64, voxel_size_m: f64) -> f64 {
    if voxels == 0 || !voxel_size_m.is_finite() || voxel_size_m <= 0.0 {
        return 0.0;
    }
    let v = voxels as f64 * voxel_size_m * voxel_size_m * voxel_size_m;
    cbrt_det(3.0 * v / (std::f64::consts::PI * REPOSE_TAN))
}

/// The world AABB a spoil search of `apex` covers, from `base` upward.
///
/// One definition, used by the candidate gather, the band pass and the paging —
/// three walks that must cover the same region or the pile is placed against a
/// field one of them has not read.
pub fn spoil_search_region(base: DVec3, apex: f64) -> (DVec3, DVec3) {
    let radius = (apex / REPOSE_TAN).max(0.0);
    (
        DVec3::new(base.x - radius, base.y, base.z - radius),
        DVec3::new(base.x + radius, base.y + apex.max(0.0), base.z + radius),
    )
}

/// **The deterministic default spoil site**: due **east** of the cut, clear of
/// its rim, at the cut's own top height.
///
/// Stated as a rule so an author can predict it and a test can pin it:
///
/// > the pile stands centred on the cut's `+X` face, offset by its own footprint
/// > radius plus [`SPOIL_GAP_M`], at the height of the top of the cut.
///
/// `+X` and not "downhill", "the nearest flat ground" or "wherever there is
/// room": every one of those reads the world, and a rule that reads the world
/// makes the committed pile depend on what had paged in. This one is a function
/// of the cut's own bounds and the count, so two sessions that dig the same pit
/// pile the same soil in the same place — the house law that *a committed level
/// must not depend on which tools the session visited*.
///
/// The caller may lower `y` onto ground it knows about (`inf-voxel` has no
/// terrain); the height returned is the cut's top, which is grade for a pit dug
/// from the surface.
///
/// # The clearance is a HINT, not the guarantee
///
/// [`pile_base_radius_m`] is the *analytic* cone radius for the count, and the
/// real footprint is wider — by the lattice, by the truncated outermost shell,
/// and by every already-solid cell the search had to grow around (81 % wider on
/// the dig the P21.3 audit measured). So this offset only keeps the heap from
/// being jammed against the rim; what actually stops soil landing back in the
/// hole is [`SpoilPlan::exclude`], which removes the cut's cells from the
/// candidate set entirely.
pub fn default_spoil_site(cut_lo: DVec3, cut_hi: DVec3, voxels: u64, voxel_size_m: f64) -> DVec3 {
    let radius = pile_base_radius_m(voxels, voxel_size_m);
    DVec3::new(
        cut_hi.x + radius + SPOIL_GAP_M,
        cut_hi.y,
        (cut_lo.z + cut_hi.z) * 0.5,
    )
}

/// A **bit-portable** cube root: Newton's method, fixed iteration count, pure
/// IEEE arithmetic.
///
/// `f64::cbrt` is a libm call, and libm is not bit-portable across platforms —
/// the same law that banned `f32::sin` from committed content in P14. The pile's
/// radius decides where a saved spoil heap stands, so it goes through this.
///
/// The argument is scaled into `[1, 8)` by exact powers of eight (which are
/// exact in binary floating point, so the scaling introduces no error at all),
/// seeded linearly, refined by `y ← (2y + x/y²)/3`, and scaled back. Twelve
/// iterations from a seed within a factor of two is roughly twice what full
/// `f64` precision needs.
pub fn cbrt_det(x: f64) -> f64 {
    // `is_finite` and not merely `is_nan`: an infinite argument would spin the
    // scaling loop below forever (`m /= 8.0` never falls under 8.0), and both
    // this function and `pile_base_radius_m` are public re-exports (P21.3
    // audit).
    if !x.is_finite() || x <= 0.0 {
        return 0.0;
    }
    let mut m = x;
    let mut k: i32 = 0;
    while m >= 8.0 {
        m /= 8.0;
        k += 1;
    }
    while m < 1.0 {
        m *= 8.0;
        k -= 1;
    }
    // Seed: the chord of ∛ across [1, 8] → [1, 2].
    let mut y = 1.0 + (m - 1.0) / 7.0;
    for _ in 0..12 {
        y = (2.0 * y + m / (y * y)) / 3.0;
    }
    // 2^k, built by repeated exact doubling/halving rather than `powi` (which is
    // also fine, but this needs no assumption about how it is lowered).
    let mut scale = 1.0f64;
    for _ in 0..k.abs() {
        if k > 0 {
            scale *= 2.0;
        } else {
            scale /= 2.0;
        }
    }
    y * scale
}

/// One candidate lattice cell: its rank and its global sample coordinate.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    rank: f64,
    g: [i32; 3],
}

impl VoxelData {
    /// Place a [`SpoilPlan`]'s voxels as a repose mound, recording the writes
    /// into `builder` so they undo with the cut that produced them.
    ///
    /// **The whole point is the count.** See the module docs for the rank order,
    /// the already-solid rule and the growth rule; the contract here is that
    /// `report.placed == plan.counts` unless the volume is so full that the
    /// search cap was reached, in which case
    /// [`shortfall`](SpoilReport::shortfall) says by how much and the caller
    /// refuses the transaction rather than committing an unbalanced one.
    ///
    /// The `builder` is shared with the carve so that **one undo step reverts
    /// the cut, the cave mouths and the pile together**. A spoil pile that
    /// outlived its excavation would be material created from nothing.
    ///
    /// Opens no heightfield holes and closes none: a pile sits *on* the ground,
    /// it does not remove it. The hole mask is the carve's business alone.
    pub fn place_spoil_into(
        &mut self,
        plan: &SpoilPlan,
        builder: &mut VoxelDeltaBuilder,
    ) -> SpoilReport {
        self.place_spoil_inner(plan, builder, None)
    }

    /// [`place_spoil_into`](Self::place_spoil_into), **paging the search region
    /// out of `store` first** (P21.3 audit).
    ///
    /// The door every editor path must use. A non-resident chunk reads as
    /// [`EMPTY_SDF`](crate::EMPTY_SDF) — which is *correct* for a key the asset
    /// does not have (that really is air) and a **silent lie** for one it does
    /// (that is rock nobody paged in). Without this, a pile placed near unpaged
    /// rock would count cells it cannot see as placeable, materialize empty
    /// chunks over keys the `.inf_voxel` already holds, and the write-back would
    /// then overwrite real geometry with a heap.
    ///
    /// Paged **inside the growth loop**, one region per step, because the search
    /// region is what grows: paging the first estimate and then growing past it
    /// would restore the same hazard at the moment the pile needed more room.
    pub fn place_spoil_into_paged(
        &mut self,
        plan: &SpoilPlan,
        builder: &mut VoxelDeltaBuilder,
        store: &dyn ChunkStore,
    ) -> SpoilReport {
        self.place_spoil_inner(plan, builder, Some(store))
    }

    fn place_spoil_inner(
        &mut self,
        plan: &SpoilPlan,
        builder: &mut VoxelDeltaBuilder,
        store: Option<&dyn ChunkStore>,
    ) -> SpoilReport {
        let mut report = SpoilReport::default();
        let total = plan.total();
        if total == 0 {
            return report;
        }
        if !plan.base.is_finite() {
            report.shortfall = plan.counts;
            return report;
        }
        let h = self.voxel_size_m();
        // Start from the analytic cone that would hold the count, plus a little
        // slack for the lattice, and grow until enough PLACEABLE cells exist.
        let mut apex = (pile_base_radius_m(total, h) * REPOSE_TAN).max(h) + 2.0 * h;
        let mut chosen: Vec<Candidate> = Vec::new();
        for _ in 0..MAX_GROWTH_STEPS {
            if let Some(s) = store {
                self.page_spoil_region(plan.base, apex, s);
            }
            // `None` = the region outgrew [`MAX_SPOIL_SEARCH_CELLS`]. **Stop**
            // rather than keep multiplying (P21.3 audit, B1): every further step
            // is a bigger region that will be rejected the same way, and the
            // 32nd one is a `(i32 span)³` product that overflows `i64` and
            // panics — while holding the shared-volumes guard, mid-transaction,
            // with the cut already in the world and no `EditCommand` describing
            // it. Exactly the un-undoable committed edit `a4e5844` ruled worst.
            let Some(cands) = self.spoil_candidates(plan, apex, total) else {
                break;
            };
            if cands.len() as u64 >= total {
                chosen = cands;
                break;
            }
            apex *= GROWTH;
        }
        if (chosen.len() as u64) < total {
            // The search ran out of room (or out of steps): the volume is solid
            // for kilometres, or the count is larger than one pile may hold. Say
            // so rather than committing a pile that does not balance.
            report.shortfall = plan.counts;
            return report;
        }
        chosen.truncate(total as usize);
        // The finished pile's apex is the last cell's rank, not the search
        // radius — a search that overshot must not report a taller pile than it
        // built, and the SDF written below is measured against this one.
        let apex_actual = chosen.last().map(|c| c.rank).unwrap_or(0.0);
        report.apex_height_m = apex_actual;
        report.base_radius_m = apex_actual / REPOSE_TAN;

        // Materials in ascending index order, each a contiguous run of the rank
        // order — bottom-up layering, deterministic by construction.
        let mut next = 0usize;
        for (m, &want) in plan.counts.iter().enumerate() {
            for _ in 0..want {
                let c = chosen[next];
                next += 1;
                let key = ChunkKey::of_sample(c.g[0], c.g[1], c.g[2]);
                if !self.is_resident(key) {
                    builder.mark_created(key);
                    report.created_chunks += 1;
                }
                let old_sdf = self.sample_global(c.g[0], c.g[1], c.g[2]);
                let old_mat = self.material_global(c.g[0], c.g[1], c.g[2]);
                let (i, j, k) = local_of(c.g[0], c.g[1], c.g[2]);
                builder.touch(key, i, j, k, old_sdf, old_mat);
                // The cone's own distance, nudged inside so the outermost shell
                // is unambiguously solid (see `PLACE_DEPTH_VOXELS`).
                let d_v = (c.rank - apex_actual) * REPOSE_COS / h - PLACE_DEPTH_VOXELS;
                let v = clamp_sdf((old_sdf as f64).min(d_v) as f32);
                let chunk = self.get_or_create_chunk(key);
                chunk.set_sample(i, j, k, v);
                chunk.set_material(i, j, k, m as u8);
                report.placed[m] += 1;
            }
        }
        self.band_spoil(plan.base, apex_actual, builder);
        report
    }

    /// Gather every placeable cell whose rank is within `apex`, in the pile's
    /// order.
    ///
    /// "Placeable" is **not currently solid**: piling onto rock makes nothing
    /// newly solid, so those cells cannot count towards the balance and are left
    /// for the mound to grow around.
    fn spoil_candidates(&self, plan: &SpoilPlan, apex: f64, want: u64) -> Option<Vec<Candidate>> {
        let base = plan.base;
        let (lo_g, hi_g) = self.spoil_search_box(base, apex)?;
        let mut out: Vec<Candidate> = Vec::with_capacity((want as usize).min(1 << 20));
        for gx in lo_g[0]..=hi_g[0] {
            for gz in lo_g[2]..=hi_g[2] {
                // The column's horizontal distance decides whether ANY cell in
                // it can be in the cone — one `sqrt` per column, not per cell.
                let w = self.grid_to_world(DVec3::new(gx as f64, 0.0, gz as f64));
                let dx = w.x - base.x;
                let dz = w.z - base.z;
                let d = (dx * dx + dz * dz).sqrt();
                let slant = d * REPOSE_TAN;
                if slant > apex {
                    continue;
                }
                // The excluded box is tested in world space, once per column for
                // XZ and once per cell for Y.
                let in_xz = plan
                    .exclude
                    .is_some_and(|(l, h)| w.x >= l.x && w.x <= h.x && w.z >= l.z && w.z <= h.z);
                for gy in lo_g[1]..=hi_g[1] {
                    let wy = self.grid_to_world(DVec3::new(0.0, gy as f64, 0.0)).y;
                    let y = wy - base.y;
                    if y < 0.0 {
                        continue;
                    }
                    let rank = slant + y;
                    if rank > apex {
                        break; // the column only climbs from here
                    }
                    if self.sample_global(gx, gy, gz) < 0.0 {
                        continue; // already rock — piling on it places nothing
                    }
                    // …and never back into the hole this soil came out of.
                    if in_xz && plan.exclude.is_some_and(|(l, h)| wy >= l.y && wy <= h.y) {
                        continue;
                    }
                    out.push(Candidate {
                        rank,
                        g: [gx, gy, gz],
                    });
                }
            }
        }
        // The total order: rank, then height, then x, then z. Distinct cells
        // never tie on all four, so the sort is a function of the set and not of
        // the walk that produced it.
        out.sort_by(|a, b| {
            a.rank
                .total_cmp(&b.rank)
                .then(a.g[1].cmp(&b.g[1]))
                .then(a.g[0].cmp(&b.g[0]))
                .then(a.g[2].cmp(&b.g[2]))
        });
        Some(out)
    }

    /// Inclusive integer sample box a spoil search of `apex` covers, or `None`
    /// when it is empty or larger than [`MAX_SPOIL_SEARCH_CELLS`].
    ///
    /// **The cell count is computed with `checked_mul`** (P21.3 audit, B1).
    /// [`integer_box`] clamps each axis to `±(i32::MAX − 4)`, so three spans
    /// multiplied straight reach ~7.9 × 10²⁸ and wrap an `i64` long before they
    /// exceed the cap — the compare then passes and the walk runs forever, or
    /// the multiply itself panics in a debug build. A `None` here is what stops
    /// the growth loop instead.
    fn spoil_search_box(&self, base: DVec3, apex: f64) -> Option<([i32; 3], [i32; 3])> {
        let (lo_w, hi_w) = spoil_search_region(base, apex);
        let (lo_g, hi_g) = integer_box(self.world_to_grid(lo_w), self.world_to_grid(hi_w))?;
        let mut cells: u64 = 1;
        for a in 0..3 {
            let span = (hi_g[a] as i64 - lo_g[a] as i64 + 1).max(0) as u64;
            cells = cells.checked_mul(span)?;
            if cells > MAX_SPOIL_SEARCH_CELLS {
                return None;
            }
        }
        (cells > 0).then_some((lo_g, hi_g))
    }

    /// Make every chunk the search region touches resident, so a cell that is
    /// absent really is air rather than rock nobody paged in.
    fn page_spoil_region(&mut self, base: DVec3, apex: f64, store: &dyn ChunkStore) {
        let Some((lo_g, hi_g)) = self.spoil_search_box(base, apex) else {
            return;
        };
        let lo = ChunkKey::of_sample(lo_g[0], lo_g[1], lo_g[2]);
        let hi = ChunkKey::of_sample(hi_g[0], hi_g[1], hi_g[2]);
        // The cell cap above already bounds the sample box; a chunk box is that
        // divided by `CHUNK_DIM³`, so this set is small by construction.
        let span = |a: i32, b: i32| (b as i64 - a as i64 + 1).max(0) as u64;
        let chunks = span(lo.x, hi.x)
            .saturating_mul(span(lo.y, hi.y))
            .saturating_mul(span(lo.z, hi.z));
        if chunks == 0 || chunks > MAX_SPOIL_SEARCH_CELLS / (CHUNK_DIM as u64).pow(3) + 8 {
            return;
        }
        let wants: std::collections::BTreeSet<ChunkKey> =
            crate::residency::chunk_range(lo, hi).into_iter().collect();
        self.request_chunks(&wants, store);
    }

    /// Write the pile's **outside** distance band, so the mesher has a gradient
    /// to interpolate against and the mound reads as a mound rather than as a
    /// staircase of cubes.
    ///
    /// Strictly `min(old, positive)`: it can only move a sample *towards* the
    /// surface from outside, never across it, so it cannot create or destroy a
    /// solid sample and the conservation identity is untouched. It also **never
    /// materializes a chunk** — an absent chunk already reads
    /// [`EMPTY_SDF`](crate::EMPTY_SDF), and creating one to store `3.9` instead
    /// would cost a whole chunk of storage for a gradient nobody can see.
    fn band_spoil(&mut self, base: DVec3, apex: f64, builder: &mut VoxelDeltaBuilder) {
        if apex.is_nan() || apex <= 0.0 {
            return;
        }
        let h = self.voxel_size_m();
        let margin = SDF_BAND as f64 * h / REPOSE_COS;
        let reach = apex + margin;
        // The same checked box the candidate gather uses — one place where three
        // `i32` spans become a cell count, so one place that can overflow.
        let Some((lo_g, hi_g)) = self.spoil_search_box(base, reach) else {
            return;
        };
        for gx in lo_g[0]..=hi_g[0] {
            for gz in lo_g[2]..=hi_g[2] {
                let w = self.grid_to_world(DVec3::new(gx as f64, 0.0, gz as f64));
                let dx = w.x - base.x;
                let dz = w.z - base.z;
                let slant = (dx * dx + dz * dz).sqrt() * REPOSE_TAN;
                if slant > reach {
                    continue;
                }
                for gy in lo_g[1]..=hi_g[1] {
                    let y = self.grid_to_world(DVec3::new(0.0, gy as f64, 0.0)).y - base.y;
                    if y < 0.0 {
                        continue;
                    }
                    let rank = slant + y;
                    if rank > reach {
                        break;
                    }
                    let d_v = (rank - apex) * REPOSE_COS / h;
                    if d_v <= 0.0 {
                        continue; // inside the cone — the placement pass owns it
                    }
                    let key = ChunkKey::of_sample(gx, gy, gz);
                    if !self.is_resident(key) {
                        continue;
                    }
                    let old = self.sample_global(gx, gy, gz);
                    let v = clamp_sdf((old as f64).min(d_v) as f32);
                    if v == old {
                        continue;
                    }
                    let mat = self.material_global(gx, gy, gz);
                    let (i, j, k) = local_of(gx, gy, gz);
                    builder.touch(key, i, j, k, old, mat);
                    if let Some(chunk) = self.get_chunk_mut(key) {
                        chunk.set_sample(i, j, k, v);
                    }
                }
            }
        }
    }
}

/// Hard ceiling on the lattice cells one spoil pass may walk.
///
/// The caller's dig budget already bounds the count; this bounds the *search*,
/// so a base point dropped a kilometre from the volume cannot turn into an
/// unbounded scan. Sixteen million cells is a 250-metre cone at half-metre
/// voxels — far past anything an editor gesture produces.
const MAX_SPOIL_SEARCH_CELLS: u64 = 16_777_216;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{VoxelChunk, CHUNK_DIM};
    use crate::ops::{VoxelOp, VoxelShape};
    use crate::residency::chunk_range;

    fn empty(voxel_m: f64) -> VoxelData {
        VoxelData::new(voxel_m)
    }

    /// Every resident chunk's bytes — the "byte-identical" object.
    fn image(v: &VoxelData) -> Vec<(ChunkKey, Vec<u8>)> {
        v.chunks()
            .map(|(&k, c)| (k, crate::asset::encode_chunk(c).unwrap()))
            .collect()
    }

    /// Independently counted solid samples across every resident chunk.
    fn solid_count(v: &VoxelData) -> u64 {
        v.chunks()
            .map(|(_, c)| {
                (0..CHUNK_DIM)
                    .flat_map(|k| {
                        (0..CHUNK_DIM).flat_map(move |j| (0..CHUNK_DIM).map(move |i| (i, j, k)))
                    })
                    .filter(|&(i, j, k)| c.sample(i, j, k) < 0.0)
                    .count() as u64
            })
            .sum()
    }

    /// …and by material.
    fn solid_by_material(v: &VoxelData) -> [u64; MATERIAL_COUNT] {
        let mut out = [0u64; MATERIAL_COUNT];
        for (_, c) in v.chunks() {
            for k in 0..CHUNK_DIM {
                for j in 0..CHUNK_DIM {
                    for i in 0..CHUNK_DIM {
                        if c.sample(i, j, k) < 0.0 {
                            out[(c.material(i, j, k) as usize).min(MATERIAL_COUNT - 1)] += 1;
                        }
                    }
                }
            }
        }
        out
    }

    fn place(v: &mut VoxelData, counts: [u64; MATERIAL_COUNT], base: DVec3) -> SpoilReport {
        let mut b = VoxelDeltaBuilder::new();
        v.place_spoil_into(&SpoilPlan::new(counts, base), &mut b)
    }

    /// **The conservation gate, in its purest form.** Whatever the count, the
    /// pile holds exactly it — per material — and the solid set grew by exactly
    /// that many samples.
    #[test]
    fn a_pile_holds_exactly_the_voxels_it_was_given() {
        for counts in [
            [1, 0, 0, 0],
            [7, 0, 0, 0],
            [64, 0, 0, 0],
            [523, 0, 0, 0],
            [1000, 250, 33, 4],
            [0, 0, 0, 991],
        ] {
            let mut v = empty(0.5);
            let before = solid_count(&v);
            let report = place(&mut v, counts, DVec3::new(10.0, 3.0, -4.0));
            assert!(report.is_exact(), "{counts:?}: {report:?}");
            assert_eq!(report.placed, counts, "{counts:?}");
            let after = solid_by_material(&v);
            assert_eq!(
                after, counts,
                "{counts:?}: the FIELD does not carry the count"
            );
            assert_eq!(
                solid_count(&v) - before,
                counts.iter().sum::<u64>(),
                "{counts:?}"
            );
            // A one-voxel pile is a single lattice cell on the base plane, so
            // its apex is legitimately zero; anything bigger has to stand up.
            assert!(report.apex_height_m >= 0.0, "{counts:?}");
            assert!(
                counts.iter().sum::<u64>() <= 1 || report.apex_height_m > 0.0,
                "{counts:?}: {report:?}"
            );
        }
    }

    /// A plan with nothing in it writes nothing at all — a cut that removed no
    /// rock spoils no rock.
    #[test]
    fn spoiling_nothing_writes_nothing() {
        let mut v = empty(0.5);
        let mut b = VoxelDeltaBuilder::new();
        let report = v.place_spoil_into(
            &SpoilPlan::new([0; MATERIAL_COUNT], DVec3::new(1.0, 2.0, 3.0)),
            &mut b,
        );
        assert!(report.is_exact());
        assert_eq!(report.total_placed(), 0);
        assert!(b.is_empty(), "an empty plan recorded an undo entry");
        assert!(v.is_empty(), "an empty plan materialized a chunk");
    }

    /// **Piling onto rock still balances.** The site is buried inside a solid
    /// block, so most of the cone's cells are already solid and place nothing;
    /// the pile has to grow around them and still land on the exact count.
    #[test]
    fn a_pile_dropped_on_solid_ground_grows_around_it_and_still_balances() {
        let mut v = empty(0.5);
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        v.clear_dirty();
        let before = solid_by_material(&v);
        // Base inside the block: every cell of the first several shells is rock.
        let counts = [0u64, 400, 0, 0];
        let site = v.grid_to_world(DVec3::splat(8.0));
        let report = place(&mut v, counts, site);
        assert!(report.is_exact(), "{report:?}");
        assert_eq!(report.placed, counts);
        let after = solid_by_material(&v);
        assert_eq!(
            after[1] - before[1],
            400,
            "the pile did not add exactly 400 solid samples of material 1"
        );
        for m in [0usize, 2, 3] {
            assert_eq!(after[m], before[m], "material {m} moved");
        }
    }

    /// **Determinism.** The same plan on two volumes built in different chunk
    /// insertion orders produces byte-identical results, and running it twice on
    /// a fresh volume produces the same bytes and the same report.
    #[test]
    fn the_pile_is_a_pure_function_of_the_plan() {
        let counts = [300u64, 120, 0, 45];
        let base = DVec3::new(-7.25, 2.5, 11.5);

        let mut a = empty(0.5);
        let ra = place(&mut a, counts, base);
        let mut b = empty(0.5);
        let rb = place(&mut b, counts, base);
        assert_eq!(ra, rb);
        assert_eq!(image(&a), image(&b));

        // Same rock, inserted back to front.
        let mut c = empty(0.5);
        let mut d = empty(0.5);
        let keys: Vec<ChunkKey> = chunk_range(ChunkKey::new(-1, 0, 1), ChunkKey::new(0, 1, 2))
            .into_iter()
            .collect();
        for &k in &keys {
            c.insert_chunk(k, VoxelChunk::solid(2));
        }
        for &k in keys.iter().rev() {
            d.insert_chunk(k, VoxelChunk::solid(2));
        }
        let rc = place(&mut c, counts, base);
        let rd = place(&mut d, counts, base);
        assert_eq!(rc, rd);
        assert_eq!(image(&c), image(&d), "insertion order changed the pile");
    }

    /// **A pile undoes byte-identically**, chunk set included — it rides the
    /// same [`VoxelDeltaBuilder`] the carve does, so one Ctrl+Z takes back the
    /// hole and the heap together.
    #[test]
    fn a_pile_reverts_byte_identically() {
        let mut v = empty(0.5);
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(0, 0, 0)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        v.clear_dirty();
        let before = image(&v);

        let mut builder = VoxelDeltaBuilder::new();
        let report = v.place_spoil_into(
            &SpoilPlan::new([250, 0, 0, 0], DVec3::new(20.0, 0.0, 20.0)),
            &mut builder,
        );
        assert!(report.is_exact());
        let after = image(&v);
        assert_ne!(after, before);

        let delta = builder.finalize(&v);
        assert_eq!(v.revert_delta(&delta), 0);
        assert_eq!(
            image(&v),
            before,
            "the pile did not undo to the exact bytes"
        );
        assert_eq!(v.apply_delta(&delta), 0);
        assert_eq!(image(&v), after, "redo must restore the pile");
    }

    /// **Cut + spoil in ONE record.** The carve and the pile share a builder, so
    /// the single delta they produce reverts both halves — and the count the
    /// carve removed is the count the pile holds.
    #[test]
    fn a_carve_and_its_spoil_balance_and_undo_as_one() {
        let mut v = empty(0.5);
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
            v.insert_chunk(key, VoxelChunk::solid(1));
        }
        v.clear_dirty();
        let before = image(&v);
        let before_solid = solid_by_material(&v);

        let mut builder = VoxelDeltaBuilder::new();
        let op = VoxelOp::carve(VoxelShape::Box {
            center: v.grid_to_world(DVec3::new(8.0, 12.0, 8.0)),
            half_extents: DVec3::new(1.5, 1.0, 1.5),
        });
        let cut = v.apply_op_into(&op, &mut builder);
        assert!(cut.total_carved() > 20, "{cut:?}");

        let (lo, hi) = op.shape.aabb_m(0.0);
        let site = default_spoil_site(lo, hi, cut.total_carved(), v.voxel_size_m());
        let spoil = v.place_spoil_into(&SpoilPlan::new(cut.carved, site), &mut builder);
        assert!(spoil.is_exact(), "{spoil:?}");
        assert_eq!(
            spoil.placed, cut.carved,
            "the pile does not hold what the cut removed"
        );
        // The solid set is unchanged in total: material moved, none was made or
        // destroyed.
        assert_eq!(
            solid_by_material(&v),
            before_solid,
            "conservation failed at the field level"
        );
        // …and the site really is clear of the cut.
        assert!(site.x > hi.x, "the default pile overlaps the cut");

        let delta = builder.finalize(&v);
        assert_eq!(v.revert_delta(&delta), 0);
        assert_eq!(
            image(&v),
            before,
            "one undo did not take back both the cut and the pile"
        );
    }

    /// **B1 (P21.3 audit): the growth loop is bounded and never overflows.**
    ///
    /// A count too large for any pile used to make step 0 exceed the cell cap,
    /// after which the `apex *= 1.4` loop ran all 32 times and multiplied three
    /// `i32`-clamped spans into a panic — while holding the shared-volumes
    /// guard, mid-transaction, with the cut already committed to the world and
    /// no `EditCommand` describing it.
    ///
    /// The measured threshold was 4_075_984 OK / 4_075_985 panic, so both sides
    /// of it are exercised here, plus a count far past any conceivable dig.
    #[test]
    fn an_impossible_count_is_a_shortfall_and_never_a_panic() {
        for total in [4_075_984u64, 4_075_985, 40_000_000, u64::MAX / 4] {
            let mut v = empty(0.5);
            let mut b = VoxelDeltaBuilder::new();
            let report = v.place_spoil_into(&SpoilPlan::new([total, 0, 0, 0], DVec3::ZERO), &mut b);
            // Either it fits (small ones may) or it is an honest shortfall — but
            // it always RETURNS.
            assert!(
                report.is_exact() || report.shortfall == [total, 0, 0, 0],
                "{total}: {report:?}"
            );
            if !report.is_exact() {
                assert_eq!(report.total_placed(), 0, "{total}: a partial pile");
                assert!(b.is_empty(), "{total}: a refused pile recorded writes");
            }
        }
    }

    /// **The pile never places back into the hole it came out of** (P21.3 audit).
    ///
    /// The default site's clearance is the *analytic* radius and the real
    /// footprint is wider, so a big dig used to refill part of its own pit — and
    /// conservation balanced perfectly while it did, because a refilled pit is
    /// an impeccably conserved place to put the soil. The guarantee is now
    /// structural: excluded cells are not candidates.
    #[test]
    fn an_excluded_box_is_never_placed_into() {
        let mut v = empty(0.5);
        // A "cut" box, and a base sitting right on its eastern rim so the pile
        // must lean back over it.
        let lo = DVec3::new(0.0, 0.0, 0.0);
        let hi = DVec3::new(6.0, 4.0, 6.0);
        let plan = SpoilPlan::new([3000, 0, 0, 0], DVec3::new(6.5, 0.0, 3.0)).excluding(lo, hi);
        let mut b = VoxelDeltaBuilder::new();
        let report = v.place_spoil_into(&plan, &mut b);
        assert!(report.is_exact(), "{report:?}");
        assert!(
            report.base_radius_m > 6.5 - hi.x,
            "the fixture pile ({:.1} m) does not even reach the exclusion — it proves nothing",
            report.base_radius_m
        );

        let mut inside = 0;
        for (&key, c) in v.chunks() {
            for k in 0..CHUNK_DIM {
                for j in 0..CHUNK_DIM {
                    for i in 0..CHUNK_DIM {
                        if c.sample(i, j, k) >= 0.0 {
                            continue;
                        }
                        let bs = key.base_sample();
                        let w = v.grid_to_world(DVec3::new(
                            (bs[0] + i as i32) as f64,
                            (bs[1] + j as i32) as f64,
                            (bs[2] + k as i32) as f64,
                        ));
                        if w.cmpge(lo).all() && w.cmple(hi).all() {
                            inside += 1;
                        }
                    }
                }
            }
        }
        assert_eq!(inside, 0, "{inside} spoil samples landed back in the cut");

        // …and without the exclusion the same pile really does intrude, so the
        // assertion above is not vacuous.
        let mut u = empty(0.5);
        let mut b2 = VoxelDeltaBuilder::new();
        let open = SpoilPlan::new([3000, 0, 0, 0], DVec3::new(6.5, 0.0, 3.0));
        assert!(u.place_spoil_into(&open, &mut b2).is_exact());
        let mut intruded = 0;
        for (&key, c) in u.chunks() {
            for k in 0..CHUNK_DIM {
                for j in 0..CHUNK_DIM {
                    for i in 0..CHUNK_DIM {
                        if c.sample(i, j, k) >= 0.0 {
                            continue;
                        }
                        let bs = key.base_sample();
                        let w = u.grid_to_world(DVec3::new(
                            (bs[0] + i as i32) as f64,
                            (bs[1] + j as i32) as f64,
                            (bs[2] + k as i32) as f64,
                        ));
                        if w.cmpge(lo).all() && w.cmple(hi).all() {
                            intruded += 1;
                        }
                    }
                }
            }
        }
        assert!(
            intruded > 0,
            "the fixture cannot intrude even unguarded — the gate above is vacuous"
        );
    }

    /// **Paging is part of the placement** (P21.3 audit): a pile that would
    /// stand where the asset holds un-paged rock pages it in first, so those
    /// cells are correctly rejected as solid instead of being counted as air and
    /// overwritten.
    #[test]
    fn a_paged_placement_sees_rock_an_unpaged_one_would_overwrite() {
        use crate::residency::MemoryChunkStore;

        // A cold store holding a solid slab right where the pile wants to stand.
        let mut source = empty(0.5);
        let mut store = MemoryChunkStore::new();
        for key in chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 0, 1)) {
            let chunk = VoxelChunk::solid(3);
            store.insert(key, &chunk).unwrap();
            source.insert_chunk(key, chunk);
        }

        let plan = SpoilPlan::new([600, 0, 0, 0], DVec3::new(4.0, 0.0, 4.0));

        // UNPAGED: the slab is invisible, so its cells are "air" and the pile
        // writes straight over the asset's own rock.
        let mut blind = empty(0.5);
        let mut b1 = VoxelDeltaBuilder::new();
        assert!(blind.place_spoil_into(&plan, &mut b1).is_exact());

        // PAGED: the same call through the paging door sees the rock, rejects
        // those cells and grows the pile around them.
        let mut seeing = empty(0.5);
        let mut b2 = VoxelDeltaBuilder::new();
        let paged = seeing.place_spoil_into_paged(&plan, &mut b2, &store);
        assert!(paged.is_exact(), "{paged:?}");
        assert!(
            paged.apex_height_m > 0.0 && seeing.chunk_count() > 0,
            "the paged pile placed nothing"
        );
        assert!(
            paged.base_radius_m > 0.0,
            "the paged pile has no footprint: {paged:?}"
        );
        // The rock is intact: every sample the store served is still solid.
        for key in source.resident_keys() {
            let got = seeing.get_chunk(key).expect("paged in");
            let want = source.get_chunk(key).unwrap();
            for k in 0..CHUNK_DIM {
                for j in 0..CHUNK_DIM {
                    for i in 0..CHUNK_DIM {
                        if want.sample(i, j, k) < 0.0 {
                            assert!(
                                got.sample(i, j, k) < 0.0,
                                "{key:?} ({i},{j},{k}): the pile hollowed the asset's own rock"
                            );
                        }
                    }
                }
            }
        }
        // …and the two runs really differ, so this is not two identical piles.
        assert_ne!(
            image(&blind),
            image(&seeing),
            "paging changed nothing — the fixture does not overlap the slab"
        );
    }

    /// The frozen trig constants really are the stated angle — checked against
    /// `std` here (a **test**, where libm's last bit does not decide committed
    /// content) so a typo in a literal cannot ship.
    #[test]
    fn the_repose_constants_are_the_stated_angle() {
        let rad = REPOSE_DEG.to_radians();
        assert!((REPOSE_TAN - rad.tan()).abs() < 1e-15, "{REPOSE_TAN}");
        assert!((REPOSE_COS - rad.cos()).abs() < 1e-15, "{REPOSE_COS}");
    }

    /// The deterministic cube root agrees with `std` to the last few bits over a
    /// wide range — it exists to be *portable*, not to be different.
    #[test]
    fn the_deterministic_cube_root_is_a_cube_root() {
        for x in [
            1e-6, 0.001, 0.5, 1.0, 2.0, 7.999, 8.0, 27.0, 1000.0, 1.5e6, 9.87e11,
        ] {
            let got = cbrt_det(x);
            let want = x.cbrt();
            assert!(
                (got - want).abs() <= want.abs() * 1e-12,
                "cbrt_det({x}) = {got}, want {want}"
            );
            // …and it really inverts the cube.
            assert!((got * got * got - x).abs() <= x * 1e-11, "{x}");
        }
        assert_eq!(cbrt_det(0.0), 0.0);
        assert_eq!(cbrt_det(-1.0), 0.0);
        assert_eq!(cbrt_det(f64::NAN), 0.0);
    }

    /// The pile's slope really is the angle of repose: the mound's height and
    /// footprint stand in the documented ratio, and a bigger pile is wider.
    #[test]
    fn the_pile_stands_at_the_angle_of_repose() {
        let mut small = empty(0.5);
        let rs = place(&mut small, [200, 0, 0, 0], DVec3::ZERO);
        let mut big = empty(0.5);
        let rb = place(&mut big, [4000, 0, 0, 0], DVec3::ZERO);
        assert!(rb.apex_height_m > rs.apex_height_m, "{rs:?} {rb:?}");
        assert!(rb.base_radius_m > rs.base_radius_m);
        // `base_radius_m == apex / tan θ` is how the field is COMPUTED, so
        // asserting it is a tautology (P21.3 audit). The claim worth making is
        // about the geometry the pile actually has: measure the placed samples
        // and check that the mound's silhouette really lies on the repose cone.
        let solid_extent = |v: &VoxelData, base: DVec3| -> (f64, f64) {
            let (mut top, mut reach) = (0.0f64, 0.0f64);
            for (&key, c) in v.chunks() {
                for k in 0..CHUNK_DIM {
                    for j in 0..CHUNK_DIM {
                        for i in 0..CHUNK_DIM {
                            if c.sample(i, j, k) >= 0.0 {
                                continue;
                            }
                            let b = key.base_sample();
                            let w = v.grid_to_world(DVec3::new(
                                (b[0] + i as i32) as f64,
                                (b[1] + j as i32) as f64,
                                (b[2] + k as i32) as f64,
                            ));
                            top = top.max(w.y - base.y);
                            let dx = w.x - base.x;
                            let dz = w.z - base.z;
                            reach = reach.max((dx * dx + dz * dz).sqrt());
                        }
                    }
                }
            }
            (top, reach)
        };
        for (label, v, r, n) in [("small", &small, rs, 200u64), ("big", &big, rb, 4000)] {
            let (top, reach) = solid_extent(v, DVec3::ZERO);
            // Every placed sample is under the cone the report describes, within
            // one voxel of lattice slack.
            assert!(
                top <= r.apex_height_m + 0.5,
                "{label}: the pile is {top:.2} m tall but reports {:.2} m",
                r.apex_height_m
            );
            assert!(
                reach <= r.base_radius_m + 0.5,
                "{label}: the pile reaches {reach:.2} m but reports {:.2} m",
                r.base_radius_m
            );
            // …and it really is a CONE and not a column or a slab: the measured
            // height over the measured footprint is the repose slope.
            if reach > 1.0 {
                let slope = top / reach;
                assert!(
                    (slope - REPOSE_TAN).abs() < 0.35,
                    "{label}: slope {slope:.3} is not the {REPOSE_TAN:.3} angle of repose \
                     ({top:.2} m over {reach:.2} m)"
                );
            }
            // The analytic cone that tall over that footprint holds roughly the
            // count — a sanity bound, not an identity (a pile is a lattice).
            let vol = std::f64::consts::PI / 3.0 * r.base_radius_m.powi(2) * r.apex_height_m;
            let cells = vol / 0.125;
            assert!(
                (0.4..2.2).contains(&(cells / n as f64)),
                "{label}: {cells:.0} lattice cells for {n} voxels"
            );
        }
    }

    /// **The tie-break order is a total order, and it is the one documented.**
    ///
    /// The pile is "a pure function of `(base, count)`" only because
    /// `(rank, height, x, z)` never ties on all four for distinct cells. Nothing
    /// asserted it, so a sort that dropped a key — or reordered them — would
    /// still pass every determinism test that compares two identical runs.
    #[test]
    fn the_candidate_order_is_total_and_as_documented() {
        let mut v = empty(1.0);
        let mut b = VoxelDeltaBuilder::new();
        // A base on the lattice, so the four cells at (±1, 0, 0) / (0, 0, ±1)
        // tie on rank AND height and can only be separated by x then z.
        let r = v.place_spoil_into(&SpoilPlan::new([5, 0, 0, 0], DVec3::ZERO), &mut b);
        assert!(r.is_exact());

        // Gather the placed cells in the order the rule says they were taken.
        let mut placed: Vec<[i32; 3]> = Vec::new();
        for (&key, c) in v.chunks() {
            for k in 0..CHUNK_DIM {
                for j in 0..CHUNK_DIM {
                    for i in 0..CHUNK_DIM {
                        if c.sample(i, j, k) < 0.0 {
                            let bs = key.base_sample();
                            placed.push([bs[0] + i as i32, bs[1] + j as i32, bs[2] + k as i32]);
                        }
                    }
                }
            }
        }
        assert_eq!(placed.len(), 5);
        // The centre cell (rank 0) must be there, and the four unit-distance
        // neighbours — not, say, two of them and one two cells out.
        assert!(placed.contains(&[0, 0, 0]), "{placed:?}");
        let ring: Vec<[i32; 3]> = vec![[-1, 0, 0], [0, 0, -1], [0, 0, 1], [1, 0, 0]];
        for cell in &ring {
            assert!(
                placed.contains(cell),
                "{cell:?} is missing — the order skipped a tied cell: {placed:?}"
            );
        }
        // Two independent runs agree, cell for cell (the order is a function of
        // the set, not of the walk).
        let mut w = empty(1.0);
        let mut b2 = VoxelDeltaBuilder::new();
        w.place_spoil_into(&SpoilPlan::new([5, 0, 0, 0], DVec3::ZERO), &mut b2);
        assert_eq!(image(&v), image(&w));
    }

    /// **Nothing is written below the base plane.** A pile grows up; a spoil
    /// site is a place to stand a heap, not a place to inject rock into the
    /// ground under it.
    #[test]
    fn a_pile_never_writes_below_its_base() {
        let mut v = empty(0.5);
        let base = DVec3::new(0.0, 5.0, 0.0);
        let report = place(&mut v, [800, 0, 0, 0], base);
        assert!(report.is_exact());
        for (&key, c) in v.chunks() {
            for k in 0..CHUNK_DIM {
                for j in 0..CHUNK_DIM {
                    for i in 0..CHUNK_DIM {
                        if c.sample(i, j, k) >= 0.0 {
                            continue;
                        }
                        let base_g = key.base_sample();
                        let w = v.grid_to_world(DVec3::new(
                            (base_g[0] + i as i32) as f64,
                            (base_g[1] + j as i32) as f64,
                            (base_g[2] + k as i32) as f64,
                        ));
                        assert!(
                            w.y >= base.y - 1e-9,
                            "a solid sample at {w:?} is below the base plane {base:?}"
                        );
                    }
                }
            }
        }
    }

    /// A non-finite base is a shortfall, not a panic and not a plane of garbage
    /// — a UI can hand this a NaN on the first frame of a drag.
    #[test]
    fn a_degenerate_base_refuses_cleanly() {
        let mut v = empty(0.5);
        let mut b = VoxelDeltaBuilder::new();
        let counts = [10u64, 0, 0, 0];
        let r = v.place_spoil_into(&SpoilPlan::new(counts, DVec3::splat(f64::NAN)), &mut b);
        assert!(!r.is_exact());
        assert_eq!(r.shortfall, counts);
        assert_eq!(r.total_placed(), 0);
        assert!(b.is_empty());
        assert!(v.is_empty());
    }

    /// The default site is a **pure function of the cut's bounds and the count**
    /// — east of the rim, clear of it, and it does not move when the world does.
    #[test]
    fn the_default_site_is_east_of_the_cut_and_clear_of_it() {
        let lo = DVec3::new(-4.0, 0.0, -2.0);
        let hi = DVec3::new(6.0, 3.0, 8.0);
        let a = default_spoil_site(lo, hi, 5000, 0.5);
        assert_eq!(
            a,
            default_spoil_site(lo, hi, 5000, 0.5),
            "not deterministic"
        );
        assert!(a.x >= hi.x + SPOIL_GAP_M, "{a:?}");
        assert_eq!(a.y, hi.y);
        assert_eq!(a.z, (lo.z + hi.z) * 0.5);
        // A bigger dig stands its pile further out, because the pile is wider.
        let b = default_spoil_site(lo, hi, 500_000, 0.5);
        assert!(b.x > a.x);
        // …and an empty dig asks for no clearance at all.
        assert_eq!(default_spoil_site(lo, hi, 0, 0.5).x, hi.x + SPOIL_GAP_M);
    }
}
