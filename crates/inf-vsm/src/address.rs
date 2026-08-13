//! The **virtual page space of one light**: where a page is, who its parent is,
//! and which entry of the indirection table describes it.
//!
//! Three tree shapes, one type, because the ROADMAP's P27.1 clause 2 names three
//! and a receiver shader that had to branch on three *types* would be three
//! shaders:
//!
//! * [`VsmTreeKind::Clipmap`] — the directional sun. Every level is the **same**
//!   `N × N` page grid and covers **twice** the world extent of the level below
//!   it, concentrically. This is `inf_terrain`'s clipmap precedent, and it is the
//!   reason a directional light cannot borrow a texture pyramid: a mip level has
//!   *fewer* tiles than the one below it, a clipmap level has exactly as many.
//! * [`VsmTreeKind::Quadtree`] — a spot light. One tree over the light's own
//!   frustum, level 0 finest with `2^(levels-1)` pages per side, halving upward
//!   to a single root page. This *is* the texture-pyramid shape.
//! * [`VsmTreeKind::Cube`] — a point light: six [`Quadtree`](VsmTreeKind::Quadtree)
//!   faces in one address space, so one light is one handle and one table block
//!   whatever its kind.
//!
//! # Level 0 is the FINEST, everywhere
//!
//! The same convention `inf_vt::TileCoord::mip` uses, chosen so the two address
//! spaces read alike for the P28.3 merge and so `parent` always means "up one
//! level, coarser". It costs one moment of surprise on the quadtree — whose
//! *root* is therefore the **last** level rather than level 0 — and it buys one
//! rule for three kinds.
//!
//! # There is no mandatory floor, and that is the load-bearing difference
//!
//! `inf-vt`'s first law is that the coarsest level of every texture is admitted
//! at registration and never evicted, because a fragment may sample *any* tile
//! and a miss would be a hole. **A shadow page has no such exposure**: the
//! marking pass marks exactly the pages the visible depth buffer needs, so a page
//! nothing marked is a page nothing samples. Pinning a clipmap's coarsest level
//! would claim `N²` pages per directional light before any evidence — **4 096**
//! at the shipped 8 k grid, which is four times the whole 1 024-page default
//! atlas, and **16 384** at 16 k, which is sixteen times it. The arithmetic is
//! pinned by [`tests::pinning_a_clipmaps_coarsest_level_costs_what_the_ruling_says`]
//! rather than quoted, because the P27.1 audit found the first draft of this
//! paragraph pairing the 16 k page count with the 8 k multiplier.
//!
//! So a page that is wanted and could not be seated resolves to its **finest
//! resident ancestor** exactly as in `inf-vt`, and to [`VSM_ENTRY_NONE`] when it
//! has none. `NONE` means *no shadow data*, which a receiver reads as **lit** —
//! a light leak under budget pressure rather than a black hole, and the honest
//! degradation for a signal whose producer is a rasterizer rather than a file.

use std::ops::Range;

/// Address of one page in one light's virtual page space.
///
/// **Its derived `Ord` is `(face, level, x, y)`; the canonical order is
/// [`entry_order`](Self::entry_order)'s `(face, level, y, x)`** — the order the
/// indirection entries are laid out in, which is also the order the marking
/// bitmask's bits run in and therefore the order a mask scan emits wants in.
/// The two differ the moment `x` and `y` do, so a `BTreeSet<VsmPage>` iterates
/// in *its* order and not the table's. `VsmResidency::apply_wants` sorts on
/// `entry_order`.
///
/// This is the `inf_vt::TileCoord` finding (P26.1 audit: derived `(mip, x, y)`
/// against a payload order of `(mip, y, x)`) met a second time in a second
/// address space, and pinned here rather than rediscovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VsmPage {
    /// Cube face (0..6) for [`VsmTreeKind::Cube`]; always 0 otherwise.
    pub face: u32,
    /// 0 = finest. See the module docs.
    pub level: u32,
    pub x: u32,
    pub y: u32,
}

impl VsmPage {
    pub const fn new(face: u32, level: u32, x: u32, y: u32) -> Self {
        Self { face, level, x, y }
    }

    /// A face-0 page — spot and directional lights, where the face is inert.
    pub const fn flat(level: u32, x: u32, y: u32) -> Self {
        Self::new(0, level, x, y)
    }

    /// This address as the table lays it out — the key to order a want set by.
    pub const fn entry_order(&self) -> (u32, u32, u32, u32) {
        (self.face, self.level, self.y, self.x)
    }
}

/// Which family of virtual page tree a light gets. See the module docs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VsmTreeKind {
    /// Directional: constant page grid, extent doubles per level.
    Clipmap,
    /// Spot: one quadtree, root at the coarsest level.
    Quadtree,
    /// Point: six quadtree faces.
    Cube,
}

impl VsmTreeKind {
    /// Faces this kind addresses.
    #[inline]
    pub const fn faces(self) -> u32 {
        match self {
            VsmTreeKind::Cube => 6,
            VsmTreeKind::Clipmap | VsmTreeKind::Quadtree => 1,
        }
    }

    /// The wire discriminant carried in a table block header.
    ///
    /// **Freeze-pinned, append-only** (the P19 wire-enum law): a shader reads
    /// this number to decide which parent rule applies, so re-ordering the
    /// variants would silently re-parent every page of every light.
    #[inline]
    pub const fn wire(self) -> u32 {
        match self {
            VsmTreeKind::Clipmap => 0,
            VsmTreeKind::Quadtree => 1,
            VsmTreeKind::Cube => 2,
        }
    }
}

/// One level's page grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmLevelDesc {
    pub pages_x: u32,
    pub pages_y: u32,
}

impl VsmLevelDesc {
    #[inline]
    pub const fn square(n: u32) -> Self {
        Self {
            pages_x: n,
            pages_y: n,
        }
    }
    /// Pages in this level's grid.
    ///
    /// **Saturating, and that is what makes the refusal reachable.** `pages_x`
    /// reaches this type from a host-set `VsmSettings::clipmap_pages_per_side`
    /// exactly as the level count reaches [`VsmLightDesc::clipmap`], so it can be
    /// absurd; `65 536²` is `2³²`, which a plain multiply wraps to **zero** in
    /// release and panics on in debug. Neither is the
    /// [`VsmError::PageSpaceTooLarge`](crate::VsmError::PageSpaceTooLarge)
    /// refusal `register_light` exists to give — a zero page count *validates*,
    /// registers, and then walks a 4-billion-page grid filling a table that has
    /// no entries.
    ///
    /// A saturated count is a count that is **at least** the ceiling, which is
    /// all a ceiling test needs.
    #[inline]
    pub const fn page_count(&self) -> u32 {
        self.pages_x.saturating_mul(self.pages_y)
    }
}

/// The maximum level count a light may declare.
///
/// Sixteen levels of a clipmap doubling each time is 65 536× the finest extent —
/// a 1 m finest texel reaches 65 km — and sixteen keeps the packed entry's 8-bit
/// level field ([`crate::table`]) unable to overflow, which is a bound the
/// encoder relies on instead of checking.
pub const MAX_VSM_LEVELS: usize = 16;

/// One light's virtual page space: the tree kind and its levels, finest first.
///
/// Holds **no matrices and no world extent** — those are the renderer's, and they
/// change every frame as the sun moves and the clipmap re-centres. What is here
/// is the shape of the address space, which does not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VsmLightDesc {
    pub kind: VsmTreeKind,
    /// Finest first. See the module docs.
    pub levels: Vec<VsmLevelDesc>,
}

impl VsmLightDesc {
    /// A directional light's clipmap: `levels` levels of `pages_per_side²`.
    ///
    /// `levels` is clamped into `1..=`[`MAX_VSM_LEVELS`] **before** the vector is
    /// allocated, exactly as [`quadtree_levels`] clamps, and the reason is not
    /// symmetry: `levels` reaches here from a host-set
    /// `VsmSettings::clipmap_levels`, and a `vec![_; u32::MAX as usize]` is an
    /// out-of-memory abort rather than the [`VsmDescError::TooManyLevels`]
    /// refusal `validate` is there to give. A refusal has to survive being asked
    /// for something absurd.
    pub fn clipmap(levels: u32, pages_per_side: u32) -> Self {
        Self {
            kind: VsmTreeKind::Clipmap,
            levels: vec![
                VsmLevelDesc::square(pages_per_side);
                levels.clamp(1, MAX_VSM_LEVELS as u32) as usize
            ],
        }
    }

    /// A spot light's quadtree: `levels` levels, `2^(levels-1)` pages per side at
    /// level 0, one root page at the coarsest.
    pub fn quadtree(levels: u32) -> Self {
        Self {
            kind: VsmTreeKind::Quadtree,
            levels: quadtree_levels(levels),
        }
    }

    /// A point light's six quadtree faces, each shaped like [`quadtree`](Self::quadtree).
    pub fn cube(levels: u32) -> Self {
        Self {
            kind: VsmTreeKind::Cube,
            levels: quadtree_levels(levels),
        }
    }

    #[inline]
    pub fn faces(&self) -> u32 {
        self.kind.faces()
    }

    #[inline]
    pub fn level_count(&self) -> u32 {
        self.levels.len() as u32
    }

    /// The coarsest level — the one with no parent.
    #[inline]
    pub fn coarsest_level(&self) -> u32 {
        self.level_count().saturating_sub(1)
    }

    /// Pages in **one** face. Saturating — see [`VsmLevelDesc::page_count`].
    pub fn face_page_count(&self) -> u32 {
        self.levels
            .iter()
            .fold(0u32, |a, l| a.saturating_add(l.page_count()))
    }

    /// Pages in the whole light — the number of indirection entries it needs and
    /// the number of bits it owns in the marking mask. Saturating, so
    /// [`crate::VsmResidency::register_light`]'s ceiling test refuses an absurd
    /// descriptor rather than being handed a wrapped one.
    pub fn page_count(&self) -> u32 {
        self.face_page_count().saturating_mul(self.faces())
    }

    /// Whether `at` names a page that exists.
    pub fn contains(&self, at: VsmPage) -> bool {
        at.face < self.faces()
            && self
                .levels
                .get(at.level as usize)
                .is_some_and(|l| at.x < l.pages_x && at.y < l.pages_y)
    }

    /// **The index of `at` in this light's page order** — faces outermost, then
    /// level, then `(y, x)`. The table entry that describes a page and the mask
    /// bit that requests it sit at the same number.
    pub fn entry_index(&self, at: VsmPage) -> Option<u32> {
        if !self.contains(at) {
            return None;
        }
        let mut base = at.face.saturating_mul(self.face_page_count());
        for l in &self.levels[..at.level as usize] {
            base = base.saturating_add(l.page_count());
        }
        let l = &self.levels[at.level as usize];
        Some(base + at.y * l.pages_x + at.x)
    }

    /// The index of `(face, level)`'s first page — the number a table level
    /// record carries so a shader indexes an entry without summing anything.
    pub fn level_first_entry(&self, face: u32, level: u32) -> Option<u32> {
        if face >= self.faces() || level as usize > self.levels.len() {
            return None;
        }
        Some(
            face.saturating_mul(self.face_page_count()).saturating_add(
                self.levels[..level as usize]
                    .iter()
                    .fold(0u32, |a, l| a.saturating_add(l.page_count())),
            ),
        )
    }

    /// **Where one clipmap level's page grid sits on the world lattice**: the
    /// index of page `(0, 0)`'s cell at that level's own page stride.
    ///
    /// `origins` is the caller's per-level table, finest first; an entry it does
    /// not have — and an empty table — means the **concentric** default,
    /// `(−pages_x/2, −pages_y/2)`, which is the arrangement every level shares
    /// when they share one snapped centre. That default is not a convention: it
    /// is the value that makes [`parent_at`](Self::parent_at) reduce to
    /// `N/4 + x/2` exactly, which
    /// [`tests::the_general_clipmap_parent_rule_reduces_to_the_concentric_one`]
    /// asserts rather than asserts about.
    ///
    /// Meaningless for a quadtree or a cube face, whose levels are nested by
    /// construction and whose parent rule is `x/2` whatever the world does.
    pub fn clip_origin(&self, origins: &[(i64, i64)], level: u32) -> (i64, i64) {
        let l = self
            .levels
            .get(level as usize)
            .copied()
            .unwrap_or(VsmLevelDesc::square(0));
        origins
            .get(level as usize)
            .copied()
            .unwrap_or((-(i64::from(l.pages_x) / 2), -(i64::from(l.pages_y) / 2)))
    }

    /// The page at the next-coarser level whose footprint contains `at`'s, or
    /// `None` when `at` is already at the coarsest — **for exactly concentric
    /// clipmap levels**. [`parent_at`](Self::parent_at) is the general rule and
    /// this is the special case `origins = &[]` names.
    ///
    /// # The two rules, and why the clipmap's is not `x / 2`
    ///
    /// A **quadtree** halves: `(x/2, y/2)`, clamped to the coarser level's grid
    /// exactly as `inf_vt::VtTextureDesc::parent` clamps, so a hand-built
    /// descriptor with a sliver level resolves rather than escaping the grid.
    ///
    /// A **clipmap** does not. Level `L+1` covers twice level `L`'s extent about
    /// the *same centre*, so level `L`'s whole grid sits in the **middle half**
    /// of level `L+1`'s: page `x` lands at `N/4 + x/2`. Using `x / 2` here would
    /// re-parent every page a quarter of a level's extent off — a shadow that
    /// falls back to the wrong ground, with no error anywhere.
    pub fn parent(&self, at: VsmPage) -> Option<VsmPage> {
        self.parent_at(&[], at)
    }

    /// **The general clipmap parent rule**: the coarser page whose *world cell*
    /// contains `at`'s, when the levels are each snapped to their own page stride
    /// and are therefore **not** concentric.
    ///
    /// Level `L` page `x` covers world cell `gx_L + x` at stride `w_L`; level
    /// `L+1`'s cells are twice as wide, so that cell sits in coarse cell
    /// `⌊(gx_L + x) / 2⌋` and the coarse *page* index is that minus `gx_{L+1}`.
    /// Floor division, not truncation — a cell index is signed and truncation
    /// would fold the two cells either side of the origin onto one parent.
    ///
    /// This is the whole of P27.3's per-level-offset change to the address space,
    /// and it **contains** the shipped concentric rule rather than replacing it:
    /// at the default origins the expression is identically `N/4 + ⌊x/2⌋` (see
    /// [`clip_origin`](Self::clip_origin)). `None` when `at` is at the coarsest
    /// level, when `at` is not in the tree, **or when the offsets put the parent
    /// outside the coarser grid** — which resolves to `VSM_ENTRY_NONE`, read as
    /// *lit*, the fail direction this phase already chose.
    pub fn parent_at(&self, origins: &[(i64, i64)], at: VsmPage) -> Option<VsmPage> {
        let up = at.level + 1;
        let l = *self.levels.get(up as usize)?;
        if !self.contains(at) {
            return None;
        }
        match self.kind {
            VsmTreeKind::Clipmap => {
                let (gx_l, gy_l) = self.clip_origin(origins, at.level);
                let (gx_p, gy_p) = self.clip_origin(origins, up);
                let px = (gx_l + i64::from(at.x)).div_euclid(2) - gx_p;
                let py = (gy_l + i64::from(at.y)).div_euclid(2) - gy_p;
                if !(0..i64::from(l.pages_x)).contains(&px)
                    || !(0..i64::from(l.pages_y)).contains(&py)
                {
                    return None;
                }
                Some(VsmPage::new(at.face, up, px as u32, py as u32))
            }
            VsmTreeKind::Quadtree | VsmTreeKind::Cube => Some(VsmPage::new(
                at.face,
                up,
                (at.x / 2).min(l.pages_x - 1),
                (at.y / 2).min(l.pages_y - 1),
            )),
        }
    }

    /// The ancestor of `at` at level `target`, following the same chain
    /// [`parent`](Self::parent) walks. **This is the page an indirection entry
    /// names**, and the CPU twin of the walk a receiver does when the table hands
    /// it a coarser level than it asked for.
    pub fn ancestor(&self, at: VsmPage, target: u32) -> Option<VsmPage> {
        self.ancestor_at(&[], at, target)
    }

    /// [`ancestor`](Self::ancestor) under per-level clipmap offsets.
    pub fn ancestor_at(&self, origins: &[(i64, i64)], at: VsmPage, target: u32) -> Option<VsmPage> {
        if !self.contains(at) || target < at.level || target as usize >= self.levels.len() {
            return None;
        }
        let mut cur = at;
        while cur.level < target {
            cur = self.parent_at(origins, cur)?;
        }
        Some(cur)
    }

    /// The half-open range of child columns of parent column `x` at level
    /// `level` — the inverse of [`parent`](Self::parent).
    pub fn child_x_range(&self, level: u32, x: u32) -> Range<u32> {
        self.child_range(&[], level, x, |l| l.pages_x, |o| o.0)
    }

    /// The half-open range of child rows of parent row `y` at level `level`.
    pub fn child_y_range(&self, level: u32, y: u32) -> Range<u32> {
        self.child_range(&[], level, y, |l| l.pages_y, |o| o.1)
    }

    /// [`child_x_range`](Self::child_x_range) under per-level clipmap offsets —
    /// the exact inverse of [`parent_at`](Self::parent_at).
    pub fn child_x_range_at(&self, origins: &[(i64, i64)], level: u32, x: u32) -> Range<u32> {
        self.child_range(origins, level, x, |l| l.pages_x, |o| o.0)
    }

    /// [`child_y_range`](Self::child_y_range) under per-level clipmap offsets.
    pub fn child_y_range_at(&self, origins: &[(i64, i64)], level: u32, y: u32) -> Range<u32> {
        self.child_range(origins, level, y, |l| l.pages_y, |o| o.1)
    }

    fn child_range(
        &self,
        origins: &[(i64, i64)],
        level: u32,
        p: u32,
        axis: fn(&VsmLevelDesc) -> u32,
        of: fn(&(i64, i64)) -> i64,
    ) -> Range<u32> {
        let parent_count = self.levels.get(level as usize).map_or(0, axis);
        let child_count = level
            .checked_sub(1)
            .and_then(|c| self.levels.get(c as usize))
            .map_or(0, axis);
        if parent_count == 0 || child_count == 0 || p >= parent_count {
            return 0..0;
        }
        match self.kind {
            // Only the middle half of a clipmap level has children at all: the
            // finer level covers a quarter of the area. Under the concentric
            // default that window is `[N/4, 3N/4)` and this expression is
            // identically `(p − N/4)·2`; under per-level offsets the window
            // slides, and it is the SAME arithmetic because both are "which
            // child cells fall in this parent cell".
            VsmTreeKind::Clipmap => {
                let gp = of(&self.clip_origin(origins, level));
                let gc = of(&self.clip_origin(origins, level - 1));
                let n = i64::from(child_count);
                let lo = (2 * (i64::from(p) + gp) - gc).clamp(0, n);
                let hi = (2 * (i64::from(p) + gp) - gc + 2).clamp(0, n);
                lo as u32..hi as u32
            }
            VsmTreeKind::Quadtree | VsmTreeKind::Cube => {
                let lo = p.saturating_mul(2);
                if lo >= child_count {
                    return lo..lo;
                }
                let hi = if p + 1 == parent_count {
                    child_count
                } else {
                    (lo + 2).min(child_count)
                };
                lo..hi
            }
        }
    }

    /// Validate the shape, so a hand-built descriptor that is not a real tree is
    /// refused at [`crate::VsmResidency::register_light`]'s door rather than
    /// mis-resolved for the rest of the session.
    pub fn validate(&self) -> Result<(), VsmDescError> {
        if self.levels.is_empty() {
            return Err(VsmDescError::EmptyTree);
        }
        if self.levels.len() > MAX_VSM_LEVELS {
            return Err(VsmDescError::TooManyLevels {
                levels: self.levels.len(),
                max: MAX_VSM_LEVELS,
            });
        }
        for (i, l) in self.levels.iter().enumerate() {
            if l.pages_x == 0 || l.pages_y == 0 {
                return Err(VsmDescError::EmptyLevel { level: i as u32 });
            }
            match self.kind {
                VsmTreeKind::Clipmap => {
                    // Every level is the same square grid, and `N % 4 == 0` is
                    // what makes `N/4 + x/2` land on a page boundary rather than
                    // between two — the parent rule is arithmetic, not a round.
                    if l.pages_x != self.levels[0].pages_x || l.pages_y != self.levels[0].pages_y {
                        return Err(VsmDescError::ClipmapLevelsDiffer { level: i as u32 });
                    }
                    if l.pages_x != l.pages_y || !l.pages_x.is_multiple_of(4) {
                        return Err(VsmDescError::ClipmapGrid {
                            pages_x: l.pages_x,
                            pages_y: l.pages_y,
                        });
                    }
                }
                VsmTreeKind::Quadtree | VsmTreeKind::Cube => {
                    if i > 0 {
                        let p = &self.levels[i - 1];
                        if l.pages_x != (p.pages_x / 2).max(1)
                            || l.pages_y != (p.pages_y / 2).max(1)
                        {
                            return Err(VsmDescError::QuadtreeChain { level: i as u32 });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// `levels` quadtree levels, finest first: `2^(levels-1)` per side down to 1.
fn quadtree_levels(levels: u32) -> Vec<VsmLevelDesc> {
    let n = levels.clamp(1, MAX_VSM_LEVELS as u32);
    (0..n)
        .map(|i| VsmLevelDesc::square(1 << (n - 1 - i)))
        .collect()
}

/// A descriptor that is not a virtual page tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VsmDescError {
    #[error("a light's page tree needs at least one level")]
    EmptyTree,
    #[error("{levels} levels is past the {max}-level ceiling")]
    TooManyLevels { levels: usize, max: usize },
    #[error("level {level} has a zero page grid")]
    EmptyLevel { level: u32 },
    #[error(
        "clipmap level {level} is not the same grid as level 0 — every clipmap level covers \
         twice the extent at the SAME page count"
    )]
    ClipmapLevelsDiffer { level: u32 },
    #[error(
        "a clipmap level is {pages_x}×{pages_y}; it must be square and a multiple of 4, or the \
         concentric parent rule `N/4 + x/2` lands between pages"
    )]
    ClipmapGrid { pages_x: u32, pages_y: u32 },
    #[error("quadtree level {level} is not half of level {} — this is not a quadtree", level - 1)]
    QuadtreeChain { level: u32 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_derived_order_is_not_the_entry_order() {
        // The `inf_vt::TileCoord` finding, met again: `(…, x, y)` and `(…, y, x)`
        // disagree the moment x and y differ, so a `BTreeSet<VsmPage>` walks the
        // table sideways.
        let a = VsmPage::flat(0, 0, 1);
        let b = VsmPage::flat(0, 1, 0);
        assert!(a < b, "derived Ord compares x before y");
        assert!(
            b.entry_order() < a.entry_order(),
            "entry order compares y before x — the OPPOSITE verdict"
        );
    }

    #[test]
    fn a_clipmap_is_the_same_grid_at_every_level() {
        let d = VsmLightDesc::clipmap(6, 8);
        d.validate().expect("a real clipmap validates");
        assert_eq!(d.level_count(), 6);
        assert_eq!(d.faces(), 1);
        for l in &d.levels {
            assert_eq!(l.page_count(), 64, "a clipmap level never shrinks");
        }
        assert_eq!(d.page_count(), 6 * 64);
        assert_eq!(d.coarsest_level(), 5);

        // The index walk is (face, level, y, x), and every page is visited once.
        let mut n = 0;
        for level in 0..d.level_count() {
            let l = d.levels[level as usize];
            for y in 0..l.pages_y {
                for x in 0..l.pages_x {
                    let at = VsmPage::flat(level, x, y);
                    assert_eq!(d.entry_index(at), Some(n), "{at:?}");
                    n += 1;
                }
            }
        }
        assert_eq!(n, d.page_count());
        assert_eq!(d.entry_index(VsmPage::flat(0, 8, 0)), None);
        assert_eq!(d.entry_index(VsmPage::flat(6, 0, 0)), None);
        assert_eq!(d.entry_index(VsmPage::new(1, 0, 0, 0)), None, "one face");
    }

    /// **The clipmap's parent is concentric, not `x / 2`** — the rule that would
    /// be wrong by a quarter of a level's extent, silently.
    #[test]
    fn a_clipmap_page_hangs_under_the_middle_half_of_the_level_above() {
        let d = VsmLightDesc::clipmap(4, 8);
        // Level 0's whole 8×8 grid sits in level 1's middle 4×4 block: columns
        // 2..6, rows 2..6.
        assert_eq!(
            d.parent(VsmPage::flat(0, 0, 0)),
            Some(VsmPage::flat(1, 2, 2))
        );
        assert_eq!(
            d.parent(VsmPage::flat(0, 1, 1)),
            Some(VsmPage::flat(1, 2, 2))
        );
        assert_eq!(
            d.parent(VsmPage::flat(0, 7, 7)),
            Some(VsmPage::flat(1, 5, 5))
        );
        // …and `x / 2` would have said (1, 3, 3) for (0, 7, 7): a whole quarter
        // of the level away.
        assert_ne!(
            d.parent(VsmPage::flat(0, 7, 7)),
            Some(VsmPage::flat(1, 3, 3))
        );
        // The coarsest level has no parent at all.
        assert_eq!(d.parent(VsmPage::flat(3, 0, 0)), None);
        // …and the ancestor chain walks two levels the same way.
        assert_eq!(
            d.ancestor(VsmPage::flat(0, 0, 0), 2),
            Some(VsmPage::flat(2, 3, 3))
        );
        assert_eq!(
            d.ancestor(VsmPage::flat(0, 0, 0), 0),
            Some(VsmPage::flat(0, 0, 0)),
            "the zero-step ancestor is the identity"
        );
        assert_eq!(d.ancestor(VsmPage::flat(2, 0, 0), 1), None, "backwards");
    }

    /// Every page of every finer level has **at most one** parent, and
    /// [`VsmLightDesc::child_range`] names it back. Under the concentric default
    /// every child has exactly one; under per-level offsets a fine page can hang
    /// off the coarser grid's edge, and `expect_total` says which world it is.
    fn assert_the_ranges_invert_the_parent_rule_at(
        d: &VsmLightDesc,
        origins: &[(i64, i64)],
        expect_total: bool,
    ) {
        d.validate().expect("a real tree");
        for face in 0..d.faces() {
            for level in 1..d.level_count() {
                let parent = d.levels[level as usize];
                let child = d.levels[level as usize - 1];
                let mut seen = vec![0u32; child.page_count() as usize];
                for y in 0..parent.pages_y {
                    for x in 0..parent.pages_x {
                        for cy in d.child_y_range_at(origins, level, y) {
                            for cx in d.child_x_range_at(origins, level, x) {
                                seen[(cy * child.pages_x + cx) as usize] += 1;
                                assert_eq!(
                                    d.parent_at(origins, VsmPage::new(face, level - 1, cx, cy)),
                                    Some(VsmPage::new(face, level, x, y)),
                                    "{:?} level {level}: ({cx},{cy}) is not ({x},{y})'s child",
                                    d.kind
                                );
                            }
                        }
                    }
                }
                assert!(
                    seen.iter().all(|&n| n <= 1),
                    "{:?} level {level}: a child was adopted {:?} times",
                    d.kind,
                    seen.iter().collect::<std::collections::BTreeSet<_>>()
                );
                if expect_total {
                    assert!(
                        seen.iter().all(|&n| n == 1),
                        "{:?} level {level}: a child was orphaned",
                        d.kind
                    );
                }
                // A child with no range entry must have no parent either — the
                // two halves are one statement, and this is what catches an
                // offset that drops a page out of both.
                for cy in 0..child.pages_y {
                    for cx in 0..child.pages_x {
                        let has_parent = d
                            .parent_at(origins, VsmPage::new(face, level - 1, cx, cy))
                            .is_some();
                        assert_eq!(
                            has_parent,
                            seen[(cy * child.pages_x + cx) as usize] == 1,
                            "{:?} level {level}: ({cx},{cy}) disagrees about its parent",
                            d.kind
                        );
                    }
                }
            }
        }
    }

    fn assert_the_ranges_invert_the_parent_rule(d: &VsmLightDesc) {
        assert_the_ranges_invert_the_parent_rule_at(d, &[], true);
    }

    /// Every child range inverts the parent rule exactly, for **both** families —
    /// including the clipmap's outer ring, which has no children at all.
    #[test]
    fn every_child_range_inverts_the_parent_rule() {
        for d in [
            VsmLightDesc::clipmap(4, 8),
            VsmLightDesc::clipmap(3, 16),
            VsmLightDesc::quadtree(5),
            VsmLightDesc::cube(4),
        ] {
            assert_the_ranges_invert_the_parent_rule(&d);
        }
    }

    /// **The general clipmap parent rule CONTAINS the concentric one** (P27.3).
    ///
    /// P27.3 snaps each clipmap level to its own page stride, which is what stops
    /// a coarse level's grid shifting on every camera step — and costs exactly
    /// one thing: the levels stop being concentric, so `N/4 + x/2` stops being
    /// the parent. The replacement is "which coarse world cell contains this fine
    /// cell", and this arm is the pin that the replacement is not a *change*:
    /// at the default origins it is the same function, page for page, over every
    /// page of every level of four grids — including the two that are multiples
    /// of four and **not** powers of two, where `N/4` is not a shift.
    #[test]
    fn the_general_clipmap_parent_rule_reduces_to_the_concentric_one() {
        for d in [
            VsmLightDesc::clipmap(4, 8),
            VsmLightDesc::clipmap(3, 16),
            VsmLightDesc::clipmap(4, 12),
            VsmLightDesc::clipmap(3, 20),
        ] {
            let n = d.levels[0].pages_x;
            let concentric: Vec<(i64, i64)> =
                vec![(-(i64::from(n) / 2), -(i64::from(n) / 2)); d.levels.len()];
            let mut checked = 0usize;
            for level in 0..d.level_count() {
                for y in 0..n {
                    for x in 0..n {
                        let at = VsmPage::flat(level, x, y);
                        assert_eq!(
                            d.parent_at(&concentric, at),
                            d.parent(at),
                            "{n}: {at:?} re-parents under the default origins"
                        );
                        assert_eq!(d.parent_at(&[], at), d.parent(at), "{n}: {at:?}");
                        checked += 1;
                    }
                }
            }
            // ANTI-VACUITY: the sweep really covered the grid, and the rule it
            // agrees with really is the concentric one rather than `x/2`.
            assert_eq!(checked, (d.level_count() * n * n) as usize);
            assert_eq!(
                d.parent(VsmPage::flat(0, n - 1, n - 1)),
                Some(VsmPage::flat(1, n / 4 + (n - 1) / 2, n / 4 + (n - 1) / 2))
            );
            // …and the offsets are load-bearing: shifting the FINE level one cell
            // re-parents half its pages — the ODD columns, because a cell and its
            // neighbour share a parent and the shift moves which of the two pairs
            // a column lands in. Page 0 is *not* one of them (the concentric
            // origin is even, so cell −N/2 and cell −N/2+1 have one parent), and
            // asserting on it is how the first draft of this arm passed with the
            // offsets deleted.
            let mut shifted = concentric.clone();
            shifted[0].0 += 1;
            assert_eq!(
                d.parent_at(&shifted, VsmPage::flat(0, 0, 0)),
                d.parent(VsmPage::flat(0, 0, 0)),
                "{n}: an even column pairs with the one above it"
            );
            assert_ne!(
                d.parent_at(&shifted, VsmPage::flat(0, 1, 0)),
                d.parent(VsmPage::flat(0, 1, 0)),
                "{n}: a level offset that moves nothing is not an offset"
            );
            let moved = (0..n)
                .filter(|&x| {
                    d.parent_at(&shifted, VsmPage::flat(0, x, 0))
                        != d.parent(VsmPage::flat(0, x, 0))
                })
                .count();
            assert_eq!(moved, (n / 2) as usize, "{n}: {moved} of {n} columns moved");
        }
    }

    /// **A shifted clipmap still has one parent per page, and the ranges still
    /// invert it** (P27.3) — the same total property, under offsets that are not
    /// concentric, at extents where `N/4` is not a shift.
    ///
    /// The offsets swept are the ones per-level snapping actually produces:
    /// `d ∈ {−1, 0, 1}` between consecutive levels, because level `L`'s centre is
    /// within `w_L/2` of the eye and level `L+1`'s within `w_L`, so the two can
    /// never be more than one and a half fine pages apart.
    #[test]
    fn a_shifted_clipmap_still_has_one_parent_per_page() {
        for n in [8u32, 12, 16, 20] {
            let d = VsmLightDesc::clipmap(4, n);
            for dx in [-1i64, 0, 1] {
                for dy in [-1i64, 0, 1] {
                    // Level L's origin is the concentric one plus L·d — the shape
                    // "every level is one cell off its parent" produces.
                    let origins: Vec<(i64, i64)> = (0..d.levels.len())
                        .map(|l| {
                            (
                                -(i64::from(n) / 2) + dx * l as i64,
                                -(i64::from(n) / 2) + dy * l as i64,
                            )
                        })
                        .collect();
                    assert_the_ranges_invert_the_parent_rule_at(&d, &origins, dx == 0 && dy == 0);
                    // A page at the CENTRE of a shifted level still has a parent —
                    // the offsets are small enough that only the rim can hang off.
                    assert!(
                        d.parent_at(&origins, VsmPage::flat(0, n / 2, n / 2))
                            .is_some(),
                        "n={n} d=({dx},{dy}): the middle of the grid lost its parent"
                    );
                }
            }
        }
    }

    /// **The odd extents** — the P26.2 law: a contract must be tested where the
    /// clamp cannot hide a disagreement.
    ///
    /// Every fixture above is a power of two, and on a power-of-two chain the two
    /// guards the parent rule carries are inert: `x/2` never leaves the coarser
    /// grid, so [`VsmLightDesc::parent`]'s documented `.min(pages - 1)` clamp
    /// never fires, and a parent's two children never run past the finer level's
    /// end, so `child_range`'s "the last parent adopts the remainder" branch
    /// agrees with `(lo + 2).min(child_count)` wherever it is taken. Measured, in
    /// the P27.1 audit: **deleting either survived the whole crate.**
    ///
    /// The three shapes below are where they are load-bearing.
    #[test]
    fn the_parent_rule_holds_where_a_clamp_cannot_hide_a_disagreement() {
        // A quadtree with SLIVER levels: `(5/2).max(1)` is 2 and `(2/2).max(1)`
        // is 1, so this is a legal chain — and page 4 of level 0 halves to
        // column 2 of a level that has two columns. The clamp is the only thing
        // between that and an address outside the grid.
        let sliver = VsmLightDesc {
            kind: VsmTreeKind::Quadtree,
            levels: vec![
                VsmLevelDesc::square(5),
                VsmLevelDesc::square(2),
                VsmLevelDesc::square(1),
            ],
        };
        assert_eq!(
            sliver.parent(VsmPage::flat(0, 4, 4)),
            Some(VsmPage::flat(1, 1, 1)),
            "the odd column escaped the coarser grid — `x / 2` is 2 and level 1 \
             has two columns"
        );
        assert_eq!(
            sliver.child_x_range(1, 1),
            2..5,
            "the last parent adopts 2, 3 AND 4"
        );
        assert_eq!(sliver.child_x_range(1, 0), 0..2);
        assert_the_ranges_invert_the_parent_rule(&sliver);

        // The same shape one level shallower, where the sliver IS the root.
        let stub = VsmLightDesc {
            kind: VsmTreeKind::Quadtree,
            levels: vec![VsmLevelDesc::square(3), VsmLevelDesc::square(1)],
        };
        assert_eq!(
            stub.parent(VsmPage::flat(0, 2, 2)),
            Some(VsmPage::flat(1, 0, 0))
        );
        assert_eq!(stub.child_x_range(1, 0), 0..3);
        assert_the_ranges_invert_the_parent_rule(&stub);

        // A clipmap grid that is a multiple of four and NOT a power of two, so
        // `N/4 + x/2` lands on 3 + x/2 rather than on a shift.
        let twelve = VsmLightDesc::clipmap(3, 12);
        assert_eq!(
            twelve.parent(VsmPage::flat(0, 0, 0)),
            Some(VsmPage::flat(1, 3, 3))
        );
        assert_eq!(
            twelve.parent(VsmPage::flat(0, 11, 11)),
            Some(VsmPage::flat(1, 8, 8))
        );
        assert!(
            twelve.child_x_range(1, 2).is_empty(),
            "column 2 is outer ring"
        );
        assert_eq!(twelve.child_x_range(1, 8), 10..12);
        assert!(twelve.child_x_range(1, 9).is_empty());
        assert_the_ranges_invert_the_parent_rule(&twelve);

        // …and a ONE-level tree, the other boundary: nothing has a parent and
        // nothing has children, which `clipmap_levels = 1` reaches from settings.
        let flat = VsmLightDesc::clipmap(1, 8);
        assert_the_ranges_invert_the_parent_rule(&flat);
        assert_eq!(flat.parent(VsmPage::flat(0, 0, 0)), None);
        assert_eq!(flat.coarsest_level(), 0);
    }

    /// **What a mandatory floor would actually cost** — the arithmetic the
    /// no-floor ruling rests on, pinned rather than quoted.
    ///
    /// The P27.1 audit found the ledger, this module's docs and the crate docs
    /// all pairing the **16 k** configuration's page count with the **8 k**
    /// configuration's multiplier ("16 384 … four times the whole default
    /// atlas"). Both numbers were real and they belong to different grids; the
    /// defence is stronger with them straight, which is why it is a correction
    /// and not a retraction.
    #[test]
    fn pinning_a_clipmaps_coarsest_level_costs_what_the_ruling_says() {
        use crate::atlas::{plan_atlas, VsmAtlasConfig};
        let default_atlas = plan_atlas(VsmAtlasConfig::default()).0.slot_count();
        assert_eq!(default_atlas, 1_024, "the 64 MiB default atlas");

        // The grid the settings actually ship on High: 64 pages a side at 128
        // texels is 8 192 virtual texels per level.
        let eight_k = VsmLightDesc::clipmap(8, 64).levels[0].page_count();
        assert_eq!(eight_k, 4_096);
        assert_eq!(eight_k / default_atlas, 4, "the 8 k floor is FOUR atlases");

        // The grid the phase's "Done when" reaches for: 128 a side, 16 k texels.
        let sixteen_k = VsmLightDesc::clipmap(8, 128).levels[0].page_count();
        assert_eq!(sixteen_k, 16_384);
        assert_eq!(
            sixteen_k / default_atlas,
            16,
            "16 384 pages is SIXTEEN default atlases, not four — the ruling's own \
             number, straight"
        );
        // Four times the largest atlas ONE 8 192² texture can hold, which is the
        // reading the first draft's "four times" was accidentally true of.
        let biggest = plan_atlas(VsmAtlasConfig {
            budget_bytes: u64::MAX / 2,
            ..Default::default()
        })
        .0
        .slot_count();
        assert_eq!(biggest, 4_096, "64 × 64 slots is one 8 192² atlas");
        assert_eq!(sixteen_k / biggest, 4);
    }

    /// ANTI-VACUITY for the arm above: a clipmap level really does have pages
    /// with **no** children, so `child_range` returning an empty range is a case
    /// the sweep reaches rather than a branch nothing takes.
    #[test]
    fn a_clipmaps_outer_ring_has_no_children() {
        let d = VsmLightDesc::clipmap(3, 8);
        assert!(d.child_x_range(1, 0).is_empty(), "column 0 is outside");
        assert!(d.child_x_range(1, 1).is_empty());
        assert_eq!(d.child_x_range(1, 2), 0..2, "column 2 is the first inside");
        assert_eq!(d.child_x_range(1, 5), 6..8, "column 5 is the last inside");
        assert!(
            d.child_x_range(1, 6).is_empty(),
            "column 6 is outside again"
        );
        assert!(d.child_x_range(1, 99).is_empty(), "off the grid");
        // …and level 0 has no children by construction (nothing is finer).
        assert!(d.child_x_range(0, 0).is_empty());
    }

    #[test]
    fn a_quadtree_halves_to_one_root_page_and_a_cube_is_six_of_them() {
        let q = VsmLightDesc::quadtree(4);
        q.validate().expect("a real quadtree");
        assert_eq!(
            q.levels,
            vec![
                VsmLevelDesc::square(8),
                VsmLevelDesc::square(4),
                VsmLevelDesc::square(2),
                VsmLevelDesc::square(1),
            ]
        );
        assert_eq!(q.face_page_count(), 64 + 16 + 4 + 1);
        assert_eq!(q.page_count(), q.face_page_count(), "one face");
        assert_eq!(
            q.parent(VsmPage::flat(0, 5, 3)),
            Some(VsmPage::flat(1, 2, 1))
        );

        let c = VsmLightDesc::cube(4);
        c.validate().expect("a real cube");
        assert_eq!(c.faces(), 6);
        assert_eq!(c.page_count(), 6 * q.face_page_count());
        // Faces are outermost in the entry order, so face 1's first page follows
        // face 0's last.
        assert_eq!(c.entry_index(VsmPage::new(1, 0, 0, 0)), Some(85));
        assert_eq!(c.level_first_entry(1, 0), Some(85));
        assert_eq!(c.level_first_entry(0, 1), Some(64));
        assert_eq!(c.level_first_entry(6, 0), None);
        // …and a face's parent chain stays on its own face.
        assert_eq!(
            c.ancestor(VsmPage::new(4, 0, 7, 7), 3),
            Some(VsmPage::new(4, 3, 0, 0))
        );
    }

    /// Every entry index of every tree kind is distinct and dense — the property
    /// the table layout AND the mask layout both rest on.
    #[test]
    fn entry_indices_are_dense_and_unique_for_every_kind() {
        for d in [
            VsmLightDesc::clipmap(5, 4),
            VsmLightDesc::quadtree(3),
            VsmLightDesc::cube(3),
        ] {
            let mut seen = std::collections::BTreeSet::new();
            for face in 0..d.faces() {
                for level in 0..d.level_count() {
                    let l = d.levels[level as usize];
                    for y in 0..l.pages_y {
                        for x in 0..l.pages_x {
                            let i = d
                                .entry_index(VsmPage::new(face, level, x, y))
                                .expect("in grid");
                            assert!(seen.insert(i), "{:?}: index {i} names two pages", d.kind);
                        }
                    }
                }
            }
            assert_eq!(seen.len() as u32, d.page_count());
            assert_eq!(seen.iter().next_back().copied(), Some(d.page_count() - 1));
        }
    }

    /// **A constructor must survive an absurd request**, because `levels` comes
    /// from a host setting rather than from content: `vec![_; u32::MAX as usize]`
    /// is an out-of-memory abort, and an abort is not the refusal `validate`
    /// exists to give.
    #[test]
    fn an_absurd_level_count_is_clamped_before_it_is_allocated() {
        for levels in [u32::MAX, u32::MAX / 2, 1_000_000, MAX_VSM_LEVELS as u32 + 1] {
            let d = VsmLightDesc::clipmap(levels, 8);
            assert_eq!(d.level_count(), MAX_VSM_LEVELS as u32, "{levels}");
            d.validate()
                .expect("a clamped clipmap is still a real tree");
            let q = VsmLightDesc::quadtree(levels);
            assert_eq!(q.level_count(), MAX_VSM_LEVELS as u32, "{levels}");
            q.validate()
                .expect("a clamped quadtree is still a real tree");
        }
        // …and zero is clamped UP, so a tree always has a level to resolve into.
        assert_eq!(VsmLightDesc::clipmap(0, 8).level_count(), 1);
        assert_eq!(VsmLightDesc::quadtree(0).level_count(), 1);
    }

    /// **The other half of that constructor takes a setting too**, and it went in
    /// unguarded — the P27.1 audit's finding.
    ///
    /// `pages_per_side` reaches [`VsmLightDesc::clipmap`] from a host-set
    /// `VsmSettings::clipmap_pages_per_side` exactly as `levels` does, and
    /// `65 536²` is `2³²`. With a plain multiply that count **wrapped to zero**
    /// in release: the descriptor validated, `register_light` compared 0 against
    /// its million-page ceiling and accepted, and the relayout then walked a
    /// four-billion-page grid writing into a table with no entries. In debug it
    /// was an overflow panic. Neither is the refusal.
    ///
    /// Saturating makes the existing refusal reachable, which is the whole fix: a
    /// count that saturates is a count that is **at least** the ceiling.
    #[test]
    fn an_absurd_page_grid_saturates_rather_than_wrapping_to_nothing() {
        // 65 536² is exactly 2³² — the wrap that read as ZERO pages.
        assert_eq!(VsmLevelDesc::square(65_536).page_count(), u32::MAX);
        assert_eq!(VsmLevelDesc::square(u32::MAX).page_count(), u32::MAX);
        // …and a grid that does NOT overflow on its own still overflows the sum
        // over sixteen levels, which is the case a per-level check would miss.
        assert_eq!(VsmLevelDesc::square(40_000).page_count(), 1_600_000_000);
        assert_eq!(
            VsmLightDesc::clipmap(16, 40_000).page_count(),
            u32::MAX,
            "sixteen levels of 1.6 G pages wrapped instead of saturating"
        );
        // A cube multiplies by six on top of that, and sixteen levels of one is
        // 8.6 G pages — the face multiply is a third place the count can wrap.
        assert_eq!(VsmLightDesc::cube(8).page_count(), 6 * 21_845);
        assert_eq!(VsmLightDesc::cube(16).page_count(), u32::MAX);
        // Every one of them is still a legal SHAPE — which is precisely why the
        // page-space ceiling rather than `validate` has to be what refuses them.
        for grid in [65_536u32, 100_000, u32::MAX - 3] {
            let d = VsmLightDesc::clipmap(8, grid);
            assert_eq!(d.validate(), Ok(()), "{grid} is a square multiple of 4");
            assert_eq!(d.page_count(), u32::MAX, "{grid}");
        }
        // The addressing helpers saturate with it, so a caller that never went
        // through the residency's door cannot make one of them panic either.
        let d = VsmLightDesc::clipmap(8, 65_536);
        assert_eq!(d.entry_index(VsmPage::flat(7, 0, 0)), Some(u32::MAX));
        assert_eq!(d.level_first_entry(0, 7), Some(u32::MAX));
    }

    #[test]
    fn a_descriptor_that_is_not_a_tree_is_refused_by_name() {
        let mut d = VsmLightDesc::clipmap(3, 8);
        d.levels[1] = VsmLevelDesc::square(4);
        assert_eq!(
            d.validate(),
            Err(VsmDescError::ClipmapLevelsDiffer { level: 1 })
        );

        let d = VsmLightDesc {
            kind: VsmTreeKind::Clipmap,
            levels: vec![VsmLevelDesc::square(6); 2],
        };
        assert_eq!(
            d.validate(),
            Err(VsmDescError::ClipmapGrid {
                pages_x: 6,
                pages_y: 6
            })
        );

        let mut d = VsmLightDesc::quadtree(4);
        d.levels.remove(1); // 8 straight to 2 — not a quadtree
        assert_eq!(d.validate(), Err(VsmDescError::QuadtreeChain { level: 1 }));

        let d = VsmLightDesc {
            kind: VsmTreeKind::Quadtree,
            levels: Vec::new(),
        };
        assert_eq!(d.validate(), Err(VsmDescError::EmptyTree));

        let d = VsmLightDesc {
            kind: VsmTreeKind::Quadtree,
            levels: vec![VsmLevelDesc::square(0)],
        };
        assert_eq!(d.validate(), Err(VsmDescError::EmptyLevel { level: 0 }));

        let d = VsmLightDesc {
            kind: VsmTreeKind::Clipmap,
            levels: vec![VsmLevelDesc::square(4); MAX_VSM_LEVELS + 1],
        };
        assert_eq!(
            d.validate(),
            Err(VsmDescError::TooManyLevels {
                levels: MAX_VSM_LEVELS + 1,
                max: MAX_VSM_LEVELS
            })
        );
    }

    /// The wire discriminants are **frozen**: a shader reads the number to pick a
    /// parent rule, so re-ordering the variants re-parents every page silently.
    #[test]
    fn the_tree_kind_wire_values_are_frozen() {
        assert_eq!(VsmTreeKind::Clipmap.wire(), 0);
        assert_eq!(VsmTreeKind::Quadtree.wire(), 1);
        assert_eq!(VsmTreeKind::Cube.wire(), 2);
        assert_eq!(VsmTreeKind::Clipmap.faces(), 1);
        assert_eq!(VsmTreeKind::Quadtree.faces(), 1);
        assert_eq!(VsmTreeKind::Cube.faces(), 6);
    }
}
