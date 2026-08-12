//! The **physical page pool**: how many interchangeable slots a byte budget
//! buys, and how they lay out in one atlas texture.
//!
//! A stored tile is the same size for every tile of every mip of every texture in
//! a pool (the container's uniform-page design, P26.1), which is the property
//! that lets the pool be a flat array of interchangeable slots with **no
//! suballocator at all**. `inf-vgeom` needs a first-fit free-list because a
//! meshlet page's four sections are different sizes; here every allocation is one
//! slot, so allocation is "take the lowest free index" and fragmentation cannot
//! exist. That is a real simplification bought by a real cost, and P26.1 measured
//! the cost: a v2 payload is **6.8× larger than v1 at 128² BC1**, 1.16× at 2048²,
//! because eight of a small texture's nine levels are one 136² tile each.

/// The page's storage format — **the pool's format, not the file's**.
///
/// These are not the same quantity and the difference is the whole transcode
/// story: on an adapter without `TEXTURE_COMPRESSION_BC` a `Bc1` container feeds
/// an `Rgba8` pool, through `TiledTextureReader::tile_rgba8`, at 8× the page
/// bytes. So this enum is not a mirror of `inf_material::TextureFormat` that
/// could drift from it — it answers a different question, and
/// `inf_material` provides the `From` that answers the *other* one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageFormat {
    Rgba8,
    Bc1,
    Bc3,
}

impl PageFormat {
    /// Bytes one page occupies at `stored_tile_size` texels per side.
    ///
    /// The stored side is a multiple of 4 by construction (136 = 34 blocks), so
    /// the block division is exact and a page is always 16-byte aligned by its
    /// own size — which is why the atlas needs no per-slot padding.
    pub fn page_bytes(&self, stored_tile_size: u32) -> u64 {
        let s = stored_tile_size as u64;
        match self {
            PageFormat::Rgba8 => s * s * 4,
            PageFormat::Bc1 => (s / 4) * (s / 4) * 8,
            PageFormat::Bc3 => (s / 4) * (s / 4) * 16,
        }
    }

    /// Whether this format needs `TEXTURE_COMPRESSION_BC`.
    pub fn needs_bc(&self) -> bool {
        !matches!(self, PageFormat::Rgba8)
    }
}

/// Default page-pool budget: **24 MiB**.
///
/// Sized against what one atlas can hold rather than against what a GPU has, so
/// the default configuration never trips the [`VtAdvisory::AtlasCapped`] warning
/// on a first run. At `136²` BC1 pages (9 248 B) that is 2 721 pages — laid out
/// 46×59 = 2 714 slots, 44 Mtexels of unique detail, comfortably inside the
/// 60×60 = 3 600 slots an 8 192² atlas holds. The same budget on the RGBA8
/// transcode fallback (73 984 B per page) buys 340 pages: the 8× cost of that
/// arm, paid in resolution rather than in a silent failure.
pub const DEFAULT_VT_BUDGET_BYTES: u64 = 24 * 1024 * 1024;

/// How a pool is configured. One atlas, one format, one slot size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtPoolConfig {
    /// The format the atlas is created in.
    pub format: PageFormat,
    /// Texels per side of a slot — the container's `STORED_TILE_SIZE` (136).
    pub stored_tile_size: u32,
    /// The VRAM ceiling for the atlas, in bytes. **Never exceeded**: the granted
    /// slot count is floored against it, and there is no second ceiling to drift.
    pub budget_bytes: u64,
    /// `Limits::max_texture_dimension_2d` of the device the atlas will live on.
    /// Probed by the mirror rather than assumed here — `inf-vt` names no `wgpu`.
    pub max_texture_dim: u32,
}

/// `Limits::default().max_texture_dimension_2d` in wgpu 30 — the value the
/// renderer requests and therefore the one a pool is planned against unless a
/// caller probes something else.
pub const DEFAULT_MAX_TEXTURE_DIM: u32 = 8192;

impl Default for VtPoolConfig {
    fn default() -> Self {
        Self {
            format: PageFormat::Bc1,
            stored_tile_size: 136,
            budget_bytes: DEFAULT_VT_BUDGET_BYTES,
            max_texture_dim: DEFAULT_MAX_TEXTURE_DIM,
        }
    }
}

/// Where the slots are in the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VtPoolGeometry {
    pub slots_x: u32,
    pub slots_y: u32,
    /// Texels per side of a slot.
    pub stored_tile_size: u32,
}

impl VtPoolGeometry {
    #[inline]
    pub fn slot_count(&self) -> u32 {
        self.slots_x * self.slots_y
    }
    #[inline]
    pub fn atlas_width(&self) -> u32 {
        self.slots_x * self.stored_tile_size
    }
    #[inline]
    pub fn atlas_height(&self) -> u32 {
        self.slots_y * self.stored_tile_size
    }
    /// The atlas texel of slot `slot`'s top-left corner.
    ///
    /// A multiple of `stored_tile_size`, which is a multiple of 4 — so a slot
    /// origin is always BC-block aligned and `write_texture` can target it
    /// directly. (`copy_*_texture` refuses a block-format origin that is not.)
    pub fn slot_origin(&self, slot: u32) -> Option<(u32, u32)> {
        if slot >= self.slot_count() {
            return None;
        }
        Some((
            (slot % self.slots_x) * self.stored_tile_size,
            (slot / self.slots_x) * self.stored_tile_size,
        ))
    }
}

/// Something a caller must be told about its pool, rather than have silently
/// applied to it (the no-silent-caps doctrine).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VtAdvisory {
    #[error(
        "the {budget_bytes} B page budget asks for {wanted_slots} pages but one \
         {max_texture_dim}² atlas holds {granted_slots}; the surplus budget is not spent. \
         Raise `max_texture_dim` if the adapter allows it, or lower the budget to match"
    )]
    AtlasCapped {
        wanted_slots: u64,
        granted_slots: u32,
        max_texture_dim: u32,
        budget_bytes: u64,
    },
    #[error(
        "the {budget_bytes} B page budget does not hold one {page_bytes} B page; the pool has \
         no slots and every texture registration will be refused"
    )]
    BudgetBelowOnePage { budget_bytes: u64, page_bytes: u64 },
}

/// Plan a pool: how many slots the budget buys and how they lay out.
///
/// **Pure arithmetic, deterministic, and it never exceeds the budget.** The slot
/// count is floored by the budget *and* by what one atlas can address, and the
/// rectangle is chosen by scanning every legal width for the one that wastes the
/// fewest pages, tie-broken toward the squarest. (Scanning is not extravagant: the
/// range is at most `max_texture_dim / stored_tile_size` — 60 at the default — and
/// it runs once per pool, not once per frame. A closed form would be
/// `ceil(sqrt(n))`, which loses 40 % of a 5-page budget to a 3×1 rectangle.)
///
/// A rectangle cannot always spend the whole budget — 2 721 pages lay out as
/// 46×59 = 2 714, leaving seven — and that residual is **arithmetic, not a cap**:
/// it is bounded by construction, it is visible in
/// [`slot_count`](VtPoolGeometry::slot_count), and no behaviour changes on
/// account of it. It therefore gets no advisory; raising one on every ordinary
/// budget would be the noise that makes the two real advisories below
/// unreadable. What *is* asserted, in this module's tests, is that the chosen
/// rectangle is **optimal** — no other legal rectangle inside the budget holds
/// more slots.
///
/// # One atlas, and what happens when the budget wants two
///
/// A `wgpu` texture is capped at `max_texture_dimension_2d`, 8 192 by default, so
/// `8192 / 136 = 60` slots per side and **3 600 slots is the ceiling of one
/// atlas** — 33.3 MB in BC1, 266 MB as RGBA8. Multiple atlas layers are deferred
/// until a measurement asks for them; until then a budget past that ceiling is
/// *reported*, not silently applied, which is the difference between a documented
/// bound and a mystery.
pub fn plan_pool(cfg: VtPoolConfig) -> (VtPoolGeometry, Vec<VtAdvisory>) {
    let mut advisories = Vec::new();
    let stored = cfg.stored_tile_size.max(4);
    let page_bytes = cfg.format.page_bytes(stored);
    let per_side = (cfg.max_texture_dim / stored).max(1);
    let wanted = if page_bytes == 0 {
        0
    } else {
        cfg.budget_bytes / page_bytes
    };

    if wanted == 0 {
        advisories.push(VtAdvisory::BudgetBelowOnePage {
            budget_bytes: cfg.budget_bytes,
            page_bytes,
        });
        return (
            VtPoolGeometry {
                slots_x: 0,
                slots_y: 0,
                stored_tile_size: stored,
            },
            advisories,
        );
    }

    let ceiling = u64::from(per_side) * u64::from(per_side);
    let n = wanted.min(ceiling) as u32;
    if wanted > ceiling {
        advisories.push(VtAdvisory::AtlasCapped {
            wanted_slots: wanted,
            granted_slots: n,
            max_texture_dim: cfg.max_texture_dim,
            budget_bytes: cfg.budget_bytes,
        });
    }

    // The best rectangle inside `n` slots: maximise the area, tie-break toward
    // the squarest. Both keys are integers, so the choice is exact and portable.
    let mut best = (1u32, 1u32);
    let mut best_key = (0u32, u32::MAX);
    for x in 1..=n.min(per_side) {
        let y = (n / x).min(per_side);
        if y == 0 {
            continue;
        }
        let key = (x * y, x.abs_diff(y));
        if key.0 > best_key.0 || (key.0 == best_key.0 && key.1 < best_key.1) {
            best_key = key;
            best = (x, y);
        }
    }
    (
        VtPoolGeometry {
            slots_x: best.0,
            slots_y: best.1,
            stored_tile_size: stored,
        },
        advisories,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_page_is_the_container_s_stored_tile() {
        // The numbers `inf_material::tiles` writes: 136² is 34×34 blocks.
        assert_eq!(PageFormat::Bc1.page_bytes(136), 34 * 34 * 8);
        assert_eq!(PageFormat::Bc3.page_bytes(136), 34 * 34 * 16);
        assert_eq!(PageFormat::Rgba8.page_bytes(136), 136 * 136 * 4);
        // …and the transcode fallback's measured cost (P26.1 audit: 8× for BC1,
        // 4× for BC3, NOT 4× for both as five documents once said).
        assert_eq!(
            PageFormat::Rgba8.page_bytes(136),
            PageFormat::Bc1.page_bytes(136) * 8
        );
        assert_eq!(
            PageFormat::Rgba8.page_bytes(136),
            PageFormat::Bc3.page_bytes(136) * 4
        );
    }

    #[test]
    fn the_pool_never_exceeds_its_budget() {
        for budget in [
            1u64,
            9_247,
            9_248,
            18_496,
            100_000,
            DEFAULT_VT_BUDGET_BYTES,
            1 << 30,
        ] {
            for format in [PageFormat::Rgba8, PageFormat::Bc1, PageFormat::Bc3] {
                let cfg = VtPoolConfig {
                    format,
                    budget_bytes: budget,
                    ..Default::default()
                };
                let (g, _) = plan_pool(cfg);
                let cost = u64::from(g.slot_count()) * format.page_bytes(136);
                assert!(
                    cost <= budget,
                    "{format:?} at {budget} B allocated {cost} B in {}×{}",
                    g.slots_x,
                    g.slots_y
                );
                assert!(g.slots_x <= 60 && g.slots_y <= 60, "8192 / 136 = 60");
            }
        }
    }

    #[test]
    fn the_default_pool_fits_one_atlas_without_an_advisory() {
        let (g, advisories) = plan_pool(VtPoolConfig::default());
        assert_eq!((g.slots_x, g.slots_y), (46, 59));
        assert_eq!(g.slot_count(), 2_714);
        assert_eq!(g.atlas_width(), 46 * 136);
        assert_eq!(g.atlas_height(), 59 * 136);
        assert!(g.atlas_width() <= 8192 && g.atlas_height() <= 8192);
        assert_eq!(advisories, Vec::new(), "the DEFAULT must be quiet");
    }

    /// **The rectangle is optimal**, not merely plausible: nothing else that fits
    /// the budget and the atlas holds more slots. Asserted by an independent
    /// exhaustive scan over both axes, so a regression in the (one-axis) planner
    /// cannot agree with the checker by sharing its arithmetic.
    #[test]
    fn the_chosen_rectangle_is_the_largest_that_fits() {
        for budget_pages in [1u64, 2, 3, 5, 7, 16, 100, 999, 2_721, 3_600, 4_000] {
            let format = PageFormat::Bc1;
            let page = format.page_bytes(136);
            let cfg = VtPoolConfig {
                format,
                budget_bytes: page * budget_pages,
                ..Default::default()
            };
            let (g, _) = plan_pool(cfg);
            let n = budget_pages.min(60 * 60) as u32;
            let mut best = 0;
            for x in 1..=60u32 {
                for y in 1..=60u32 {
                    if x * y <= n {
                        best = best.max(x * y);
                    }
                }
            }
            assert_eq!(
                g.slot_count(),
                best,
                "{budget_pages} pages: {}×{} is not the largest rectangle",
                g.slots_x,
                g.slots_y
            );
        }
    }

    #[test]
    fn a_budget_past_one_atlas_is_reported_not_silently_applied() {
        let cfg = VtPoolConfig {
            budget_bytes: 512 * 1024 * 1024,
            ..Default::default()
        };
        let (g, advisories) = plan_pool(cfg);
        assert_eq!(g.slot_count(), 3_600, "60 × 60 is one atlas");
        let capped = advisories
            .iter()
            .find(|a| matches!(a, VtAdvisory::AtlasCapped { .. }))
            .expect("the cap is named");
        assert!(
            capped.to_string().contains("3600"),
            "the advisory says what was granted: {capped}"
        );
    }

    #[test]
    fn a_budget_below_one_page_is_named_rather_than_rounded_up() {
        let cfg = VtPoolConfig {
            budget_bytes: 4_096,
            ..Default::default()
        };
        let (g, advisories) = plan_pool(cfg);
        assert_eq!(g.slot_count(), 0);
        assert!(matches!(
            advisories.as_slice(),
            [VtAdvisory::BudgetBelowOnePage { .. }]
        ));
    }

    #[test]
    fn slot_origins_tile_the_atlas_and_are_block_aligned() {
        let (g, _) = plan_pool(VtPoolConfig::default());
        let mut seen = std::collections::BTreeSet::new();
        for slot in 0..g.slot_count() {
            let o = g.slot_origin(slot).expect("in range");
            assert!(seen.insert(o), "slot {slot} reuses origin {o:?}");
            assert_eq!(o.0 % 4, 0);
            assert_eq!(o.1 % 4, 0);
            assert!(o.0 + g.stored_tile_size <= g.atlas_width());
            assert!(o.1 + g.stored_tile_size <= g.atlas_height());
        }
        assert_eq!(g.slot_origin(g.slot_count()), None);
    }

    /// The rectangle scan beats the obvious closed form, which is why it exists.
    #[test]
    fn the_rectangle_scan_wastes_fewer_pages_than_ceil_sqrt() {
        // 5 pages: `ceil(sqrt(5)) = 3` wide gives 3×1 = 3 slots and loses 2 of
        // the 5 the budget paid for. The scan finds 1×5 and loses none.
        let cfg = VtPoolConfig {
            format: PageFormat::Bc1,
            budget_bytes: PageFormat::Bc1.page_bytes(136) * 5,
            ..Default::default()
        };
        let (g, _) = plan_pool(cfg);
        assert_eq!((g.slots_x, g.slots_y), (1, 5));
        assert_eq!(g.slot_count(), 5, "a strip spends the whole budget");
        // 2 721 is the shape the closed form is *nearly* right on, and still is
        // not: `ceil(sqrt)` = 53 wide → 53×51 = 2 703.
        let (g, _) = plan_pool(VtPoolConfig::default());
        assert!(g.slot_count() > 2_703, "the scan beats 53×51");
    }
}
