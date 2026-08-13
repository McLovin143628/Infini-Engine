//! The **physical page atlas**: how many interchangeable `Depth32Float` slots a
//! byte budget buys, and how they lay out in one texture.
//!
//! Every page of every level of every light is the same size, which is the
//! property that lets the atlas be a flat array of interchangeable slots with
//! **no suballocator at all** — `inf_vt::pool`'s argument, and it holds here for
//! a stronger reason: a shadow page's size is not inherited from a file's tile
//! geometry, it is *chosen*, so it can simply be made uniform.
//!
//! # The page geometry is a ruling, not a default
//!
//! [`VSM_PAGE_SIZE`] and [`VSM_PAGE_BORDER`] are measured decisions with a memo
//! behind them (`docs/memos/p27-1-page-geometry.md`), not tuning constants. The
//! short version, because the numbers belong next to the code they constrain:
//!
//! * **128 texels**, because 128 divides `Limits::default()`'s
//!   `max_texture_dimension_2d` of 8 192 exactly — 64 slots a side with **zero**
//!   remainder, against the 60-and-32-texels-wasted a 136-texel page leaves — and
//!   because a 128² `Depth32Float` page is exactly 64 KiB, so a budget in
//!   mebibytes is a whole number of pages.
//! * **no border**, and this is the half that is not obvious. `inf-vt` bakes a
//!   4-texel ring so a filtered sample at a tile edge reads real neighbours. A
//!   shadow page's border would have to be *rasterized*, which is cheap — and
//!   would have to be **re-rasterized whenever a neighbouring page's casters
//!   moved**, which is not: it couples every cached page to its eight
//!   neighbours' invalidation and turns P27's own "moving one object invalidates
//!   exactly the pages its bounds touch" into a statement about **9** pages.
//!   Measured against the alternative in the memo; P27.4 pays for it with
//!   per-tap page-table resolution at page edges instead.

use crate::table::MAX_SLOT_INDEX;

/// Bytes one texel of a shadow page occupies. `Depth32Float`, always — there is
/// no format decision here and deliberately no enum offering one: a shadow page
/// is a depth attachment the raster writes, so the format is fixed by what the
/// hardware will render into rather than by what the content was cooked as.
pub const VSM_PAGE_BYTES_PER_TEXEL: u64 = 4;

/// **Payload texels per side of a page.** 128 — see the module docs.
pub const VSM_PAGE_SIZE: u32 = 128;

/// **Border texels on each side of a page.** Zero — see the module docs.
pub const VSM_PAGE_BORDER: u32 = 0;

/// `Limits::default().max_texture_dimension_2d` in wgpu 30 — the value the
/// renderer requests and therefore the one an atlas is planned against unless a
/// caller probes something else. (`inf_vt::DEFAULT_MAX_TEXTURE_DIM`'s twin; the
/// two crates do not share a constant because neither depends on the other.)
pub const VSM_DEFAULT_MAX_TEXTURE_DIM: u32 = 8192;

/// Default atlas budget: **64 MiB** — 1 024 pages at the ruled geometry, which
/// lay out 32×32 with **nothing** left over, in a 4 096² texture.
///
/// Sized against what one atlas holds rather than against what a GPU has, on
/// `inf_vt::DEFAULT_VT_BUDGET_BYTES`'s reasoning, so the default configuration
/// never trips [`VsmAdvisory::AtlasCapped`] on a first run: 8 192² holds 4 096
/// pages (256 MiB) and this asks for a quarter of them.
pub const DEFAULT_VSM_BUDGET_BYTES: u64 = 64 * 1024 * 1024;

/// How an atlas is configured. One texture, one format, one slot size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmAtlasConfig {
    /// Payload texels per side of a page ([`VSM_PAGE_SIZE`]).
    pub page_size: u32,
    /// Border texels per side ([`VSM_PAGE_BORDER`]).
    pub border: u32,
    /// The VRAM ceiling for the atlas, in bytes. **Never exceeded**: the granted
    /// slot count is floored against it, and there is no second ceiling to drift.
    pub budget_bytes: u64,
    /// `Limits::max_texture_dimension_2d` of the device the atlas will live on.
    /// Probed by the mirror rather than assumed here — `inf-vsm` names no `wgpu`.
    pub max_texture_dim: u32,
}

impl Default for VsmAtlasConfig {
    fn default() -> Self {
        Self {
            page_size: VSM_PAGE_SIZE,
            border: VSM_PAGE_BORDER,
            budget_bytes: DEFAULT_VSM_BUDGET_BYTES,
            max_texture_dim: VSM_DEFAULT_MAX_TEXTURE_DIM,
        }
    }
}

impl VsmAtlasConfig {
    /// Texels per side of a stored page: `border + page_size + border`.
    #[inline]
    pub const fn stored_page_size(&self) -> u32 {
        self.page_size + 2 * self.border
    }

    /// Bytes one page occupies.
    #[inline]
    pub const fn page_bytes(&self) -> u64 {
        let s = self.stored_page_size() as u64;
        s * s * VSM_PAGE_BYTES_PER_TEXEL
    }
}

/// Where the slots are in the atlas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VsmAtlasGeometry {
    pub slots_x: u32,
    pub slots_y: u32,
    /// Texels per side of a slot (`border + page_size + border`).
    pub stored_page_size: u32,
}

impl VsmAtlasGeometry {
    #[inline]
    pub fn slot_count(&self) -> u32 {
        self.slots_x * self.slots_y
    }
    #[inline]
    pub fn atlas_width(&self) -> u32 {
        self.slots_x * self.stored_page_size
    }
    #[inline]
    pub fn atlas_height(&self) -> u32 {
        self.slots_y * self.stored_page_size
    }
    /// The atlas texel of slot `slot`'s top-left corner — the origin a per-page
    /// `set_viewport` / `set_scissor` takes in P27.2.
    pub fn slot_origin(&self, slot: u32) -> Option<(u32, u32)> {
        if slot >= self.slot_count() {
            return None;
        }
        Some((
            (slot % self.slots_x) * self.stored_page_size,
            (slot / self.slots_x) * self.stored_page_size,
        ))
    }
}

/// Something a caller must be told about its atlas rather than have silently
/// applied to it (the no-silent-caps doctrine).
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VsmAdvisory {
    #[error(
        "the {budget_bytes} B page budget asks for {wanted_slots} shadow pages but one \
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
        "the {budget_bytes} B page budget does not hold one {page_bytes} B shadow page; the \
         atlas has no slots and every page request will be deferred"
    )]
    BudgetBelowOnePage { budget_bytes: u64, page_bytes: u64 },
    #[error(
        "the {budget_bytes} B page budget asks for {wanted_slots} shadow pages and an \
         indirection entry addresses {granted_slots}; a slot index is 16 bits. Raising \
         `max_texture_dim` cannot help — this is the entry format"
    )]
    SlotIndexCapped {
        wanted_slots: u64,
        granted_slots: u32,
        budget_bytes: u64,
    },
}

/// Plan an atlas: how many slots the budget buys and how they lay out.
///
/// **Pure arithmetic, deterministic, and it never exceeds the budget.** The slot
/// count is floored by the budget *and* by what one atlas can address, and the
/// rectangle is the one that wastes the fewest pages, tie-broken toward the
/// squarest — `inf_vt::plan_pool`'s scan, for the reason that function's docs
/// give (a closed `ceil(sqrt(n))` loses 40 % of a 5-page budget to a 3×1
/// rectangle). At the ruled geometry the residual is **zero** for any
/// power-of-two page count, which is one of the two things 128 buys.
pub fn plan_atlas(cfg: VsmAtlasConfig) -> (VsmAtlasGeometry, Vec<VsmAdvisory>) {
    let mut advisories = Vec::new();
    let stored = cfg.stored_page_size().max(1);
    let page_bytes = cfg.page_bytes().max(1);
    let per_side = (cfg.max_texture_dim / stored).max(1);
    let wanted = cfg.budget_bytes / page_bytes;

    if wanted == 0 {
        advisories.push(VsmAdvisory::BudgetBelowOnePage {
            budget_bytes: cfg.budget_bytes,
            page_bytes,
        });
        return (
            VsmAtlasGeometry {
                slots_x: 0,
                slots_y: 0,
                stored_page_size: stored,
            },
            advisories,
        );
    }

    let atlas_ceiling = u64::from(per_side) * u64::from(per_side);
    // The second ceiling, and it is the entry format rather than the adapter: a
    // packed entry carries its slot in 16 bits, so 65 536 slots is the limit
    // however large a texture the device allows. Enforced rather than remarked
    // on, because `pack_entry` masks and the 65 537th page would alias onto the
    // first, silently, in release.
    let entry_ceiling = u64::from(MAX_SLOT_INDEX) + 1;
    let ceiling = atlas_ceiling.min(entry_ceiling);
    let n = wanted.min(ceiling) as u32;
    if wanted > ceiling {
        advisories.push(if atlas_ceiling <= entry_ceiling {
            VsmAdvisory::AtlasCapped {
                wanted_slots: wanted,
                granted_slots: n,
                max_texture_dim: cfg.max_texture_dim,
                budget_bytes: cfg.budget_bytes,
            }
        } else {
            VsmAdvisory::SlotIndexCapped {
                wanted_slots: wanted,
                granted_slots: n,
                budget_bytes: cfg.budget_bytes,
            }
        });
    }

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
        VsmAtlasGeometry {
            slots_x: best.0,
            slots_y: best.1,
            stored_page_size: stored,
        },
        advisories,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The page-geometry ruling, as arithmetic** — the two numbers the memo
    /// rests on, asserted here so a "tuning" edit fails a test that names the
    /// document it would contradict.
    #[test]
    fn the_ruled_page_geometry_is_the_one_the_memo_measured() {
        assert_eq!(VSM_PAGE_SIZE, 128);
        assert_eq!(
            VSM_PAGE_BORDER, 0,
            "docs/memos/p27-1-page-geometry.md: a rasterized border couples every \
             cached page to its eight neighbours' invalidation"
        );
        let cfg = VsmAtlasConfig::default();
        assert_eq!(cfg.stored_page_size(), 128);
        assert_eq!(cfg.page_bytes(), 64 * 1024, "a page is exactly 64 KiB");
        // 128 divides 8 192 exactly: 64 slots a side and NOTHING wasted. The
        // alternative the memo weighs — `inf-vt`'s 136 — leaves 32 texels.
        assert_eq!(VSM_DEFAULT_MAX_TEXTURE_DIM % VSM_PAGE_SIZE, 0);
        assert_eq!(VSM_DEFAULT_MAX_TEXTURE_DIM / VSM_PAGE_SIZE, 64);
        assert_eq!(
            VSM_DEFAULT_MAX_TEXTURE_DIM % 136,
            32,
            "the 136-texel residual"
        );
    }

    #[test]
    fn the_default_atlas_is_square_and_spends_its_whole_budget() {
        let (g, advisories) = plan_atlas(VsmAtlasConfig::default());
        assert_eq!((g.slots_x, g.slots_y), (32, 32));
        assert_eq!(g.slot_count(), 1_024);
        assert_eq!(g.atlas_width(), 4_096);
        assert_eq!(g.atlas_height(), 4_096);
        assert_eq!(
            u64::from(g.slot_count()) * VsmAtlasConfig::default().page_bytes(),
            DEFAULT_VSM_BUDGET_BYTES,
            "the budget bought a whole number of pages and the rectangle spent \
             every one of them — the residual `inf_vt::plan_pool` documents is ZERO here"
        );
        assert_eq!(advisories, Vec::new(), "the DEFAULT must be quiet");
    }

    #[test]
    fn the_atlas_never_exceeds_its_budget() {
        for budget in [
            1u64,
            65_535,
            65_536,
            131_072,
            1_000_000,
            DEFAULT_VSM_BUDGET_BYTES,
            1 << 30,
        ] {
            let cfg = VsmAtlasConfig {
                budget_bytes: budget,
                ..Default::default()
            };
            let (g, _) = plan_atlas(cfg);
            let cost = u64::from(g.slot_count()) * cfg.page_bytes();
            assert!(
                cost <= budget,
                "{budget} B allocated {cost} B in {}×{}",
                g.slots_x,
                g.slots_y
            );
            assert!(g.slots_x <= 64 && g.slots_y <= 64, "8192 / 128 = 64");
        }
    }

    /// **The rectangle is optimal**, not merely plausible — asserted by an
    /// independent exhaustive scan over both axes, so a regression in the
    /// (one-axis) planner cannot agree with the checker by sharing its arithmetic.
    #[test]
    fn the_chosen_rectangle_is_the_largest_that_fits() {
        let cfg0 = VsmAtlasConfig::default();
        for budget_pages in [1u64, 2, 3, 5, 7, 17, 100, 999, 1_024, 4_096, 5_000] {
            let cfg = VsmAtlasConfig {
                budget_bytes: cfg0.page_bytes() * budget_pages,
                ..cfg0
            };
            let (g, _) = plan_atlas(cfg);
            let n = budget_pages.min(64 * 64) as u32;
            let mut best = 0;
            for x in 1..=64u32 {
                for y in 1..=64u32 {
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
        let (g, advisories) = plan_atlas(VsmAtlasConfig {
            budget_bytes: 1024 * 1024 * 1024,
            ..Default::default()
        });
        assert_eq!(g.slot_count(), 4_096, "64 × 64 is one atlas");
        let capped = advisories
            .iter()
            .find(|a| matches!(a, VsmAdvisory::AtlasCapped { .. }))
            .expect("the cap is named");
        assert!(
            capped.to_string().contains("4096"),
            "the advisory says what was granted: {capped}"
        );
    }

    /// The entry format is a **different** ceiling with different advice, so it
    /// gets a different advisory — unreachable on every adapter that exists,
    /// which is precisely why it is enforced rather than remarked on.
    #[test]
    fn the_sixteen_bit_slot_index_floors_the_atlas_before_it_can_alias() {
        let cfg0 = VsmAtlasConfig::default();
        // The largest device anything reports: 32 768² is 256 slots a side and
        // 65 536 slots — the whole 16-bit range, exactly, and no advisory.
        let (g, advisories) = plan_atlas(VsmAtlasConfig {
            budget_bytes: cfg0.page_bytes() * 65_536,
            max_texture_dim: 32_768,
            ..cfg0
        });
        assert_eq!(g.slot_count(), 65_536, "256 × 256");
        assert_eq!(advisories, Vec::new());

        // Past it, on a device that does not exist yet: floored, named, and NOT
        // blamed on the atlas.
        let (g, advisories) = plan_atlas(VsmAtlasConfig {
            budget_bytes: cfg0.page_bytes() * 200_000,
            max_texture_dim: 65_536,
            ..cfg0
        });
        assert_eq!(g.slot_count(), 65_536);
        let a = advisories.first().expect("the cap is named");
        assert!(
            matches!(a, VsmAdvisory::SlotIndexCapped { .. }),
            "the atlas held 262 144 slots — the entry format is what capped it: {a}"
        );
        assert!(
            a.to_string().contains("16 bits") && !a.to_string().contains("atlas holds"),
            "and the advice must not be `raise max_texture_dim`: {a}"
        );
    }

    #[test]
    fn a_budget_below_one_page_is_named_rather_than_rounded_up() {
        let (g, advisories) = plan_atlas(VsmAtlasConfig {
            budget_bytes: 4_096,
            ..Default::default()
        });
        assert_eq!(g.slot_count(), 0);
        assert!(matches!(
            advisories.as_slice(),
            [VsmAdvisory::BudgetBelowOnePage { .. }]
        ));
    }

    #[test]
    fn slot_origins_tile_the_atlas_without_overlapping() {
        let (g, _) = plan_atlas(VsmAtlasConfig::default());
        let mut seen = std::collections::BTreeSet::new();
        for slot in 0..g.slot_count() {
            let o = g.slot_origin(slot).expect("in range");
            assert!(seen.insert(o), "slot {slot} reuses origin {o:?}");
            assert!(o.0 + g.stored_page_size <= g.atlas_width());
            assert!(o.1 + g.stored_page_size <= g.atlas_height());
        }
        assert_eq!(g.slot_origin(g.slot_count()), None);
    }

    /// A **bordered** geometry still plans, because the memo's rejected
    /// alternative has to remain measurable rather than become unrepresentable —
    /// and the numbers it costs are the ones the memo quotes.
    #[test]
    fn the_rejected_bordered_geometry_costs_what_the_memo_says() {
        let bordered = VsmAtlasConfig {
            border: 4,
            ..Default::default()
        };
        assert_eq!(bordered.stored_page_size(), 136);
        assert_eq!(bordered.page_bytes(), 136 * 136 * 4);
        // 11.42 % of every page is ring rather than shadow.
        let waste = (bordered.page_bytes() - VsmAtlasConfig::default().page_bytes()) as f64
            / bordered.page_bytes() as f64;
        assert!(
            (waste - 0.1142).abs() < 5e-4,
            "the bordered page wastes {waste:.4} of itself, not 0.1142"
        );
        // …and the same budget buys 121 fewer pages: 907 pages of ring-and-all
        // against 1 024, laid out 21×43 because 907 is PRIME and 60 is the side
        // limit — so four of them are lost to the rectangle as well.
        let (g, _) = plan_atlas(bordered);
        assert_eq!((g.slots_x, g.slots_y), (21, 43));
        assert_eq!(g.slot_count(), 903);
        assert_eq!(
            plan_atlas(VsmAtlasConfig::default()).0.slot_count() - g.slot_count(),
            121
        );
    }
}
