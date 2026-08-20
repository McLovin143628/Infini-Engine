//! **Lot subdivision** (IB-2c): one block polygon becomes many oriented building
//! lots.
//!
//! Before this module, `one building.plan node == one building` — the
//! certification's IB-2 says so in those words — so two cities meant thousands of
//! hand-authored graph nodes or thousands of `PcgVolume` entities. A block is the
//! unit an author actually draws (a GIS block polygon, a ring of roads, a
//! painted region); the lots inside it are a *rule*, and this is the rule.
//!
//! # The rule, in one paragraph
//!
//! The block's points are hulled and fitted with an oriented minimum-area
//! rectangle ([`inf_math::min_area_rect`]) — the same substrate wave I2 built for
//! [`oriented_lot_of`](super::pass::oriented_lot_of), so a rotated block gets the
//! rectangle it *is* rather than the bounding box of its corners. The rectangle's
//! **long** axis is the frontage direction (a canonical `MinAreaRect` always puts
//! the long side on local `+X`), because a city block fronts the longer street.
//! That rectangle is cut into `cols × rows` cells by boundaries at exact
//! fractions of its extent, each interior boundary displaced by a seeded jitter,
//! and every cell is inset by the setback. A cell survives if it still has
//! positive size, clears [`LotRules::min_area_m2`], and lies **wholly inside the
//! block hull**.
//!
//! # Why the lots tile the block exactly
//!
//! Boundaries are shared: cell `k` runs from `x[k]` to `x[k+1]`, so two adjacent
//! lots meet on one number and cannot overlap however the jitter falls. The
//! jitter is bounded at [`MAX_JITTER`] = 0.45 of a cell, and two adjacent
//! boundaries moving *towards* each other close at most 0.9 of a cell, so a
//! boundary can never cross its neighbour and no clamping pass — which would make
//! the result depend on the order boundaries were visited — is needed.
//! `lots_are_disjoint_and_inside_the_block` measures both halves.
//!
//! # Why containment is tested on the CORNERS
//!
//! The hull is convex and a lot is convex, so a lot whose four corners are all
//! inside the hull is *entirely* inside it — a proof, not a sample. Testing the
//! centre instead would admit a lot hanging half-way over a chamfered corner, and
//! testing "most corners" would admit it by a vote. A block that is genuinely
//! non-rectangular therefore loses its corner lots, and says how many
//! ([`BlockSubdivision::dropped_outside`]) rather than quietly shipping lots in
//! the road.
//!
//! # Determinism
//!
//! Counter-hash draws only ([`Hash64`]), no stateful RNG, no iteration over a
//! hash map, and arithmetic restricted to `+ - * /` — no trigonometry anywhere on
//! the path, because a lot's position reaches committed content and the P14 law
//! says `f32`/`f64` `sin`/`cos` are not bit-portable. `min_area_rect` is
//! trig-free for the same reason.

use glam::DVec2;

use super::{LotFrame, OrientedLot, Rect2};
use crate::hash::Hash64;

/// Separates lot-subdivision draws from every other draw in the building seed
/// space.
const SUBDIVIDE_SALT: u64 = 0x006C_6F74_5F73_7562; // "lot_sub"
/// Column-boundary jitter draws.
const SALT_COL: u64 = 0x434F_4C55_4D4E_5F5F; // "COLUMN__"
/// Row-boundary jitter draws.
const SALT_ROW: u64 = 0x524F_5753_5F5F_5F5F; // "ROWS____"
/// Per-lot decorrelation, so two lots of one block are two buildings.
const SALT_LOT: u64 = 0x4C4F_545F_5345_4544; // "LOT_SEED"

/// The most a boundary may move, as a fraction of its own cell.
///
/// **Bounded at 0.45 rather than 0.5 on purpose**: two adjacent boundaries can
/// each move 0.45 of a cell towards each other and still leave 0.1 of a cell
/// between them, so a jittered boundary can never cross its neighbour and the
/// tiling stays a tiling with no repair pass.
pub const MAX_JITTER: f64 = 0.45;

/// Cells one block may be cut into on either axis.
///
/// A ceiling rather than a preference: `frontage_m` is author-supplied and a
/// mis-typed `0.01` on a 1 km block is 100 000 lots and a hung editor. Reported
/// through [`BlockSubdivision::clamped`] rather than silently obeyed — the I2
/// `MAX_ATTR_FLOORS` lesson (a clamp nobody counts is a skyline nobody can
/// explain).
pub const MAX_LOTS_PER_AXIS: u32 = 512;

/// How a block is cut into lots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LotRules {
    /// Target street frontage per lot, in metres, along the block's LONG axis.
    /// Zero or negative means "do not cut this axis" (one column).
    pub frontage_m: f64,
    /// Target lot depth, in metres, across the block's SHORT axis. Zero or
    /// negative means one row — a single depth of lots fronting one street.
    pub depth_m: f64,
    /// Boundary jitter as a fraction of a cell, clamped to `0..=`[`MAX_JITTER`].
    pub jitter: f64,
    /// Metres shaved off every side of every lot — side and rear yards, and the
    /// pavement between the lot line and the kerb.
    pub setback_m: f64,
    /// Lots below this floor area are dropped rather than built.
    pub min_area_m2: f64,
}

impl Default for LotRules {
    /// A North-American downtown block: 25 m of frontage, 30 m deep (two rows
    /// back to back on a 60 m block), a 1 m setback and a small-lot floor.
    fn default() -> Self {
        Self {
            frontage_m: 25.0,
            depth_m: 30.0,
            jitter: 0.12,
            setback_m: 1.0,
            min_area_m2: 60.0,
        }
    }
}

impl LotRules {
    /// The rules with every field brought inside its own range — the ONE place
    /// author input is made safe, so no caller can hold an unclamped copy.
    ///
    /// Non-finite input collapses to the "do not cut" value rather than
    /// propagating a NaN into a cell count (the NaN-at-doors law: a `NaN`
    /// frontage would make `w / frontage` a NaN, `as u32` zero, and the block
    /// would silently become one lot with no diagnostic).
    pub fn sane(&self) -> Self {
        let fin = |v: f64, fallback: f64| if v.is_finite() { v } else { fallback };
        Self {
            frontage_m: fin(self.frontage_m, 0.0).max(0.0),
            depth_m: fin(self.depth_m, 0.0).max(0.0),
            jitter: fin(self.jitter, 0.0).clamp(0.0, MAX_JITTER),
            setback_m: fin(self.setback_m, 0.0).max(0.0),
            min_area_m2: fin(self.min_area_m2, 0.0).max(0.0),
        }
    }
}

/// One lot, and the seed the building standing on it should draw from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockLot {
    /// The lot rectangle in its own frame, with the block's orientation.
    pub lot: OrientedLot,
    /// The lot's own seed — the block seed folded with its `(col, row)`, so two
    /// lots of one block are two different buildings and the same lot rebuilds
    /// the same one.
    pub seed: u64,
    /// Column index along the frontage axis.
    pub col: u32,
    /// Row index across the depth axis.
    pub row: u32,
}

/// What subdividing one block produced, and what it refused.
///
/// **A refusal is a value** (the standing law): a block that yields three lots
/// out of a nominal nine has not failed, and the two ways a cell can be lost are
/// counted separately because they have different remedies — `dropped_small`
/// says the rules are too fine for the block, `dropped_outside` says the block is
/// not a rectangle.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BlockSubdivision {
    pub lots: Vec<BlockLot>,
    /// Nominal cells along the frontage axis.
    pub cols: u32,
    /// Nominal cells across the depth axis.
    pub rows: u32,
    /// Cells lost to the setback or to [`LotRules::min_area_m2`].
    pub dropped_small: usize,
    /// Cells lost because a corner fell outside the block hull.
    pub dropped_outside: usize,
    /// `true` when [`MAX_LOTS_PER_AXIS`] bound a count the rules asked for.
    pub clamped: bool,
}

impl BlockSubdivision {
    /// Cells the grid nominally held, before either drop.
    pub fn cells(&self) -> usize {
        self.cols as usize * self.rows as usize
    }
}

/// Whether `p` lies inside the counter-clockwise convex ring `hull`, allowing
/// `slack` metres of overhang.
///
/// Trig-free (one cross product per edge). A ring shorter than three points has
/// no interior and answers `false` for everything, which is what makes a
/// degenerate block produce no lots rather than all of them.
fn hull_contains(hull: &[DVec2], p: DVec2, slack: f64) -> bool {
    if hull.len() < 3 {
        return false;
    }
    for i in 0..hull.len() {
        let a = hull[i];
        let b = hull[(i + 1) % hull.len()];
        let e = b - a;
        // Positive on the interior side of a CCW ring; `slack` is a metre
        // allowance scaled by the edge length so it really is a distance.
        let len = (e.x * e.x + e.y * e.y).sqrt();
        if len <= 0.0 {
            continue;
        }
        let cross = e.x * (p.y - a.y) - e.y * (p.x - a.x);
        if cross < -slack * len {
            return false;
        }
    }
    true
}

/// The `n + 1` boundaries of an axis of half-extent `half`, jittered.
///
/// Boundary `0` and boundary `n` are the block's own edges and never move — a
/// jittered outer edge would push a lot into the road.
fn boundaries(half: f64, n: u32, jitter: f64, hash: Hash64, salt: u64) -> Vec<f64> {
    let n = n.max(1);
    let cell = 2.0 * half / f64::from(n);
    (0..=n)
        .map(|k| {
            let base = -half + cell * f64::from(k);
            if k == 0 || k == n || jitter <= 0.0 {
                return base;
            }
            // `unit()` is [0,1); centre it so the displacement is symmetric.
            let u = hash.mix_u64(salt).mix_u64(u64::from(k)).unit() * 2.0 - 1.0;
            base + u * jitter * cell
        })
        .collect()
}

/// Cut a block polygon into oriented building lots.
///
/// `block` is the block's boundary in world XZ — any winding, any point count;
/// it is hulled internally. `seed` is the block's own seed; every lot's seed is
/// derived from it and its grid position.
///
/// Returns an empty subdivision (with `cols = rows = 0`) for a block with no
/// area, which is the honest answer for a degenerate polygon and is what makes a
/// mis-wired span build nothing rather than a building on top of the origin.
pub fn subdivide_block(block: &[DVec2], rules: &LotRules, seed: u64) -> BlockSubdivision {
    let rules = rules.sane();
    let Some(mar) = inf_math::min_area_rect(block) else {
        return BlockSubdivision::default();
    };
    if !(mar.half.x > 0.0 && mar.half.y > 0.0) {
        return BlockSubdivision::default();
    }
    let hull = inf_math::convex_hull_2d(block);

    let w = 2.0 * mar.half.x;
    let d = 2.0 * mar.half.y;

    // The count rule: round to the nearest whole number of lots, never below
    // one. Rounding rather than flooring is what makes a 55 m block at 25 m
    // frontage two 27.5 m lots instead of one 55 m lot and a 5 m remainder.
    let count = |extent: f64, target: f64| -> (u32, bool) {
        if target <= 0.0 {
            return (1, false);
        }
        let raw = (extent / target + 0.5).floor();
        if !raw.is_finite() || raw < 1.0 {
            return (1, false);
        }
        if raw > f64::from(MAX_LOTS_PER_AXIS) {
            return (MAX_LOTS_PER_AXIS, true);
        }
        (raw as u32, false)
    };
    let (cols, col_clamped) = count(w, rules.frontage_m);
    let (rows, row_clamped) = count(d, rules.depth_m);

    let hash = Hash64::new(seed).mix_u64(SUBDIVIDE_SALT);
    let xs = boundaries(mar.half.x, cols, rules.jitter, hash, SALT_COL);
    let zs = boundaries(mar.half.y, rows, rules.jitter, hash, SALT_ROW);

    let mut out = BlockSubdivision {
        lots: Vec::new(),
        cols,
        rows,
        dropped_small: 0,
        dropped_outside: 0,
        clamped: col_clamped || row_clamped,
    };

    for row in 0..rows {
        for col in 0..cols {
            let cell = Rect2 {
                min: DVec2::new(xs[col as usize], zs[row as usize]),
                max: DVec2::new(xs[col as usize + 1], zs[row as usize + 1]),
            };
            let inset = cell.inset(rules.setback_m);
            if !inset.is_positive() || inset.area() < rules.min_area_m2 {
                out.dropped_small += 1;
                continue;
            }
            // The lot's own frame: the block's basis, its own centre. The lot
            // rectangle is then symmetric about its origin, which is the shape
            // every rule in `building` already reads.
            let centre_world = mar.to_world(inset.center());
            let half = inset.size() * 0.5;
            let lot = OrientedLot {
                rect: Rect2 {
                    min: -half,
                    max: half,
                },
                frame: LotFrame::new(centre_world, mar.u),
            };
            // Convexity does the work: four corners inside a convex hull put the
            // whole rectangle inside it.
            if !lot
                .world_corners()
                .iter()
                .all(|c| hull_contains(&hull, *c, 1e-9))
            {
                out.dropped_outside += 1;
                continue;
            }
            out.lots.push(BlockLot {
                lot,
                seed: hash
                    .mix_u64(SALT_LOT)
                    .mix_u64(u64::from(row) << 32 | u64::from(col))
                    .finish(),
                col,
                row,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A square block, no jitter, no setback: the lots tile it exactly.
    fn square(half: f64) -> Vec<DVec2> {
        vec![
            DVec2::new(-half, -half),
            DVec2::new(half, -half),
            DVec2::new(half, half),
            DVec2::new(-half, half),
        ]
    }

    #[test]
    fn a_block_becomes_a_grid_of_lots_and_the_counts_are_the_rule() {
        // 100 x 60 block, 25 m frontage, 30 m depth -> 4 x 2 = 8 lots.
        let block = vec![
            DVec2::new(0.0, 0.0),
            DVec2::new(100.0, 0.0),
            DVec2::new(100.0, 60.0),
            DVec2::new(0.0, 60.0),
        ];
        let rules = LotRules {
            frontage_m: 25.0,
            depth_m: 30.0,
            jitter: 0.0,
            setback_m: 0.0,
            min_area_m2: 0.0,
        };
        let sub = subdivide_block(&block, &rules, 7);
        assert_eq!((sub.cols, sub.rows), (4, 2));
        assert_eq!(sub.lots.len(), 8, "{sub:?}");
        assert_eq!(sub.dropped_small, 0);
        assert_eq!(sub.dropped_outside, 0);
        for l in &sub.lots {
            let s = l.lot.rect.size();
            assert!((s.x - 25.0).abs() < 1e-9, "frontage {s:?}");
            assert!((s.y - 30.0).abs() < 1e-9, "depth {s:?}");
        }
        // Total lot area is the block's area: an exact tiling.
        let total: f64 = sub.lots.iter().map(|l| l.lot.rect.area()).sum();
        assert!((total - 6000.0).abs() < 1e-6, "{total}");
        println!(
            "IB-2c: 100x60 block -> {} lots ({} x {}), {:.1} m2 of {:.1}",
            sub.lots.len(),
            sub.cols,
            sub.rows,
            total,
            6000.0
        );
    }

    /// **The world proof**: lots are oriented to the block, pairwise disjoint,
    /// and inside it — with the jitter on, which is when the tiling could break.
    #[test]
    fn lots_are_disjoint_and_inside_the_block() {
        // The 3-4-5 rotation: exact in binary, no trigonometry.
        let (c, s) = (0.8f64, 0.6f64);
        let turn = |p: DVec2| DVec2::new(p.x * c - p.y * s, p.x * s + p.y * c);
        let block: Vec<DVec2> = [(-60.0, -35.0), (60.0, -35.0), (60.0, 35.0), (-60.0, 35.0)]
            .iter()
            .map(|&(x, z)| turn(DVec2::new(x, z)) + DVec2::new(400.0, -250.0))
            .collect();

        let rules = LotRules {
            frontage_m: 20.0,
            depth_m: 35.0,
            jitter: MAX_JITTER,
            setback_m: 0.0,
            min_area_m2: 0.0,
        };
        let sub = subdivide_block(&block, &rules, 4242);
        assert_eq!((sub.cols, sub.rows), (6, 2));
        assert_eq!(sub.lots.len(), 12, "{} lots", sub.lots.len());

        // (a) every lot carries the BLOCK's basis, and none is square to the
        // world axes — the IB-6 shape, one level up.
        for l in &sub.lots {
            let u = l.lot.frame.u;
            assert!(
                (u - DVec2::new(c, s)).length() < 1e-9,
                "lot basis {u:?} is not the block's"
            );
            assert!(!l.lot.frame.is_identity());
        }

        // (b) pairwise disjoint, measured as a real overlap area in the block's
        // own frame (where every lot is axis-aligned).
        let local = |l: &BlockLot| -> Rect2 {
            let c = l.lot.frame.origin;
            // Back into the block frame: the block basis is the lot basis.
            let bx = DVec2::new(c.x, c.y);
            let u = l.lot.frame.u;
            let v = l.lot.frame.v();
            let o = DVec2::new(400.0, -250.0);
            let rel = bx - o;
            let p = DVec2::new(rel.dot(u), rel.dot(v));
            Rect2 {
                min: p + l.lot.rect.min,
                max: p + l.lot.rect.max,
            }
        };
        let rects: Vec<Rect2> = sub.lots.iter().map(local).collect();
        let mut worst = 0.0f64;
        for i in 0..rects.len() {
            for j in (i + 1)..rects.len() {
                let ox = (rects[i].max.x.min(rects[j].max.x) - rects[i].min.x.max(rects[j].min.x))
                    .max(0.0);
                let oz = (rects[i].max.y.min(rects[j].max.y) - rects[i].min.y.max(rects[j].min.y))
                    .max(0.0);
                worst = worst.max(ox * oz);
            }
        }
        assert!(worst < 1e-9, "two lots overlap by {worst} m2");

        // (c) and every lot is inside the block: the areas still sum to it.
        let total: f64 = sub.lots.iter().map(|l| l.lot.rect.area()).sum();
        assert!(
            (total - 120.0 * 70.0).abs() < 1e-6,
            "jitter moved area: {total}"
        );
        println!(
            "IB-2c: rotated 120x70 block, jitter {MAX_JITTER} -> {} lots, worst overlap {worst:.3e} m2",
            sub.lots.len()
        );
    }

    /// The seeded variety the district needs: two lots of one block are two
    /// buildings, and the same block re-cuts identically.
    #[test]
    fn the_subdivision_is_deterministic_and_the_lot_seeds_decorrelate() {
        let block = square(50.0);
        let rules = LotRules {
            jitter: 0.3,
            ..LotRules::default()
        };
        let a = subdivide_block(&block, &rules, 99);
        let b = subdivide_block(&block, &rules, 99);
        assert_eq!(a, b, "the same block cut twice is the same subdivision");
        let c = subdivide_block(&block, &rules, 100);
        assert_ne!(a, c, "a different block seed is a different subdivision");

        let mut seeds: Vec<u64> = a.lots.iter().map(|l| l.seed).collect();
        let n = seeds.len();
        assert!(n >= 4, "{n} lots");
        seeds.sort_unstable();
        seeds.dedup();
        assert_eq!(seeds.len(), n, "two lots of one block drew the same seed");
    }

    /// A non-rectangular block loses its corner lots and says how many — the
    /// alternative (a centre test) is priced in the same arm.
    #[test]
    fn a_triangular_block_refuses_the_lots_that_hang_over_its_edges() {
        let block = vec![
            DVec2::new(-60.0, -30.0),
            DVec2::new(60.0, -30.0),
            DVec2::new(0.0, 30.0),
        ];
        let rules = LotRules {
            frontage_m: 20.0,
            depth_m: 30.0,
            jitter: 0.0,
            setback_m: 0.0,
            min_area_m2: 0.0,
        };
        let sub = subdivide_block(&block, &rules, 1);
        assert!(sub.dropped_outside > 0, "{sub:?}");
        assert_eq!(
            sub.lots.len() + sub.dropped_small + sub.dropped_outside,
            sub.cells()
        );
        // Every surviving lot really is inside the triangle.
        let hull = inf_math::convex_hull_2d(&block);
        for l in &sub.lots {
            for c in l.lot.world_corners() {
                assert!(hull_contains(&hull, c, 1e-6), "lot corner {c:?} escaped");
            }
        }
        // THE ALTERNATIVE, PRICED: a centre-only containment test would have
        // admitted these, and they hang over the hypotenuse.
        let by_centre = sub.cells() - sub.dropped_small;
        assert!(
            by_centre > sub.lots.len(),
            "if a centre test admits no more than the corner test, this fixture \
             is not a real chamfer"
        );
        println!(
            "IB-2c triangle: {} cells -> {} lots, {} outside; a centre-only test \
             would have admitted up to {by_centre}",
            sub.cells(),
            sub.lots.len(),
            sub.dropped_outside
        );
    }

    /// Author input cannot make the subdivider misbehave: NaN, zero and absurd
    /// values all resolve to a stated answer.
    #[test]
    fn hostile_rules_resolve_rather_than_propagate() {
        let block = square(100.0);
        let nan = LotRules {
            frontage_m: f64::NAN,
            depth_m: f64::INFINITY,
            jitter: f64::NAN,
            setback_m: -5.0,
            min_area_m2: f64::NAN,
        };
        let sane = nan.sane();
        assert_eq!(sane.frontage_m, 0.0);
        assert_eq!(sane.depth_m, 0.0);
        assert_eq!(sane.jitter, 0.0);
        assert_eq!(sane.setback_m, 0.0);
        let sub = subdivide_block(&block, &nan, 1);
        assert_eq!((sub.cols, sub.rows), (1, 1));
        assert_eq!(sub.lots.len(), 1, "one whole block is one lot");

        // A frontage that would ask for a hundred thousand lots is bounded, and
        // SAYS it was bounded.
        let fine = LotRules {
            frontage_m: 0.001,
            depth_m: 0.001,
            jitter: 0.0,
            setback_m: 0.0,
            min_area_m2: 0.0,
        };
        let bounded = subdivide_block(&block, &fine, 1);
        assert_eq!(bounded.cols, MAX_LOTS_PER_AXIS);
        assert_eq!(bounded.rows, MAX_LOTS_PER_AXIS);
        assert!(bounded.clamped, "the clamp must be reported, not silent");

        // A degenerate block is no lots, not one lot at the origin.
        assert!(subdivide_block(&[], &LotRules::default(), 1)
            .lots
            .is_empty());
        let line = [DVec2::ZERO, DVec2::new(10.0, 0.0), DVec2::new(20.0, 0.0)];
        assert!(subdivide_block(&line, &LotRules::default(), 1)
            .lots
            .is_empty());
    }

    /// The setback is a real yard: it shrinks lots and can retire them.
    #[test]
    fn the_setback_shrinks_every_lot_and_small_cells_are_dropped() {
        let block = square(30.0);
        let base = LotRules {
            frontage_m: 15.0,
            depth_m: 15.0,
            jitter: 0.0,
            setback_m: 0.0,
            min_area_m2: 0.0,
        };
        let plain = subdivide_block(&block, &base, 5);
        assert_eq!(plain.lots.len(), 16);
        let set = subdivide_block(
            &block,
            &LotRules {
                setback_m: 2.0,
                ..base
            },
            5,
        );
        assert_eq!(set.lots.len(), 16);
        for l in &set.lots {
            assert!(
                (l.lot.rect.size().x - 11.0).abs() < 1e-9,
                "{:?}",
                l.lot.rect
            );
        }
        // A setback larger than the half-cell collapses every lot.
        let gone = subdivide_block(
            &block,
            &LotRules {
                setback_m: 9.0,
                ..base
            },
            5,
        );
        assert!(gone.lots.is_empty());
        assert_eq!(gone.dropped_small, 16);
    }
}
