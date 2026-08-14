//! **One stamp domain** (P28.3, clause 1).
//!
//! Before this crate the tree carried three process-global monotone counters
//! with character-identical semantics and character-identical doc blocks —
//! `inf_vgeom::stream::NEXT_RESIDENCY_STAMP`, `inf_vt::residency::NEXT_VT_STAMP`
//! and `inf_vsm::residency::NEXT_VSM_STAMP` — each with a comment saying the
//! other two existed and that P28.3 was where they would merge. This is that
//! merge, and it is one `static` and one function.
//!
//! # Why one counter is more than tidiness
//!
//! Three counters agree on the one property each of them was built for (never
//! decreasing) and disagree on the one the arbiter needs: **an order between
//! two consumers' stamps.** With three domains, "this cluster page was touched
//! more recently than that texture tile" is not a question that has an answer —
//! the two numbers are drawn from different sequences and comparing them is
//! comparing two clocks that were started at different times. Cross-system
//! eviction ([`crate::couple`]) is exactly that comparison, so it needs exactly
//! one sequence.
//!
//! # A stamp is not a measurement
//!
//! The law all three crates already carried, restated once here because it is
//! now one law over one number:
//!
//! * a stamp identifies *a* state; it does not describe one;
//! * the only legal operations are `!=` against a previously-observed stamp
//!   (the "has this moved since I last looked?" a GPU cache asks) and `<`
//!   between two stamps **within one process** (the LRU order);
//! * a stamp must **never** enter a trace, a golden, or a comparison between
//!   runs. Ordering it across processes, subtracting it, or printing it are all
//!   ways of pretending it describes something.
//!
//! Merging the three makes the law *easier* to break in one specific way and
//! that is worth stating: a per-crate counter advanced only when that crate did
//! something, so a test that (wrongly) printed one saw a small number that
//! looked stable. One shared counter advances whenever *any* consumer touches
//! anything, so the same mistake now produces obviously-garbage values. That is
//! the better failure.
//!
//! # Why not the terrain, voxel and journal counters too
//!
//! `inf_terrain::data::NEXT_TILE_VERSION`, `inf_voxel`'s two and
//! `inf_dcc::journal::NEXT_GENERATION` are **content** versions: they stamp
//! *what a payload is*, so a GPU mirror can ask "are these the bytes I
//! uploaded". A residency stamp records *when a slot was last wanted*, which is
//! what an LRU orders on. The two never meet — nothing evicts a terrain tile by
//! comparing its version against a texture tile's recency — so merging them
//! would couple three more crates to the streamer to buy an ordering nobody
//! asks for. Enumerated rather than summarised, and counted the same way:
//! **seven** counters exist — the three residency stamps that merge here, and
//! four content versions that do not (`inf_terrain`'s one, `inf_voxel`'s two,
//! `inf_dcc`'s one). Two *kinds*, seven counters; the split is by the question
//! each one answers, not by how many crates they live in. (The first draft said
//! "five", which contradicted the enumeration in the sentence above it — the
//! P28.3 audit's arithmetic finding.)

use std::sync::atomic::{AtomicU64, Ordering};

/// A residency stamp. Monotone, process-global, and **never a measurement**.
pub type Stamp = u64;

/// The one counter.
///
/// Starts at 1 so `0` is safe as "nothing uploaded yet" — the value all three
/// merged counters used for that, kept.
static NEXT_STAMP: AtomicU64 = AtomicU64::new(1);

/// The next stamp. Never 0.
///
/// `Relaxed` is the right ordering and was in all three predecessors: the only
/// guarantee needed is that no two calls return the same value, which
/// `fetch_add` gives on its own. Nothing is published *through* a stamp — the
/// state it labels is behind the same `&mut` the caller already holds.
#[inline]
pub fn next_stamp() -> Stamp {
    NEXT_STAMP.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strictly increasing, never 0, and unique — the three properties every
    /// merged counter's doc block claimed, now asserted once.
    #[test]
    fn the_domain_is_strictly_increasing_and_never_zero() {
        let a = next_stamp();
        let b = next_stamp();
        assert!(a >= 1, "0 is reserved for 'nothing uploaded yet'");
        assert!(b > a, "{b} did not follow {a}");
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..1000 {
            assert!(seen.insert(next_stamp()), "a stamp repeated");
        }
    }

    /// **The property three counters could not offer**: two consumers' stamps
    /// are comparable, because they are drawn from one sequence.
    ///
    /// Written as the cross-system question `couple` asks — "was the tile
    /// touched after the cluster page?" — because with three domains that
    /// question has no answer at all, and an arbiter that asked it anyway would
    /// get a stable, meaningless one.
    #[test]
    fn two_consumers_stamps_order_against_each_other() {
        let geometry_touch = next_stamp();
        let texture_touch = next_stamp();
        let second_geometry_touch = next_stamp();
        assert!(texture_touch > geometry_touch);
        assert!(second_geometry_touch > texture_touch);
        // …and the order is the order the touches happened in, which is the
        // only claim an LRU makes.
        let mut by_recency = [
            ("texture", texture_touch),
            ("geometry", geometry_touch),
            ("geometry2", second_geometry_touch),
        ];
        by_recency.sort_by_key(|&(_, s)| s);
        assert_eq!(
            by_recency.map(|(n, _)| n),
            ["geometry", "texture", "geometry2"]
        );
    }
}
