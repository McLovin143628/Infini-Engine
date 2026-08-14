//! **One want pipeline** (P28.3, clause 1): the lanes a want can arrive in, and
//! the normalization every consumer runs before admission.
//!
//! `inf-vt` and `inf-vsm` each shipped this function, spelled differently for
//! the same reason and with the same two-pass structure. It is written once
//! here because the *order* is the thing that has to be right, and because the
//! P28.2 audit found the half of it that was not: [`admit`](crate::admit) reads
//! the lane, and before this batch nothing did once a want was resident.

/// How badly something is wanted — **the primary sort key**, lower first.
///
/// A rank, not a policy: the consumer still decides what to want and in which
/// lane. Three lanes exist and the third has no producer yet (P28.4), which is
/// stated rather than implied — the `VSM_PRIORITY_SPECULATIVE` treatment.
pub type Lane = u8;

/// **The analytic floor.** Computed on the CPU from `(camera, bounds,
/// registry)` every frame, with no GPU readback in it, so a dropped feedback
/// frame degrades to exactly this. Residency may never fall below it while the
/// budget allows — which is what [`crate::admit::admit_by_lane`] enforces
/// against every other lane, including against a *resident* want of a weaker
/// one.
pub const LANE_FLOOR: Lane = 0;

/// **A refinement the GPU feedback asked for.** Served after the whole floor,
/// and evictable *by* the floor: this is the lane whose residency used to
/// outrank a floor miss (the P28.2 audit's priority-blind protection order).
pub const LANE_FEEDBACK: Lane = 1;

/// **A speculative want the predictor produced** — P28.4's dead reckoning,
/// whose contract is that it enters *"at strictly lower priority than the
/// analytic floor and feedback"*.
///
/// **It has no producer in this tree**, deliberately: the pipeline carries the
/// lane now and P28.4 supplies the producer. A sort with two ranks is a sort
/// that cannot see a third being mis-ordered, so the lane ships with the walk
/// that orders it and with
/// [`admit::tests::a_predicted_want_never_takes_a_floor_or_feedback_slot`](crate::admit)
/// exercising it, rather than as a reserved constant.
pub const LANE_PREDICT: Lane = 2;

/// **Normalize a want set**: dedup on the address, keeping the *strongest*
/// lane, and order lane-major with the consumer's payload order inside each
/// lane.
///
/// `key` is the consumer's own scan order — `inf-vt`'s payload order (the tile
/// directory's `(mip, y, x)`, not `TileCoord`'s derived `(mip, x, y)`, which
/// walks the file backwards) and `inf-vsm`'s entry order (the order its table
/// and its marking bitmask share). Both are "one forward scan of the container
/// per lane", which is why the tie-break is the payload's and not the sort's
/// convenience.
///
/// Two passes, and the order of the two is the whole point:
///
/// 1. sort by `(key, lane)` and dedup on `key` alone. [`Vec::dedup_by`] keeps
///    the **first** of a run, which after that sort is the **lowest** lane — so
///    an address the floor and the feedback both ask for is one floor want, not
///    two wants. (Asserted directly: the survivor's lane, not merely the count,
///    is the claim that matters.)
/// 2. **stable** sort by lane alone. Rust's `sort_by_key` is stable, so the
///    payload order established above survives inside each lane.
pub fn normalize<W: Copy, K: Ord>(
    wants: &[W],
    key: impl Fn(&W) -> K,
    lane: impl Fn(&W) -> Lane,
) -> Vec<W> {
    let mut out: Vec<W> = wants.to_vec();
    out.sort_by(|a, b| key(a).cmp(&key(b)).then(lane(a).cmp(&lane(b))));
    out.dedup_by(|a, b| key(a) == key(b));
    out.sort_by_key(&lane);
    out
}

/// The half-open ranges of one lane each, over a list [`normalize`] produced.
///
/// A free function rather than an inline `while` in the admission walk so the
/// run-splitting can be exercised on its own — including the two cases a
/// hand-written loop gets wrong, an empty list and a single-lane list.
pub fn lane_runs<W>(wants: &[W], lane: impl Fn(&W) -> Lane) -> Vec<(Lane, usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < wants.len() {
        let l = lane(&wants[i]);
        let mut j = i + 1;
        while j < wants.len() && lane(&wants[j]) == l {
            j += 1;
        }
        out.push((l, i, j));
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A want, as small as one can be: an address and a lane.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct W(u32, Lane);

    fn norm(w: &[W]) -> Vec<W> {
        normalize(w, |x| x.0, |x| x.1)
    }

    /// **The survivor of a dedup is the STRONGEST lane**, which is a different
    /// claim from "the duplicate went away" and the one that matters.
    ///
    /// The P26.4 audit found the weaker version of this arm in `inf-vt`: "two
    /// wants become one" is satisfied by *either* survivor, so reversing the
    /// tie-break left it green while the tile was served in the wrong class.
    #[test]
    fn a_duplicate_keeps_the_strongest_lane_whichever_order_it_arrives_in() {
        for input in [
            vec![W(7, LANE_FEEDBACK), W(7, LANE_FLOOR)],
            vec![W(7, LANE_FLOOR), W(7, LANE_FEEDBACK)],
            vec![W(7, LANE_PREDICT), W(7, LANE_FEEDBACK), W(7, LANE_FLOOR)],
        ] {
            let out = norm(&input);
            assert_eq!(out, vec![W(7, LANE_FLOOR)], "input {input:?}");
        }
        // …and a predictor duplicate of a feedback want is a feedback want.
        assert_eq!(
            norm(&[W(3, LANE_PREDICT), W(3, LANE_FEEDBACK)]),
            vec![W(3, LANE_FEEDBACK)]
        );
    }

    /// Lane-major, payload order inside a lane — and the fixture is built so
    /// the two orders are **opposed**, because a fixture where they agree
    /// cannot see the sort at all.
    #[test]
    fn the_order_is_lane_major_and_payload_ordered_inside_a_lane() {
        // Feedback wants at the LOW addresses, floor wants at the high ones.
        let input = [
            W(10, LANE_FEEDBACK),
            W(90, LANE_FLOOR),
            W(20, LANE_FEEDBACK),
            W(80, LANE_FLOOR),
            W(50, LANE_PREDICT),
        ];
        assert_eq!(
            norm(&input),
            vec![
                W(80, LANE_FLOOR),
                W(90, LANE_FLOOR),
                W(10, LANE_FEEDBACK),
                W(20, LANE_FEEDBACK),
                W(50, LANE_PREDICT),
            ]
        );
        // ANTI-VACUITY: the feedback addresses really do precede the floor's in
        // payload order, so the assertion above is about the lane and not about
        // an order that happened to agree.
        assert!(10 < 80, "the fixture's lanes and addresses are not opposed");
    }

    /// The want set is a **function of its contents, not of its arrival
    /// order** — the property that lets a host append per instance.
    #[test]
    fn normalization_does_not_depend_on_the_order_wants_arrive_in() {
        let base = [
            W(4, LANE_FLOOR),
            W(9, LANE_FEEDBACK),
            W(1, LANE_PREDICT),
            W(4, LANE_FEEDBACK),
            W(9, LANE_FLOOR),
        ];
        let want = norm(&base);
        let mut rotated = base;
        for _ in 0..base.len() {
            rotated.rotate_left(1);
            assert_eq!(norm(&rotated), want, "rotation {rotated:?}");
        }
        let mut reversed = base;
        reversed.reverse();
        assert_eq!(norm(&reversed), want);
    }

    /// The run splitter, including the two shapes a hand-written loop drops.
    #[test]
    fn the_lane_runs_partition_the_list_exactly_once() {
        let empty: [W; 0] = [];
        assert!(lane_runs(&empty, |x| x.1).is_empty());
        let one = [W(1, LANE_FLOOR), W(2, LANE_FLOOR)];
        assert_eq!(lane_runs(&one, |x| x.1), vec![(LANE_FLOOR, 0, 2)]);
        let three = norm(&[
            W(1, LANE_FLOOR),
            W(2, LANE_FEEDBACK),
            W(3, LANE_PREDICT),
            W(4, LANE_FEEDBACK),
        ]);
        let runs = lane_runs(&three, |x| x.1);
        assert_eq!(
            runs,
            vec![
                (LANE_FLOOR, 0, 1),
                (LANE_FEEDBACK, 1, 3),
                (LANE_PREDICT, 3, 4)
            ]
        );
        // The runs tile the list: no want is in two runs and none is in none.
        assert_eq!(
            runs.iter().map(|(_, a, b)| b - a).sum::<usize>(),
            three.len()
        );
        assert_eq!(runs.first().map(|r| r.1), Some(0));
        assert_eq!(runs.last().map(|r| r.2), Some(three.len()));
    }
}
