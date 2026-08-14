//! **The admission walk** — one implementation, two slot-pool consumers, and
//! the place the P28.2 audit's priority-blind protection order is fixed.
//!
//! # The defect, as the audit measured it
//!
//! `inf-vt` and `inf-vsm` both ran the same three steps: normalize the wants,
//! **touch and protect everything already resident**, then admit the misses
//! against whatever slots are left. Step 2 is where the rank was lost — it
//! protected every resident want *of any lane* before any miss *of any lane*
//! was offered a slot, so:
//!
//! > a refinement that got there first outranks a floor want that has not, and
//! > on the audit's fixture a decoy feedback class costs the pairing its finest
//! > page.
//!
//! The protection was also **unfalsifiable in place**: one want class over a
//! comfortable budget never contests a slot, so an `apply_wants` that protected
//! *nothing* passed every arm in the invariant gate.
//!
//! # The fix: protect per lane, admit per lane
//!
//! [`admit_by_lane`] walks the lanes in ascending order and, for each,
//! **protects that lane's residents and then admits that lane's misses**. So a
//! floor miss may take a resident *refinement's* slot (it is not protected
//! yet), and a refinement miss may never take a floor tile's (it was protected
//! in the previous round). The guarantee P28.2 needs is strictly stronger than
//! before rather than merely preserved: a resident cluster page's tiles are
//! floor wants, so they now outrank every refinement in the pool, resident or
//! not.
//!
//! Three properties survive the change and are asserted, because each is one a
//! plausible implementation loses:
//!
//! * **a transaction never evicts what it just admitted** — an admitted slot
//!   joins `protected` immediately, in every lane;
//! * **the deferral count is exact and O(1)** — once acquisition fails nothing
//!   frees a slot, so every remaining miss in every remaining lane is deferred
//!   without retrying;
//! * **a resident want is touched even after the pool is exhausted** — running
//!   out of slots must not make a lane's residents look old, or the *next*
//!   frame evicts exactly what this frame proved was wanted.

use std::collections::BTreeSet;

use crate::lane::{lane_runs, Lane};

/// A slot the pool handed out, and what (if anything) had to leave for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Acquired<K> {
    pub slot: u32,
    /// The occupant that was unseated, or `None` when the slot was free.
    pub evicted: Option<K>,
}

/// A residency the arbiter can admit into: a flat array of interchangeable
/// slots with an LRU victim rule.
///
/// Implemented by `inf_vt::VtResidency` and `inf_vsm::VsmResidency`, and
/// deliberately **not** by `inf_vgeom::VgeomStreamer` — see the crate docs: a
/// meshlet page is a variable number of bytes across four suballocated pools
/// and its residency is a prefix, not a set, so calling it a slot would be a
/// lie the arbiter then reasons on.
pub trait SlotPool {
    /// What this pool calls an address. `Copy` because the walk carries it by
    /// value through three collections and a page address is two or three
    /// integers in both consumers.
    type Key: Copy + Eq;

    /// The slot holding `key` exactly, or `None` when it is not paged in.
    ///
    /// **Exactly**, not "resolves to": a want whose address falls back to an
    /// ancestor is a miss, which is the whole reason there is anything to
    /// admit.
    fn resident_slot(&self, key: &Self::Key) -> Option<u32>;

    /// Mark `slot` as wanted now — the LRU's only input.
    fn touch(&mut self, slot: u32);

    /// The lowest free slot, or the least-recently-stamped evictable one,
    /// never a `protected` slot and never a pinned one. `None` when neither
    /// exists.
    fn acquire(&mut self, protected: &BTreeSet<u32>) -> Option<Acquired<Self::Key>>;

    /// Put `key` in `slot`.
    fn seat(&mut self, slot: u32, key: Self::Key);
}

/// What one [`admit_by_lane`] did, in the order it did it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmitLog<K> {
    /// `(slot, key)` seated, in admission order.
    pub admits: Vec<(u32, K)>,
    /// `(slot, key)` unseated, in the order the slots were taken.
    pub evicts: Vec<(u32, K)>,
    /// Wants that found no slot. **Never a silent cap.**
    pub deferred: u32,
}

impl<K> Default for AdmitLog<K> {
    fn default() -> Self {
        Self {
            admits: Vec::new(),
            evicts: Vec::new(),
            deferred: 0,
        }
    }
}

/// **The walk.** `wants` must already be lane-major — the output of
/// [`normalize`](crate::lane::normalize) — and free of addresses the pool
/// cannot hold (a consumer counts those itself, because an out-of-range want
/// and an unknown handle are two different defects and one number would hide
/// which).
///
/// `protected` starts from the caller's own set, which is how a consumer keeps
/// a slot the arbiter must not touch: `inf-vt` passes the root pages it seated
/// at registration, whose bytes this frame is about to emit.
pub fn admit_by_lane<P: SlotPool>(
    pool: &mut P,
    wants: &[(Lane, P::Key)],
    protected: &mut BTreeSet<u32>,
) -> AdmitLog<P::Key> {
    let mut log = AdmitLog::default();
    // Once acquisition fails nothing in this transaction frees a slot — the
    // protected set only grows and every seat is immediately protected — so
    // every later miss fails too. Counting them without retrying is the same
    // answer for O(1) instead of O(n · slots), and it must survive across
    // LANES, not only within one.
    let mut exhausted = false;
    for (_, from, to) in lane_runs(wants, |(l, _)| *l) {
        // (a) this lane's residents: touched and protected, *before* this lane
        //     admits anything and *after* every stronger lane already has.
        let mut misses: Vec<P::Key> = Vec::new();
        for (_, key) in &wants[from..to] {
            match pool.resident_slot(key) {
                Some(slot) => {
                    protected.insert(slot);
                    pool.touch(slot);
                }
                None => misses.push(*key),
            }
        }
        // (b) this lane's misses, against the slots that are left.
        for (i, key) in misses.iter().enumerate() {
            if exhausted {
                log.deferred += (misses.len() - i) as u32;
                break;
            }
            let Some(a) = pool.acquire(protected) else {
                exhausted = true;
                log.deferred += (misses.len() - i) as u32;
                break;
            };
            if let Some(gone) = a.evicted {
                log.evicts.push((a.slot, gone));
            }
            pool.seat(a.slot, *key);
            // Immediately, and in every lane: a transaction may never evict
            // what it just brought in.
            protected.insert(a.slot);
            log.admits.push((a.slot, *key));
        }
    }
    log
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::{normalize, LANE_FEEDBACK, LANE_FLOOR, LANE_PREDICT};
    use crate::stamp::{next_stamp, Stamp};

    /// The smallest honest slot pool: `n` interchangeable slots, an occupant
    /// per slot, an LRU with a total tie-break on the slot index.
    ///
    /// Written here rather than borrowed from a consumer precisely because the
    /// walk must be exercised against a pool the walk's own author wrote — the
    /// consumers' arms exercise the walk *through* their own residencies, and
    /// an error the two share would be invisible to both.
    struct TestPool {
        occupant: Vec<Option<u32>>,
        stamp: Vec<Stamp>,
        pinned: Vec<bool>,
    }

    impl TestPool {
        fn new(n: usize) -> Self {
            Self {
                occupant: vec![None; n],
                stamp: vec![0; n],
                pinned: vec![false; n],
            }
        }
        fn resident(&self, key: u32) -> bool {
            self.occupant.iter().any(|o| *o == Some(key))
        }
        /// A slot nothing may evict — `inf-vt`'s pinned root, in miniature.
        fn pin(&mut self, slot: u32, pinned: bool) {
            self.pinned[slot as usize] = pinned;
        }
    }

    impl SlotPool for TestPool {
        type Key = u32;
        fn resident_slot(&self, key: &u32) -> Option<u32> {
            self.occupant
                .iter()
                .position(|o| o.as_ref() == Some(key))
                .map(|i| i as u32)
        }
        fn touch(&mut self, slot: u32) {
            self.stamp[slot as usize] = next_stamp();
        }
        fn acquire(&mut self, protected: &BTreeSet<u32>) -> Option<Acquired<u32>> {
            if let Some(i) = self.occupant.iter().position(Option::is_none) {
                return Some(Acquired {
                    slot: i as u32,
                    evicted: None,
                });
            }
            let victim = self
                .occupant
                .iter()
                .enumerate()
                .filter(|(i, o)| {
                    o.is_some() && !self.pinned[*i] && !protected.contains(&(*i as u32))
                })
                .min_by_key(|(i, _)| (self.stamp[*i], *i))
                .map(|(i, _)| i as u32)?;
            let gone = self.occupant[victim as usize].take();
            Some(Acquired {
                slot: victim,
                evicted: gone,
            })
        }
        fn seat(&mut self, slot: u32, key: u32) {
            self.occupant[slot as usize] = Some(key);
            self.stamp[slot as usize] = next_stamp();
        }
    }

    fn run(pool: &mut TestPool, wants: &[(Lane, u32)]) -> AdmitLog<u32> {
        let norm = normalize(wants, |(_, k)| *k, |(l, _)| *l);
        let mut protected = BTreeSet::new();
        admit_by_lane(pool, &norm, &mut protected)
    }

    /// **THE FIX, asserted as the audit stated the defect**: a floor want that
    /// is not resident outranks a refinement that is.
    ///
    /// Three slots, all held by resident refinements, and one floor want
    /// arrives. Before this walk the floor want was deferred — every slot was
    /// protected in step 2 — and the surface stayed at its fallback for as long
    /// as the feedback kept asking for the same three tiles, which is for ever.
    #[test]
    fn a_floor_want_outranks_a_resident_refinement() {
        let mut pool = TestPool::new(3);
        // Seat three refinements and let them be the whole pool.
        let warm = run(
            &mut pool,
            &[
                (LANE_FEEDBACK, 10),
                (LANE_FEEDBACK, 11),
                (LANE_FEEDBACK, 12),
            ],
        );
        assert_eq!(warm.admits.len(), 3);
        assert_eq!(warm.deferred, 0);

        // Now the floor asks for one tile, and the feedback asks for its three
        // again — exactly the steady state the defect was invisible in.
        let txn = run(
            &mut pool,
            &[
                (LANE_FLOOR, 99),
                (LANE_FEEDBACK, 10),
                (LANE_FEEDBACK, 11),
                (LANE_FEEDBACK, 12),
            ],
        );
        assert!(
            pool.resident(99),
            "the floor want was refused a slot by a resident refinement: {txn:?}"
        );
        assert_eq!(txn.admits.len(), 1, "{txn:?}");
        assert_eq!(txn.evicts.len(), 1, "one refinement made room: {txn:?}");
        assert_eq!(txn.deferred, 1, "the displaced refinement is the deferral");
        // …and the refinement that left is the least recently touched of the
        // three, not an arbitrary one.
        assert_eq!(txn.evicts[0].1, 10, "{txn:?}");
    }

    /// The other direction, which is the one that must NOT change: a
    /// refinement can never take a floor tile's slot, resident or not.
    #[test]
    fn a_refinement_never_takes_a_floor_tiles_slot() {
        let mut pool = TestPool::new(3);
        let warm = run(
            &mut pool,
            &[(LANE_FLOOR, 1), (LANE_FLOOR, 2), (LANE_FLOOR, 3)],
        );
        assert_eq!(warm.admits.len(), 3);
        let txn = run(
            &mut pool,
            &[
                (LANE_FLOOR, 1),
                (LANE_FLOOR, 2),
                (LANE_FLOOR, 3),
                (LANE_FEEDBACK, 77),
            ],
        );
        assert!(txn.admits.is_empty() && txn.evicts.is_empty(), "{txn:?}");
        assert_eq!(txn.deferred, 1);
        for k in [1u32, 2, 3] {
            assert!(pool.resident(k), "the floor lost tile {k}: {txn:?}");
        }
    }

    /// The third lane, which has no producer, is ordered against both the
    /// others — so the day P28.4 supplies one, the sort is already right.
    #[test]
    fn a_predicted_want_never_takes_a_floor_or_feedback_slot() {
        let mut pool = TestPool::new(2);
        run(&mut pool, &[(LANE_FLOOR, 1), (LANE_FEEDBACK, 2)]);
        let txn = run(
            &mut pool,
            &[(LANE_PREDICT, 8), (LANE_FLOOR, 1), (LANE_FEEDBACK, 2)],
        );
        assert_eq!(txn.deferred, 1, "{txn:?}");
        assert!(!pool.resident(8));
        assert!(pool.resident(1) && pool.resident(2));

        // …and it DOES get a slot when one is free, or the arm above is a
        // statement about a full pool rather than about the lane.
        let mut roomy = TestPool::new(3);
        run(&mut roomy, &[(LANE_FLOOR, 1), (LANE_FEEDBACK, 2)]);
        let txn = run(
            &mut roomy,
            &[(LANE_PREDICT, 8), (LANE_FLOOR, 1), (LANE_FEEDBACK, 2)],
        );
        assert_eq!(txn.admits.len(), 1, "{txn:?}");
        assert!(roomy.resident(8));
    }

    /// **A transaction never evicts what it just admitted**, in any lane —
    /// including across a lane boundary, which is the case the per-lane walk
    /// introduced and the single-pass one could not reach.
    #[test]
    fn a_transaction_never_evicts_what_it_just_admitted() {
        let mut pool = TestPool::new(2);
        // Four misses into two slots, split across two lanes: the feedback
        // lane runs with both slots already taken by this same transaction.
        let txn = run(
            &mut pool,
            &[
                (LANE_FLOOR, 1),
                (LANE_FLOOR, 2),
                (LANE_FEEDBACK, 3),
                (LANE_FEEDBACK, 4),
            ],
        );
        assert_eq!(txn.admits.len(), 2);
        assert_eq!(txn.deferred, 2);
        assert!(txn.evicts.is_empty(), "the pool ate its own work: {txn:?}");
        let admitted: BTreeSet<u32> = txn.admits.iter().map(|(_, k)| *k).collect();
        for (_, gone) in &txn.evicts {
            assert!(!admitted.contains(gone));
        }
    }

    /// **A resident want is touched even after the pool is exhausted.**
    ///
    /// Otherwise a frame that over-asks makes its own residents look old, and
    /// the *next* frame evicts exactly what this one proved was wanted — a
    /// thrash that every counter reports as healthy churn.
    ///
    /// Reaching the case takes a **pinned** slot, and that is a property of the
    /// fix rather than of the fixture: a miss in an earlier lane can always
    /// take a later lane's resident slot (it is not protected yet — that is the
    /// whole point), so the only resident that can outlive an exhausting lane
    /// is one nothing may evict. Which is exactly `inf-vt`'s pinned root, and
    /// exactly the want a frame must not let go stale.
    #[test]
    fn exhaustion_does_not_stop_a_resident_want_from_being_touched() {
        let mut pool = TestPool::new(3);
        run(&mut pool, &[(LANE_FLOOR, 1), (LANE_FLOOR, 2), (LANE_FLOOR, 3)]);
        let root = pool.resident_slot(&1).expect("just seated");
        pool.pin(root, true);
        let before = pool.stamp[root as usize];

        // Three floor misses into two evictable slots: two land, the third
        // finds every slot pinned or protected. The FEEDBACK lane then wants
        // the pinned tile, which is resident and must still be touched.
        let txn = run(
            &mut pool,
            &[
                (LANE_FLOOR, 40),
                (LANE_FLOOR, 41),
                (LANE_FLOOR, 42),
                (LANE_FEEDBACK, 1),
            ],
        );
        assert_eq!(txn.deferred, 1, "the pool did not run out: {txn:?}");
        assert!(pool.resident(1), "a pinned slot was evicted: {txn:?}");
        assert!(
            pool.stamp[root as usize] > before,
            "the pool ran out and the resident want in the next lane was never \
             touched: {txn:?}"
        );

        // …and the consequence, which is what the stamp is for: with the pin
        // lifted, the tile last frame wanted is NOT the first victim.
        pool.pin(root, false);
        let txn = run(&mut pool, &[(LANE_FLOOR, 50)]);
        assert_eq!(txn.evicts.len(), 1, "{txn:?}");
        assert_ne!(
            txn.evicts[0].0, root,
            "the tile the last frame wanted was the first the next frame took"
        );
    }

    /// The deferral count is **exact** across lanes, and the walk stops
    /// retrying once it is hopeless.
    #[test]
    fn the_deferral_count_is_exact_across_every_remaining_lane() {
        let mut pool = TestPool::new(1);
        let txn = run(
            &mut pool,
            &[
                (LANE_FLOOR, 1),
                (LANE_FLOOR, 2),
                (LANE_FEEDBACK, 3),
                (LANE_PREDICT, 4),
                (LANE_PREDICT, 5),
            ],
        );
        assert_eq!(txn.admits.len(), 1);
        assert_eq!(
            txn.deferred, 4,
            "one floor, one feedback and two predicted wants went unserved: {txn:?}"
        );
    }

    /// The caller's own protected set is honoured — the door `inf-vt` seats its
    /// pinned roots through.
    #[test]
    fn a_callers_protected_slot_is_never_a_victim() {
        let mut pool = TestPool::new(2);
        run(&mut pool, &[(LANE_FLOOR, 1), (LANE_FLOOR, 2)]);
        let mut protected = BTreeSet::new();
        protected.insert(0u32);
        protected.insert(1u32);
        let norm = normalize(&[(LANE_FLOOR, 9u32)], |(_, k)| *k, |(l, _)| *l);
        let txn = admit_by_lane(&mut pool, &norm, &mut protected);
        assert!(txn.admits.is_empty() && txn.evicts.is_empty());
        assert_eq!(txn.deferred, 1);
    }

    /// **Function of state, not history**: two routes to one residency produce
    /// one arbitration.
    ///
    /// The same three tiles reached by two different want sequences, then the
    /// same fourth want offered to both — identical admits, evicts and
    /// deferrals. This is the crate-level law stated in the crate docs, checked
    /// where the walk can break it (the LRU order is the only place history
    /// could leak in, and it is a *stamp* order, which two routes that touched
    /// in the same order agree on).
    #[test]
    fn two_routes_to_one_residency_arbitrate_identically() {
        let mut a = TestPool::new(4);
        run(&mut a, &[(LANE_FLOOR, 1)]);
        run(&mut a, &[(LANE_FLOOR, 1), (LANE_FLOOR, 2)]);
        run(&mut a, &[(LANE_FLOOR, 1), (LANE_FLOOR, 2), (LANE_FLOOR, 3)]);

        let mut b = TestPool::new(4);
        run(&mut b, &[(LANE_FLOOR, 1), (LANE_FLOOR, 2), (LANE_FLOOR, 3)]);
        // Re-touch in the same order so the two LRU orders match; the claim is
        // about the arbitration being a function of (state, wants), and the LRU
        // order is part of the state.
        run(&mut b, &[(LANE_FLOOR, 1)]);
        run(&mut b, &[(LANE_FLOOR, 1), (LANE_FLOOR, 2)]);
        run(&mut b, &[(LANE_FLOOR, 1), (LANE_FLOOR, 2), (LANE_FLOOR, 3)]);

        let wants = [
            (LANE_FLOOR, 7u32),
            (LANE_FLOOR, 8),
            (LANE_FEEDBACK, 9),
            (LANE_FLOOR, 1),
        ];
        let ta = run(&mut a, &wants);
        let tb = run(&mut b, &wants);
        assert_eq!(ta, tb, "two routes to one residency disagree");
        assert!(!ta.evicts.is_empty(), "the fixture did not contest a slot");
    }
}
