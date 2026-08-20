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
//!   frame evicts exactly what this frame proved was wanted;
//! * **a transaction never re-admits what it just evicted** — the P28.3 audit's
//!   finding, and the one the per-lane walk introduced: a miss takes an
//!   *unwanted* slot before a weaker lane's resident one, so a tile this
//!   transaction is about to want again is not evicted, moved and re-uploaded
//!   for nothing. The rank is unchanged; only the choice between equals is.

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
    ///
    /// **A `None` must unseat nothing**, and that is a contract rather than an
    /// observation: [`admit_by_lane`] asks twice — once over a set that also
    /// spares the slots this transaction's own wants occupy, then over
    /// `protected` alone — so an implementation that evicted before deciding it
    /// had failed would evict twice for one admit.
    fn acquire(&mut self, protected: &BTreeSet<u32>) -> Option<Acquired<Self::Key>>;

    /// Put `key` in `slot`.
    fn seat(&mut self, slot: u32, key: Self::Key);
}

/// **How much one walk may admit** — the per-frame upload throttle (island wave
/// I4, IB-16).
///
/// Wave T's T33b refused a throttle *on purpose* — *"there is no per-frame time
/// budget and no per-frame admission or upload throttle in the VT loop; the only
/// budget is a byte residency ceiling… adding one is not a local change — it
/// re-opens a measurement, and the named tripwire tests are built to go red when
/// it lands."* This is that change, made deliberately, and every named tripwire
/// is re-aimed in the same commit.
///
/// It lives here rather than in `inf-vt` because [`admit_by_lane`] is the one
/// admission walk both page systems run, and a throttle that only one of them
/// obeyed would be a second answer to "how much does a frame page in".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdmitBudget {
    /// Maximum seats this walk may fill. Wants past it are **deferred, not
    /// dropped** — they are re-offered on the next frame's want set, so a tile
    /// arrives late and never never.
    pub max_admits: u32,
}

impl AdmitBudget {
    /// No throttle — the pre-IB-16 behaviour, and what a consumer with no
    /// upload budget passes.
    pub const UNLIMITED: Self = Self {
        max_admits: u32::MAX,
    };

    /// A budget of `bytes` over pages of `page_bytes` each.
    ///
    /// **Bytes in, seats out**, and the conversion is the point: every page in a
    /// virtual-texture pool is the same size, so a byte budget over uniform pages
    /// *is* a seat count — but the size depends on the format, and an RGBA8
    /// transcode page is **8×** a BC1 one. A seat budget would therefore throttle
    /// a BC-less adapter eight times as hard in bytes as a BC one while looking
    /// identical, which is exactly the kind of number that reads as fair and is
    /// not.
    ///
    /// Never zero: a budget that admits nothing is a page that never arrives, and
    /// "late, never never" is the whole contract. `0` bytes means unlimited, on
    /// the convention every other budget in this tree uses.
    pub fn from_bytes(bytes: u64, page_bytes: u64) -> Self {
        if bytes == 0 || page_bytes == 0 {
            return Self::UNLIMITED;
        }
        Self {
            max_admits: u32::try_from((bytes / page_bytes).max(1)).unwrap_or(u32::MAX),
        }
    }
}

impl Default for AdmitBudget {
    fn default() -> Self {
        Self::UNLIMITED
    }
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
    /// Of [`deferred`](Self::deferred), how many were held back by the
    /// [`AdmitBudget`] rather than by the pool being full (IB-16).
    ///
    /// The two are different defects wearing one number: **pool-full** is a
    /// residency problem — the working set does not fit and the answer is a
    /// bigger pool — while **throttled** is flow control, and the answer is to
    /// wait one frame. Reporting them together would make a burst look like a
    /// capacity failure, which is precisely the diagnosis a streaming engine
    /// must not get wrong.
    pub throttled: u32,
}

impl<K> Default for AdmitLog<K> {
    fn default() -> Self {
        Self {
            admits: Vec::new(),
            evicts: Vec::new(),
            deferred: 0,
            throttled: 0,
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
    budget: AdmitBudget,
) -> AdmitLog<P::Key> {
    let mut log = AdmitLog::default();
    // **The slots this transaction's own wants are sitting in, in any lane** —
    // `reserved` is `protected` plus those, and a miss tries it *first*.
    //
    // Without it the walk is correct and wasteful in one reachable case (the
    // P28.3 audit measured it on a real `VtResidency`): a lane-0 miss picks the
    // LRU victim over every unprotected slot, and a later lane's resident want
    // can be older than a slot nothing wants at all — so the transaction evicts
    // a tile it is about to want again, re-admits it into a different slot, and
    // re-uploads its bytes for nothing. That also made `evicts` name an address
    // that was resident again by the time the transaction returned, which is
    // not what the field says it is.
    //
    // The rank is unchanged, and that is the point: a floor miss may still take
    // a resident refinement's slot — it just takes an *unwanted* slot first if
    // one exists. When none does, the fallback below is the pre-audit
    // behaviour exactly, which is why `a_floor_want_outranks_a_resident_
    // refinement` still measures what it always did.
    let mut reserved: BTreeSet<u32> = protected.clone();
    reserved.extend(wants.iter().filter_map(|(_, k)| pool.resident_slot(k)));
    // Once acquisition fails nothing in this transaction frees a slot — the
    // protected set only grows and every seat is immediately protected — so
    // every later miss fails too. Counting them without retrying is the same
    // answer for O(1) instead of O(n · slots), and it must survive across
    // LANES, not only within one. It is the **fallback** acquire that decides
    // this: the reserved attempt failing means only that every spare slot is
    // spoken for, not that the pool is out.
    let mut exhausted = false;
    for (lane, from, to) in lane_runs(wants, |(l, _)| *l) {
        // **THE FLOOR IS NEVER THROTTLED** (IB-16). The budget bounds
        // *refinement*, which is where a burst lives; the analytic floor is the
        // mandatory residency class and is bounded by construction
        // (`VT_FLOOR_MAX_TILES` per visible surface), so deferring one of its
        // tiles would sacrifice the floor to flow control — which is what
        // `a_refinement_never_evicts_the_floor_and_a_dropped_mask_is_the_floors_trace`
        // exists to forbid, and would break P28.2's rule that the two halves of
        // one cluster transaction are seated at the same depth of the frame (a
        // cluster's tiles ride this lane). Measured: throttling the floor made
        // `one_transaction_admits_a_cluster_page_and_its_tiles` retract a page
        // and left the P28.3 load at 3 of 5 pages resident.
        let throttled_lane = lane > crate::lane::LANE_FLOOR;
        // (a) this lane's residents: touched and protected, *before* this lane
        //     admits anything and *after* every stronger lane already has.
        let mut misses: Vec<P::Key> = Vec::new();
        for (_, key) in &wants[from..to] {
            match pool.resident_slot(key) {
                Some(slot) => {
                    protected.insert(slot);
                    reserved.insert(slot);
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
            // **The throttle** (IB-16), tested here and not before the lane loop
            // so that a stronger lane's residents are still touched and
            // protected: a budget must decide what is UPLOADED this frame and
            // must never change what outranks what. A tile held back here is
            // re-offered on the next frame's want set — late, never never.
            if throttled_lane && log.admits.len() as u32 >= budget.max_admits {
                let held = (misses.len() - i) as u32;
                log.deferred += held;
                log.throttled += held;
                break;
            }
            // A slot nothing in this transaction wants, or — failing that —
            // the weakest thing the rank allows. `acquire` is documented to
            // return `None` without unseating anything, which is what makes
            // the second attempt safe rather than a double eviction.
            let Some(a) = pool.acquire(&reserved).or_else(|| pool.acquire(protected)) else {
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
            reserved.insert(a.slot);
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
            self.occupant.contains(&Some(key))
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
        admit_by_lane(pool, &norm, &mut protected, AdmitBudget::UNLIMITED)
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
    }

    /// **A transaction never re-admits what it just evicted**, and nothing it
    /// reports as evicted is resident when it returns.
    ///
    /// The P28.3 audit measured this on a real `VtResidency`: the per-lane walk
    /// let a lane-0 miss take the LRU victim over *every* unprotected slot, and
    /// a later lane's resident want can be older than a slot nothing wants at
    /// all. The transaction then evicted a tile it was about to want again,
    /// re-admitted it into a different slot, and re-uploaded its bytes for
    /// nothing — two evictions and two admissions to move one tile in and one
    /// out — while `evicts` named an address that was resident again by the
    /// time the caller read it.
    ///
    /// The fixture is the ordinary frame that produces it: a tile **demoted**
    /// from the floor lane to the feedback lane on the frame a new floor tile
    /// arrives. Nothing exotic, and nothing a budget prevents.
    ///
    /// The other half is the anti-vacuity one: the transaction has to have
    /// evicted something, or this is a statement about an empty list — which is
    /// exactly what the arm above turned out to be before this one existed.
    #[test]
    fn a_transaction_never_re_admits_what_it_just_evicted() {
        let mut pool = TestPool::new(4);
        // Seat in a known stamp order: 1 and 2 at the floor, then 3 and 4 at
        // the feedback lane, so 1 < 2 < 3 < 4 and the pool is full.
        run(
            &mut pool,
            &[
                (LANE_FLOOR, 1),
                (LANE_FLOOR, 2),
                (LANE_FEEDBACK, 3),
                (LANE_FEEDBACK, 4),
            ],
        );
        // Tile 2 is DEMOTED to feedback and a new floor tile arrives. Tile 2's
        // slot is the oldest thing the walk may take, and 3 and 4 are wanted by
        // nothing this frame.
        let txn = run(
            &mut pool,
            &[(LANE_FLOOR, 1), (LANE_FLOOR, 9), (LANE_FEEDBACK, 2)],
        );
        assert!(
            !txn.evicts.is_empty(),
            "the pool was never contested, so this arm asserts nothing: {txn:?}"
        );
        let admitted: BTreeSet<u32> = txn.admits.iter().map(|(_, k)| *k).collect();
        for (slot, gone) in &txn.evicts {
            assert!(
                !admitted.contains(gone),
                "slot {slot}'s occupant {gone} was evicted and re-admitted in one \
                 transaction — its bytes are re-uploaded for nothing: {txn:?}"
            );
            assert!(
                !pool.resident(*gone),
                "{gone} is reported EVICTED and is resident: {txn:?}"
            );
        }
        // …and the demoted tile kept its slot rather than being moved, which is
        // the whole of what the reserved set buys.
        assert!(
            pool.resident(2) && pool.resident(9) && pool.resident(1),
            "{txn:?}"
        );
        assert_eq!(txn.admits.len(), 1, "one tile came in, once: {txn:?}");
        assert_eq!(txn.evicts.len(), 1, "one tile left, once: {txn:?}");
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
        run(
            &mut pool,
            &[(LANE_FLOOR, 1), (LANE_FLOOR, 2), (LANE_FLOOR, 3)],
        );
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
        let txn = admit_by_lane(&mut pool, &norm, &mut protected, AdmitBudget::UNLIMITED);
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

    /// **The throttle defers rather than drops, and it says which** (IB-16).
    ///
    /// A budget of two seats against four misses must seat two, defer two, and
    /// report BOTH of those deferrals as throttled — because a pool with spare
    /// slots reporting "deferred" and nothing else is indistinguishable from a
    /// pool that is full, and the two want opposite fixes.
    ///
    /// **Feedback lane, not floor**: the floor is exempt by construction (see
    /// the walk), so a floor-lane fixture here would report a throttle that
    /// never fires.
    #[test]
    fn the_admit_budget_defers_the_tail_and_names_it_throttled() {
        let mut pool = TestPool::new(16);
        let wants: Vec<(Lane, u32)> = (0..4).map(|k| (LANE_FEEDBACK, k)).collect();
        let norm = normalize(&wants, |(_, k)| *k, |(l, _)| *l);
        let mut protected = BTreeSet::new();
        let txn = admit_by_lane(
            &mut pool,
            &norm,
            &mut protected,
            AdmitBudget { max_admits: 2 },
        );
        assert_eq!(txn.admits.len(), 2, "the budget must bound the seats");
        assert_eq!(txn.deferred, 2, "the tail is deferred, not dropped");
        assert_eq!(
            txn.throttled, 2,
            "every deferral here is flow control — the pool has 16 slots and four wants, so nothing was deferred for want of a slot"
        );

        // …and the FLOOR is exempt: the same over-ask in lane 0 seats whole.
        let mut floor_pool = TestPool::new(16);
        let floor: Vec<(Lane, u32)> = (0..4).map(|k| (LANE_FLOOR, k + 100)).collect();
        let norm_floor = normalize(&floor, |(_, k)| *k, |(l, _)| *l);
        let mut protected_floor = BTreeSet::new();
        let txn_floor = admit_by_lane(
            &mut floor_pool,
            &norm_floor,
            &mut protected_floor,
            AdmitBudget { max_admits: 2 },
        );
        assert_eq!(
            txn_floor.admits.len(),
            4,
            "the floor lane must ignore the upload budget — it is the mandatory residency class and a cluster's tiles ride it"
        );
        assert_eq!(txn_floor.throttled, 0);

        // …and the very next walk seats the rest: LATE, never never.
        let mut protected = BTreeSet::new();
        let txn = admit_by_lane(
            &mut pool,
            &norm,
            &mut protected,
            AdmitBudget { max_admits: 2 },
        );
        assert_eq!(txn.admits.len(), 2);
        assert_eq!(
            txn.deferred, 0,
            "the deferred tail arrived on the next walk"
        );
        assert_eq!(txn.throttled, 0);
    }

    /// A byte budget over uniform pages is a seat count — and the conversion has
    /// to notice that an RGBA8 page is eight times a BC1 one, or a BC-less
    /// adapter is throttled eight times as hard while the number looks the same.
    #[test]
    fn a_byte_budget_becomes_seats_at_the_pages_own_size() {
        const BC1: u64 = 9_248;
        const RGBA8: u64 = 73_984;
        assert_eq!(AdmitBudget::from_bytes(1 << 20, BC1).max_admits, 113);
        assert_eq!(AdmitBudget::from_bytes(1 << 20, RGBA8).max_admits, 14);
        // Zero bytes is the tree's "no ceiling" convention, not "admit nothing".
        assert_eq!(AdmitBudget::from_bytes(0, BC1), AdmitBudget::UNLIMITED);
        // …and a budget under one page still admits one: a page that never
        // arrives is not "late".
        assert_eq!(AdmitBudget::from_bytes(1, BC1).max_admits, 1);
    }
}
