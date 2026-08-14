//! **Cross-system eviction** (P28.3, clause 3): a group and its members age
//! together, or they do not age at all.
//!
//! # What P28.2 built, and what it did not own
//!
//! P28.2 bound one cluster page's geometry to its texture tiles inside one
//! transaction, and made the pairing survive *between* transactions by folding
//! every resident page's tiles into the same want set as the analytic floor —
//! *"put them in a second transaction and that stops being true on the very
//! next frame"*. That mechanism is right and it is kept verbatim. What it was
//! missing is an owner: the map from a page to its tiles lived inside a render
//! node, the "is every member still there" question had no answer outside the
//! gate's fixture, and nothing could ask it about a *third* consumer.
//!
//! This is that map, in the arbiter, as a general group:
//!
//! * [`Coupling::couple`] declares that a group's members must be wanted while
//!   the group is resident;
//! * [`Coupling::wants`] emits every member of every coupled group in one lane,
//!   so one transaction touches all of them and their stamps land in one epoch
//!   — which is what "age together" means once there is [one stamp
//!   domain](crate::stamp);
//! * [`Coupling::retract`] and [`Coupling::retain`] drop a group, and its
//!   members stop being wanted **in the same frame**, so they age out together
//!   rather than the tiles outliving the page that needed them;
//! * [`breaches`] is the invariant, computable outside any fixture: for every
//!   coupled group, every member is resident.
//!
//! # Aging together is a want, not a stamp copy
//!
//! The tempting implementation writes one stamp into every member's slot. It is
//! wrong for a reason worth recording: a slot's stamp is the *residency's*
//! bookkeeping and the arbiter does not own a residency — writing into one from
//! here would mean two writers for one field, and the second one runs at a
//! different point in the frame. Wanting them together achieves the same thing
//! through the door that already exists (`apply_wants` touches every resident
//! want), and it composes with the lane order: a group's members are floor
//! wants, so since P28.3's [admission walk](crate::admit) they outrank every
//! refinement in the pool, resident or not.
//!
//! # The bound, stated
//!
//! A group's members are the addresses **the group itself names** — a cluster
//! page's tiles section, which the cook wrote. A shadow page a cluster casts
//! into is *not* a member and cannot be: which pages a caster reaches is
//! decided by a per-page frustum test that runs over the pages that are already
//! resident, after the marking mask has been read, so producing it at the
//! arbiter's sync point means deriving next frame's page set from last frame's
//! casters. That is a *prediction*, and a prediction enters at
//! [`LANE_PREDICT`](crate::lane::LANE_PREDICT) — P28.4's lane, with no producer
//! here. Recorded rather than approximated.

use std::collections::{BTreeMap, BTreeSet};

use crate::lane::Lane;

/// Groups whose members must live and die with them.
///
/// `G` is what the owning consumer calls a group (`inf-render` uses
/// `(asset id, page index)`) and `M` what the member consumer calls an address
/// (`(texture guid, TileCoord)`). Neither is inspected here: the arbiter moves
/// addresses, it does not read them.
#[derive(Debug, Clone)]
pub struct Coupling<G: Ord + Copy, M: Ord + Copy> {
    groups: BTreeMap<G, Vec<M>>,
}

impl<G: Ord + Copy, M: Ord + Copy> Default for Coupling<G, M> {
    fn default() -> Self {
        Self {
            groups: BTreeMap::new(),
        }
    }
}

impl<G: Ord + Copy, M: Ord + Copy> Coupling<G, M> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare `group`'s members. Replaces any previous declaration, because a
    /// re-cooked or re-paged group is a different set and patching one into the
    /// other is how a member outlives the reason it was wanted.
    pub fn couple(&mut self, group: G, members: Vec<M>) {
        self.groups.insert(group, members);
    }

    /// Drop `group`, returning what it held. Its members stop being wanted from
    /// this moment, which is the eviction half of "they age together".
    pub fn retract(&mut self, group: &G) -> Option<Vec<M>> {
        self.groups.remove(group)
    }

    /// Keep only the groups `live` accepts; returns how many were dropped.
    ///
    /// The frame-shaped door: a consumer whose residency shrank hands in its
    /// surviving set once rather than calling [`retract`](Self::retract) per
    /// page, so a group cannot be forgotten by a caller that forgot a branch.
    pub fn retain(&mut self, live: impl Fn(&G) -> bool) -> usize {
        let before = self.groups.len();
        self.groups.retain(|g, _| live(g));
        before - self.groups.len()
    }

    /// Forget every group.
    pub fn clear(&mut self) {
        self.groups.clear();
    }

    /// The members of one group, or an empty slice.
    pub fn members(&self, group: &G) -> &[M] {
        self.groups.get(group).map_or(&[], Vec::as_slice)
    }

    /// Every coupled group, in key order.
    pub fn groups(&self) -> impl Iterator<Item = (&G, &[M])> {
        self.groups.iter().map(|(g, m)| (g, m.as_slice()))
    }

    /// Coupled groups.
    pub fn len(&self) -> usize {
        self.groups.len()
    }

    pub fn is_empty(&self) -> bool {
        self.groups.is_empty()
    }

    /// Members across every group, **deduplicated** — the number a
    /// cross-system byte accounting is keyed on, and deliberately not the sum
    /// of the per-group lengths: one texture tile is shared by every page that
    /// samples it, and counting it once per page is the double-count P28.2's
    /// ledger already refuses for bytes.
    pub fn member_count(&self) -> usize {
        self.members_deduped().len()
    }

    /// **Every member of every coupled group, once, in address order**, at
    /// `lane`.
    ///
    /// Address order rather than group order because the consumer's admission
    /// walk wants one forward scan of its container per lane — the same reason
    /// [`normalize`](crate::lane::normalize) sorts on the payload's order.
    pub fn wants(&self, lane: Lane) -> Vec<(Lane, M)> {
        self.members_deduped()
            .into_iter()
            .map(|m| (lane, m))
            .collect()
    }

    fn members_deduped(&self) -> Vec<M> {
        let set: BTreeSet<M> = self.groups.values().flatten().copied().collect();
        set.into_iter().collect()
    }
}

/// **The invariant**: every member of every coupled group is resident.
///
/// Returns the breaches — `(group, member)` pairs that are not — rather than a
/// `bool`, because a gate that fails should name the page and the tile, and
/// because "how many" is the number a runtime advisory would carry.
///
/// This is the P28.2 invariant, lifted out of the render node's fixture to a
/// function anything can call over any two consumers. It is the arbiter's own
/// oracle only when `resident` is derived independently of the coupling — which
/// is what `inf-render`'s gate does and what this signature makes possible.
pub fn breaches<G: Ord + Copy, M: Ord + Copy>(
    coupling: &Coupling<G, M>,
    resident: impl Fn(&M) -> bool,
) -> Vec<(G, M)> {
    let mut out = Vec::new();
    for (g, members) in coupling.groups() {
        for m in members {
            if !resident(m) {
                out.push((*g, *m));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lane::{LANE_FLOOR, LANE_PREDICT};

    fn fixture() -> Coupling<(u8, u8), u16> {
        let mut c = Coupling::new();
        c.couple((1, 0), vec![10, 11, 12]);
        c.couple((1, 1), vec![12, 13]); // 12 is shared, deliberately
        c.couple((2, 0), vec![20]);
        c
    }

    /// A member shared by two groups is **one** want, and the shared member
    /// survives one of the two groups being retracted.
    #[test]
    fn a_shared_member_is_wanted_once_and_survives_its_second_owner() {
        let mut c = fixture();
        assert_eq!(
            c.wants(LANE_FLOOR),
            vec![
                (LANE_FLOOR, 10),
                (LANE_FLOOR, 11),
                (LANE_FLOOR, 12),
                (LANE_FLOOR, 13),
                (LANE_FLOOR, 20)
            ]
        );
        assert_eq!(c.member_count(), 5, "the shared tile was counted twice");
        // Retract the group that introduced 12 — it is still wanted, by the
        // other one. A per-group teardown that forgot this evicts a tile out
        // from under a page that is still resident, which is the invariant.
        assert_eq!(c.retract(&(1, 0)), Some(vec![10, 11, 12]));
        let w: Vec<u16> = c.wants(LANE_FLOOR).into_iter().map(|(_, m)| m).collect();
        assert_eq!(w, vec![12, 13, 20]);
    }

    /// Retracting a group stops its members being wanted **in the same call** —
    /// the eviction half of "they age together".
    #[test]
    fn a_retracted_groups_members_stop_being_wanted_at_once() {
        let mut c = fixture();
        assert_eq!(c.len(), 3);
        assert_eq!(c.retain(|(a, _)| *a == 1), 1, "group (2,0) should have gone");
        let w: Vec<u16> = c.wants(LANE_FLOOR).into_iter().map(|(_, m)| m).collect();
        assert_eq!(w, vec![10, 11, 12, 13]);
        c.clear();
        assert!(c.is_empty() && c.wants(LANE_FLOOR).is_empty());
    }

    /// Re-coupling **replaces** rather than merges: a page that was re-paged at
    /// a different level names different tiles, and keeping the old ones is a
    /// member that outlives the reason it was wanted.
    #[test]
    fn re_coupling_replaces_the_member_set() {
        let mut c = fixture();
        c.couple((1, 0), vec![99]);
        assert_eq!(c.members(&(1, 0)), &[99]);
        let w: Vec<u16> = c.wants(LANE_FLOOR).into_iter().map(|(_, m)| m).collect();
        assert_eq!(w, vec![12, 13, 20, 99], "the old members lingered");
    }

    /// The invariant names the pair that broke it, and is empty when it holds.
    #[test]
    fn the_invariant_names_the_group_and_the_member_that_broke_it() {
        let c = fixture();
        assert!(breaches(&c, |_| true).is_empty());
        let gone = breaches(&c, |m| *m != 12);
        // 12 belongs to two groups, so it breaks the invariant twice — and both
        // are reported, because "which page lost its tile" is the question.
        assert_eq!(gone, vec![((1, 0), 12), ((1, 1), 12)]);
        // ANTI-VACUITY: an empty coupling cannot breach, so a gate that only
        // ever sees one is asserting nothing.
        let empty: Coupling<(u8, u8), u16> = Coupling::new();
        assert!(breaches(&empty, |_| false).is_empty());
        assert_eq!(empty.member_count(), 0);
    }

    /// The lane is the caller's, so the same coupling can be a floor demand in
    /// one consumer and a speculation in another the day one exists.
    #[test]
    fn the_wants_carry_the_lane_they_were_asked_for() {
        let c = fixture();
        assert!(c.wants(LANE_PREDICT).iter().all(|(l, _)| *l == LANE_PREDICT));
        assert!(c.wants(LANE_FLOOR).iter().all(|(l, _)| *l == LANE_FLOOR));
    }
}
