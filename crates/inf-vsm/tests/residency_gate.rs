//! **The shadow-page residency gate** (P27.1): determinism, fallback
//! correctness, LRU honesty, the budget, and the law `inf-vt` does not have —
//! that a page with no resident ancestor resolves to **nothing**, exactly.
//!
//! Every arm here runs with **no adapter**, which is the point of building the
//! residency GPU-free: these are the assertions that would otherwise be locked
//! behind a `SKIP: no GPU` on the CI legs that have no GPU.
//!
//! # The shadow model
//!
//! The fallback arms do not ask the residency what is resident. They rebuild that
//! from the **transaction stream alone** — admits insert, evicts remove — and
//! then resolve every virtual page by walking the tree upward from scratch. So
//! the comparison is between two independent statements about the world, and a
//! defect in either fails it. Checking the table against a recompute *inside* the
//! residency would be checking an algorithm against itself.
//!
//! And it checks **the whole table after every transaction**, never a sample: a
//! fallback pointer that goes wrong goes wrong somewhere specific, and a version
//! that spot-checked a dozen addresses would be satisfied by an implementation
//! that never propagated past the first level.

use std::collections::{BTreeMap, BTreeSet};

use inf_vsm::{
    VsmAtlasConfig, VsmLightDesc, VsmLightHandle, VsmPage, VsmResidency, VsmResolved,
    VsmTransaction, VsmWant,
};

/// An atlas holding exactly `pages` shadow pages.
fn atlas(pages: u64) -> VsmResidency {
    let cfg = VsmAtlasConfig {
        budget_bytes: VsmAtlasConfig::default().page_bytes() * pages,
        ..Default::default()
    };
    let (r, advisories) = VsmResidency::new(cfg);
    assert!(advisories.is_empty(), "unexpected advisory: {advisories:?}");
    assert_eq!(r.geometry().slot_count() as u64, pages, "{pages} pages");
    r
}

/// SplitMix64 — a seeded generator with no dependency and no platform behaviour.
/// The churn has to be *reproducible*, not statistically excellent.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u32) -> u32 {
        if n == 0 {
            0
        } else {
            (self.next() % u64::from(n)) as u32
        }
    }
}

/// What the transaction stream says is resident: `(light, page) -> slot`.
#[derive(Default)]
struct Shadow(BTreeMap<(u32, VsmPage), u32>);

impl Shadow {
    fn apply(&mut self, txn: &VsmTransaction) {
        for e in &txn.evicts {
            let gone = self.0.remove(&(e.light.0, e.page));
            assert_eq!(
                gone,
                Some(e.slot),
                "the stream evicted {:?} from slot {} but never put it there",
                e.page,
                e.slot
            );
        }
        for a in &txn.admits {
            let previous = self.0.insert((a.light.0, a.page), a.slot);
            assert_eq!(
                previous, None,
                "the stream admitted {:?} twice without evicting it",
                a.page
            );
        }
    }

    /// Resolve `at` by walking the tree upward — the brute force. `None` when
    /// nothing in the chain is resident, which is a legal answer here and is the
    /// whole difference from `inf-vt`'s model.
    fn resolve(
        &self,
        light: u32,
        desc: &VsmLightDesc,
        origins: &[(i64, i64)],
        at: VsmPage,
    ) -> Option<VsmResolved> {
        let mut cur = at;
        loop {
            if let Some(&slot) = self.0.get(&(light, cur)) {
                return Some(VsmResolved { slot, page: cur });
            }
            cur = desc.parent_at(origins, cur)?;
        }
    }
}

/// How the world looked at one check — the reachability counters that keep the
/// arms below from passing vacuously.
#[derive(Default, Clone, Copy)]
struct WorldShape {
    /// Addresses that resolved to nothing at all.
    none: u64,
    /// Addresses that resolved to an ancestor **more than one level** above.
    deep: u64,
}

/// Check the whole world: every entry of every table, the slot bijection, and
/// the fallback law in both directions.
fn assert_world(res: &VsmResidency, shadow: &Shadow, what: &str) -> WorldShape {
    let mut by_slot: BTreeMap<u32, (u32, VsmPage)> = BTreeMap::new();
    for (&(light, page), &slot) in &shadow.0 {
        assert!(
            by_slot.insert(slot, (light, page)).is_none(),
            "{what}: slot {slot} holds two pages"
        );
        assert!(
            res.is_resident(VsmLightHandle(light), page),
            "{what}: the stream put {page:?} in slot {slot} and the residency disagrees"
        );
        assert_eq!(
            res.slot_occupant(slot),
            Some((VsmLightHandle(light), page)),
            "{what}: slot {slot}"
        );
    }
    let mut shape = WorldShape::default();
    for l in 0..res.light_count() as u32 {
        let handle = VsmLightHandle(l);
        let desc = res.desc(handle).expect("registered").clone();
        // The brute-force walk uses the light's OWN per-level offsets, read off
        // the residency — so a clipmap that snapped its levels is checked against
        // the tree it actually has rather than against the concentric one it had
        // at registration (P27.3).
        let origins = res.clip_origins(handle).to_vec();
        for face in 0..desc.faces() {
            for level in 0..desc.level_count() {
                let g = desc.levels[level as usize];
                for y in 0..g.pages_y {
                    for x in 0..g.pages_x {
                        let at = VsmPage::new(face, level, x, y);
                        let brute = shadow.resolve(l, &desc, &origins, at);
                        assert_eq!(
                            res.resolve(handle, at),
                            brute,
                            "{what}: light {l} {at:?} — the maintained entry and the \
                             brute-force walk disagree"
                        );
                        match brute {
                            None => shape.none += 1,
                            Some(r) => {
                                // The answer is genuinely an ancestor of what was
                                // asked for, not merely something resident.
                                assert!(r.page.level >= at.level);
                                assert_eq!(
                                    desc.ancestor_at(&origins, at, r.page.level),
                                    Some(r.page)
                                );
                                shape.deep += u64::from(r.page.level > at.level + 1);
                                // **The premise that makes an evict cost nothing on
                                // the GPU**: no live entry addresses a slot that is
                                // not holding a page. The mirror writes nothing when
                                // a page leaves, so the day this stops being true a
                                // frame samples an unreachable slot's stale depth.
                                assert_eq!(
                                    res.slot_occupant(r.slot),
                                    Some((handle, r.page)),
                                    "{what}: {at:?} resolves to slot {} and nothing \
                                     lives there",
                                    r.slot
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(
        res.stats().resident as usize,
        shadow.0.len(),
        "{what}: resident count"
    );
    assert!(
        res.resident_bytes() <= res.capacity_bytes(),
        "{what}: resident past capacity"
    );
    assert!(
        res.capacity_bytes() <= res.config().budget_bytes,
        "{what}: the atlas outgrew its budget"
    );
    shape
}

/// One of each tree kind, so a bug that only shows on a clipmap has somewhere to
/// show — and so the face axis is exercised by something.
fn fixtures() -> Vec<VsmLightDesc> {
    vec![
        VsmLightDesc::clipmap(4, 8),
        VsmLightDesc::quadtree(4),
        VsmLightDesc::cube(3),
    ]
}

/// What one scripted churn did.
struct Churn {
    trace: String,
    res: VsmResidency,
    shadow: Shadow,
    /// Rounds in which the atlas ran out of slots. **The reachability counter**:
    /// the "never evict what this transaction admitted" law can only be broken by
    /// a transaction that needs a slot it cannot have.
    pressed: usize,
    /// Rounds in which something was evicted.
    evicting: usize,
    /// The world shape summed over **every** round, not the last one's.
    ///
    /// Summed because both numbers are reachability counters and a single round
    /// is the wrong window for either: a deep fallback needs a page whose parent
    /// left while its grandparent stayed, which is a transient state the last
    /// round of a churn is under no obligation to be in. Measured — the last
    /// round's `deep` really is zero on this seed (the first cut of this file
    /// kept only the last round and failed on it), and the property it exists to
    /// check happened **11 669** times on the way there, against 75 702
    /// addresses that resolved to nothing.
    shape: WorldShape,
}

/// Run one scripted churn.
fn churn(seed: u64, pages: u64, rounds: usize) -> Churn {
    let mut res = atlas(pages);
    let mut handles = Vec::new();
    for d in fixtures() {
        handles.push(res.register_light(d).expect("a legal tree"));
    }
    let mut rng = Rng(seed);
    let mut shadow = Shadow::default();
    let mut trace = String::new();
    let (mut pressed, mut evicting) = (0usize, 0usize);
    let mut shape = WorldShape::default();
    for round in 0..rounds {
        // Up to 40 wants against an atlas of 8–20 slots, so a round routinely
        // asks for more than it can hold. That is not incidental: it is the only
        // way the "never evict what this transaction admits" law is *reachable*.
        let n = rng.below(40) + 1;
        let mut wants = Vec::new();
        for _ in 0..n {
            let h = handles[rng.below(handles.len() as u32) as usize];
            let desc = res.desc(h).expect("registered").clone();
            let level = rng.below(desc.level_count());
            let g = desc.levels[level as usize];
            wants.push(VsmWant::new(
                h,
                VsmPage::new(
                    rng.below(desc.faces()),
                    level,
                    rng.below(g.pages_x),
                    rng.below(g.pages_y),
                ),
            ));
        }
        let txn = res.apply_wants(&wants);
        trace.push_str(&format!("-- round {round}\n"));
        trace.push_str(&txn.trace());
        pressed += usize::from(txn.deferred > 0);
        evicting += usize::from(!txn.evicts.is_empty());

        let admitted: BTreeSet<(u32, VsmPage)> =
            txn.admits.iter().map(|a| (a.light.0, a.page)).collect();
        for e in &txn.evicts {
            assert!(
                !admitted.contains(&(e.light.0, e.page)),
                "round {round}: {:?} was admitted and evicted by one transaction",
                e.page
            );
        }
        shadow.apply(&txn);
        for a in &txn.admits {
            assert!(
                res.is_resident(a.light, a.page),
                "round {round}: {:?} was admitted and is not resident",
                a.page
            );
        }
        let round_shape = assert_world(&res, &shadow, &format!("seed {seed}, round {round}"));
        shape.none += round_shape.none;
        shape.deep += round_shape.deep;
    }
    Churn {
        trace,
        res,
        shadow,
        pressed,
        evicting,
        shape,
    }
}

/// **Determinism.** Two runs of one want sequence produce byte-identical
/// transaction streams — in the *same process*, so the global stamp counter is at
/// a different value the second time round and cannot be hiding in the output.
#[test]
fn a_want_sequence_produces_a_byte_identical_transaction_stream() {
    for pages in [8u64, 12, 20] {
        let a = churn(0xC0FF_EE01, pages, 60);
        let b = churn(0xC0FF_EE01, pages, 60);
        assert_eq!(a.trace, b.trace, "{pages} pages: two runs disagree");
        assert!(
            a.evicting > 20 && a.pressed > 0,
            "{pages} pages: {} evicting rounds, {} over-asked — the churn never \
             pressed the atlas",
            a.evicting,
            a.pressed
        );
        // A different seed must NOT produce the same stream, or the comparison
        // above is satisfied by any constant.
        let c = churn(0xC0FF_EE02, pages, 60);
        assert_ne!(a.trace, c.trace, "{pages} pages: two scripts agree");
    }
}

/// The result does not depend on the order the caller assembled the frame in,
/// which is the property the sort actually buys.
///
/// (No worker-pool matrix here on purpose: `apply_wants` is serial, and **a pool
/// matrix over a serial program measures nothing** — the P22.4 law.)
#[test]
fn the_transaction_does_not_depend_on_the_order_wants_arrive_in() {
    let mut a = atlas(24);
    let mut b = atlas(24);
    let mut handles = Vec::new();
    for d in fixtures() {
        handles.push(a.register_light(d.clone()).unwrap());
        b.register_light(d).unwrap();
    }
    let mut rng = Rng(7);
    for _ in 0..40 {
        let mut wants = Vec::new();
        for _ in 0..10 {
            let h = handles[rng.below(3) as usize];
            let desc = a.desc(h).unwrap().clone();
            let level = rng.below(desc.level_count());
            let g = desc.levels[level as usize];
            wants.push(VsmWant::new(
                h,
                VsmPage::new(
                    rng.below(desc.faces()),
                    level,
                    rng.below(g.pages_x),
                    rng.below(g.pages_y),
                ),
            ));
        }
        // The same set, reversed, with every want repeated once — all three of
        // which a marking-mask scan followed by a per-light append produces.
        let mut shuffled: Vec<VsmWant> = wants.iter().rev().copied().collect();
        shuffled.extend(wants.iter().copied());
        assert_eq!(
            a.apply_wants(&wants).trace(),
            b.apply_wants(&shuffled).trace(),
            "the want order leaked into the transaction"
        );
    }
}

/// **The fallback pointers survive churn**, in both directions.
///
/// The heavy arm: 200 rounds against an atlas small enough to thrash, with the
/// whole table re-derived and compared after every single transaction (that check
/// lives in [`assert_world`], called from [`churn`]).
#[test]
fn every_address_resolves_to_the_finest_resident_ancestor_or_to_nothing() {
    // 12 pages against three lights whose trees hold 256 + 85 + 126 pages, so the
    // atlas can never be a superset and both answers — an ancestor and NONE — are
    // reachable on every round.
    let c = churn(0x5EED_1234, 12, 200);
    let res = &c.res;
    assert!(
        res.stats().admits > 200,
        "the churn admitted {} pages",
        res.stats().admits
    );
    assert!(
        res.stats().evicts > 100,
        "the churn evicted {} pages",
        res.stats().evicts
    );
    assert!(
        c.pressed > 10 && c.evicting > 100,
        "{} over-asked rounds, {} evicting rounds — the laws asserted per \
         transaction were never reachable",
        c.pressed,
        c.evicting
    );
    // ANTI-VACUITY, both directions. `none` proves the sentinel path is exercised
    // (a residency that resolved everything would be `inf-vt`'s, and this crate's
    // whole third law would be untested); `deep` proves the propagation reaches
    // past one level (an implementation that only ever re-pointed the immediate
    // children would satisfy every equality above).
    assert!(
        c.shape.none > 0,
        "no address of {} checked resolved to NONE — the sentinel is untested",
        c.shape.none + c.shape.deep
    );
    assert!(
        c.shape.deep > 0,
        "{} addresses resolved more than one level up — the propagation was never \
         more than one link deep",
        c.shape.deep
    );
    // …and the residency really did stay inside its slots.
    assert!(res.stats().resident <= 12);
    assert_eq!(res.stats().resident as usize, c.shadow.0.len());
}

/// **A clipmap that snapped its levels independently still resolves exactly**
/// (P27.3).
///
/// P27.3 gives each clipmap level its own snapped centre, which is what stops a
/// coarse level's grid shifting on every camera step. The price is that the
/// levels stop being concentric, so the whole maintained fallback chain is
/// against a *different* parent rule — and a maintained structure that is not
/// recomputed when its rule changes is a structure that silently points at the
/// wrong ground.
///
/// So: churn a residency into a state with real fallbacks, move the offsets, and
/// re-check **every** entry of the table against the independent walk under the
/// new offsets. The anti-vacuity is the one that matters: the new offsets must
/// re-parent something, or the arm is asserting that nothing changed.
#[test]
fn moving_a_clipmaps_level_offsets_re_points_the_whole_fallback_chain() {
    let mut c = churn(0x0FF5_E750, 12, 60);
    // The clipmap is fixture 0 (see `fixtures`), and it is the only kind offsets
    // mean anything for.
    let h = VsmLightHandle(0);
    let desc = c.res.desc(h).expect("registered").clone();
    let n = desc.levels[0].pages_x;
    assert!(
        matches!(desc.kind, inf_vsm::VsmTreeKind::Clipmap),
        "fixture 0 is the clipmap"
    );
    assert_eq!(c.res.clip_origins(h), &[] as &[(i64, i64)], "concentric");

    // Each level one cell further off than the one below it — the shape per-level
    // snapping produces when the camera sits between two coarse pages.
    let origins: Vec<(i64, i64)> = (0..desc.levels.len())
        .map(|l| {
            (
                -(i64::from(n) / 2) + l as i64,
                -(i64::from(n) / 2) - l as i64,
            )
        })
        .collect();

    // How many addresses the two rules disagree about, BEFORE anything is applied
    // — the arm's anti-vacuity, computed from the descriptor rather than observed.
    let re_parented = (1..desc.level_count())
        .flat_map(|level| {
            let g = desc.levels[level as usize - 1];
            (0..g.pages_y).flat_map(move |y| (0..g.pages_x).map(move |x| (level - 1, x, y)))
        })
        .filter(|&(level, x, y)| {
            let at = VsmPage::flat(level, x, y);
            desc.parent_at(&origins, at) != desc.parent(at)
        })
        .count();
    assert!(
        re_parented > 100,
        "the offsets re-parent only {re_parented} pages — the arm would pass with \
         `parent_at` ignoring its argument"
    );

    assert!(c.res.set_clip_origins(h, &origins), "the offsets moved");
    assert!(
        !c.res.set_clip_origins(h, &origins),
        "setting the same offsets twice recomputed the table a second time"
    );
    assert_eq!(c.res.clip_origins(h), origins.as_slice());
    // The whole table, against the independent walk, under the new rule.
    let shape = assert_world(&c.res, &c.shadow, "after the levels snapped");
    assert!(
        shape.none > 0,
        "nothing resolved to NONE after the shift — the sentinel path went untested"
    );

    // …and a further transaction still maintains it: the recompute is a starting
    // point, not a one-off repair.
    let txn = c.res.apply_wants(&[
        VsmWant::new(h, VsmPage::flat(0, 1, 1)),
        VsmWant::new(h, VsmPage::flat(1, 3, 2)),
        VsmWant::new(h, VsmPage::flat(2, 4, 4)),
    ]);
    c.shadow.apply(&txn);
    assert_world(
        &c.res,
        &c.shadow,
        "after a transaction under the new offsets",
    );
}

/// **LRU honesty**, with the timeline written out so the expected victim is a
/// consequence rather than an observation.
#[test]
fn the_least_recently_touched_page_is_the_one_that_leaves() {
    let mut res = atlas(3);
    let h = res.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
    // Frame 1: nothing at all — a registration seats no page.
    let t1 = res.apply_wants(&[]);
    assert!(t1.admits.is_empty() && t1.layout_rebuilt);

    // Frame 2: three pages, admitted in ENTRY order into the three free slots.
    let a = VsmPage::flat(0, 0, 0);
    let b = VsmPage::flat(0, 1, 0);
    let c = VsmPage::flat(0, 2, 0);
    let t2 = res.apply_wants(&[VsmWant::new(h, c), VsmWant::new(h, a), VsmWant::new(h, b)]);
    assert_eq!(
        t2.admits
            .iter()
            .map(|x| (x.slot, x.page))
            .collect::<Vec<_>>(),
        vec![(0, a), (1, b), (2, c)],
        "admits are in entry order, into the lowest free slots"
    );

    // Frame 3: touch `a` only. Nothing moves.
    let t3 = res.apply_wants(&[VsmWant::new(h, a)]);
    assert!(t3.admits.is_empty() && t3.evicts.is_empty());

    // Frame 4: a fourth page. `a` was touched most recently, `b` least — so `b`
    // leaves, and `c` (touched before `a`, after `b`) stays.
    let d = VsmPage::flat(0, 0, 1);
    let t4 = res.apply_wants(&[VsmWant::new(h, d)]);
    assert_eq!(
        t4.evicts
            .iter()
            .map(|e| (e.slot, e.page))
            .collect::<Vec<_>>(),
        vec![(1, b)],
        "the least recently touched page is not the one that left"
    );
    assert_eq!(
        t4.admits
            .iter()
            .map(|x| (x.slot, x.page))
            .collect::<Vec<_>>(),
        vec![(1, d)],
        "the freed slot is reused immediately"
    );
    assert!(res.is_resident(h, a) && res.is_resident(h, c));
    assert!(!res.is_resident(h, b));
    // `b` had no resident ancestor, so it now resolves to NOTHING — and `d`,
    // resident, resolves to itself.
    assert_eq!(res.resolve(h, d).map(|r| r.page), Some(d));
    assert_eq!(res.resolve(h, b), None);
}

/// **The budget is honoured exactly, and the shortfall is a number.**
#[test]
fn an_over_ask_defers_and_says_so() {
    let mut res = atlas(4);
    let h = res.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
    res.apply_wants(&[]);
    let wants: Vec<VsmWant> = (0..3)
        .flat_map(|x| (0..2).map(move |y| VsmWant::new(h, VsmPage::flat(0, x, y))))
        .collect();
    assert_eq!(wants.len(), 6);
    let txn = res.apply_wants(&wants);
    assert_eq!(txn.admits.len(), 4, "four free slots, four admits");
    assert_eq!(txn.deferred, 2);
    assert!(res.stats().budget_clamped);
    assert_eq!(res.stats().resident, 4, "every slot is in use, none over");
    assert_eq!(res.resident_bytes(), res.capacity_bytes());
    assert!(res.capacity_bytes() <= res.config().budget_bytes);

    // Nothing admitted this transaction was evicted by it — the atlas ran out
    // rather than eating its own work.
    assert!(txn.evicts.is_empty());
    for a in &txn.admits {
        assert!(res.is_resident(a.light, a.page));
    }

    // The deferred wants are not forgotten state: ask again next frame and the
    // atlas refuses again, identically, rather than thrashing.
    let again = res.apply_wants(&wants);
    assert_eq!(again.admits.len(), 0);
    assert_eq!(again.deferred, 2);
    assert!(again.evicts.is_empty(), "it thrashed: {}", again.trace());
}

/// **Nothing is pinned, and that is a property rather than an omission.**
///
/// `inf-vt`'s twin arm asserts a root is never a victim. The claim here is the
/// opposite one and it has to be checked, because an implementation that quietly
/// pinned the coarsest level would pass every other arm in this file while
/// claiming `N²` pages per directional light for ever.
#[test]
fn the_coarsest_level_is_evictable_like_anything_else() {
    let mut res = atlas(2);
    let h = res.register_light(VsmLightDesc::quadtree(3)).unwrap();
    res.apply_wants(&[]);
    let root = VsmPage::flat(2, 0, 0);
    let txn = res.apply_wants(&[VsmWant::new(h, root)]);
    assert_eq!(txn.admits.len(), 1);
    assert!(res.is_resident(h, root));

    // Two finer pages into a two-slot atlas: the root is the LRU victim and goes.
    let txn = res.apply_wants(&[
        VsmWant::new(h, VsmPage::flat(0, 0, 0)),
        VsmWant::new(h, VsmPage::flat(0, 1, 1)),
    ]);
    assert_eq!(
        txn.evicts.iter().map(|e| e.page).collect::<Vec<_>>(),
        vec![root],
        "the coarsest page was pinned: {}",
        txn.trace()
    );
    assert!(!res.is_resident(h, root));
    assert_eq!(res.resolve(h, root), None);
    assert_eq!(res.stats().resident, 2, "no third slot appeared");
}

/// The table image is **reproducible**, and its *layout* is route-independent.
///
/// What is pinned is what the mirror actually depends on: the same script twice
/// produces the same words (either route), and registering a light is the only
/// thing that moves the layout. Slot assignment *is* residency state, so two
/// routes to one resident set can legitimately hold different slots and both be
/// right — asserting otherwise would be a gate aimed at something it does not
/// name (the P23 law).
#[test]
fn the_table_image_is_reproducible_and_its_layout_is_route_independent() {
    let target = [
        VsmPage::flat(0, 0, 0),
        VsmPage::flat(0, 3, 2),
        VsmPage::flat(1, 3, 3),
    ];
    let build = |detour: bool| -> Vec<u32> {
        let mut res = atlas(12);
        let h = res.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
        res.apply_wants(&[]);
        if detour {
            // A long way round: page in a lot of other pages first, let them age
            // out, and only then ask for the target set.
            for x in 0..8 {
                for y in 0..2 {
                    res.apply_wants(&[VsmWant::new(h, VsmPage::flat(0, x, y))]);
                }
            }
            for level in 1..3 {
                res.apply_wants(&[VsmWant::new(h, VsmPage::flat(level, 4, 4))]);
            }
        }
        let wants: Vec<VsmWant> = target.iter().map(|p| VsmWant::new(h, *p)).collect();
        for _ in 0..24 {
            res.apply_wants(&wants);
        }
        res.table_words().to_vec()
    };
    let a = build(false);
    assert_eq!(a, build(false), "the direct route twice gave two images");
    let detoured = build(true);
    assert_eq!(detoured, build(true), "the detour twice gave two images");
    assert_eq!(detoured.len(), a.len(), "the layout followed the route");
    assert_ne!(
        detoured, a,
        "the two routes reached the same image, so this arm is comparing a \
         constant and the reproducibility above proves nothing"
    );

    // …and `resolved_table` — the door the mirror's readback arm compares GPU
    // bytes against — agrees with `resolve`, address for address, including the
    // addresses that resolve to nothing.
    let mut res = atlas(12);
    let h = res.register_light(VsmLightDesc::clipmap(3, 8)).unwrap();
    res.apply_wants(&target.map(|p| VsmWant::new(h, p)));
    let table = inf_vsm::resolved_table(&res);
    assert_eq!(table.len(), res.desc(h).unwrap().page_count() as usize);
    assert!(
        table.values().any(Option::is_some) && table.values().any(Option::is_none),
        "the resolved table is all one answer, so comparing it proves nothing"
    );
    for (&(l, page), got) in &table {
        assert_eq!(*got, res.resolve(VsmLightHandle(l), page), "{page:?}");
    }
}
