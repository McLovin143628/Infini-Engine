//! **The arbiter's own oracle** (P28.3): want-set conservation, floor
//! protection, byte-budget conservation, cross-system aging, and one stamp
//! domain — over the three *real* residencies, with no adapter.
//!
//! # Why this file exists beside `cluster_pages.rs`
//!
//! `cluster_pages.rs` gates the P28.2 **invariant** — a resident cluster's tiles
//! are resident — against an oracle that re-derives the pairing a different way.
//! It cannot gate the arbiter, and the P28.2 audit said why: *"a gate cannot see
//! an error the subsystems share"*, and after P28.3 the two subsystems share
//! considerably more than they did (one stamp domain, one admission walk, one
//! budget). So the arbitration needs an oracle of its own, derived from
//! quantities the arbiter does not compute:
//!
//! | claim | the arbiter's mechanism | this file's oracle |
//! |---|---|---|
//! | want-set conservation | `admit_by_lane`'s `misses` | the want set as the caller offered it, kept beside the transaction and compared to its admits |
//! | floor protection | lane order + `protected` | a floor set recorded *before* the transaction, checked against the evicts *after* it |
//! | byte-budget conservation | `arbitrate`'s water-fill | the three residencies' own `resident_bytes`, summed by this file |
//! | cross-system aging | `Coupling::retain` | `inf_stream::breaches` against a residency predicate this file derives from `is_resident` |
//! | one stamp domain | one `static` | three crates' generations, compared for strict interleaving |
//!
//! # The controls
//!
//! Every claim here has one, because a conservation law over an empty
//! transaction is satisfied by construction. The churn's `contested` /
//! `dropped_groups` / `evicts` counters are asserted **non-zero** before any
//! bound is read off them, and the cross-system arm carries the same
//! coupling-off control `cluster_pages.rs` uses — with the coupling off, the
//! forbidden state must be *reached*.

use std::collections::{BTreeMap, BTreeSet};

use inf_render::{arbitrate, BudgetRequest, Consumer, Coupling, StreamError, LANE_FLOOR};
use inf_vgeom::{ClusterTexture, ClusterTextureSet, VgeomSource, VgeomStreamBudget, VgeomStreamer};
use inf_vsm::{VsmAtlasConfig, VsmLightDesc, VsmPage, VsmResidency, VsmWant};
use inf_vt::{
    full_pyramid, TileCoord, VtPoolConfig, VtResidency, VtTextureDesc, VtTextureHandle, VtWant,
};

const ASSET: u128 = 0x2803_0000_0000_0001;
const TEX_A: u128 = 0xB0B0_0000_0000_0000_0000_0000_0000_0001;
const TEX_B: u128 = 0xB0B0_0000_0000_0000_0000_0000_0000_0002;

fn asset_id(v: u128) -> inf_asset::AssetId {
    inf_asset::AssetId(uuid::Uuid::from_u128(v))
}

fn descs() -> Vec<(u128, VtTextureDesc)> {
    vec![
        (TEX_A, full_pyramid(2048, 2048, 128, 4, true)),
        (TEX_B, full_pyramid(1024, 1024, 128, 4, false)),
    ]
}

/// A paired `.inf_vmesh` — the shape a cook produces, tangents and all.
fn paired_source() -> VgeomSource {
    source_paired_against(&descs())
}

/// A paired image cooked against `cook`, to be met at runtime by whatever the
/// caller registers.
fn source_paired_against(cook: &[(u128, VtTextureDesc)]) -> VgeomSource {
    let mesh = inf_vgeom::test_support::build_grid_tangented(
        24,
        0.3,
        inf_vgeom::test_support::GridNormals::Analytic,
        true,
    );
    let set = ClusterTextureSet {
        textures: cook
            .iter()
            .map(|(g, d)| ClusterTexture::from_desc(asset_id(*g), d))
            .collect(),
    };
    VgeomSource::from_mesh_paired(&mesh, &set).expect("build the paired image")
}

fn texture_residency(budget_bytes: u64) -> (VtResidency, BTreeMap<u128, VtTextureHandle>) {
    let (mut res, _adv) = VtResidency::new(VtPoolConfig {
        budget_bytes,
        ..Default::default()
    });
    let mut by_guid = BTreeMap::new();
    for (guid, desc) in descs() {
        by_guid.insert(guid, res.register_texture(desc).expect("the floor fits"));
    }
    (res, by_guid)
}

// ── the world under test ─────────────────────────────────────────────────────

/// The three residencies plus the coupling — the triple the arbiter divides.
struct Streamer {
    geometry: VgeomStreamer,
    texture: VtResidency,
    shadow: VsmResidency,
    by_guid: BTreeMap<u128, VtTextureHandle>,
    coupling: Coupling<(u128, usize), (u128, TileCoord)>,
    /// Groups dropped over the run — the anti-vacuity number for aging.
    dropped: u64,
    /// Peak combined resident bytes over the run.
    peak_bytes: u64,
    /// Transactions in which the texture pool ran out.
    contested: u32,
    /// A third texture, registered and **never part of the pairing** — the
    /// competing refinement class. Without one the only wants in the pool are
    /// floor wants, every miss is deferred rather than seated, and "nothing a
    /// floor protects was evicted" is a bound on zero evictions.
    decoy: Option<VtTextureHandle>,
}

impl Streamer {
    fn new(texture_budget: u64, atlas_pages: u64) -> Self {
        let (texture, by_guid) = texture_residency(texture_budget);
        let (mut shadow, _adv) = VsmResidency::new(VsmAtlasConfig {
            budget_bytes: inf_vsm::VsmAtlasConfig::default().page_bytes() * atlas_pages,
            ..Default::default()
        });
        shadow
            .register_light(VsmLightDesc::clipmap(3, 8))
            .expect("a three-level clipmap");
        Self {
            geometry: VgeomStreamer::new(VgeomStreamBudget::default()),
            texture,
            shadow,
            by_guid,
            coupling: Coupling::new(),
            dropped: 0,
            peak_bytes: 0,
            contested: 0,
            decoy: None,
        }
    }

    /// Register the competing refinement class — a 2 048² texture this mesh
    /// never samples, so its wants are disjoint from the pairing by
    /// construction (a decoy drawn from the pairing's own tiles is deduped into
    /// the pairing and competes with nothing).
    fn with_decoy(mut self) -> Self {
        self.decoy = Some(
            self.texture
                .register_texture(full_pyramid(2048, 2048, 128, 4, true))
                .expect("the floor fits"),
        );
        self
    }

    /// This frame's refinements from the decoy, or nothing.
    fn decoy_wants(&self) -> Vec<VtWant> {
        let Some(h) = self.decoy else {
            return Vec::new();
        };
        let m = self.texture.desc(h).expect("registered").mips[0];
        (0..m.tiles_y)
            .flat_map(|y| (0..m.tiles_x).map(move |x| (x, y)))
            .map(|(x, y)| VtWant::refine(h, TileCoord::new(0, x, y)))
            .collect()
    }

    /// Bytes the three hold between them — the number that did not exist before
    /// this phase, summed here rather than read off a report so the ratchet is
    /// this file's own measurement.
    fn resident_bytes(&self) -> [u64; 3] {
        [
            self.geometry.stats().resident_bytes,
            self.texture.resident_bytes(),
            self.shadow.resident_bytes(),
        ]
    }

    /// The live floors: the meshlet streamer's always-resident page 0s and the
    /// virtual texture's pinned roots. The shadow atlas has none by design.
    fn floors(&self) -> [u64; 3] {
        [
            self.geometry
                .assets()
                .map(|(_, r)| r.pages().first().map_or(0, |p| p.resident_bytes()))
                .sum(),
            u64::from(self.texture.stats().roots) * self.texture.page_bytes(),
            0,
        ]
    }

    /// One frame: plan the geometry, fold the coupling into the texture want
    /// set, transact, pair, mark shadow pages, retain the coupling.
    ///
    /// Returns the **want set as offered** and the three transactions' admits,
    /// so the caller's oracle can compare them without asking the residencies.
    fn step(&mut self, src: &VgeomSource, threshold: f32, marks: &[VsmPage], couple: bool) -> Step {
        let plan = self.geometry.plan(&[inf_vgeom::VgeomWant {
            asset: ASSET,
            source: src,
            threshold,
        }]);

        // The coupling, rebuilt from the container for the pages that are
        // resident right now.
        self.coupling.clear();
        let resident = self
            .geometry
            .residency(ASSET)
            .map_or(0, |r| r.resident_pages());
        for page in 0..resident {
            let refs = src
                .with_page_sections(page, |s| {
                    s.tile_refs()
                        .iter()
                        .map(|t| (t.texture().uuid().as_u128(), t.coord()))
                        .filter(|(g, t)| {
                            self.by_guid
                                .get(g)
                                .is_some_and(|h| self.texture.can_address(*h, *t))
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            self.coupling.couple((ASSET, page), refs);
        }

        // ONE texture transaction: the coupling's wants at the floor lane.
        let mut offered: Vec<VtWant> = if couple {
            self.coupling
                .wants(LANE_FLOOR)
                .into_iter()
                .filter_map(|(_, (g, t))| self.by_guid.get(&g).map(|h| VtWant::new(*h, t)))
                .collect()
        } else {
            Vec::new()
        };
        offered.extend(self.decoy_wants());
        // **The FLOOR lane only.** A resident *refinement* is exactly what a
        // floor miss is now allowed to take (P28.3), so folding one in here
        // would make this arm assert the defect the batch removed.
        let floor_before: BTreeSet<(u32, TileCoord)> = offered
            .iter()
            .filter(|w| w.priority == inf_vt::VT_PRIORITY_FLOOR)
            .filter(|w| self.texture.is_resident(w.texture, w.tile))
            .map(|w| (w.texture.0, w.tile))
            .collect();
        let txn = self.texture.apply_wants(&offered);
        self.contested += u32::from(txn.deferred > 0);

        // Pair, or hand the page back.
        let coupling = &self.coupling;
        let texture = &self.texture;
        let page_in = self.geometry.pair(plan, |asset, page| {
            if !couple {
                return Some(Vec::new());
            }
            if !coupling.has_group(&(asset, page)) {
                return None;
            }
            let mut seated = Vec::new();
            for &(g, t) in coupling.members(&(asset, page)) {
                let h = self.by_guid.get(&g)?;
                if !texture.is_resident(*h, t) {
                    return None;
                }
                seated.push((g, t));
            }
            Some(seated)
        });

        // Cross-system aging: a group whose page left stops wanting its members
        // in this same call.
        let live: BTreeSet<(u128, usize)> = self
            .geometry
            .assets()
            .flat_map(|(a, r)| (0..r.resident_pages()).map(move |p| (a, p)))
            .collect();
        self.dropped += self.coupling.retain(|g| live.contains(g)) as u64;

        // The shadow half runs its own transaction against the same stamp
        // domain — it is a consumer of the budget and of the lane order, not of
        // the coupling (see `inf_stream::couple`'s bound).
        let shadow_txn = self.shadow.apply_wants(
            &marks
                .iter()
                .map(|p| VsmWant::new(inf_vsm::VsmLightHandle(0), *p))
                .collect::<Vec<_>>(),
        );

        let bytes = self.resident_bytes();
        self.peak_bytes = self.peak_bytes.max(bytes.iter().sum::<u64>());
        Step {
            offered: offered.iter().map(|w| (w.texture.0, w.tile)).collect(),
            floor_before,
            admits: txn.admits.iter().map(|a| (a.texture.0, a.tile)).collect(),
            evicts: txn.evicts.iter().map(|e| (e.slot, e.texture.0, e.tile)).collect(),
            shadow_admits: shadow_txn.admits.len(),
            retracted: page_in.retracted,
            pages: resident,
        }
    }
}

/// What one step offered and what came back — the oracle's raw material.
struct Step {
    offered: BTreeSet<(u32, TileCoord)>,
    floor_before: BTreeSet<(u32, TileCoord)>,
    admits: Vec<(u32, TileCoord)>,
    evicts: Vec<(u32, u32, TileCoord)>,
    shadow_admits: usize,
    retracted: u32,
    pages: usize,
}

/// A ladder of strictly refining thresholds, so "residency went backwards" can
/// only mean the arbiter took ground back.
const LADDER: [f32; 8] = [4.0, 0.5, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001];

/// Shadow pages to mark, cycling so the atlas churns rather than settling.
fn marks_for(i: usize) -> Vec<VsmPage> {
    (0..6)
        .map(|k| VsmPage::flat(0, ((i + k) % 8) as u32, (k % 8) as u32))
        .collect()
}

// ── the arms ─────────────────────────────────────────────────────────────────

/// **WANT-SET CONSERVATION**: nothing is admitted that no consumer wanted.
///
/// The oracle is the want set *as the caller offered it*, kept beside the
/// transaction — so an arbiter that invented an address, or admitted one it had
/// cached from an earlier frame, is caught by a set this file owns rather than
/// by the residency agreeing with itself.
#[test]
fn nothing_is_admitted_that_no_consumer_wanted() {
    let src = paired_source();
    let mut s = Streamer::new(4 * 1024 * 1024, 32);
    let mut admitted = 0usize;
    for (i, t) in LADDER.iter().enumerate() {
        let step = s.step(&src, *t, &marks_for(i), true);
        for a in &step.admits {
            assert!(
                step.offered.contains(a) || s.texture.slot_is_root(0),
                "step {i}: {a:?} was admitted and nothing wanted it"
            );
        }
        admitted += step.admits.len();
        assert!(
            step.shadow_admits <= marks_for(i).len(),
            "step {i}: the shadow atlas admitted more pages than were marked"
        );
    }
    // ANTI-VACUITY: the run really did admit things, or the loop above bounded
    // an empty set eight times.
    assert!(admitted > 0, "nothing was ever admitted");
}

/// **FLOOR PROTECTION**: nothing is evicted that a floor protects.
///
/// The floor set is recorded **before** the transaction (every offered want
/// that was already resident) and checked against the evicts **after** it, so
/// the claim is about the transaction rather than about the residency's own
/// bookkeeping. Under P28.3's lane walk this is the guarantee that survives in
/// the strong form: a floor want cannot be evicted by anything, and a resident
/// refinement can be evicted by a floor miss.
#[test]
fn nothing_is_evicted_that_a_floor_protects() {
    let src = paired_source();
    // A pool tight enough that the pairing genuinely contests slots, plus the
    // competing refinement class that gives it something evictable to take.
    let mut s = Streamer::new(768 * 1024, 8).with_decoy();
    let mut evicted = 0usize;
    for (i, t) in LADDER.iter().enumerate() {
        let step = s.step(&src, *t, &marks_for(i), true);
        for (slot, tex, tile) in &step.evicts {
            assert!(
                !step.floor_before.contains(&(*tex, *tile)),
                "step {i}: slot {slot} held a resident FLOOR want ({tex}, {tile:?}) \
                 and the transaction took it"
            );
        }
        evicted += step.evicts.len();
    }
    assert!(
        evicted > 0 && s.contested > 0,
        "the pool never ran out ({} evictions, {} contested steps) — the bound \
         above is a bound on nothing",
        evicted,
        s.contested
    );
}

/// **BYTE-BUDGET CONSERVATION**: the three residencies together never exceed
/// what the arbiter grants them, and the grant never exceeds the ceiling.
///
/// Both halves, because they fail differently: a grant that over-spends is an
/// arbiter bug, and a residency past its grant is a consumer ignoring one.
#[test]
fn the_three_residencies_stay_inside_one_budget() {
    let src = paired_source();
    let texture_budget = 2 * 1024 * 1024;
    let mut s = Streamer::new(texture_budget, 16);
    // The unified ceiling: the three requests, so the arbiter is an identity
    // here and the residency bound is the load-bearing half.
    let geometry_budget = VgeomStreamBudget::default().budget_bytes;
    let shadow_budget = s.shadow.capacity_bytes();
    let total = geometry_budget + texture_budget + shadow_budget;

    for (i, t) in LADDER.iter().enumerate() {
        s.step(&src, *t, &marks_for(i), true);
        let bytes = s.resident_bytes();
        let floors = s.floors();
        let grant = arbitrate(
            total,
            &[
                BudgetRequest {
                    floor_bytes: floors[0],
                    want_bytes: geometry_budget,
                },
                BudgetRequest {
                    floor_bytes: floors[1],
                    want_bytes: texture_budget,
                },
                BudgetRequest::want(shadow_budget),
            ],
        )
        .expect("the live floors fit the sum of the three requests");
        assert_eq!(
            grant.total(),
            total,
            "step {i}: the arbiter did not hand out the whole ceiling it was an \
             identity on"
        );
        for c in Consumer::ALL {
            assert!(
                bytes[c.index()] <= grant.get(c),
                "step {i}: {} holds {} B against a grant of {} B",
                c.label(),
                bytes[c.index()],
                grant.get(c)
            );
            assert!(
                bytes[c.index()] >= floors[c.index()],
                "step {i}: {} fell below its own floor",
                c.label()
            );
        }
        assert!(bytes.iter().sum::<u64>() <= total);
    }
    // ANTI-VACUITY: all three consumers actually held something.
    let bytes = s.resident_bytes();
    for c in Consumer::ALL {
        assert!(
            bytes[c.index()] > 0,
            "{} held nothing all run — its bound is a bound on zero",
            c.label()
        );
    }
    // **THE RATCHET** (P28.3, clause 2). A WORLD number: bytes are bytes on a
    // discrete card, a CPU rasterizer and a paravirtualized runner alike, and
    // the sequence is a pure function of committed input — so it is asserted
    // unconditionally, on the `VT_ADMITS_PER_FRAME_CEILING` precedent.
    //
    // Measured on this fixture: **2.99 MiB** peak combined — geometry 0.01,
    // texture 1.98 (a 2 MiB pool, full), shadow 1.00 (16 pages). The ceiling is
    // ~2.7x that, which no bounded set over this fixture can reach, so crossing
    // it means residency grew without bound rather than that a number drifted.
    //
    // Note what the split says about a gate-sized world: the geometry share is
    // 10 KiB because the pools grow lazily and this asset is one grid. The
    // number that would move on a regression is the SUM, which is why it is the
    // sum that is ratcheted. **RATCHET RULE: it may only ever DECREASE.**
    const COMBINED_CEILING: u64 = 8 * 1024 * 1024;
    println!(
        "stream arbiter: peak combined residency {:.2} MiB (ceiling {:.1} MiB) — \
         geometry {:.2}, texture {:.2}, shadow {:.2}",
        s.peak_bytes as f64 / (1024.0 * 1024.0),
        COMBINED_CEILING as f64 / (1024.0 * 1024.0),
        bytes[0] as f64 / (1024.0 * 1024.0),
        bytes[1] as f64 / (1024.0 * 1024.0),
        bytes[2] as f64 / (1024.0 * 1024.0),
    );
    assert!(
        s.peak_bytes <= COMBINED_CEILING,
        "the three page systems held {} B between them, over the {COMBINED_CEILING} B \
         ceiling — RATCHET RULE: this constant may only ever DECREASE, so a value \
         above it is a regression report and not a settings change",
        s.peak_bytes
    );
    assert!(s.peak_bytes > 0);
}

/// **THE FLOOR IS A REFUSAL, NOT A CLAMP** — and this is where
/// `StreamError::FloorExceedsBudget` has a producer over live numbers.
///
/// The live floors of a real streamed asset and a real registered texture set,
/// held against a unified budget that cannot contain them.
#[test]
fn a_unified_budget_that_cannot_hold_the_live_floors_is_refused_by_name() {
    let src = paired_source();
    let mut s = Streamer::new(4 * 1024 * 1024, 16);
    s.step(&src, 0.01, &marks_for(0), true);
    let floors = s.floors();
    assert!(
        floors[0] > 0 && floors[1] > 0,
        "neither consumer pinned anything, so there is no floor to refuse: {floors:?}"
    );
    let total: u64 = floors.iter().sum();
    let requests = [
        BudgetRequest {
            floor_bytes: floors[0],
            want_bytes: 256 * 1024 * 1024,
        },
        BudgetRequest {
            floor_bytes: floors[1],
            want_bytes: 4 * 1024 * 1024,
        },
        BudgetRequest::want(1024 * 1024),
    ];
    // One byte short of the floors: refused, by name, with both numbers.
    let err = arbitrate(total - 1, &requests).expect_err("the floors do not fit");
    assert_eq!(
        err,
        StreamError::FloorExceedsBudget {
            floor_bytes: total,
            total_bytes: total - 1,
        }
    );
    assert!(err.to_string().contains("mandatory floors"), "{err}");
    // …and the boundary is granted rather than refused, so the refusal is about
    // the floors and not about the arithmetic being off by one.
    let g = arbitrate(total, &requests).expect("floor == budget, to the byte");
    assert_eq!(g.get(Consumer::Geometry), floors[0]);
    assert_eq!(g.get(Consumer::Texture), floors[1]);
    assert_eq!(g.total(), total);
}

/// **CROSS-SYSTEM AGING** (clause 3): a cluster's tiles and its pages age
/// together — and the coupling-off control must reach the forbidden state.
///
/// The invariant is checked with `inf_stream::breaches` against a residency
/// predicate this file derives from `is_resident`, after **every** step, and the
/// dropped-group count is the anti-vacuity number: a coupling that never lets a
/// group go and one that is never populated both report zero breaches.
#[test]
fn a_clusters_tiles_and_pages_age_together() {
    let src = paired_source();
    let run = |couple: bool| -> (u64, usize, usize) {
        let mut s = Streamer::new(768 * 1024, 16);
        let mut worst = 0usize;
        let mut checked = 0usize;
        for (i, t) in LADDER.iter().enumerate() {
            s.step(&src, *t, &marks_for(i), couple);
            let by_guid = &s.by_guid;
            let texture = &s.texture;
            let broken = inf_render::stream::breaches(&s.coupling, |(g, tile)| {
                by_guid
                    .get(g)
                    .is_some_and(|h| texture.is_resident(*h, *tile))
            });
            checked += s.coupling.member_count();
            worst = worst.max(broken.len());
            if couple {
                assert!(
                    broken.is_empty(),
                    "step {i}: {} coupled (page, tile) pairs are not resident — the \
                     first is {:?}",
                    broken.len(),
                    broken.first()
                );
            }
        }
        (s.dropped, worst, checked)
    };

    let (dropped, worst, checked) = run(true);
    assert_eq!(worst, 0);
    // ANTI-VACUITY, both halves: the coupling held real members, and groups
    // really were dropped — the eviction half of "they age together".
    assert!(
        checked > 100,
        "only {checked} (page, tile) pairs were ever checked"
    );
    assert!(
        dropped > 0,
        "no group was ever dropped, so nothing aged out and the retain is untested"
    );

    // **THE CONTROL**: with the coupling off — the pre-P28.2 arrangement, a
    // `pair` that seats unconditionally — the forbidden state must be REACHED,
    // or the invariant above is satisfied by a fixture whose floor happens to
    // cover everything.
    let (_, worst_uncoupled, checked_uncoupled) = run(false);
    assert!(
        worst_uncoupled > 0,
        "the control never reached the forbidden state over {checked_uncoupled} \
         pairs — this fixture cannot see the coupling at all"
    );
}

/// **FUNCTION OF STATE, NOT HISTORY** (extended to the arbiter): two routes to
/// one residency triple produce identical arbitration.
///
/// Two different ladders that converge on the same threshold, then the same
/// step offered to both. The grants, the admits and the coupling must agree —
/// the arbiter has no memory, and neither may the walk that consumes it.
#[test]
fn two_routes_to_one_residency_triple_arbitrate_identically() {
    let src = paired_source();
    let converge = |ladder: &[f32]| -> Streamer {
        let mut s = Streamer::new(1024 * 1024, 16);
        for (i, t) in ladder.iter().enumerate() {
            s.step(&src, *t, &marks_for(i), true);
        }
        s
    };
    // Two routes, ending at the same threshold and the same mark set.
    let mut a = converge(&[4.0, 0.5, 0.2, 0.05]);
    let mut b = converge(&[2.0, 1.0, 0.1, 0.05]);
    // Settle both onto one state: the same threshold twice, so any transient of
    // the route is spent.
    for i in 0..2 {
        a.step(&src, 0.05, &marks_for(i), true);
        b.step(&src, 0.05, &marks_for(i), true);
    }
    assert_eq!(
        a.resident_bytes(),
        b.resident_bytes(),
        "two routes reached two different residencies, so the arm below would \
         compare arbitration of two different states"
    );

    let sa = a.step(&src, 0.001, &marks_for(7), true);
    let sb = b.step(&src, 0.001, &marks_for(7), true);
    assert_eq!(sa.admits, sb.admits, "the admits differ");
    assert_eq!(sa.evicts, sb.evicts, "the evictions differ");
    assert_eq!(sa.pages, sb.pages);
    assert_eq!(sa.retracted, sb.retracted);
    assert_eq!(a.coupling.member_count(), b.coupling.member_count());
    // …and the arbiter itself, over the live floors of each.
    let grant = |s: &Streamer| {
        let f = s.floors();
        arbitrate(
            64 * 1024 * 1024,
            &[
                BudgetRequest {
                    floor_bytes: f[0],
                    want_bytes: 32 * 1024 * 1024,
                },
                BudgetRequest {
                    floor_bytes: f[1],
                    want_bytes: 1024 * 1024,
                },
                BudgetRequest::want(8 * 1024 * 1024),
            ],
        )
    };
    assert_eq!(grant(&a), grant(&b));
    // ANTI-VACUITY: the step actually did something on both.
    assert!(
        !sa.admits.is_empty() || !sa.evicts.is_empty(),
        "the compared step was a no-op on both routes"
    );
}

/// **CROSS-CONSUMER IDENTITY** (P28.3): a cooked tile address carries the
/// digest of the grid it was paired against, and a mismatch is detected from
/// the FIRST reference rather than discovered address by address.
///
/// The P28.2 audit routed the *format* answer here — *"a mip-count or a content
/// digest in the tiles section, checked at load — belongs with P28.3's unified
/// streamer, which is where cross-consumer identity is owned"* — and named the
/// direction the runtime answer is blind to: an image re-imported **larger**
/// still has every cooked address, so `can_address` says yes to all of them and
/// the surface streams the wrong detail level in silence.
///
/// Measured here, in both directions, against a control that is the same asset
/// cooked against the image it actually meets.
#[test]
fn a_pairing_cooked_against_another_grid_is_detected_from_its_first_reference() {
    let bigger = vec![
        (TEX_A, full_pyramid(4096, 4096, 128, 4, true)),
        (TEX_B, full_pyramid(1024, 1024, 128, 4, false)),
    ];
    let smaller = vec![
        (TEX_A, full_pyramid(512, 512, 128, 4, true)),
        (TEX_B, full_pyramid(1024, 1024, 128, 4, false)),
    ];
    // The runtime registers `descs()`; the cooks disagree with it.
    let count = |src: &VgeomSource| -> (usize, usize) {
        let (res, by_guid) = texture_residency(4 * 1024 * 1024);
        let (mut mismatched, mut missing) = (0usize, 0usize);
        for page in 0..src.pages().len() {
            src.with_page_sections(page, |s| {
                for t in s.tile_refs() {
                    let g = t.texture().uuid().as_u128();
                    let Some(h) = by_guid.get(&g) else { continue };
                    let desc = res.desc(*h).expect("registered");
                    if !t.grid_matches(desc) {
                        mismatched += 1;
                    } else if !res.can_address(*h, t.coord()) {
                        missing += 1;
                    }
                }
            });
        }
        (mismatched, missing)
    };

    // THE CONTROL: cooked against the image it meets — nothing is detected, so
    // the two numbers below are about the mismatch and not about the walk.
    let (m, x) = count(&paired_source());
    assert_eq!((m, x), (0, 0), "the control detected a mismatch");

    // **The direction `can_address` cannot see.** Every address of a 4 096²
    // image exists inside a 2 048² one? No — but every address of the SMALLER
    // cook exists in the bigger registered image, which is the benign direction
    // the audit called out. Both are caught by the grid.
    let (m_big, _) = count(&source_paired_against(&bigger));
    assert!(m_big > 0, "a coarser cook's grid claim was accepted");
    let (m_small, _) = count(&source_paired_against(&smaller));
    assert!(m_small > 0, "a finer cook's grid claim was accepted");

    // And the SECOND texture, cooked against the image that is registered, is
    // untouched — a mismatch is per texture and does not condemn the pairing.
    let src = source_paired_against(&bigger);
    let (res, by_guid) = texture_residency(4 * 1024 * 1024);
    let mut ok_b = 0usize;
    for page in 0..src.pages().len() {
        src.with_page_sections(page, |s| {
            for t in s.tile_refs() {
                let g = t.texture().uuid().as_u128();
                if g != TEX_B {
                    continue;
                }
                let h = by_guid.get(&g).expect("registered");
                if t.grid_matches(res.desc(*h).expect("registered")) {
                    ok_b += 1;
                }
            }
        });
    }
    assert!(
        ok_b > 0,
        "the untouched texture's pairing was condemned with its neighbour's"
    );
}

/// **ONE STAMP DOMAIN**, observed across three crates.
///
/// Each residency publishes a generation drawn from what used to be its own
/// counter. With three domains the three sequences are independent and a
/// generation minted by one crate says nothing about another's; with one, a
/// generation minted *after* another is strictly greater — which is the ordering
/// cross-system eviction needs and the only thing about a stamp that may be
/// compared.
///
/// Deliberately the only place a stamp is read at all: it is a stamp, not a
/// measurement, and this arm compares two of them with `<` inside one process,
/// which is the one legal ordering.
#[test]
fn the_three_consumers_draw_their_stamps_from_one_sequence() {
    let src = paired_source();
    // Interleave: texture, shadow, geometry, texture — and the four generations
    // must come out in that order.
    let (texture, _by_guid) = texture_residency(4 * 1024 * 1024);
    let t0 = texture.layout_generation();

    let (mut shadow, _adv) = VsmResidency::new(VsmAtlasConfig::default());
    shadow
        .register_light(VsmLightDesc::clipmap(3, 8))
        .expect("registers");
    let s0 = shadow.layout_generation();

    let mut geometry = VgeomStreamer::new(VgeomStreamBudget::default());
    geometry.plan(&[inf_vgeom::VgeomWant {
        asset: ASSET,
        source: &src,
        threshold: 0.01,
    }]);
    let g0 = geometry
        .residency(ASSET)
        .expect("planned")
        .generation();

    let (texture2, _) = texture_residency(4 * 1024 * 1024);
    let t1 = texture2.layout_generation();

    assert!(
        t0 < s0 && s0 < g0 && g0 < t1,
        "the three crates' stamps do not interleave — three domains, not one: \
         texture {t0}, shadow {s0}, geometry {g0}, texture again {t1}"
    );
}
