//! Meshlet **streaming** behavior (P18.2): the residency-clamped cut, and the
//! `VgeomStreamer` that produces the floors the cut is clamped to.
//!
//! `tests/dag.rs` pins the *unclamped* cut invariant — at every threshold the
//! selected meshlets form a complete, crack-free surface. This file extends that
//! proof over the second axis streaming introduces: the **residency floor**. A
//! frame may legally draw while only a prefix of an asset's pages is in VRAM, so
//! the property that has to hold is no longer "the finest surface" but "a complete
//! surface at every (threshold, floor) pair" — softer detail, never a hole.
//!
//! The watertightness technique below is deliberately the *same* edge bookkeeping
//! `dag.rs` uses (a crack shows up as a single-used interior edge, a hole as a
//! missing rim edge), so a regression reads identically on both axes and neither
//! test can drift into asserting a weaker property than the other.

use std::collections::BTreeMap;

// `dense_mesh` is the SHARED generator (portable trig — see `test_support`'s
// header). This file used to carry a hand-copied body, because `test_support` was
// `#[cfg(test)]`-internal and an integration test cannot see that; the crate's own
// dev-dependency now turns on the `test-support` feature instead.
use inf_vgeom::test_support::dense_mesh;
use inf_vgeom::{
    build_vgeom_asset, PoolBlock, VgeomAssetReader, VgeomMesh, VgeomSource, VgeomStreamBudget,
    VgeomStreamer, VgeomWant, NOT_RESIDENT,
};

// ─────────────────────────── the fixture ───────────────────────────────────

/// Grid resolution of the shared fixture. Chosen because it is the smallest one
/// that produces the *interesting* DAG: five LOD levels, five streaming pages, and
/// roots at more than one level. See [`fixture`].
const GRID: usize = 26;

/// The fixture mesh and its v2 image, **plus the assertions that it is still the
/// hard case**.
///
/// A shallower grid degenerates into a DAG whose roots all sit at the coarsest
/// level, and every one of these tests would still pass on it while proving much
/// less: the clamp's difficult case is a root *below* the residency floor (a region
/// that failed to simplify part-way up), which is precisely what page 0 exists to
/// keep resident. Pinning the shape here means a future tweak to the builder that
/// flattens the fixture fails loudly instead of quietly hollowing out the suite.
fn fixture() -> (VgeomMesh, VgeomSource) {
    let mesh = dense_mesh(GRID);
    let src = VgeomSource::from_mesh(&mesh).expect("the fixture lays out as a v2 image");
    assert!(
        src.pages().len() >= 4,
        "the fixture must page into a real prefix ladder, got {} pages",
        src.pages().len()
    );
    let mut root_levels: Vec<u8> = mesh
        .meshlets
        .iter()
        .filter(|m| m.is_root())
        .map(|m| m.lod_level)
        .collect();
    root_levels.sort_unstable();
    root_levels.dedup();
    assert!(
        root_levels.len() >= 2,
        "the fixture must have roots at more than one LOD level (the case page 0 \
         exists for), got {root_levels:?}"
    );
    (mesh, src)
}

// ─────────────────────────── edge helpers (mirrors dag.rs) ─────────────────

fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Undirected edge → incident-triangle count over a set of meshlets.
fn edge_counts(
    mesh: &VgeomMesh,
    meshlets: impl Iterator<Item = usize>,
) -> BTreeMap<(u32, u32), u32> {
    let mut counts: BTreeMap<(u32, u32), u32> = BTreeMap::new();
    for mi in meshlets {
        let m = &mesh.meshlets[mi];
        for t in 0..m.triangle_count as usize {
            let [a, b, c] = mesh.triangle(mi, t);
            for (u, v) in [(a, b), (b, c), (c, a)] {
                *counts.entry(edge_key(u, v)).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// The set of single-used edges (the outer boundary) of an edge-count map.
fn boundary_of(counts: &BTreeMap<(u32, u32), u32>) -> Vec<(u32, u32)> {
    counts
        .iter()
        .filter(|(_, &c)| c == 1)
        .map(|(&e, _)| e)
        .collect()
}

/// Assert the given meshlet set is a watertight, complete surface whose outer
/// boundary equals `expected_boundary`: every edge is used once (only on the
/// boundary) or twice (interior manifold), with no single-used interior edge
/// (a crack) and no missing boundary edge (a hole). Identical to
/// `dag.rs::assert_watertight`, with the failing (threshold, floor) named.
fn assert_watertight(
    mesh: &VgeomMesh,
    meshlets: &[usize],
    expected_boundary: &[(u32, u32)],
    what: &str,
) {
    let counts = edge_counts(mesh, meshlets.iter().copied());
    for (&e, &c) in &counts {
        assert!(
            c == 1 || c == 2,
            "{what}: edge {e:?} used {c} times — non-manifold selection"
        );
    }
    assert_eq!(
        boundary_of(&counts),
        expected_boundary,
        "{what}: selected surface boundary differs from the mesh rim (crack or hole)"
    );
}

fn lod0_indices(mesh: &VgeomMesh) -> Vec<usize> {
    mesh.meshlets
        .iter()
        .enumerate()
        .filter(|(_, m)| m.lod_level == 0)
        .map(|(i, _)| i)
        .collect()
}

/// The mesh rim: the single-used edges of the finest (LOD-0) full surface. Every
/// legal cut, clamped or not, must reproduce exactly this boundary.
fn rim_of(mesh: &VgeomMesh) -> Vec<(u32, u32)> {
    let lod0 = lod0_indices(mesh);
    let rim = boundary_of(&edge_counts(mesh, lod0.iter().copied()));
    assert!(
        !rim.is_empty(),
        "the grid fixture has an open outer boundary"
    );
    rim
}

// ─────────────────────────── cut helpers ───────────────────────────────────

fn max_finite_error(mesh: &VgeomMesh) -> f32 {
    mesh.meshlets
        .iter()
        .map(|m| m.error)
        .filter(|e| e.is_finite())
        .fold(0.0f32, f32::max)
}

/// A dense threshold sweep across the whole error range, plus every group error
/// and its immediate neighborhood — interval endpoints are where a cut could go
/// wrong — plus one threshold beyond the top (which must select exactly the
/// roots). Mirrors `dag.rs::cut_invariant_holds_at_every_threshold`'s sweep;
/// negative thresholds are clamped away because screen-space error is a
/// non-negative quantity and the cut is undefined below zero.
fn threshold_sweep(mesh: &VgeomMesh, steps: usize) -> Vec<f32> {
    let max_err = max_finite_error(mesh);
    let mut out: Vec<f32> = (0..=steps)
        .map(|k| max_err * 1.2 * (k as f32) / (steps as f32))
        .collect();
    for g in &mesh.groups {
        out.push(g.error);
        out.push((g.error - 1e-5).max(0.0));
        out.push(g.error + 1e-5);
    }
    out.push(max_err * 10.0);
    out
}

fn selected_at(mesh: &VgeomMesh, threshold: f32, floor: u8) -> Vec<usize> {
    mesh.select_with_residency(threshold, floor)
        .map(|(i, _)| i)
        .collect()
}

fn selected_triangles(mesh: &VgeomMesh, threshold: f32, floor: u8) -> usize {
    mesh.select_with_residency(threshold, floor)
        .map(|(_, m)| m.triangle_count as usize)
        .sum()
}

/// Every residency floor the streamer can ever report for `src`: one per page
/// prefix length, taken from the finest page in the prefix — exactly what
/// `AssetResidency::floor_lod` computes.
fn every_floor(src: &VgeomSource) -> Vec<u8> {
    (1..=src.pages().len())
        .map(|p| src.pages()[p - 1].floor_lod as u8)
        .collect()
}

/// Assert the selected set picks **exactly one** meshlet on every root-to-leaf
/// chain of the DAG.
///
/// A "chain" is the walk that starts at a LOD-0 meshlet and repeatedly steps to
/// *one* of the meshlets its group produced, until it reaches a root (which has no
/// group and therefore terminates the walk). The half-open intervals
/// `[error, parent_error)` of the meshlets on such a chain tile `[0, +∞)`, so a
/// chain with two selections is an overlap (double-drawn geometry) and a chain
/// with none is a hole. Enumerating the chains would be exponential, so this is
/// the equivalent bottom-up fold: `none[i]` = every chain from `i` upward selects
/// nothing, `one[i]` = every chain from `i` upward selects exactly one.
///
/// `assemble` lays meshlets out coarsest-level-first, so a group's produced
/// (coarser) meshlets always sit at *lower* global indices than its members — one
/// ascending pass therefore visits every parent before its child.
fn assert_exactly_one_per_path(mesh: &VgeomMesh, selected: &[bool], what: &str) {
    let n = mesh.meshlets.len();
    let mut none = vec![false; n];
    let mut one = vec![false; n];
    for (i, m) in mesh.meshlets.iter().enumerate() {
        let sel = selected[i];
        if m.is_root() {
            none[i] = !sel;
            one[i] = sel;
            continue;
        }
        let g = &mesh.groups[m.group as usize];
        let lo = g.produced_start as usize;
        let hi = lo + g.produced_count as usize;
        assert!(
            hi <= i,
            "{what}: group {} produced meshlets [{lo}, {hi}) are not coarser than member {i}",
            m.group
        );
        let all_none = (lo..hi).all(|p| none[p]);
        let all_one = (lo..hi).all(|p| one[p]);
        none[i] = !sel && all_none;
        one[i] = if sel { all_none } else { all_one };
    }
    for (i, m) in mesh.meshlets.iter().enumerate() {
        if m.lod_level == 0 {
            assert!(
                one[i],
                "{what}: the chains above LOD-0 meshlet {i} do not each select exactly one \
                 meshlet (a hole or a double-draw)"
            );
        }
    }
}

fn selection_flags(mesh: &VgeomMesh, selected: &[usize]) -> Vec<bool> {
    let mut flags = vec![false; mesh.meshlets.len()];
    for &i in selected {
        flags[i] = true;
    }
    flags
}

// ─────────────────────────── 1. the gate ───────────────────────────────────

/// **The never-a-hole gate.** For every residency floor the streamer can produce
/// and a dense sweep of thresholds, the clamped cut is a legal draw set: it is
/// non-empty, it names only meshlets that are actually paged in, the triangles it
/// selects form a watertight surface with the mesh's own rim, and it picks exactly
/// one meshlet on every root-to-leaf chain of the DAG.
///
/// This is the property the whole paging design exists to make unreachable-to-
/// violate. Streaming introduces a second axis (the floor) that the pre-P18.2 cut
/// invariant never covered, and a partially resident frame is a *normal* frame —
/// so "the surface is complete at every threshold" has to become "the surface is
/// complete at every (threshold, floor) pair", or a camera move during a load
/// shows a hole in the world. Brute-forcing every floor rather than a sampled few
/// is deliberate: floors come from the page ladder, which is short, and a bug in
/// the clamp would most plausibly hide at one specific level boundary.
#[test]
fn never_a_hole_at_every_residency_and_threshold() {
    let (mesh, src) = fixture();
    let floors = every_floor(&src);
    let rim = rim_of(&mesh);
    let thresholds = threshold_sweep(&mesh, 100);

    for &floor in &floors {
        for &t in &thresholds {
            let what = format!("floor {floor}, threshold {t}");
            let sel = selected_at(&mesh, t, floor);

            // (a) a legal cut always draws something.
            assert!(!sel.is_empty(), "{what}: selected nothing");

            // (b) it may only name meshlets whose page is in VRAM.
            for &i in &sel {
                assert!(
                    mesh.meshlets[i].resident_at(floor),
                    "{what}: selected meshlet {i} (lod {}) is not resident",
                    mesh.meshlets[i].lod_level
                );
            }

            // (c) the triangles form a closed, consistent surface.
            assert_watertight(&mesh, &sel, &rim, &what);

            // (d) exactly one meshlet per root-to-leaf chain.
            assert_exactly_one_per_path(&mesh, &selection_flags(&mesh, &sel), &what);
        }
    }
}

// ─────────────────────────── 2. the golden equivalence ─────────────────────

/// At full residency the clamped cut is **the same set**, not merely an equivalent
/// surface, as the unclamped one.
///
/// Every committed golden image was rendered by the pre-streaming cut, and the
/// streaming path is only allowed to be a strict generalization of it: a cold frame
/// pages an asset in fully within budget, which means `floor_lod == 0`, which must
/// select bit-for-bit what `select` selected before. If this drifts, the goldens
/// stop measuring what they were blessed against and every image in the suite would
/// have to be re-blessed to hide it — so this equality is the one that keeps the
/// golden suite honest.
#[test]
fn full_residency_equals_the_unclamped_cut() {
    let mesh = dense_mesh(GRID);
    for &t in &threshold_sweep(&mesh, 250) {
        let clamped = selected_at(&mesh, t, 0);
        let plain: Vec<usize> = mesh.select(t).map(|(i, _)| i).collect();
        assert_eq!(
            clamped, plain,
            "select_with_residency(t = {t}, floor = 0) must equal select(t) exactly"
        );
    }
}

// ─────────────────────────── 3. degradation direction ──────────────────────

/// Clamping to a coarser floor only ever **coarsens** the surface: it never
/// smuggles in more geometry than the finer floor would have drawn.
///
/// The clamp works by treating a meshlet at or below the floor as having zero
/// error, which makes it win the cut against detail that is not paged in. That is a
/// rule which *adds* selections, so the direction of the trade has to be pinned
/// explicitly: losing residency must cost detail and save triangles, never the
/// reverse. A regression here would mean a low-VRAM machine draws *more* than a
/// fully resident one — the budget would stop being a budget.
#[test]
fn clamping_only_ever_coarsens() {
    let (mesh, src) = fixture();
    // Page floors descend (coarsest page first); walk them fine → coarse.
    let mut floors = every_floor(&src);
    floors.sort_unstable();
    floors.dedup();

    for &t in &threshold_sweep(&mesh, 100) {
        for w in floors.windows(2) {
            let (fine, coarse) = (w[0], w[1]);
            let fine_tris = selected_triangles(&mesh, t, fine);
            let coarse_tris = selected_triangles(&mesh, t, coarse);
            assert!(
                coarse_tris <= fine_tris,
                "threshold {t}: floor {coarse} drew {coarse_tris} triangles, more than the finer \
                 floor {fine}'s {fine_tris} — clamping must not refine"
            );
        }
    }
}

// ─────────────────────────── 4. CPU ↔ streamer agreement ───────────────────

/// The streamer's reported floor is exactly the floor the CPU clamp needs: a
/// meshlet has a pool slot in the remap table **iff** `resident_at(floor_lod())`
/// says it does.
///
/// `floor_lod()` is a single scalar, and it is the *only* thing the cull shader is
/// told about residency — everything else it learns from the remap table the
/// streamer uploaded. So the two have to describe the same set, or the shader will
/// either select a meshlet whose bytes are not in the pools (garbage geometry) or
/// skip one that is (a hole). Driving a real `VgeomStreamer` over a scripted walk
/// with a budget that forces partial residency exercises the agreement across
/// acquisition, hysteresis and eviction, not just at the two endpoints. The GPU
/// parity test extends exactly this equality into the shader.
#[test]
fn streamer_residency_matches_the_cpu_clamp() {
    let (mesh, src) = fixture();
    let max_err = max_finite_error(&mesh);
    const ASSET: u128 = 0x7;

    // Half the asset's bytes, one page per sync: residency spends most of the walk
    // strictly between "roots only" and "everything".
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget {
        budget_bytes: src.total_resident_bytes() / 2,
        max_loads_per_sync: 1,
        ..Default::default()
    });

    // A scripted approach → retreat → snap-close → snap-far walk.
    let mut walk: Vec<f32> = Vec::new();
    for k in 0..12 {
        walk.push(max_err * (12 - k) as f32 / 6.0);
    }
    for k in 0..12 {
        walk.push(max_err * k as f32 / 6.0);
    }
    walk.extend_from_slice(&[0.0, f32::MAX / 4.0, 0.0]);

    let rim = rim_of(&mesh);
    let mut seen_partial = false;
    for (step, &t) in walk.iter().enumerate() {
        streamer.plan(&[VgeomWant {
            asset: ASSET,
            source: &src,
            threshold: t,
        }]);
        let res = streamer
            .residency(ASSET)
            .expect("the asset is in the frame");
        assert!(
            res.resident_pages() >= 1,
            "step {step}: the root page is the floor and is granted unconditionally"
        );
        if res.resident_pages() < src.pages().len() {
            seen_partial = true;
        }
        let floor = u8::try_from(res.floor_lod()).expect("lod levels are u8 in the model");

        for (i, m) in mesh.meshlets.iter().enumerate() {
            assert_eq!(
                res.remap()[i] != NOT_RESIDENT,
                m.resident_at(floor),
                "step {step} (floor {floor}): meshlet {i} (lod {}, root {}) — the remap table \
                 and the CPU clamp disagree about residency",
                m.lod_level,
                m.is_root()
            );
        }

        // And the cut taken at that floor is drawable: every selected meshlet has a
        // pool slot, and the surface is still whole.
        let sel = selected_at(&mesh, t, floor);
        for &i in &sel {
            assert_ne!(
                res.remap()[i],
                NOT_RESIDENT,
                "step {step}: the cut selected meshlet {i}, which is not in the pools"
            );
        }
        assert_watertight(&mesh, &sel, &rim, &format!("step {step}, floor {floor}"));
    }
    assert!(
        seen_partial,
        "the budget must actually force partial residency for this test to mean anything"
    );
}

// ─────────────────────────── 5. suballocator safety ────────────────────────

/// No two resident pages ever share a pool unit, across **every** asset sharing the
/// pools.
///
/// `PoolAllocator`'s own tests prove one allocator never double-hands-out a block;
/// this pins the property one level up, where it actually matters: four pools are
/// shared by every vmesh in the frame, and pages come and go as assets enter,
/// refine, coarsen and leave. An overlap would not crash — it would silently
/// overwrite another mesh's vertices with this one's, which is the single worst
/// failure mode a suballocator has because it looks like a content bug. The want
/// walk is randomised on a fixed seed so it churns the free list (assets leaving
/// the frame free interior runs that later pages must reuse) while staying exactly
/// reproducible.
#[test]
fn pool_blocks_never_overlap_across_assets() {
    let (mesh, src) = fixture();
    let max_err = max_finite_error(&mesh);
    let assets: [u128; 4] = [0x11, 0x22, 0x33, 0x44];

    // Two assets' worth of budget for four assets: guaranteed contention, so the
    // free list is exercised rather than a monotonically growing pool.
    let mut streamer = VgeomStreamer::new(VgeomStreamBudget {
        budget_bytes: src.total_resident_bytes() * 2,
        max_loads_per_sync: 2,
        ..Default::default()
    });

    let mut rng = 0x5eed_1234_u64;
    let mut next = move || {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        rng >> 33
    };

    for step in 0..60u32 {
        let mut wants: Vec<VgeomWant<'_>> = Vec::new();
        for &asset in &assets {
            if next().is_multiple_of(5) {
                continue; // this asset left the frame this step
            }
            wants.push(VgeomWant {
                asset,
                source: &src,
                threshold: max_err * (next() % 200) as f32 / 100.0,
            });
        }
        streamer.plan(&wants);

        // Every live block, bucketed by which pool it came from — offsets are only
        // comparable within a pool.
        let mut per_pool: Vec<Vec<(u128, usize, PoolBlock)>> = vec![Vec::new(); 4];
        for (asset, res) in streamer.assets() {
            for page in 0..res.resident_pages() {
                let b = *res.blocks(page).expect("a resident page has its blocks");
                per_pool[0].push((asset, page, b.meshlets));
                per_pool[1].push((asset, page, b.mlverts));
                per_pool[2].push((asset, page, b.mltris));
                per_pool[3].push((asset, page, b.vertices));
            }
        }
        const POOL: [&str; 4] = ["meshlets", "mlverts", "mltris", "vertices"];
        for (pool, blocks) in per_pool.iter().enumerate() {
            for (i, a) in blocks.iter().enumerate() {
                for b in &blocks[i + 1..] {
                    assert!(
                        !a.2.overlaps(&b.2),
                        "step {step}, pool {}: asset {:#x} page {} block {:?} overlaps asset \
                         {:#x} page {} block {:?}",
                        POOL[pool],
                        a.0,
                        a.1,
                        a.2,
                        b.0,
                        b.1,
                        b.2
                    );
                }
            }
        }
        assert!(
            streamer.pools().used_bytes() <= streamer.pools().limit_bytes(),
            "step {step}: the pools handed out more than their limits"
        );
    }
}

// ─────────────────────────── 6. corruption degrades, never breaks ──────────

/// A page whose bytes cannot be staged blocks its asset from there on and the frame
/// degrades to the coarser pages that did load — it never becomes a hole, and it
/// never becomes a per-frame retry storm.
///
/// Bad bytes are not hypothetical: a truncated download, a bit-rotted pack or a
/// half-written cook all reach the streamer as a page that parses (the directory is
/// fine) but cannot be made sense of. The two failure modes worth pinning are the
/// obvious one — geometry disappearing — and the quiet one: a streamer that keeps
/// re-requesting a page it can never load burns the whole load budget every frame
/// forever, which shows up as a mysterious permanent stutter rather than as an
/// error. The blocked-set makes the failure a one-time, permanent, *quiet*
/// downgrade, and the residency it leaves behind is still a valid prefix, so the
/// cut at the resulting floor is still whole.
#[test]
fn a_corrupt_page_blocks_and_degrades() {
    let (mesh, _) = fixture();
    let mut bytes = build_vgeom_asset(&mesh)
        .expect("the fixture lays out as a v2 image")
        .into_bytes();

    // Corrupt a NON-root page: page 0 is the always-resident floor, and the point of
    // the test is that the floor survives the failure.
    const CORRUPT: usize = 2;
    let mlverts_off = {
        let reader = VgeomAssetReader::new(bytes.as_slice()).expect("the fresh image parses");
        assert!(
            reader.pages().len() > CORRUPT + 1,
            "need a page below the corrupt one so the truncation is observable"
        );
        let e = reader.pages()[CORRUPT];
        assert!(
            e.mlvert_count > 0,
            "the corrupt page must have micro indices"
        );
        e.mlverts_off as usize
    };
    // Point one micro vertex index past every page's vertex block. The page
    // *directory* stays byte-for-byte valid — this survives `parse` and only fails
    // in `stage_page`, which is exactly the "loads, then cannot be used" shape.
    bytes[mlverts_off..mlverts_off + 4].copy_from_slice(&u32::MAX.to_le_bytes());

    let src = VgeomSource::from_image(bytes).expect("a corrupt section still parses");
    const ASSET: u128 = 0x1;
    let want = |threshold: f32| VgeomWant {
        asset: ASSET,
        source: &src,
        threshold,
    };

    let mut streamer = VgeomStreamer::new(VgeomStreamBudget::default());
    let plan = streamer.plan(&[want(0.0)]); // threshold 0 wants every page

    assert!(
        plan.failed
            .iter()
            .any(|(a, p, msg)| *a == ASSET && *p == CORRUPT && msg.contains("non-resident vertex")),
        "the plan must report the staging failure, got {:?}",
        plan.failed
    );

    let res = streamer
        .residency(ASSET)
        .expect("the asset is in the frame");
    assert_eq!(
        res.blocked_from(),
        Some(CORRUPT),
        "the failing page must be recorded as the block point"
    );
    assert_eq!(
        res.resident_pages(),
        CORRUPT,
        "residency must truncate to a prefix strictly below the corrupt page"
    );
    for page in 0..res.resident_pages() {
        assert!(
            res.blocks(page).is_some(),
            "page {page} is resident and must own its blocks"
        );
    }
    assert!(
        res.blocks(res.resident_pages()).is_none(),
        "nothing past the prefix may hold blocks"
    );

    // The failure is permanent and silent: five more plans re-request nothing.
    let failed_after_first = streamer.stats().failed;
    assert_eq!(failed_after_first, 1, "exactly one page failed to stage");
    for round in 0..5 {
        streamer.plan(&[want(0.0)]);
        assert_eq!(
            streamer.stats().failed,
            failed_after_first,
            "round {round}: a blocked page must never be re-requested"
        );
        assert_eq!(
            streamer.residency(ASSET).unwrap().resident_pages(),
            CORRUPT,
            "round {round}: the truncated prefix must be stable"
        );
    }

    // And the degraded frame still draws a complete surface at every threshold.
    let floor = u8::try_from(streamer.residency(ASSET).unwrap().floor_lod())
        .expect("lod levels are u8 in the model");
    let rim = rim_of(&mesh);
    for &t in &threshold_sweep(&mesh, 60) {
        let what = format!("blocked floor {floor}, threshold {t}");
        let sel = selected_at(&mesh, t, floor);
        assert!(!sel.is_empty(), "{what}: selected nothing");
        assert_watertight(&mesh, &sel, &rim, &what);
        assert_exactly_one_per_path(&mesh, &selection_flags(&mesh, &sel), &what);
    }
}

// ─────────────────────────── 7. v1 is not a second-class path ──────────────

/// A v1 (bare bincode) payload and a v2 paged image stream **identically** — same
/// resident page counts, same floors, same remap tables, step for step.
///
/// "v1 keeps loading forever" is a promise about packs cooked before P18.2, and the
/// cheap way to keep it would be a second, simpler code path for old payloads —
/// which is how a compatibility promise quietly becomes "it opens, and then behaves
/// differently". Lifting v1 into the *same* v2 image at open time is what makes the
/// streaming code have exactly one shape to be right about, and this test is what
/// pins that the lift is faithful rather than merely successful.
#[test]
fn v1_and_v2_sources_stream_identically() {
    let (mesh, _) = fixture();
    let v1 = inf_asset::encode(&mesh).expect("v1 bincode encode");
    let v2 = build_vgeom_asset(&mesh)
        .expect("the fixture lays out as a v2 image")
        .into_bytes();

    let lifted = VgeomSource::from_payload(v1).expect("a v1 payload lifts");
    let paged = VgeomSource::from_image(v2).expect("a v2 image opens");

    // The lift is not an approximation: it produces the very same image.
    assert_eq!(
        lifted.payload().expect("owned bytes").as_ref(),
        paged.payload().expect("owned bytes").as_ref(),
        "lifting v1 must reproduce the v2 image byte for byte"
    );

    let max_err = max_finite_error(&mesh);
    // Portable trig (the P14 LAW): the walk is the same sequence on every
    // platform, so the two traces below are compared over identical inputs
    // everywhere, not merely over inputs identical to themselves.
    let walk: Vec<f32> = (0..24)
        .map(|k| max_err * (inf_math::psin64(k as f64 * 0.7).abs() * 2.0) as f32)
        .collect();
    let budget = paged.total_resident_bytes() / 2;

    let trace = |src: &VgeomSource| {
        let mut streamer = VgeomStreamer::new(VgeomStreamBudget {
            budget_bytes: budget,
            max_loads_per_sync: 2,
            ..Default::default()
        });
        let mut out = Vec::new();
        for &t in &walk {
            streamer.plan(&[VgeomWant {
                asset: 0x9,
                source: src,
                threshold: t,
            }]);
            let res = streamer.residency(0x9).expect("the asset is in the frame");
            out.push((
                res.resident_pages(),
                res.floor_lod(),
                res.resident_bytes(),
                res.remap().to_vec(),
            ));
        }
        out
    };

    let from_v1 = trace(&lifted);
    let from_v2 = trace(&paged);
    assert!(
        from_v1.iter().any(|s| s.0 > 1 && s.0 < paged.pages().len()),
        "the walk must pass through partial residency to be worth comparing"
    );
    assert_eq!(
        from_v1, from_v2,
        "a v1 payload and a v2 image must produce identical residency traces"
    );
}
