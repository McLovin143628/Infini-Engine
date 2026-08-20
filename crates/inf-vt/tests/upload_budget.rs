//! **IB-16 — the per-frame VT upload budget: a burst is SMOOTHED, never dropped.**
//!
//! Wave T's T33b refused a throttle on purpose, and the refusal is the spec this
//! file has to meet: *"there is no per-frame time budget and no per-frame
//! admission or upload throttle in the VT loop; the only budget is a byte
//! residency ceiling. That is deliberate and it is coupled to a measured
//! ruling… Adding one is not a local change — it re-opens a measurement, and the
//! named tripwire tests are built to go red when it lands. Do it deliberately or
//! not at all."* The AAA-readiness certification relayed it as IB-16, *"the most
//! 60 fps-relevant carried item in the texture stack"*.
//!
//! # The three claims
//!
//! 1. **Bounded.** No frame writes more than the budget's bytes.
//! 2. **Late, never never.** Every wanted tile becomes resident, in a bounded
//!    number of frames, with no want silently forgotten. A throttle that dropped
//!    would be a residency ceiling with a worse name.
//! 3. **Told apart.** A deferral caused by flow control and a deferral caused by
//!    a full pool are counted separately, because they want opposite fixes — one
//!    more frame's patience against a bigger pool — and a streaming engine that
//!    confuses them gives the author the wrong instruction.
//!
//! and, beside them, the advisory: sustained over-subscription is not a burst,
//! and is reported as sustained rather than absorbed for ever in silence.

use std::collections::BTreeSet;

use inf_vt::{
    full_pyramid, PageFormat, TileCoord, VtPoolConfig, VtResidency, VtWant, VT_PRIORITY_FEEDBACK,
    VT_SUSTAINED_THROTTLE_FRAMES,
};

/// A **refinement** want — the class the budget throttles.
///
/// The floor is exempt by construction (`inf_stream::admit_by_lane`): it is the
/// mandatory residency class, it is bounded per visible surface, and a cluster's
/// tiles ride it, so deferring one would sacrifice the floor to flow control and
/// split a P28.2 transaction across frames. A burst is a *refinement* burst.
fn refine(h: inf_vt::VtTextureHandle, tile: TileCoord) -> VtWant {
    VtWant::new(h, tile).with_priority(VT_PRIORITY_FEEDBACK)
}

const PAGE: u32 = 136;

fn page_bytes() -> u64 {
    PageFormat::Bc1.page_bytes(PAGE)
}

/// A pool of `pages` BC1 slots with an upload budget of `per_frame` pages.
fn pool(pages: u64, per_frame: u64) -> VtResidency {
    let (r, advisories) = VtResidency::new(VtPoolConfig {
        format: PageFormat::Bc1,
        stored_tile_size: PAGE,
        budget_bytes: page_bytes() * pages,
        max_texture_dim: 8192,
        trilinear: false,
        upload_budget_bytes: page_bytes() * per_frame,
    });
    assert!(advisories.is_empty(), "unexpected advisory: {advisories:?}");
    r
}

/// **The burst proof.** Ask for far more than one frame may write, and require
/// every tile to arrive — late, and never never.
#[test]
fn a_burst_is_smoothed_across_frames_and_nothing_is_dropped() {
    const SLOTS: u64 = 64;
    const PER_FRAME: u64 = 4;
    let mut res = pool(SLOTS, PER_FRAME);
    let h = res
        .register_texture(full_pyramid(1024, 1024, 128, 4, true))
        .unwrap();
    // Seat the roots first, so what follows is measured against a settled pool.
    res.apply_wants(&[]);
    let seated_roots = res.stats().roots;

    // A whip-pan's worth of level-0 tiles: 40 of them, against a four-page
    // frame. Well inside the pool, so nothing here can be deferred for want of
    // a slot — every deferral must be the throttle.
    let wants: Vec<VtWant> = (0..8)
        .flat_map(|x| (0..5).map(move |y| refine(h, TileCoord::new(0, x, y))))
        .collect();
    let asked: BTreeSet<(u32, u32)> = wants.iter().map(|w| (w.tile.x, w.tile.y)).collect();
    assert_eq!(asked.len(), 40);
    assert!(
        (asked.len() as u64) + u64::from(seated_roots) <= SLOTS,
        "the burst must FIT — otherwise a deferral could be capacity, and this \
         arm would not be about the throttle at all"
    );

    let mut frames = 0usize;
    let mut worst_bytes = 0u64;
    let mut total_throttled = 0u32;
    let mut arrived: BTreeSet<(u32, u32)> = BTreeSet::new();
    while arrived.len() < asked.len() {
        let txn = res.apply_wants(&wants);
        frames += 1;
        worst_bytes = worst_bytes.max(res.stats().upload_bytes);
        total_throttled += txn.throttled;
        for a in &txn.admits {
            arrived.insert((a.tile.x, a.tile.y));
        }
        // (3) told apart: this pool has spare slots, so every deferral here is
        //     flow control and none of it is capacity.
        assert_eq!(
            txn.throttled,
            txn.deferred,
            "frame {frames}: {} deferrals of which only {} were throttled — the \
             pool has {SLOTS} slots and the burst is {} tiles, so a deferral for \
             want of a slot is impossible and this count is wrong",
            txn.deferred,
            txn.throttled,
            asked.len()
        );
        assert!(
            frames < 200,
            "the burst did not drain in 200 frames — 'late' has become 'never'"
        );
    }

    println!(
        "IB-16 burst: {} tiles at {PER_FRAME} pages/frame drained in {frames} \
         frames; worst upload {worst_bytes} B/frame (budget {} B), {total_throttled} \
         page-deferrals, all flow control",
        asked.len(),
        page_bytes() * PER_FRAME
    );

    // (1) bounded.
    assert!(
        worst_bytes <= page_bytes() * PER_FRAME,
        "a frame wrote {worst_bytes} B against a {} B budget",
        page_bytes() * PER_FRAME
    );
    // (2) late, never never — and the ceiling is the arithmetic, not a guess:
    //     40 tiles at 4 a frame is 10 frames.
    assert_eq!(arrived, asked, "a wanted tile never arrived");
    assert_eq!(
        frames,
        (asked.len() as u64).div_ceil(PER_FRAME) as usize,
        "the burst took {frames} frames where the budget's own arithmetic says \
         it should take {}",
        (asked.len() as u64).div_ceil(PER_FRAME)
    );
    // Anti-vacuity: the throttle really bit. A budget wider than the burst would
    // satisfy every clause above while measuring nothing.
    assert!(
        total_throttled > 0,
        "nothing was ever throttled — the budget is wider than the burst and \
         this arm proves nothing"
    );
}

/// **An unthrottled pool admits the burst whole** — the control, and the proof
/// that `upload_budget_bytes: 0` really is the pre-IB-16 behaviour.
///
/// Every gate written before this wave configures `0`, and the claim that none
/// of their transactions moved rests on this being exactly true rather than
/// nearly.
#[test]
fn an_unthrottled_pool_is_the_pre_ib16_behaviour_exactly() {
    let mut res = pool(64, 0);
    let h = res
        .register_texture(full_pyramid(1024, 1024, 128, 4, true))
        .unwrap();
    res.apply_wants(&[]);
    let wants: Vec<VtWant> = (0..8)
        .flat_map(|x| (0..5).map(move |y| refine(h, TileCoord::new(0, x, y))))
        .collect();
    let txn = res.apply_wants(&wants);
    assert_eq!(txn.admits.len(), 40, "the whole burst in one transaction");
    assert_eq!(txn.deferred, 0);
    assert_eq!(txn.throttled, 0);
    assert_eq!(res.stats().throttled_frames, 0);
    assert!(res.upload_advisory().is_none());
}

/// **Sustained demand is not a burst, and is reported as sustained.**
///
/// The advisory counts a *run* of throttled frames and resets the moment one
/// drains, so a whip pan never raises it and a working set that permanently
/// wants more bandwidth than the budget grants always does.
#[test]
fn sustained_over_subscription_raises_an_advisory_and_a_burst_does_not() {
    // One page a frame against a want set that grows faster than that: the
    // camera keeps asking for new tiles, so the run never resets.
    let mut res = pool(256, 1);
    let h = res
        .register_texture(full_pyramid(1024, 1024, 128, 4, true))
        .unwrap();
    res.apply_wants(&[]);

    // The level-0 grid is 8 x 8; a want set laid out ROW-MAJOR over it stays
    // inside the texture however far it grows. The first version of this arm
    // walked `x` alone past 15, so from frame 8 on the new wants were
    // OUT OF RANGE — counted separately, never offered a slot, never throttled —
    // and the run reset while the arm still read as demand outrunning supply.
    let tile = |k: u32| TileCoord::new(0, k % 8, k / 8);

    let mut raised_at = None;
    for frame in 0..(VT_SUSTAINED_THROTTLE_FRAMES + 5) {
        // Two new tiles a frame against a one-page budget: demand outruns supply
        // for ever, which is what "sustained" means.
        let wants: Vec<VtWant> = (0..=frame)
            .flat_map(|i| [refine(h, tile(i * 2)), refine(h, tile(i * 2 + 1))])
            .collect();
        let txn = res.apply_wants(&wants);
        assert_eq!(
            txn.out_of_range, 0,
            "frame {frame} asked for a tile outside the texture — an              out-of-range want is never offered a slot and so is never              throttled, which would make this arm reset its own run"
        );
        if raised_at.is_none() && res.upload_advisory().is_some() {
            raised_at = Some(frame);
        }
    }
    let advisory = res.upload_advisory();
    println!(
        "IB-16 sustained: advisory first raised at frame {:?} (threshold \
         {VT_SUSTAINED_THROTTLE_FRAMES} frames): {advisory:?}",
        raised_at
    );
    assert!(
        advisory.is_some(),
        "demand outran the budget for {} frames and nothing was reported",
        VT_SUSTAINED_THROTTLE_FRAMES + 5
    );
    assert_eq!(
        raised_at,
        Some(VT_SUSTAINED_THROTTLE_FRAMES - 1),
        "the advisory fired at the wrong frame — it must be a run of exactly \
         {VT_SUSTAINED_THROTTLE_FRAMES} throttled frames, counted from the first"
    );

    // …and a BURST does not raise it: the same budget, a want set that drains.
    let mut res = pool(256, 1);
    let h = res
        .register_texture(full_pyramid(1024, 1024, 128, 4, true))
        .unwrap();
    res.apply_wants(&[]);
    let burst: Vec<VtWant> = (0..4).map(|i| refine(h, TileCoord::new(0, i, 0))).collect();
    for _ in 0..(VT_SUSTAINED_THROTTLE_FRAMES + 5) {
        res.apply_wants(&burst);
        assert!(
            res.upload_advisory().is_none(),
            "a four-tile burst at one page a frame raised the SUSTAINED advisory \
             — the run is not being reset when a frame drains, so every throttle \
             eventually reads as a content problem"
        );
    }
    assert_eq!(
        res.stats().throttled_frames,
        0,
        "the run must be back at zero once the burst has drained"
    );
}
