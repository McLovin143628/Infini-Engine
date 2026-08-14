//! **The A/B whip-pan gate** (P28.4, clause 3): a scripted 360° whip-pan run
//! twice — predictor OFF, predictor ON — with a bit-exact trace per arm and
//! fallback-frame counters that must fall strictly.
//!
//! # Why this file has no adapter
//!
//! The ROADMAP's clause is *"A/B inside the gate, counters not pixels"*, and the
//! counters it names are WORLD-class: which tiles a camera justifies, which of
//! them residency seated, and how many frames went by with a justified tile at
//! its fallback. Every one of those is a pure function of `(camera path, scene,
//! budget)` and none needs a device — so this runs on every CI leg and is
//! exhaustive over the path rather than sampled at whatever frame rate a runner
//! managed.
//!
//! # What is modelled, and how faithfully
//!
//! The **floor** is the shipped `analytic_floor`, unchanged. The **refinement**
//! cannot be: its wants come off a mask the GPU wrote, read back
//! `READBACK_LATENCY_FRAMES` later. It is modelled by exactly that rule — the
//! level a surface's footprint justified **two ticks ago**, for the surfaces
//! that were visible two ticks ago — which is what `vt_feedback.wgsl` marks and
//! when the ring lets the CPU see it. That lag is not an implementation detail
//! here; it is the entire thing the predictor exists to close, and a model
//! without it would make this file a tautology in the other direction.
//!
//! # The oracle
//!
//! A gate cannot see an error the subsystems share, so nothing here is borrowed
//! from the producer:
//!
//! * **the blur counter is re-derived** — for every surface visible at tick `t`,
//!   this file computes the level its footprint justifies with arithmetic
//!   written out below, and asks the residency whether those tiles are seated.
//!   It never calls `vt_stream::count_fallbacks`, and it does not ask *which
//!   class* wanted a tile: what a player sees is whether the level is there.
//! * **the prediction is replayed** — re-derived from the recorded camera path
//!   with its own secant, arc and Rodrigues, and compared against the pose the
//!   producer used. It never calls `inf_math::dead_reckon` to check
//!   `inf_math::dead_reckon`.

use std::collections::BTreeSet;

use glam::{DVec3, Vec3};
use inf_render::{
    analytic_floor, justified_mip, ndc_margin, on_screen, projection_scale, screen_diameter_px,
    speculative_wants, RenderView, VtCoverage, VtPoolConfig, VtTextureSet, VtTextures, VtWant,
    READBACK_LATENCY_FRAMES, VT_FEEDBACK_MAX_TILES,
};
use inf_vt::{TileCoord, VtTextureHandle};

// ── the script ───────────────────────────────────────────────────────────────

/// Ticks the whole run lasts — 260 at the shipped 60 Hz step is 4.3 s.
const TICKS: u64 = 260;
/// Ticks the camera stands still before the whip, so the history is full and
/// residency has settled. A predictor measured from a cold pool would be
/// measuring the warm-up.
const WARM: u64 = 40;
/// Ticks the 360° sweep takes: 3 s, i.e. **120°/s** average.
const SWEEP: u64 = 180;
/// Ticks of the sweep spent accelerating, and the same again decelerating.
///
/// The ramps are not decoration: a constant-rate turn is the case dead
/// reckoning is exactly right about, and a fixture made only of those would
/// measure a tautology. A third of this sweep is a rate the predictor is
/// provably wrong about.
const RAMP: u64 = 30;
/// The shipped horizon, in committed ticks — **0** since P28.5, i.e. the
/// committed pose with the speculative lane still running. See
/// [`inf_render::DEFAULT_PREDICT_HORIZON_TICKS`] and
/// `docs/memos/p28-5-lead-time-ruling.md` for the ruling and its measurement.
const HORIZON: u32 = inf_render::DEFAULT_PREDICT_HORIZON_TICKS;
/// **The lead the ROADMAP's clause prescribed** — 18 ticks, 300 ms at 60 Hz.
///
/// The refuted arm of the A/B, kept by name so
/// `a_lead_time_costs_this_fixture_what_the_lane_earns_it` runs the shipped
/// zero against the prescription rather than against a literal.
const LEAD: u32 = inf_render::ROADMAP_PREDICT_HORIZON_TICKS;

/// **The whip-pan's yaw at `tick`**, radians — a closed form, so the camera at
/// tick `t` does not depend on the camera at tick `t − 1` and the oracle can
/// seek to any tick without replaying the path.
///
/// Piecewise-constant angular acceleration: still, ramp up, constant, ramp down,
/// still. The constant rate `w` is set so the phases integrate to exactly one
/// revolution: `w · (RAMP/2 + hold + RAMP/2) = 2π`.
fn yaw_at(tick: u64) -> f64 {
    let hold = (SWEEP - 2 * RAMP) as f64;
    let w = std::f64::consts::TAU / (RAMP as f64 + hold);
    if tick <= WARM {
        return 0.0;
    }
    let u = (tick - WARM).min(SWEEP) as f64;
    let r = RAMP as f64;
    if u < r {
        w * u * u / (2.0 * r)
    } else if u < r + hold {
        w * r * 0.5 + w * (u - r)
    } else {
        let v = u - r - hold;
        w * r * 0.5 + w * hold + w * (v - v * v / (2.0 * r))
    }
}

/// The camera at `tick`: a whip-pan about `up`, with a slow drift so the path is
/// not a pure rotation about one point.
///
/// Portable trig, because the pose feeds a want set two hosts are claimed to
/// agree about (the P14 law).
fn whip_view(tick: u64) -> RenderView {
    let a = yaw_at(tick);
    RenderView {
        origin: inf_math::FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.012 * tick as f64, 1.6, 0.0),
        forward: Vec3::new(inf_math::psin64(a) as f32, 0.0, -inf_math::pcos64(a) as f32)
            .normalize(),
        up: Vec3::Y,
        fov_y: 60_f32.to_radians(),
        near: 0.1,
        width: 1280,
        height: 720,
        ortho: None,
    }
}

// ── the scene ────────────────────────────────────────────────────────────────

/// Surfaces on the ring — **three clusters of four**, not twelve evenly spaced.
///
/// An even ring keeps the arrival rate constant; clusters make it bursty, which
/// is the shape a prefetcher is for and the shape a fixed lag hurts most.
const CLUSTERS: [f64; 3] = [0.0, 2.0943951023931953, 4.1887902047863905];
const PER_CLUSTER: usize = 4;
const RING_RADIUS: f64 = 14.0;
/// Big enough that the surface's footprint justifies a level with **more than
/// `VT_FLOOR_MAX_TILES` tiles**, which is the only regime in which the floor and
/// the refinement settle on different mips and the two classes are separable at
/// all. Measured: 1 069 px, mip 1 of a 2 048² (64 tiles) against the floor's
/// mip 2 (16).
const SURFACE_RADIUS: f32 = 12.0;

fn surface_count() -> usize {
    CLUSTERS.len() * PER_CLUSTER
}

/// The registry: one 2 048² texture per ring surface.
///
/// **800 pages**, which is comfortably over the steady demand and is the point:
/// a pool under the demand makes every frame a fallback frame, in both arms,
/// for a reason no predictor can touch (see
/// `a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so`). What this
/// fixture measures is *latency*, so it is deliberately not a budget fixture.
fn library(pages: u64) -> (VtTextures, Vec<VtTextureHandle>) {
    let (mut lib, _adv) = VtTextures::new(VtPoolConfig {
        budget_bytes: pages * 8192,
        ..Default::default()
    });
    let handles = (0..surface_count())
        .map(|_| {
            lib.residency_mut()
                .register_texture(inf_vt::full_pyramid(2048, 2048, 128, 4, true))
                .expect("the mandatory floor fits")
        })
        .collect();
    (lib, handles)
}

/// The coverage list — **camera-free**, exactly as `scene_coverage` is, so the
/// committed and predicted cameras are handed the same scene.
fn coverage(handles: &[VtTextureHandle]) -> Vec<VtCoverage> {
    let mut out = Vec::new();
    for (c, base) in CLUSTERS.iter().enumerate() {
        for k in 0..PER_CLUSTER {
            let a = base + (k as f64 - 1.5) * 0.11;
            let h = handles[c * PER_CLUSTER + k];
            out.push(VtCoverage {
                centre: Vec3::new(
                    (RING_RADIUS * inf_math::psin64(a)) as f32,
                    1.6,
                    (-RING_RADIUS * inf_math::pcos64(a)) as f32,
                ),
                radius: SURFACE_RADIUS,
                set: VtTextureSet {
                    albedo: h.0 + 1,
                    ..VtTextureSet::NONE
                },
                vgeom: false,
            });
        }
    }
    out
}

// ── the oracle: what a surface visibly needs, re-derived ─────────────────────

type Addr = (u32, u32, u32, u32);

/// **The level every visible surface's footprint justifies, and its tiles** —
/// this file's own derivation, not the producer's want set.
///
/// Written out rather than borrowed because it is the measurement the whole
/// gate rests on: a surface is *blurry* exactly when the level its screen
/// footprint justifies is not resident, whichever class did or did not ask for
/// it. The rule is `justified_mip` walked coarser until the level fits the
/// refinement's cap — the shipped rule, re-derived, so a one-sided edit fails
/// here (the `cluster_pages` oracle's discipline).
fn justified_tiles(
    res: &inf_vt::VtResidency,
    view: &RenderView,
    cov: &[VtCoverage],
) -> Vec<(VtTextureHandle, TileCoord)> {
    let view_proj = view.view_proj();
    let scale = projection_scale(view);
    let eye = view.eye_local();
    let half_h = (view.height as f32 * 0.5).max(1.0);
    let mut out = Vec::new();
    for c in cov {
        let px = screen_diameter_px(c.centre, c.radius, eye, scale);
        if !on_screen(&view_proj, c.centre, c.radius, ndc_margin(px, half_h)) {
            continue;
        }
        for slot in c.set.slots() {
            if slot == 0 {
                continue;
            }
            let h = VtTextureHandle(slot - 1);
            let Some(desc) = res.desc(h) else { continue };
            let extent = desc.mips[0].width.max(desc.mips[0].height);
            let mut lv = justified_mip(extent, px, desc.mip_count());
            while lv + 1 < desc.mip_count()
                && desc.mips[lv as usize].tile_count() > VT_FEEDBACK_MAX_TILES
            {
                lv += 1;
            }
            let m = desc.mips[lv as usize];
            for y in 0..m.tiles_y {
                for x in 0..m.tiles_x {
                    out.push((h, TileCoord::new(lv, x, y)));
                }
            }
        }
    }
    out
}

fn addr(t: VtTextureHandle, c: TileCoord) -> Addr {
    (t.0, c.mip, c.x, c.y)
}

// ── one arm ──────────────────────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
struct Arm {
    /// One line a tick, **by address and never by slot** (the P27 rule): a slot
    /// index is an allocation detail two arms may legitimately disagree about.
    trace: Vec<String>,
    /// **THE COUNTER**: tiles a visible surface's footprint justified and the
    /// pool did not hold, summed over frames.
    blur: u64,
    /// …and the frames at least one of them fell in.
    blur_frames: u64,
    /// Justified tiles offered to the measurement at all — the **anti-vacuity**
    /// number for the two above.
    justified: u64,
    floor_fallback: u64,
    floor_fallback_frames: u64,
    predict_wants: u64,
    admits: u64,
    evicts: u64,
    deferred: u64,
    /// **The world invariant's counter**: **floor** wants that were resident
    /// before a transaction and were evicted by it.
    ///
    /// The floor and not the whole proved set, and the difference is the rank
    /// working rather than failing: a floor miss taking a resident refinement's
    /// slot is exactly what P28.3's lane walk was built to allow, and counting
    /// it as a breach would assert the defect that batch removed. What may never
    /// happen is a resident **floor** tile losing its slot — lane 0's residents
    /// are protected before lane 0's own misses are offered anything.
    floor_breaches: u64,
    /// Per tick, how many justified tiles were resident — the series the ON arm
    /// must dominate the OFF arm on, everywhere.
    sharp: Vec<usize>,
    /// Per tick, the proved (floor ∪ feedback) wants that were resident after
    /// the transaction — the SET the ON arm must contain the OFF arm's.
    proved_resident: Vec<BTreeSet<Addr>>,
    /// Per tick, the pose the producer predicted, recorded for the oracle.
    predicted: Vec<Option<(DVec3, DVec3)>>,
    /// **The arrival window** (P28.4 audit): tiles justified by a surface that
    /// is visible *now* and was not visible one tick ago, and how many of them
    /// the pool did not hold.
    ///
    /// The only class a *lead time* can serve. Everything else in this file
    /// measures surfaces that are already on screen, and for those "ask for what
    /// is needed now" is unbeatable in a loop with no admission throttle — which
    /// is the ruling `a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so`
    /// already reached for the floor. If dead reckoning earns anything anywhere,
    /// it is here.
    arrival_blur: u64,
    arrival_justified: u64,
    arrival_frames: u64,
    /// **Speculative wants this transaction actually SEATED**, and the ticks at
    /// least one was — the "there was headroom" half of the equality's
    /// anti-vacuity (P28.4 audit).
    spec_admits: u64,
    spec_admit_ticks: u64,
    /// **Speculative wants still unseated after the transaction**, and the ticks
    /// at least one was — the "the pool pushed back" half.
    ///
    /// Both are needed and neither is enough. An equality between the two arms'
    /// proved sets proves the rank protects the classes above it only if the
    /// lane below was *pressing* on them; measured under a lane that was never
    /// admitted, or never refused, it is an equality between two inert runs.
    spec_denied: u64,
    spec_denied_ticks: u64,
}

fn run(predict: Option<u32>, pages: u64) -> Arm {
    let (mut lib, handles) = library(pages);
    let cov = coverage(&handles);
    let mut hist = inf_math::CameraHistory::new();
    let mut arm = Arm::default();

    for tick in 0..TICKS {
        let view = whip_view(tick);
        assert!(
            hist.commit(inf_math::CameraSample {
                tick,
                eye: view.eye_world,
                forward: view.forward.as_dvec3(),
                up: view.up.as_dvec3(),
            }),
            "tick {tick} was refused by the history"
        );

        let mut wants = analytic_floor(&lib, &view, &cov);
        let floor_end = wants.len();

        // **The refinement lane, at the ring's own latency.** The mask the CPU
        // reads at tick `t` was written at `t − READBACK_LATENCY_FRAMES`, and it
        // could only mark surfaces that were visible then. Both halves of that
        // are the lag the predictor exists to close.
        if tick >= READBACK_LATENCY_FRAMES {
            let past = whip_view(tick - READBACK_LATENCY_FRAMES);
            wants.extend(
                justified_tiles(lib.residency(), &past, &cov)
                    .into_iter()
                    .map(|(h, t)| VtWant::refine(h, t)),
            );
        }
        let proved_end = wants.len();

        let p = predict.and_then(|h| inf_math::dead_reckon(&hist, h));
        arm.predicted.push(p.map(|p| (p.eye, p.forward)));
        let mut speculated: BTreeSet<Addr> = BTreeSet::new();
        if let Some(p) = &p {
            let spec = speculative_wants(lib.residency(), &view, &cov, p);
            arm.predict_wants += spec.len() as u64;
            speculated = spec.iter().map(|w| addr(w.texture, w.tile)).collect();
            wants.extend(spec);
        }

        // The floor's resident set **as offered**, kept beside the transaction —
        // the arbiter does not compute this and cannot agree with it by
        // accident.
        let floor_before: BTreeSet<Addr> = wants[..floor_end]
            .iter()
            .filter(|w| lib.residency().is_resident(w.texture, w.tile))
            .map(|w| addr(w.texture, w.tile))
            .collect();

        let txn = lib.residency_mut().apply_wants(&wants);
        for e in &txn.evicts {
            if floor_before.contains(&addr(e.texture, e.tile)) {
                arm.floor_breaches += 1;
            }
        }
        // **The two regimes the equality below rests on** (P28.4 audit), read
        // off the transaction and the pool rather than inferred: how many
        // speculative addresses this transaction seated, and how many it left
        // unseated. `residency_never_falls_below_the_proved_classes_under_
        // speculation` is a statement about a lane that was both admitted
        // somewhere and refused somewhere; without these it is a statement about
        // whichever of the two happened.
        let seated = txn
            .admits
            .iter()
            .filter(|a| speculated.contains(&addr(a.texture, a.tile)))
            .count() as u64;
        arm.spec_admits += seated;
        arm.spec_admit_ticks += u64::from(seated > 0);
        let denied = speculated
            .iter()
            .filter(|(t, mip, x, y)| {
                !lib.residency()
                    .is_resident(VtTextureHandle(*t), TileCoord::new(*mip, *x, *y))
            })
            .count() as u64;
        arm.spec_denied += denied;
        arm.spec_denied_ticks += u64::from(denied > 0);
        arm.proved_resident.push(
            wants[..proved_end]
                .iter()
                .filter(|w| lib.residency().is_resident(w.texture, w.tile))
                .map(|w| addr(w.texture, w.tile))
                .collect(),
        );

        // **This file's own measurement**: what a surface visible NOW needs, and
        // whether it is there. No want class is consulted.
        let need = justified_tiles(lib.residency(), &view, &cov);
        let (mut blur, mut sharp) = (0u64, 0usize);
        for (h, t) in &need {
            if lib.residency().is_resident(*h, *t) {
                sharp += 1;
            } else {
                blur += 1;
            }
        }
        arm.justified += need.len() as u64;
        arm.blur += blur;
        arm.blur_frames += u64::from(blur > 0);
        arm.sharp.push(sharp);

        // **The arrival window**: the surfaces that entered view on THIS tick.
        // Visibility is a property of the camera path alone, so "was it visible
        // a tick ago" is asked of `whip_view(tick - 1)` and of nothing residency
        // holds.
        if tick > 0 {
            let seen_before: BTreeSet<u32> =
                justified_tiles(lib.residency(), &whip_view(tick - 1), &cov)
                    .iter()
                    .map(|(h, _)| h.0)
                    .collect();
            let mut arriving_blur = 0u64;
            for (h, t) in &need {
                if seen_before.contains(&h.0) {
                    continue;
                }
                arm.arrival_justified += 1;
                arriving_blur += u64::from(!lib.residency().is_resident(*h, *t));
            }
            arm.arrival_blur += arriving_blur;
            arm.arrival_frames += u64::from(arriving_blur > 0);
        }

        let floor_miss = wants[..floor_end]
            .iter()
            .filter(|w| !lib.residency().is_resident(w.texture, w.tile))
            .count() as u64;
        arm.floor_fallback += floor_miss;
        arm.floor_fallback_frames += u64::from(floor_miss > 0);
        arm.admits += txn.admits.len() as u64;
        arm.evicts += txn.evicts.len() as u64;
        arm.deferred += u64::from(txn.deferred);

        arm.trace.push(format!(
            "{tick} b{blur}/{} f{floor_miss} a{} e{} d{}",
            need.len(),
            txn.admits.len(),
            txn.evicts.len(),
            txn.deferred
        ));
    }
    arm
}

/// The shipped fixture's pool.
const PAGES: u64 = 800;

// ── the arms ─────────────────────────────────────────────────────────────────

/// **Bit-exact per arm, and the two arms differ.**
///
/// Both halves matter. Determinism alone is satisfied by a predictor that does
/// nothing; difference alone is satisfied by one that is random. The second
/// assertion is what makes the first a statement about this batch.
#[test]
fn each_whip_pan_arm_replays_to_itself_and_the_two_arms_diverge() {
    let off = run(None, PAGES);
    assert_eq!(off.trace, run(None, PAGES).trace, "OFF is not reproducible");
    let on = run(Some(HORIZON), PAGES);
    assert_eq!(
        on.trace,
        run(Some(HORIZON), PAGES).trace,
        "ON is not reproducible"
    );
    assert_ne!(
        off.trace, on.trace,
        "the predictor changed nothing — the A/B below would be a comparison of \
         one arm with itself"
    );
    assert_eq!(off.trace.len(), TICKS as usize);
}

/// **THE ROADMAP'S CLAUSE**: fallback-frame counters strictly reduced, with the
/// OFF leg carrying its own anti-vacuity.
#[test]
fn the_predictor_strictly_reduces_a_whip_pans_fallback_frames() {
    let off = run(None, PAGES);
    let on = run(Some(HORIZON), PAGES);

    for (name, a) in [("OFF", &off), ("ON ", &on)] {
        println!(
            "{name}  blur {}/{} over {} frames | floor {} over {} frames | \
             admits {} evicts {} deferred {} | speculation {}",
            a.blur,
            a.justified,
            a.blur_frames,
            a.floor_fallback,
            a.floor_fallback_frames,
            a.admits,
            a.evicts,
            a.deferred,
            a.predict_wants
        );
    }

    // ANTI-VACUITY, on the OFF leg, before anything is read off it.
    assert!(
        off.blur_frames > 0,
        "the OFF arm was never blurry — the fixture cannot miss and the \
         reduction below would be a bound on zero"
    );
    assert!(
        off.blur_frames < TICKS,
        "every frame is blurry — the pool is under the steady demand, which is \
         a budget problem the predictor cannot fix and must not be credited for"
    );
    assert!(off.justified > 0 && on.predict_wants > 0);

    // THE CLAUSE.
    assert!(
        on.blur_frames < off.blur_frames,
        "fallback frames did not strictly fall: {} ON against {} OFF",
        on.blur_frames,
        off.blur_frames
    );
    assert!(
        on.blur < off.blur,
        "fallback tiles did not strictly fall: {} ON against {} OFF",
        on.blur,
        off.blur
    );
}

/// **THE REFUTATION THAT CHOSE WHAT TO PREFETCH**, kept as an arm because it is
/// the reason `VT_PREDICT_MAX_TILES` is the refinement's cap and not the
/// floor's.
///
/// `apply_wants` seats a miss the frame it is offered and there is no per-frame
/// admission throttle, so the floor's fallback is `max(0, demand − pool)` — a
/// function of two numbers a prediction changes neither of. Run over a
/// deliberately undersized pool, where the floor misses on every frame, the two
/// arms must come out **identical on every floor counter** while the predictor
/// is demonstrably running.
#[test]
fn a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so() {
    const STARVED: u64 = 96;
    let off = run(None, STARVED);
    let on = run(Some(HORIZON), STARVED);
    println!(
        "starved pool ({STARVED} pages): floor {} over {} frames both arms; \
         speculation offered {}",
        off.floor_fallback, off.floor_fallback_frames, on.predict_wants
    );
    assert!(
        off.floor_fallback_frames == TICKS,
        "the pool is not actually saturated, so this arm proves nothing"
    );
    assert!(on.predict_wants > 0, "the predictor was not even running");
    assert_eq!(
        (on.floor_fallback, on.floor_fallback_frames),
        (off.floor_fallback, off.floor_fallback_frames),
        "the floor's fallback moved under saturation — if this ever fails, an \
         admission throttle has appeared and VT_PREDICT_MAX_TILES needs \
         re-deciding"
    );
}

/// **The world invariant, at full speculative pressure** — the P26.4 floor law
/// extended to the lane P28.4 added.
#[test]
fn residency_never_falls_below_the_proved_classes_under_speculation() {
    let off = run(None, PAGES);
    let on = run(Some(HORIZON), PAGES);
    assert_eq!(
        off.floor_breaches, 0,
        "the OFF arm evicted a resident FLOOR tile"
    );
    assert_eq!(
        on.floor_breaches, 0,
        "speculation cost a resident FLOOR tile its slot"
    );
    // ANTI-VACUITY: the pool was genuinely contested, or "nothing was evicted"
    // is a statement about an empty list.
    assert!(
        on.evicts > 0 && off.evicts > 0,
        "the pool was never contested"
    );

    // **`residency ⊇ floor ∪ feedback` under full speculative pressure**, as a
    // SET and per tick — not as a count, which two different sets of the same
    // size satisfy.
    let mut gained = 0usize;
    for (t, (a, b)) in off
        .proved_resident
        .iter()
        .zip(&on.proved_resident)
        .enumerate()
    {
        assert!(
            a.is_subset(b),
            "tick {t}: speculation cost the proved classes {:?}",
            a.difference(b).take(4).collect::<Vec<_>>()
        );
        gained += b.len() - a.len();
    }
    // **The containment came out an EQUALITY**, and that is the result rather
    // than a weakness in the arm. Every proved want is offered a slot before any
    // speculative one, so a pool that can seat it seats it in both arms and a
    // pool that cannot seats it in neither: a strictly-lower lane is *exactly*
    // neutral to the classes above it, not merely non-harmful. Printed rather
    // than asserted as an equality, because the day speculation legitimately
    // hands the feedback a tile it had already fetched, this must not fail.
    //
    // The anti-vacuity is therefore about the pressure, not about the gain: the
    // proved sets have to be large, the pool has to have refused something, and
    // the predictor has to have asked.
    println!(
        "proved residency: identical at every tick, ON holds {gained} extra tile-frames; \
         proved set peaks at {} tiles, {} deferrals under {} speculative wants; \
         speculation seated {} tiles over {} ticks and was refused {} over {} ticks",
        on.proved_resident
            .iter()
            .map(BTreeSet::len)
            .max()
            .unwrap_or(0),
        on.deferred,
        on.predict_wants,
        on.spec_admits,
        on.spec_admit_ticks,
        on.spec_denied,
        on.spec_denied_ticks
    );
    assert!(
        on.proved_resident
            .iter()
            .map(BTreeSet::len)
            .max()
            .unwrap_or(0)
            > 100,
        "the proved set is too small for the containment to mean anything"
    );
    assert!(
        on.deferred > 0 && on.predict_wants > 0,
        "nothing was refused a slot, so nothing was ranked"
    );

    // **THE ANTI-VACUITY THAT DECIDES WHAT THE EQUALITY MEANS** (P28.4 audit).
    //
    // Identical-because-protected and identical-because-inert are the same
    // measurement, and the counters above cannot tell them apart: `deferred` and
    // `predict_wants` are both satisfied by a lane that offered 137 584 wants
    // and was handed nothing, which is a run in which the rank was never tested.
    // What has to be true is that the lane got in somewhere — so there were
    // ticks with headroom, and the equality is not "the pool was full all
    // along" — AND that it was turned away somewhere, so there were ticks
    // without, and the equality is not "the pool was never contested". Only
    // under both is "a strictly-lower lane is exactly neutral to the classes
    // above it" a statement about the rank.
    //
    // Measured here: seated on 22 of the 260 ticks, refused on 106.
    assert!(
        on.spec_admit_ticks > 0,
        "no speculative want was ever SEATED — the two arms are identical \
         because the lane below never got in, not because the rank held it out \
         of the classes above"
    );
    assert!(
        on.spec_denied_ticks > 0,
        "no speculative want was ever REFUSED — the pool was never under \
         pressure, so nothing was ranked and the equality is arithmetic"
    );

    // **And the same claim is NOT made per tick about `sharp`**, deliberately,
    // because it is false and the reason is worth recording. `sharp` counts
    // tiles a *currently visible* surface's footprint justifies; the refinement
    // lane only asks for the footprint of two ticks ago, so on a turning camera
    // some of those tiles are resident and wanted by **nobody this frame** — and
    // an unwanted slot is exactly what a speculative miss is supposed to take
    // first (the P28.3 audit's `reserved` fix). Speculation can therefore lose
    // the visible set a tile it was about to want anyway, on individual ticks,
    // while never touching a tile any class has actually asked for. The claim
    // that survives is the aggregate one.
    let (a, b) = (
        off.sharp.iter().sum::<usize>(),
        on.sharp.iter().sum::<usize>(),
    );
    assert!(
        b > a,
        "the predictor did not leave the visible set sharper overall: {a} → {b}"
    );
}

/// **THE ORACLE — an independent replay of the recorded history.**
///
/// The prediction is re-derived here from the recorded camera path, longhand:
/// the secant over the six-tick window, the arc between its ends, Rodrigues
/// about their cross product. It is *not* `inf_math::dead_reckon` called twice.
///
/// **It replays at [`LEAD`] and not at the shipped [`HORIZON`], and that is the
/// point** (P28.5 audit). This arm is an oracle of the *formula*, and at
/// `h = 0` the formula collapses: the secant is scaled by nothing, the arc is
/// turned by nothing, and both spellings return the newest committed pose. Run
/// at the shipped zero it reported a worst error of `0e0` — not because the two
/// derivations agree, but because neither of them does anything. The lead-time
/// ruling would have silently retired this oracle; running it at the
/// prescription keeps the rotation half live, which is the half a second
/// spelling can actually disagree about.
///
/// `led` is the guard that says so: a prediction that never leaves the
/// committed pose is the vacuous shape, and it fails here rather than passing
/// quietly.
#[test]
fn the_prediction_replays_from_the_recorded_history_alone() {
    let on = run(Some(LEAD), PAGES);
    let window = inf_math::PREDICT_HISTORY as u64;
    let (mut checked, mut worst, mut turned) = (0usize, 0.0f64, 0usize);
    let mut led = 0usize;

    for tick in 0..TICKS {
        let Some((eye, forward)) = on.predicted[tick as usize] else {
            assert!(
                tick + 1 < window,
                "tick {tick} had a full window and no prediction"
            );
            continue;
        };
        let oldest = tick.saturating_sub(window - 1);
        let (a, b) = (whip_view(oldest), whip_view(tick));
        let span = (tick - oldest) as f64;
        let h = f64::from(LEAD);

        assert_eq!(
            b.eye_world + (b.eye_world - a.eye_world) * (h / span),
            eye,
            "tick {tick}: the eye is not a secant over the window"
        );

        // The arc, longhand — normalized ends, because the directions arrive as
        // f32 and `|f| = 1` only to f32.
        let fa = a.forward.as_dvec3().normalize();
        let fb = b.forward.as_dvec3().normalize();
        let axis = fa.cross(fb);
        let raw = b.forward.as_dvec3();
        let want = if axis.length() <= 1e-9 {
            raw
        } else {
            turned += 1;
            let axis = axis.normalize();
            let arc = inf_math::pacos64(fa.dot(fb).clamp(-1.0, 1.0));
            let turn = (arc * h / span).min(inf_math::PREDICT_MAX_TURN);
            let (s, c) = (inf_math::psin64(turn), inf_math::pcos64(turn));
            raw * c + axis.cross(raw) * s + axis * (axis.dot(raw) * (1.0 - c))
        };
        let err = (want - forward).length();
        worst = worst.max(err);
        assert!(
            err < 1e-12,
            "tick {tick}: the replayed direction is {err} from the producer's"
        );
        // Did the prediction actually LEAD the committed pose on this tick?
        led += usize::from((forward - raw).length() > 1e-6);
        checked += 1;
    }
    println!(
        "oracle: {checked} predictions replayed ({turned} through a real arc, \
         {led} that actually led the committed pose at h={LEAD}), worst \
         direction error {worst:e}"
    );
    assert!(
        checked as u64 > TICKS - window,
        "the replay checked almost nothing"
    );
    assert!(
        turned > SWEEP as usize / 2,
        "almost nothing in the path actually turned — the replay never exercised \
         the rotation half"
    );
    // **ANTI-VACUITY, and it is the guard the shipped horizon needed.**
    // `turned` counts ticks whose HISTORY turned, which is a property of the
    // fixture and stays true at any horizon — so it cannot see a lead of zero.
    // This counts ticks whose PREDICTION turned, which is a property of the
    // thing under test.
    assert!(
        led > SWEEP as usize / 2,
        "only {led} of {checked} predictions differed from the committed pose — \
         the replay is comparing two spellings of the identity, which is what \
         this oracle reads as at a zero lead"
    );
}

/// **The horizon sweep**, printed for the memo and asserted as a band.
///
/// The ROADMAP names 200–500 ms. Every horizon in that band must strictly beat
/// OFF — a band in which some members lose is a band the ROADMAP should not
/// have named, and that half of P28.4's claim survives intact.
///
/// **What P28.5 changes is the second half.** P28.4 asserted the shipped
/// horizon was no worse than either *end of the band*; the band is flat (nine
/// frames of 131 separate best from worst), so that comparison could not see
/// what the zero-lead control then showed — every member of the band loses to
/// no lead at all. The assertion is now the ruling: the **shipped** horizon
/// beats every member of the band, one row at a time. The day a member wins,
/// this goes red and `DEFAULT_PREDICT_HORIZON_TICKS` is wrong.
#[test]
fn every_horizon_in_the_roadmaps_band_beats_the_predictor_being_off() {
    let off = run(None, PAGES);
    let mut table = Vec::new();
    for ms in [200u32, 250, 300, 350, 400, 450, 500] {
        let ticks = inf_math::horizon_ticks(ms, 60);
        let on = run(Some(ticks), PAGES);
        table.push((ms, ticks, on.blur_frames, on.blur));
        assert!(
            on.blur_frames < off.blur_frames,
            "{ms} ms ({ticks} ticks) did not beat OFF: {} against {}",
            on.blur_frames,
            off.blur_frames
        );
    }
    println!(
        "horizon sweep (OFF = {} blur frames, {} blur tiles):",
        off.blur_frames, off.blur
    );
    for (ms, ticks, frames, tiles) in &table {
        println!("  {ms:>3} ms / {ticks:>2} ticks -> {frames} frames, {tiles} tiles");
    }
    // **THE P28.5 RULING**, asserted against every row rather than against the
    // two ends: the shipped horizon is not in this band and beats all of it.
    let shipped = run(Some(HORIZON), PAGES);
    assert!(
        shipped.blur_frames < off.blur_frames,
        "the shipped horizon does not beat OFF at all"
    );
    for (ms, ticks, frames, tiles) in &table {
        assert!(
            shipped.blur_frames <= *frames,
            "the ROADMAP's {ms} ms ({ticks} ticks) BEATS the shipped horizon \
             ({} ticks): {} frames / {} tiles against {frames} / {tiles}. The \
             lead-time ruling in docs/memos/p28-5-lead-time-ruling.md is \
             refuted and DEFAULT_PREDICT_HORIZON_TICKS has to be re-decided.",
            HORIZON,
            shipped.blur_frames,
            shipped.blur
        );
    }
}

/// **THE CONTROL THE A/B DID NOT RUN, AND THE RULING IT FORCES** (P28.4 audit).
///
/// `the_predictor_strictly_reduces_a_whip_pans_fallback_frames` compares the
/// speculative lane against **no lane at all**, so what it measures is the lane
/// — a CPU-side want set at the *refinement's* cap — and not the dead reckoning
/// inside it. The control that separates the two is a horizon of **zero**, which
/// is reachable through the shipped API and needs no mutation: `dead_reckon` at
/// `h = 0` scales the secant by nothing and turns by nothing, so it returns the
/// newest committed pose exactly. Same lane, same cap, same rank, no lead.
///
/// Measured on this fixture, blur frames / blur tiles against OFF's 131 /
/// 19 872 and an arrival window of 384 / 1 728 over 15 frames:
///
/// | lead | blur frames | blur tiles | arrival blur |
/// |---|---|---|---|
/// | **0 ticks** | **105** | **18 752** | **64**, over 2 frames |
/// | 3 | 108 | 18 800 | 96 |
/// | 6 | 113 | 18 912 | 144 |
/// | 12 | 115 | 18 976 | 176 |
/// | **18 (shipped)** | 115 | 18 976 | 176, over 7 frames |
/// | 24 | 112 | 18 896 | 128 |
/// | 36 | 124 | 19 152 | 224 |
///
/// **The lead time is a cost on this fixture, not the win** — and the arrival
/// window, the only class a lead can serve, says it louder than the aggregate
/// does. That is not a defect in the predictor; it is
/// `a_saturated_floor_cannot_be_prefetched_and_the_arm_says_so` one class up.
/// `apply_wants` seats a miss the frame it is offered, out of the same pool,
/// with no admission throttle and no fetch latency — a page is sampleable the
/// frame it is admitted — so *having asked earlier* buys nothing anywhere in
/// this loop, and every want spent on where the camera will be is a slot not
/// spent on where it is.
///
/// **P28.5 acted on it** (`docs/memos/p28-5-lead-time-ruling.md`): the shipped
/// [`inf_render::DEFAULT_PREDICT_HORIZON_TICKS`] is now **0** — the lane at the
/// committed pose, which is the measured win — and the ROADMAP's prescribed 18
/// is kept as [`LEAD`], the arm this compares against. So the two runs below
/// have swapped names since P28.4 and the *inequality is unchanged*: no lead
/// beats a lead, on the aggregate and on the arrival window both.
///
/// Asserted rather than written down, in the shape the floor's refutation
/// already uses: **the day a lead time wins, this arm goes red and the ruling is
/// re-opened by a test instead of by memory.** What would make it win is a
/// throttle or a real latency between "admitted" and "sampleable"; neither
/// exists in this tree today (P28.3 §8 re-measured the loader and left it
/// alone).
#[test]
fn a_lead_time_costs_this_fixture_what_the_lane_earns_it() {
    let off = run(None, PAGES);
    // The shipped arm IS the zero-lead control since P28.5 — asserted rather
    // than assumed, so a default that drifts off zero is caught here and not
    // silently turned into a comparison of a lead with itself.
    assert_eq!(
        HORIZON, 0,
        "the shipped horizon left zero — this arm's control is no longer the \
         committed pose and the ruling has to be re-measured"
    );
    let zero = run(Some(HORIZON), PAGES);
    let shipped = run(Some(LEAD), PAGES);

    // **The whole lead sweep, printed** — the table in this arm's own header,
    // reproduced by the arm rather than remembered from an audit. The
    // assertions below are only about the two ends that matter; the rows in
    // between are what make the shape monotonic rather than a pair of points.
    let mut sweep: Vec<(String, Arm)> = vec![("OFF ".into(), off.clone())];
    for lead in [HORIZON, 3, 6, 12, LEAD, 24, 36] {
        let a = if lead == HORIZON {
            zero.clone()
        } else if lead == LEAD {
            shipped.clone()
        } else {
            run(Some(lead), PAGES)
        };
        sweep.push((format!("h={lead:<3}"), a));
    }
    for (name, a) in &sweep {
        println!(
            "{name} blur {}/{} over {} frames | ARRIVAL blur {}/{} over {} frames | \
             speculation {} offered, {} seated over {} ticks",
            a.blur,
            a.justified,
            a.blur_frames,
            a.arrival_blur,
            a.arrival_justified,
            a.arrival_frames,
            a.predict_wants,
            a.spec_admits,
            a.spec_admit_ticks
        );
    }

    // ANTI-VACUITY, in both directions: a horizon of zero really is the
    // committed pose, and the shipped horizon really is somewhere else. Without
    // the second, "the control ties the shipped arm" would also be what two
    // identical runs look like.
    //
    // Read **mid-sweep**, not at the last tick: the path ends still, and a still
    // camera turns by nothing at every horizon, so a check there is satisfied by
    // any `h` at all — measured, an `h + 1` mutation is invisible at tick 259
    // and dies here.
    //
    // `1e-6` and not equality, because `pcos64(0.0)` is a portable
    // approximation and is not bit-identically 1: the residual is f32-epsilon
    // scale, well under a tile boundary, and the two runs' want sets come out
    // the same as a committed-camera mutation's to the last page.
    const TURNING: u64 = WARM + RAMP + 30;
    let committed = whip_view(TURNING).forward.as_dvec3();
    let at_zero = zero.predicted[TURNING as usize]
        .expect("a full window predicts")
        .1;
    let at_shipped = shipped.predicted[TURNING as usize]
        .expect("a full window predicts")
        .1;
    assert!(
        (at_zero - committed).length() < 1e-6,
        "h = 0 did not return the committed direction ({at_zero:?} against \
         {committed:?}), so this is not the control it claims to be"
    );
    assert!(
        (at_shipped - committed).length() > 0.5,
        "the LEAD arm predicted the committed pose ({at_shipped:?}), so the two \
         arms are the same camera and the comparison is empty"
    );
    assert!(zero.predict_wants > 0 && off.arrival_justified > 0);

    // **The lane earns the win** — which is what the A/B arm next door
    // measures, stated here as the half of it that is about the lane.
    assert!(
        zero.blur_frames < off.blur_frames,
        "the speculative lane at the refinement's cap did not beat OFF even with \
         no lead to pay for: {} against {}",
        zero.blur_frames,
        off.blur_frames
    );

    // **…and the lead costs.** `>=` rather than `>` because the two are
    // allowed to tie; what is refused is the reading that the lead is what won.
    assert!(
        shipped.blur_frames >= zero.blur_frames,
        "the ROADMAP's {LEAD}-tick lead now BEATS the shipped zero ({} against \
         {}) — a throttle or a fetch latency has appeared between `apply_wants` \
         and a sampleable page, and P28.5's ruling on what the predictor buys \
         has to be rewritten",
        shipped.blur_frames,
        zero.blur_frames
    );
    assert!(
        shipped.arrival_blur >= zero.arrival_blur,
        "the lead now wins the ARRIVAL window ({} against {}) — the same \
         re-opening, on the class a lead time is actually for",
        shipped.arrival_blur,
        zero.arrival_blur
    );
}

/// The tier clamp, pinned where the predictor's knob is decided.
#[test]
fn the_predictor_is_clamped_off_on_low_and_on_mobile() {
    let high = inf_render::RenderTier::High.apply(inf_render::RenderSettings::default());
    assert!(
        high.stream.predict.enabled,
        "High clamped the predictor off"
    );
    let low = inf_render::RenderTier::Low.apply(inf_render::RenderSettings::default());
    assert!(!low.stream.predict.enabled);
    let mobile = inf_render::RenderTier::clamp_mobile(inf_render::RenderSettings::default());
    assert!(!mobile.stream.predict.enabled);
    // Only ever off: a caller that already disabled it on High keeps it off, and
    // `apply` stays idempotent.
    let mut asked_off = inf_render::RenderSettings::default();
    asked_off.stream.predict.enabled = false;
    assert!(
        !inf_render::RenderTier::High
            .apply(asked_off)
            .stream
            .predict
            .enabled
    );
    assert_eq!(
        inf_render::RenderTier::Low.apply(low),
        low,
        "the clamp is not idempotent"
    );
}

/// **THE TRUTH ORACLE** (P28.4 audit, inherited item 13) — how far the
/// prediction is from where the camera **actually went**.
///
/// The P28.4 audit's verdict on `the_prediction_replays_from_the_recorded_
/// history_alone` is that it is an oracle and that it measures **conformance**:
/// it is a second longhand implementation of the same specification, so its
/// `4.04e-16` is the distance between two spellings of one formula and says
/// nothing about the world. *"A truth oracle would compare the prediction
/// against `whip_view(tick + h)` and is not built."* This is that comparison,
/// and it needs no residency at all — a prediction is a pose, and this file
/// already has the closed form for where the camera is at any tick.
///
/// Three things it establishes, and each is load-bearing for a different claim:
///
/// * **`h = 0` is the committed pose, exactly.** The shipped horizon since
///   P28.5, so the zero-lead control's whole meaning is that at `h = 0` the
///   prediction *is* the camera. Measured against the truth rather than against
///   a rearrangement of the same algebra.
/// * **The error grows with the lead**, monotonically in the worst case. This
///   is the accuracy statement behind the lead-time ruling: the reckoner is not
///   broken, it is being asked a question whose answer is further away.
/// * **It is right about what it assumes and wrong about what it does not.**
///   Restricted to the constant-rate hold — the phase where "the rate holds" is
///   true — the error collapses; over the acceleration ramps it does not. A
///   dead-reckoner that scored the same in both would not be dead-reckoning.
#[test]
fn the_prediction_is_measured_against_where_the_camera_actually_went() {
    /// Angle between two unit directions, radians, in portable trig.
    fn angle(a: DVec3, b: DVec3) -> f64 {
        inf_math::pacos64(a.normalize().dot(b.normalize()).clamp(-1.0, 1.0))
    }
    // The constant-rate hold: after the acceleration ramp, before the brake —
    // the phase dead reckoning's own premise is true in.
    let holding = |t: u64| t > WARM + RAMP && t < WARM + SWEEP - RAMP;
    // The acceleration ramps, where the premise is false by construction.
    let ramping =
        |t: u64| (t > WARM && t <= WARM + RAMP) || (WARM + SWEEP - RAMP..WARM + SWEEP).contains(&t);

    let mut table: Vec<(u32, f64, f64, f64, f64, u64)> = Vec::new();
    for h in [HORIZON, 3, 6, 12, LEAD, 24, 36] {
        let mut hist = inf_math::CameraHistory::new();
        let (mut worst, mut worst_hold, mut worst_ramp, mut worst_eye) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut turned = 0u64;
        for tick in 0..TICKS {
            let v = whip_view(tick);
            hist.commit(inf_math::CameraSample {
                tick,
                eye: v.eye_world,
                forward: v.forward.as_dvec3(),
                up: v.up.as_dvec3(),
            });
            let Some(p) = inf_math::dead_reckon(&hist, h) else {
                continue;
            };
            // **The world**, not a re-derivation of the predictor: the closed
            // form for where this camera is at tick + h.
            let truth = whip_view(tick + u64::from(h));
            let err = angle(p.forward, truth.forward.as_dvec3());
            worst = worst.max(err);
            worst_eye = worst_eye.max((p.eye - truth.eye_world).length());
            if holding(tick) && holding(tick + u64::from(h)) {
                worst_hold = worst_hold.max(err);
            }
            if ramping(tick) {
                worst_ramp = worst_ramp.max(err);
            }
            turned += u64::from(angle(v.forward.as_dvec3(), truth.forward.as_dvec3()) > 1e-3);
        }
        table.push((h, worst, worst_hold, worst_ramp, worst_eye, turned));
    }

    println!("prediction accuracy against whip_view(tick + h) — worst, radians:");
    for (h, worst, hold, ramp, eye, turned) in &table {
        // **Scientific, not six decimals** (P28.5 audit): the h = 0 row is the
        // one this table exists to establish, its value is ~1e-8, and `{:.6}`
        // printed it as `0.000000` — so the ledger's headline number could not
        // be read off the arm that measured it. An arm prints what its own
        // header tabulates (the P28.4 audit's ruling, met again).
        println!(
            "  h={h:<3} worst {worst:.3e} rad ({:.4}°) | hold {hold:.3e} | ramp {ramp:.3e} | \
             eye {eye:.3e} m | {turned} ticks where the camera actually moved",
            worst.to_degrees()
        );
    }

    // **`h = 0` is the committed pose.** `1e-6` and not equality, for the reason
    // `a_lead_time_costs_this_fixture_what_the_lane_earns_it` gives at the same
    // tolerance: `psin64(0)`/`pcos64(0)` are portable approximations of 0 and 1
    // and `dead_reckon` runs its Rodrigues rotation regardless, so the residual
    // is arithmetic and not a lead. Measured **2.1 × 10⁻⁸ rad** — 1.2 × 10⁻⁶
    // degrees, some eleven orders under a tile boundary. The eye is **exact**:
    // this path's drift is linear and a secant extrapolates linear motion with
    // no error at all, which is why every row below is an *angular* number.
    let zero = table[0];
    assert_eq!(zero.0, 0, "the shipped horizon left zero");
    assert!(
        zero.1 < 1e-6 && zero.4 < 1e-6,
        "h = 0 is not the committed pose ({:.3e} rad, {:.3e} m) — the zero-lead \
         control in `a_lead_time_costs_this_fixture_what_the_lane_earns_it` is \
         not the control it claims to be",
        zero.1,
        zero.4
    );
    // …and `turned` says the same thing from the other side: at zero lead the
    // truth IS this tick, so there is no tick at which the camera moved over
    // the horizon. It is the one row for which that count must be zero.
    assert_eq!(
        zero.5, 0,
        "h = 0 saw the camera move over its horizon, which is a horizon"
    );

    // ANTI-VACUITY: under every non-zero lead the path has to have moved, or
    // "the error grows" is a statement about a still camera.
    for (h, .., turned) in table.iter().skip(1) {
        assert!(
            *turned > SWEEP / 2,
            "h={h}: the camera barely moved over the horizon ({turned} ticks), \
             so nothing below measures a prediction"
        );
    }

    // **The error grows with the lead**, worst case, every step of the ladder.
    for w in table.windows(2) {
        assert!(
            w[1].1 > w[0].1,
            "the worst error did not grow from h={} to h={}: {:.6} against \
             {:.6} rad",
            w[0].0,
            w[1].0,
            w[0].1,
            w[1].1
        );
    }

    // **Right about what it assumes, wrong about what it does not.** At the
    // ROADMAP's own lead, the constant-rate hold is where the premise holds and
    // the ramps are where it does not, and the two must not be the same number.
    let lead = table
        .iter()
        .find(|r| r.0 == LEAD)
        .expect("the prescribed lead is in the table");
    assert!(
        lead.3 > lead.2 * 4.0,
        "the acceleration ramps cost no more accuracy than the constant-rate \
         hold ({:.6} against {:.6} rad) — this fixture is not exercising the \
         regime dead reckoning is provably wrong about",
        lead.3,
        lead.2
    );
}
