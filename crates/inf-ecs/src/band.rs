//! **The simulation's collider band** (IB-2a): which static structural colliders
//! a fixed step is allowed to hold.
//!
//! # The measurement this exists for
//!
//! The AAA-readiness certification measured a P19 town at **12 850 static
//! colliders → 4.663 ms per fixed step, 0.363 µs each**, and the town is *seven*
//! buildings. The engine's streamed step budget is
//! `inf_player::budget::STREAMED_STEP_BUDGET_MS` = **4.0 ms**, so the whole
//! simulation affords on the order of **11 000** static colliders — about seven
//! grammar buildings, against the two cities and five towns the island needs.
//! Nothing about that arithmetic improves by making the solver faster; the fix is
//! to stop handing it buildings nobody can touch.
//!
//! # Why this is legal, and what makes it so
//!
//! The standing law is *visibility filters what is DRAWN, never what is
//! SIMULATED*. A band that read the camera would break it outright: two players
//! looking in different directions would simulate different worlds, and PIE would
//! stop equalling shipping the moment the editor's free camera moved.
//!
//! So the band's anchors are [`streaming_sources`] — entities carrying
//! [`StreamingSource`], which is exactly the
//! set P16's cell activation already reads to decide which parts of the world
//! exist at all. The active collider set is therefore a pure function of the
//! world's own contents: same entities, same positions, same colliders, in every
//! host, on every machine, on every replay.
//!
//! # Why a radius and not a partition CELL
//!
//! The obvious reading of "band by the partition" is to admit the colliders of
//! active cells. The measurement refuses it: `DEFAULT_CELL_SIZE_M` is **256 m**
//! and `DEFAULT_ACTIVATION_RADIUS_M` is 256 m, so the active set is at least a
//! 3 × 3 block of 256 m cells — 590 000 m² — and a downtown block holds roughly
//! one building per 700 m² of ground. That is ~840 buildings of full interiors:
//! two orders of magnitude past the budget, i.e. the same failure with more
//! bookkeeping. The cells decide what *exists*; the band decides what is
//! *solid*, and they are different questions at different scales.
//!
//! # Why the anchors are QUANTIZED
//!
//! The band's membership is a change stamp — the P19.5 mechanism — and a stamp
//! that moved every step would re-describe the whole active set sixty times a
//! second, which is precisely the 11.62 ms regression P19.5's memo records. So
//! anchors are snapped to a [`BAND_LATTICE_M`] lattice and the band is computed
//! about the snapped point, which means membership changes only when a source
//! crosses a lattice line — roughly once a second at walking pace.
//!
//! The cost is stated rather than hidden: a source may be up to
//! `BAND_LATTICE_M · √2 / 2` ≈ **11.3 m** from the point the band is measured
//! about, so the effective near radius varies by that much. It is a *radius*
//! budget, not a correctness one, and the near radius is chosen with it in hand.
//!
//! The second cost is the lattice's own **edge**, and it is a bound rather than
//! a saving: "once a second at walking pace" is true of a source *travelling*
//! and false of one parked on a lattice line, whose sub-millimetre jitter
//! crosses it every step. Hysteresis would fix it and is refused — a band that
//! depended on the history of positions rather than on the positions would stop
//! being a pure function of sim state, which is the whole reason PIE equals
//! shipping here. So the thrash is measured instead, and what is asserted is
//! that it alternates between exactly **two** bands (the two cells the source is
//! between): never wrong, never wandering, re-described until the source moves
//! off the line. See `a_source_parked_on_a_lattice_line_rebands_every_step`.
//!
//! # It fails OPEN
//!
//! A world with no streaming source at all — every committed level before the
//! island, every unit fixture — is [`SimBand::unbounded`]: everything is solid,
//! exactly as before I3. Same for a world whose only sources carry non-finite
//! positions. Dropping colliders is the dangerous direction (a body falls through
//! the world and keeps falling); keeping them is merely slow.

use glam::{DQuat, DVec3};

use crate::components::{GlobalTransform, Guid, StreamingSource, Transform};
use crate::world::EcsWorld;
use inf_math::Tier;

/// Metres of full-detail collider band around each streaming source.
///
/// **Measured, not chosen.** `inf-physics`'s
/// `the_radius_sweep_states_what_the_default_buys` builds a thousand-building
/// city (427 351 solids) and prices every candidate against the 11 019 colliders
/// a 4.0 ms step buys at the certification's own 0.363 µs each:
///
/// | near_m | colliders | cert ms |
/// |---|---|---|
/// | 16 | 1 000 | 0.363 |
/// | 32 | 1 563 | 0.567 |
/// | 48 | 4 241 | 1.539 |
/// | **64** | **7 513** | **2.727** |
/// | 96 | 14 666 | 5.324 |
/// | 128 | 21 544 | 7.820 |
///
/// 64 m is the widest radius on that table inside the budget, and the arm fails
/// if the shipped default stops being one. It is also comfortably past what
/// gameplay needs — the band is recomputed every fixed step, so the radius only
/// has to exceed a step's travel plus the [`BAND_LATTICE_M`] slop.
pub const DEFAULT_COLLIDER_NEAR_M: f64 = 64.0;

/// Metres beyond which a grouped structure loses even its shell collider.
///
/// Generous, because a shell is one collider: a thousand shells is a thousand
/// colliders, well inside a budget that affords eleven thousand, and a body that
/// leaves the band is a body that would otherwise fall through a city block.
pub const DEFAULT_COLLIDER_FAR_M: f64 = 1_024.0;

/// The lattice anchors snap to, in metres.
///
/// Sixteen because it is a power of two (so the snap is exact in binary at every
/// scale the island reaches) and because the resulting ±11.3 m of radius slop is
/// small beside [`DEFAULT_COLLIDER_NEAR_M`] while still making a re-band a
/// once-a-second event at walking pace rather than a per-step one.
pub const BAND_LATTICE_M: f64 = 16.0;

/// Which static structural colliders this step may hold.
#[derive(Debug, Clone, PartialEq)]
pub struct SimBand {
    /// Lattice-snapped anchor positions, ascending — the band is measured about
    /// these, never about the raw source positions, so membership and
    /// [`stamp`](Self::stamp) can never disagree.
    anchors: Vec<DVec3>,
    near_m: f64,
    far_m: f64,
    unbounded: bool,
    stamp: u64,
}

impl Default for SimBand {
    fn default() -> Self {
        Self::unbounded()
    }
}

impl SimBand {
    /// Everything is solid — the pre-I3 behaviour, and the answer for a world
    /// with no streaming source.
    pub fn unbounded() -> Self {
        Self {
            anchors: Vec::new(),
            near_m: f64::INFINITY,
            far_m: f64::INFINITY,
            unbounded: true,
            stamp: 0,
        }
    }

    /// The band a world's own streaming sources define.
    pub fn from_world(world: &EcsWorld, near_m: f64, far_m: f64) -> Self {
        Self::from_anchors(
            streaming_sources(world).into_iter().map(|(p, _)| p),
            near_m,
            far_m,
        )
    }

    /// The band a set of anchor positions defines.
    ///
    /// Non-finite anchors are dropped; if that leaves none, the band is
    /// [`unbounded`](Self::unbounded) — failing open, per the module docs.
    /// Anchors are snapped, deduplicated and sorted, so the band is a pure
    /// function of the *set* of source positions and not of the order a world
    /// walk produced them in.
    pub fn from_anchors(anchors: impl IntoIterator<Item = DVec3>, near_m: f64, far_m: f64) -> Self {
        let mut snapped: Vec<[i64; 2]> = anchors
            .into_iter()
            .filter(|p| p.x.is_finite() && p.z.is_finite())
            .map(|p| [lattice(p.x), lattice(p.z)])
            .collect();
        if snapped.is_empty() || !near_m.is_finite() || !far_m.is_finite() {
            return Self::unbounded();
        }
        snapped.sort_unstable();
        snapped.dedup();

        // FNV-1a over the lattice coordinates and both radii. A stamp, not an
        // identity: the only legal operations on it are `==` against a
        // previously observed one.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        let mut fold = |v: u64| {
            for b in v.to_le_bytes() {
                h ^= u64::from(b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        };
        for c in &snapped {
            fold(c[0] as u64);
            fold(c[1] as u64);
        }
        fold(near_m.to_bits());
        fold(far_m.to_bits());

        Self {
            anchors: snapped
                .iter()
                .map(|c| DVec3::new(unlattice(c[0]), 0.0, unlattice(c[1])))
                .collect(),
            near_m,
            far_m,
            unbounded: false,
            stamp: h,
        }
    }

    /// `true` when nothing is banded — every collider is solid.
    #[inline]
    pub fn is_unbounded(&self) -> bool {
        self.unbounded
    }

    /// The band's membership stamp. Equal stamps mean equal membership, which is
    /// what lets the physics bridge retain an unchanged volume's colliders
    /// instead of re-describing them.
    ///
    /// `0` for an unbounded band, which is also the stamp a bridge that has never
    /// seen a band holds — so the first sync of a banded world always
    /// re-describes.
    #[inline]
    pub fn stamp(&self) -> u64 {
        self.stamp
    }

    /// The lattice-snapped anchors the band is measured about.
    #[inline]
    pub fn anchors(&self) -> &[DVec3] {
        &self.anchors
    }

    /// The near and far radii, in metres.
    #[inline]
    pub fn radii(&self) -> (f64, f64) {
        (self.near_m, self.far_m)
    }

    /// The tier an oriented box takes in this band.
    ///
    /// An unbounded band answers [`Tier::Near`] for everything — including for a
    /// box whose position is not finite, because refusing it here would silently
    /// change the pre-I3 behaviour of a fixture that has one. A *banded* world
    /// does refuse it: `distance_xz_to_box` returns NaN and
    /// [`inf_math::tier_at`] reads NaN as [`Tier::Out`].
    #[inline]
    pub fn tier(&self, center: DVec3, half: DVec3, rotation: DQuat) -> Tier {
        if self.unbounded {
            return Tier::Near;
        }
        inf_math::tier_at(
            inf_math::nearest_xz_to_box(&self.anchors, center, half, rotation),
            self.near_m,
            self.far_m,
        )
    }
}

#[inline]
fn lattice(v: f64) -> i64 {
    let q = (v / BAND_LATTICE_M).floor();
    // Clamp rather than wrap: an anchor at 10^18 m is a broken world, and a
    // wrapped index would put it next to the origin.
    if q <= -(i64::MAX as f64) {
        i64::MIN + 1
    } else if q >= i64::MAX as f64 {
        i64::MAX
    } else {
        q as i64
    }
}

#[inline]
fn unlattice(c: i64) -> f64 {
    (c as f64 + 0.5) * BAND_LATTICE_M
}

/// **Where the simulation says it is** — every entity carrying a
/// [`StreamingSource`], as
/// `(world position, radius)`, ordered by stable [`Guid`].
///
/// # The one authority
///
/// This is the world walk P16's cell activation runs and the collider band runs,
/// and it is here — in the ECS facade — rather than in either consumer, because
/// the two answering differently is the defect that would make a cell active and
/// its buildings intangible (or the reverse). It takes **no camera argument**, so
/// a caller cannot pass one by accident; that is the same guard
/// `CellStreaming::sync_sim` states about itself.
///
/// Sorted by `Guid` so the result is a function of the world's *contents*, not
/// of bevy's archetype iteration order.
pub fn streaming_sources(world: &EcsWorld) -> Vec<(DVec3, f64)> {
    let w = world.world();
    let mut out: Vec<(uuid::Uuid, DVec3, f64)> = w
        .iter_entities()
        .filter_map(|e| {
            let guid = e.get::<Guid>()?.0;
            let src = e.get::<StreamingSource>()?;
            let p = e
                .get::<GlobalTransform>()
                .map(|g| g.translation())
                .or_else(|| e.get::<Transform>().map(|t| t.translation.into()))
                .unwrap_or(DVec3::ZERO);
            Some((guid, p, src.radius_m))
        })
        .collect();
    out.sort_by_key(|(g, _, _)| *g);
    out.into_iter().map(|(_, p, r)| (p, r)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(x: f64, z: f64) -> (DVec3, DVec3, DQuat) {
        (
            DVec3::new(x, 0.0, z),
            DVec3::new(5.0, 8.0, 5.0),
            DQuat::IDENTITY,
        )
    }

    /// **It fails open.** No sources means no banding, which is what keeps every
    /// pre-I3 fixture and every committed level behaving exactly as before.
    #[test]
    fn a_world_with_no_streaming_source_bands_nothing() {
        let b = SimBand::from_anchors(Vec::<DVec3>::new(), 64.0, 1024.0);
        assert!(b.is_unbounded());
        assert_eq!(b.stamp(), 0);
        let (c, h, r) = boxed(1e9, -1e9);
        assert_eq!(b.tier(c, h, r), Tier::Near, "an unbounded band holds all");
        // …and so does a world whose only source is at a non-finite position:
        // dropping colliders is the direction that drops a body through a city.
        let nan = SimBand::from_anchors([DVec3::new(f64::NAN, 0.0, 0.0)], 64.0, 1024.0);
        assert!(nan.is_unbounded());
        // A non-finite radius is refused the same way.
        assert!(SimBand::from_anchors([DVec3::ZERO], f64::NAN, 1024.0).is_unbounded());
    }

    /// The three tiers over a real anchor, measured from the box's own face.
    #[test]
    fn the_band_tiers_a_box_by_its_nearest_anchor() {
        let b = SimBand::from_anchors([DVec3::ZERO], 64.0, 256.0);
        assert!(!b.is_unbounded());
        // Snapped to the cell centre: (8, 8) for an anchor at the origin.
        assert_eq!(b.anchors(), [DVec3::new(8.0, 0.0, 8.0)]);

        let near = boxed(60.0, 8.0); // face at x=55, 47 m from the anchor
        assert_eq!(b.tier(near.0, near.1, near.2), Tier::Near);
        let far = boxed(200.0, 8.0); // face at 195, 187 m away
        assert_eq!(b.tier(far.0, far.1, far.2), Tier::Far);
        let out = boxed(400.0, 8.0);
        assert_eq!(b.tier(out.0, out.1, out.2), Tier::Out);

        // A second anchor near the far box promotes it — the NEAREST wins.
        let two = SimBand::from_anchors([DVec3::ZERO, DVec3::new(200.0, 0.0, 8.0)], 64.0, 256.0);
        assert_eq!(two.tier(far.0, far.1, far.2), Tier::Near);
        assert_ne!(two.stamp(), b.stamp(), "a new anchor is a new membership");
    }

    /// **The stamp is what makes banding affordable**: it must be stable while a
    /// source moves inside a lattice cell, and must move when it leaves one.
    #[test]
    fn the_stamp_is_stable_within_a_lattice_cell_and_moves_across_one() {
        let at = |x: f64| SimBand::from_anchors([DVec3::new(x, 0.0, 0.0)], 64.0, 1024.0);
        let base = at(0.0);
        // Anywhere inside [0, 16) is the same band, membership included.
        for x in [0.0, 1.0, 7.999, 15.999] {
            assert_eq!(at(x).stamp(), base.stamp(), "x = {x} re-stamped");
            assert_eq!(at(x).anchors(), base.anchors());
        }
        assert_ne!(at(16.0).stamp(), base.stamp(), "crossing a line re-stamps");
        assert_ne!(at(-0.001).stamp(), base.stamp());

        // The radii are part of membership: the same anchors under different
        // radii are a different active set and must not be retained across.
        assert_ne!(
            SimBand::from_anchors([DVec3::ZERO], 64.0, 1024.0).stamp(),
            SimBand::from_anchors([DVec3::ZERO], 65.0, 1024.0).stamp()
        );
        assert_ne!(
            SimBand::from_anchors([DVec3::ZERO], 64.0, 1024.0).stamp(),
            SimBand::from_anchors([DVec3::ZERO], 64.0, 1025.0).stamp()
        );

        // …and it is a function of the SET, not of the order or of duplicates.
        let a = SimBand::from_anchors(
            [DVec3::new(0.0, 0.0, 0.0), DVec3::new(300.0, 0.0, 40.0)],
            64.0,
            1024.0,
        );
        let b = SimBand::from_anchors(
            [
                DVec3::new(300.0, 0.0, 40.0),
                DVec3::new(1.0, 5.0, 1.0),
                DVec3::new(0.0, 0.0, 0.0),
            ],
            64.0,
            1024.0,
        );
        assert_eq!(a, b, "order, height and duplicates must not move the band");
    }

    /// **The quantization cost, measured** — the module doc's ±11.3 m, stated as
    /// a number rather than as an adjective.
    #[test]
    fn the_lattice_slop_is_bounded_by_half_a_cell_diagonal() {
        let bound = BAND_LATTICE_M * std::f64::consts::SQRT_2 / 2.0;
        let mut worst: f64 = 0.0;
        // Sweep a source across two cells on both axes and measure how far the
        // band's own anchor is from where the source really is.
        for i in 0..64 {
            for j in 0..64 {
                let p = DVec3::new(
                    -20.0 + f64::from(i) * 0.75,
                    0.0,
                    -20.0 + f64::from(j) * 0.75,
                );
                let b = SimBand::from_anchors([p], 64.0, 1024.0);
                let a = b.anchors()[0];
                let d = ((a.x - p.x).powi(2) + (a.z - p.z).powi(2)).sqrt();
                worst = worst.max(d);
            }
        }
        assert!(
            worst <= bound + 1e-9,
            "{worst} m exceeds the {bound} m bound"
        );
        assert!(
            worst > bound * 0.5,
            "the sweep never approached the bound ({worst} m) — it is not \
             measuring the corner case it claims to"
        );
        println!("IB-2a lattice slop: worst {worst:.3} m against the {bound:.3} m bound");
    }

    /// **The lattice has an edge, and a source parked on it re-bands every
    /// step** (I3 audit) — measured and bounded rather than claimed away.
    ///
    /// The quantization buys "membership changes on a lattice crossing instead
    /// of every step", which is true of a source *travelling*: at walking pace a
    /// crossing is a once-a-second event. It is not true of a source sitting on
    /// a lattice line with sub-millimetre jitter — a standing character whose
    /// solver pose oscillates — which crosses on *every* step and re-describes
    /// the whole active set each time. That is the P19.5 regression's own
    /// mechanism, met from the other side.
    ///
    /// **It is not fixed with hysteresis, deliberately.** Hysteresis means the
    /// band depends on the *history* of positions rather than on the positions,
    /// and the band being a pure function of sim state is the entire reason
    /// PIE == shipping holds on a drive-through. A stateful band would agree
    /// with itself in each host and diverge between them the first time one of
    /// them started mid-trace.
    ///
    /// So the cost is stated: the thrash alternates between exactly **two**
    /// bands — the two cells the source is between — so the active set is never
    /// wrong and never wanders; it is re-described at the fixed-step rate for as
    /// long as the source parks there. The bound this arm holds is that
    /// two-ness.
    #[test]
    fn a_source_parked_on_a_lattice_line_rebands_every_step() {
        // A metre-scale jitter around a lattice line, which is what a solver
        // gives a body that is standing still.
        let line = BAND_LATTICE_M * 3.0;
        let mut stamps = Vec::new();
        for step in 0..60 {
            let x = line + if step % 2 == 0 { -0.001 } else { 0.001 };
            stamps.push(SimBand::from_anchors([DVec3::new(x, 0.0, 0.0)], 64.0, 1024.0).stamp());
        }
        let distinct: std::collections::BTreeSet<u64> = stamps.iter().copied().collect();
        let changes = stamps.windows(2).filter(|w| w[0] != w[1]).count();
        println!(
            "IB-2a lattice edge: a source jittering +/-1 mm across a line re-stamps \
             {changes} times in {} steps, over {} distinct bands",
            stamps.len(),
            distinct.len()
        );
        assert_eq!(
            distinct.len(),
            2,
            "a parked source produced {} bands — the thrash is a wander, not the \
             two cells it is between",
            distinct.len()
        );
        assert_eq!(
            changes,
            stamps.len() - 1,
            "the arm is not measuring the edge"
        );

        // THE CONTROL, and it is what says the quantization does its job at all:
        // the same jitter a metre away from the line re-stamps NEVER.
        let inside: Vec<u64> = (0..60)
            .map(|step| {
                let x = line + 1.0 + if step % 2 == 0 { -0.001 } else { 0.001 };
                SimBand::from_anchors([DVec3::new(x, 0.0, 0.0)], 64.0, 1024.0).stamp()
            })
            .collect();
        assert!(
            inside.windows(2).all(|w| w[0] == w[1]),
            "the same jitter away from the line re-stamped — the lattice is \
             buying nothing"
        );
    }
}
