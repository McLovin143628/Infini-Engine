//! **inf-water** (Ring 0) — the pure math behind every water surface the engine
//! draws or simulates: Gerstner [`wave`]s, spline [`river`] frames, and [`shore`]
//! queries against terrain heights.
//!
//! # What this crate is for
//!
//! One evaluator, three bodies. An **ocean** is an unbounded plane carrying a
//! wind-driven Gerstner sum; a **lake** is a bounded rectangle carrying a small
//! ripple; a **river** is a spline ribbon carrying a ripple that travels
//! *downstream* rather than with the wind. They differ in their footprint and in
//! the frame the waves are evaluated in — not in the wave model — which is why
//! there is one [`WaveField`] here and one water shader in `inf-render`.
//!
//! # The sim never reads render state
//!
//! [`WaterSurface::height_at`] is the API P20.2's buoyancy will call **from the
//! fixed step**, and it is designed for that now rather than retro-fitted later.
//! It is pure `f64`, allocation-free, has no notion of a camera, a frame or a
//! GPU, and takes its time as an explicit `t` in seconds — the level's clock
//! (`ResolvedSky::cloud_time_s`), never a wall clock and never a frame counter.
//! That is the P16.3 sim/render split applied before there is a simulation to
//! split: a buoyancy force computed from anything the renderer owns would make
//! the physics trace depend on the frame rate, and no replay gate would survive
//! it.
//!
//! Everything here is bit-portable. Trigonometry goes through
//! [`inf_math::portable`]'s `psin64`/`pcos64` (the P14 LAW: `f64::sin` is not
//! bit-identical across platforms), integer hashing replaces every RNG, and
//! iteration counts are fixed rather than convergence-tested — so two runs, two
//! machines and two processes agree to the bit.
//!
//! # Units
//!
//! SI throughout, per the units doctrine: metres, seconds, m/s, radians. A water
//! *level* is an absolute world elevation in metres, not an offset from anything.

pub mod river;
pub mod shore;
pub mod wave;

use glam::{DVec2, DVec3};

pub use river::{RiverFrame, RiverPath, RiverProfile, RiverSample, UphillSpan};
pub use shore::{shore_blend, shore_class, shore_distance, ShoreClass};
pub use wave::{Wave, WaveField, WaveSpec, GRAVITY_M_S2, MAX_WAVES};

/// A water body's geometry and its wave field — everything a height query needs.
///
/// Built by each host from the ECS `WaterBody` component (both projectors build
/// it the same way, which is what the MIRROR gate pins) and consumed by the
/// renderer, the cook and — from P20.2 — the fixed step.
#[derive(Clone, Debug, PartialEq)]
pub enum WaterSurface {
    /// An **unbounded** plane at `level_m`, displaced by a wind-driven Gerstner
    /// sum evaluated in world XZ.
    ///
    /// Unbounded is the honest model: an ocean does not have an authored edge,
    /// and giving it one would mean a level's sea stops at a rectangle a player
    /// can walk to. What is bounded is how much of it the *renderer* tessellates,
    /// which is a camera-following concern and lives in the pass.
    Ocean {
        /// Still-water elevation, metres.
        level_m: f64,
        /// The sea state.
        waves: WaveField,
    },
    /// A **bounded** rectangle in world XZ at `level_m`, carrying a ripple.
    ///
    /// A lake is authored as a region because it *is* one: it has banks, and the
    /// banks are where its surface stops. The waves are evaluated in world XZ
    /// exactly as an ocean's are, so two adjacent lakes at the same level do not
    /// show a seam.
    Lake {
        /// Still-water elevation, metres.
        level_m: f64,
        /// Centre of the region in world XZ, metres.
        center: DVec2,
        /// Half-extent of the region in world XZ, metres.
        half_extent: DVec2,
        /// The ripple.
        waves: WaveField,
    },
    /// A spline **ribbon** with a width/depth profile and a flow speed.
    ///
    /// Its waves are evaluated in the river's own `(s, lateral)` frame — arc
    /// length along the flow, offset across it — so the ripple travels
    /// **downstream** whichever way the river bends. Evaluating it in world XZ
    /// (the obvious shortcut) would make a river that turns a corner show its
    /// ripple crossing it sideways.
    River {
        /// The built centreline.
        path: RiverPath,
        /// The ripple, in river-local coordinates.
        waves: WaveField,
    },
}

impl WaterSurface {
    /// The still-water elevation used for footprint tests and for the renderer's
    /// base plane, metres. For a river this is the **start** of the centreline —
    /// a river's surface follows its spline, so a single number is only a
    /// representative one, and callers that need the real elevation ask
    /// [`height_at`](Self::height_at).
    pub fn reference_level_m(&self) -> f64 {
        match self {
            WaterSurface::Ocean { level_m, .. } | WaterSurface::Lake { level_m, .. } => *level_m,
            WaterSurface::River { path, .. } => {
                path.frames.first().map(|f| f.center.y).unwrap_or(0.0)
            }
        }
    }

    /// The wave field this body carries.
    pub fn waves(&self) -> &WaveField {
        match self {
            WaterSurface::Ocean { waves, .. }
            | WaterSurface::Lake { waves, .. }
            | WaterSurface::River { waves, .. } => waves,
        }
    }

    /// Whether the world XZ point `p` lies over this body's surface.
    ///
    /// The footprint is the **still-water** one: an ocean is everywhere, a lake is
    /// its rectangle, a river is between its banks. Displacement is deliberately
    /// not taken into account — a wave crest that overhangs the bank by 30 cm is
    /// not a reason for a query at the water's edge to flicker in and out between
    /// fixed steps.
    pub fn contains(&self, p: DVec2) -> bool {
        match self {
            WaterSurface::Ocean { .. } => true,
            WaterSurface::Lake {
                center,
                half_extent,
                ..
            } => {
                (p.x - center.x).abs() <= half_extent.x.max(0.0)
                    && (p.y - center.y).abs() <= half_extent.y.max(0.0)
            }
            WaterSurface::River { path, .. } => path.sample(p).is_some_and(|s| s.inside()),
        }
    }

    /// **The height query.** Absolute world elevation of the water surface over
    /// `p` at time `t` (seconds), metres — or `None` when `p` is outside this
    /// body's footprint.
    ///
    /// This is the function P20.2's buoyancy samples in the fixed step. Pure,
    /// allocation-free (a river's `O(frames)` scan aside), deterministic and
    /// bit-portable; see the crate docs for why that matters and
    /// [`WaveField::height_at`] for how the Gerstner inverse is solved.
    pub fn height_at(&self, p: DVec2, t: f64) -> Option<f64> {
        match self {
            WaterSurface::Ocean { level_m, waves } => Some(level_m + waves.height_at(p, t)),
            WaterSurface::Lake { level_m, waves, .. } => {
                if !self.contains(p) {
                    return None;
                }
                Some(level_m + waves.height_at(p, t))
            }
            WaterSurface::River { path, waves } => {
                let s = path.sample(p)?;
                if !s.inside() {
                    return None;
                }
                Some(s.surface_y + waves.height_at(river_uv(&s), t))
            }
        }
    }

    /// The unit surface normal over `p` at time `t`, or `None` outside the
    /// footprint.
    ///
    /// A river's normal is returned in **world** space: the ripple normal is
    /// computed in the river's `(s, lateral)` frame and then rotated onto the
    /// frame's tangent/across vectors, so a river running north tilts its wavelets
    /// north rather than east.
    pub fn normal_at(&self, p: DVec2, t: f64) -> Option<DVec3> {
        match self {
            WaterSurface::Ocean { waves, .. } => Some(waves.normal(p, t)),
            WaterSurface::Lake { waves, .. } => {
                if !self.contains(p) {
                    return None;
                }
                Some(waves.normal(p, t))
            }
            WaterSurface::River { path, waves } => {
                let s = path.sample(p)?;
                if !s.inside() {
                    return None;
                }
                let local = waves.normal(river_uv(&s), t);
                // Rotate the local (along, up, across) normal onto the frame.
                let along = DVec3::new(s.tangent.x, 0.0, s.tangent.z);
                let along = if along.length() > 0.0 {
                    along.normalize()
                } else {
                    DVec3::X
                };
                let across = DVec3::new(-along.z, 0.0, along.x);
                let n = along * local.x + DVec3::Y * local.y + across * local.z;
                Some(if n.length() > 0.0 {
                    n.normalize()
                } else {
                    DVec3::Y
                })
            }
        }
    }

    /// How far the world point `world` is **below** this body's surface, metres —
    /// positive when submerged, `None` outside the footprint.
    ///
    /// The buoyancy primitive: displaced volume is an integral of this over a
    /// body's cross-section, and P20.2 needs exactly one number per sample point.
    pub fn submersion_m(&self, world: DVec3, t: f64) -> Option<f64> {
        let h = self.height_at(DVec2::new(world.x, world.z), t)?;
        Some(h - world.y)
    }

    /// Surface flow velocity at `p`, m/s in world XZ — the river's tangent times
    /// its speed, and **zero** for still bodies.
    ///
    /// Zero rather than "the wave orbital velocity" for oceans and lakes in v1:
    /// a Gerstner particle's orbit averages to no net transport, and reporting the
    /// instantaneous orbital velocity as a *current* would push a floating body
    /// across the sea at wave speed. Stokes drift is real and is a P20.2 question,
    /// stated here so the zero is a decision rather than an omission.
    pub fn flow_at(&self, p: DVec2) -> Option<DVec2> {
        match self {
            WaterSurface::Ocean { .. } => Some(DVec2::ZERO),
            WaterSurface::Lake { .. } => {
                if self.contains(p) {
                    Some(DVec2::ZERO)
                } else {
                    None
                }
            }
            WaterSurface::River { path, .. } => path.flow_at(p),
        }
    }

    /// The **crest factor** over `p`, `[0, 1]` — the wave-folding measure the
    /// renderer turns into foam. `None` outside the footprint.
    pub fn crest_at(&self, p: DVec2, t: f64) -> Option<f64> {
        match self {
            WaterSurface::Ocean { waves, .. } => Some(waves.crest(p, t)),
            WaterSurface::Lake { waves, .. } => {
                if !self.contains(p) {
                    return None;
                }
                Some(waves.crest(p, t))
            }
            WaterSurface::River { path, waves } => {
                let s = path.sample(p)?;
                if !s.inside() {
                    return None;
                }
                Some(waves.crest(river_uv(&s), t))
            }
        }
    }
}

/// The river-local wave parameter: arc length along the flow, offset across it.
#[inline]
fn river_uv(s: &RiverSample) -> DVec2 {
    DVec2::new(s.s, s.lateral_m)
}

/// The **highest** water surface over `p` among `bodies`, and which body it came
/// from — or `None` when `p` is over none of them.
///
/// The rule for overlapping bodies, defined once so a renderer, a cook and a
/// physics step cannot each invent their own: the *topmost* surface wins, and a
/// tie goes to the earlier body in the list. That makes the answer a function of
/// the water itself rather than of a traversal order, which is the same property
/// `inf_ecs::sky::sky_authority` exists to give the sky.
pub fn highest_surface(bodies: &[WaterSurface], p: DVec2, t: f64) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, b) in bodies.iter().enumerate() {
        if let Some(h) = b.height_at(p, t) {
            if best.is_none_or(|(_, bh)| h > bh) {
                best = Some((i, h));
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_math::spline::SplineInterp;

    fn ocean() -> WaterSurface {
        WaterSurface::Ocean {
            level_m: 4.0,
            waves: WaveField::from_spec(&WaveSpec::default()),
        }
    }

    fn lake() -> WaterSurface {
        WaterSurface::Lake {
            level_m: 12.0,
            center: DVec2::new(100.0, -50.0),
            half_extent: DVec2::new(30.0, 20.0),
            waves: WaveField::from_spec(&WaveSpec::ripple(0.04, 6.0, 11)),
        }
    }

    fn river() -> WaterSurface {
        let points = vec![
            DVec3::new(0.0, 20.0, 0.0),
            DVec3::new(60.0, 16.0, 10.0),
            DVec3::new(120.0, 12.0, -10.0),
            DVec3::new(180.0, 8.0, 0.0),
        ];
        WaterSurface::River {
            path: RiverPath::from_points(
                &points,
                false,
                SplineInterp::CatmullRom,
                &RiverProfile {
                    width_start_m: 8.0,
                    width_end_m: 14.0,
                    depth_start_m: 1.0,
                    depth_end_m: 2.5,
                    flow_speed_m_s: 1.8,
                },
            ),
            waves: WaveField::from_spec(&WaveSpec::ripple(0.06, 4.0, 5)),
        }
    }

    #[test]
    fn an_ocean_is_everywhere_and_a_lake_is_not() {
        let o = ocean();
        let l = lake();
        for p in [
            DVec2::ZERO,
            DVec2::new(1e6, -1e6),
            DVec2::new(-500.0, 900.0),
        ] {
            assert!(o.contains(p));
            assert!(o.height_at(p, 3.0).is_some());
        }
        assert!(l.contains(DVec2::new(100.0, -50.0)));
        assert!(l.contains(DVec2::new(130.0, -30.0)), "the corner is inside");
        assert!(!l.contains(DVec2::new(131.0, -50.0)));
        assert!(l.height_at(DVec2::new(131.0, -50.0), 0.0).is_none());
    }

    #[test]
    fn the_height_query_stays_near_the_level() {
        let o = ocean();
        let bound = o.waves().max_amplitude_m();
        for i in 0..300 {
            let p = DVec2::new(i as f64 * 1.7 - 200.0, i as f64 * -0.9);
            let h = o.height_at(p, i as f64 * 0.05).unwrap();
            assert!((h - 4.0).abs() <= bound + 1e-9, "h = {h}");
        }
    }

    #[test]
    fn submersion_is_the_buoyancy_primitive() {
        let l = lake();
        let p = DVec2::new(100.0, -50.0);
        let surface = l.height_at(p, 7.0).unwrap();
        let below = DVec3::new(p.x, surface - 2.0, p.y);
        let above = DVec3::new(p.x, surface + 2.0, p.y);
        assert!((l.submersion_m(below, 7.0).unwrap() - 2.0).abs() < 1e-9);
        assert!((l.submersion_m(above, 7.0).unwrap() + 2.0).abs() < 1e-9);
        assert!(l.submersion_m(DVec3::new(1000.0, 0.0, 0.0), 7.0).is_none());
    }

    #[test]
    fn a_river_surface_follows_its_spline_and_flows_downstream() {
        let r = river();
        let start = r.height_at(DVec2::new(0.0, 0.0), 0.0).unwrap();
        let end = r.height_at(DVec2::new(180.0, 0.0), 0.0).unwrap();
        assert!(start > end, "the river must run downhill: {start} → {end}");
        // The flow points along the river, not along the world axes.
        let mid = r.flow_at(DVec2::new(60.0, 10.0)).unwrap();
        assert!((mid.length() - 1.8).abs() < 1e-6, "speed {}", mid.length());
        assert!(mid.x > 0.0, "the river flows toward +X here");
        // Outside the banks there is no river.
        assert!(r.height_at(DVec2::new(60.0, 60.0), 0.0).is_none());
        assert!(r.flow_at(DVec2::new(60.0, 60.0)).is_none());
    }

    #[test]
    fn normals_are_unit_and_upward_for_every_body() {
        for (name, body, p) in [
            ("ocean", ocean(), DVec2::new(13.0, -7.0)),
            ("lake", lake(), DVec2::new(105.0, -45.0)),
            ("river", river(), DVec2::new(60.0, 10.0)),
        ] {
            for i in 0..40 {
                let n = body.normal_at(p, i as f64 * 0.25).unwrap();
                assert!((n.length() - 1.0).abs() < 1e-9, "{name}");
                assert!(n.y > 0.0, "{name}");
            }
        }
        assert!(lake().normal_at(DVec2::new(999.0, 0.0), 0.0).is_none());
    }

    #[test]
    fn still_bodies_report_no_current() {
        assert_eq!(ocean().flow_at(DVec2::new(5.0, 5.0)), Some(DVec2::ZERO));
        assert_eq!(lake().flow_at(DVec2::new(100.0, -50.0)), Some(DVec2::ZERO));
        assert!(lake().flow_at(DVec2::new(0.0, 0.0)).is_none());
    }

    #[test]
    fn queries_are_bit_reproducible() {
        let bodies = [ocean(), lake(), river()];
        let sample = || {
            bodies
                .iter()
                .flat_map(|b| {
                    (0..40).map(move |i| {
                        let p = DVec2::new(i as f64 * 4.5, i as f64 * -1.25 - 20.0);
                        let t = i as f64 * 0.37;
                        (
                            b.height_at(p, t).map(f64::to_bits),
                            b.crest_at(p, t).map(f64::to_bits),
                            b.normal_at(p, t).map(|n| n.y.to_bits()),
                        )
                    })
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(sample(), sample());
    }

    #[test]
    fn the_topmost_surface_wins_and_ties_go_to_the_first() {
        let low = WaterSurface::Lake {
            level_m: 1.0,
            center: DVec2::ZERO,
            half_extent: DVec2::splat(10.0),
            waves: WaveField::default(),
        };
        let high = WaterSurface::Lake {
            level_m: 5.0,
            center: DVec2::ZERO,
            half_extent: DVec2::splat(10.0),
            waves: WaveField::default(),
        };
        let tie = WaterSurface::Lake {
            level_m: 5.0,
            center: DVec2::ZERO,
            half_extent: DVec2::splat(10.0),
            waves: WaveField::default(),
        };
        assert_eq!(
            highest_surface(&[low.clone(), high.clone()], DVec2::ZERO, 0.0),
            Some((1, 5.0))
        );
        // Order-independent: the same body wins whichever way they are listed.
        assert_eq!(
            highest_surface(&[high.clone(), low.clone()], DVec2::ZERO, 0.0),
            Some((0, 5.0))
        );
        // A tie goes to the earlier entry.
        assert_eq!(
            highest_surface(&[high, tie], DVec2::ZERO, 0.0),
            Some((0, 5.0))
        );
        // Outside everything.
        assert!(highest_surface(&[low], DVec2::new(100.0, 0.0), 0.0).is_none());
        assert!(highest_surface(&[], DVec2::ZERO, 0.0).is_none());
    }

    #[test]
    fn a_reference_level_exists_for_every_body() {
        assert_eq!(ocean().reference_level_m(), 4.0);
        assert_eq!(lake().reference_level_m(), 12.0);
        assert!((river().reference_level_m() - 20.0).abs() < 1e-9);
    }
}
