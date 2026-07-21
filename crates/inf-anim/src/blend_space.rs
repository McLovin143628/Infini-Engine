//! Blend spaces (P11.2): 1D/2D parametric clip blends.
//!
//! A **blend space** maps a continuous parameter (speed, direction, aim pitch…)
//! onto a weighted blend of sample clips placed in that parameter space. Moving
//! the parameter smoothly cross-fades between the nearest samples, so one
//! authored graph of "idle / walk / run" clips yields a continuum of locomotion.
//!
//! Everything here is **pure and deterministic** (a function of its inputs
//! alone), matching [`crate::pose`] — no globals, no allocation-order surprises,
//! so it can move onto the parallel pose hot loop later without a rewrite.
//!
//! # Phase synchronization (v1)
//!
//! When two locomotion clips of *different* lengths blend (a 1 s walk vs. a
//! 0.6 s run), sampling both at the same absolute second would let their feet
//! drift out of step and slide. v1 therefore blends by **normalized phase**: a
//! single phase `φ ∈ [0,1)` is derived from the play-head `t` against the
//! weight-blended duration of the contributing clips, and each clip is sampled
//! at `φ · clip.duration`. The gait cycles stay locked regardless of clip
//! length. (A per-clip sync-marker scheme — aligning specific foot-plant events
//! rather than uniform phase — is the documented follow-up.)
//!
//! # 2D weighting (v1)
//!
//! True 2D blend spaces triangulate the sample points (Delaunay) and
//! barycentrically weight the enclosing triangle. v1 ships the simpler
//! **inverse-distance weighting of the `k = 3` nearest** samples: robust,
//! allocation-light, exact at a sample, and good enough for aim/locomotion
//! grids. The Delaunay upgrade is a seam left for later (swap
//! [`blend_weights_2d`] for a triangulator; the sampling path is unchanged).

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::clip::AnimClip;
use crate::pose::{blend_poses, sample_clip, Pose};
use crate::skeleton::Skeleton;

/// A raw 16-byte clip GUID reference (this crate stays `uuid`-free — the editor
/// records the dependency edge; see [`crate::asset::AnimClipAsset`]).
pub type ClipRef = [u8; 16];

/// Distances below this (in parameter units) count as "exactly at" a sample, so
/// a blend space returns that clip alone (exactness + no divide-by-zero).
const EXACT_EPS: f64 = 1e-9;

/// One sample of a [`BlendSpace1D`]: a clip pinned at a scalar parameter value.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BlendEntry1D {
    /// The parameter value at which this clip is at full weight.
    pub pos: f64,
    /// The clip played at this sample point.
    pub clip: ClipRef,
}

/// A 1D blend space: samples along a single named parameter axis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlendSpace1D {
    /// The Blueprint/actor variable name whose value selects the blend point.
    pub param: String,
    /// Sample clips. Need not be pre-sorted; sampling sorts by `pos`.
    pub entries: Vec<BlendEntry1D>,
}

/// One sample of a [`BlendSpace2D`]: a clip pinned at a 2D parameter point.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BlendEntry2D {
    /// The 2D parameter coordinate at which this clip is at full weight.
    pub pos: [f64; 2],
    /// The clip played at this sample point.
    pub clip: ClipRef,
}

/// A 2D blend space: samples over a plane of two named parameter axes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlendSpace2D {
    /// The two variable names (x, y) whose values select the blend point.
    pub params: (String, String),
    /// Sample clips scattered over the parameter plane.
    pub entries: Vec<BlendEntry2D>,
}

impl BlendSpace1D {
    /// A blend space over `param` with the given (pos, clip) samples.
    pub fn new(param: impl Into<String>, entries: Vec<BlendEntry1D>) -> Self {
        Self {
            param: param.into(),
            entries,
        }
    }
}

impl BlendSpace2D {
    /// A blend space over `(x, y)` params with the given samples.
    pub fn new(x: impl Into<String>, y: impl Into<String>, entries: Vec<BlendEntry2D>) -> Self {
        Self {
            params: (x.into(), y.into()),
            entries,
        }
    }
}

/// The weighted contribution of each 1D sample at `param`: `(entry_index,
/// weight)` pairs whose weights sum to 1 (empty if there are no entries).
///
/// At most two entries contribute (the bracketing pair); the ends **clamp** (a
/// parameter past the last sample yields that sample at full weight). An exact
/// hit on a sample returns it alone.
pub fn blend_weights_1d(space: &BlendSpace1D, param: f64) -> Vec<(usize, f64)> {
    let n = space.entries.len();
    if n == 0 {
        return Vec::new();
    }
    // Indices sorted by parameter position (stable so equal positions keep order).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        space.entries[a]
            .pos
            .partial_cmp(&space.entries[b].pos)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let first = order[0];
    let last = order[n - 1];
    if param <= space.entries[first].pos {
        return vec![(first, 1.0)];
    }
    if param >= space.entries[last].pos {
        return vec![(last, 1.0)];
    }
    // Find the bracketing pair in sorted order.
    for w in order.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let a = space.entries[lo].pos;
        let b = space.entries[hi].pos;
        if param >= a && param <= b {
            let span = b - a;
            let frac = if span > EXACT_EPS {
                (param - a) / span
            } else {
                0.0
            };
            let mut out = Vec::with_capacity(2);
            if 1.0 - frac > 0.0 {
                out.push((lo, 1.0 - frac));
            }
            if frac > 0.0 {
                out.push((hi, frac));
            }
            return out;
        }
    }
    // Unreachable given the clamp guards, but stay defensive.
    vec![(first, 1.0)]
}

/// The weighted contribution of each 2D sample at `params`: inverse-distance
/// weighting of the `k = 3` nearest samples, normalized to sum to 1 (empty if
/// there are no entries). An exact hit on a sample returns it alone.
pub fn blend_weights_2d(space: &BlendSpace2D, params: DVec2) -> Vec<(usize, f64)> {
    let n = space.entries.len();
    if n == 0 {
        return Vec::new();
    }
    // (index, squared distance), nearest first.
    let mut dists: Vec<(usize, f64)> = space
        .entries
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let d = DVec2::new(e.pos[0], e.pos[1]) - params;
            (i, d.length_squared())
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Exact (or near-exact) hit → that sample alone.
    if dists[0].1 <= EXACT_EPS * EXACT_EPS {
        return vec![(dists[0].0, 1.0)];
    }

    let k = dists.len().min(3);
    let raw: Vec<(usize, f64)> = dists[..k].iter().map(|&(i, d2)| (i, 1.0 / d2)).collect();
    let total: f64 = raw.iter().map(|&(_, w)| w).sum();
    if total <= 0.0 {
        return vec![(dists[0].0, 1.0)];
    }
    raw.into_iter().map(|(i, w)| (i, w / total)).collect()
}

/// Incrementally blend a set of `(pose, weight)` contributions into one pose.
///
/// Folds pairwise with a running normalized alpha so N weighted poses combine
/// correctly (lerp translation/scale, slerp rotation). Zero-weight entries are
/// dropped. `None` only when nothing contributes.
fn weighted_blend(items: Vec<(Pose, f64)>) -> Option<Pose> {
    let mut it = items.into_iter().filter(|(_, w)| *w > 0.0);
    let (mut acc, mut total) = it.next()?;
    for (pose, w) in it {
        total += w;
        let alpha = (w / total) as f32;
        acc = blend_poses(&acc, &pose, alpha);
    }
    Some(acc)
}

/// Resolve a weight list into sampled, phase-synced, blended poses.
///
/// `weights` are `(entry_index, weight)`; `clip_of(index)` yields the clip for
/// an entry (missing clips drop out and the remaining weights re-normalize via
/// the incremental blend). Returns the skeleton's rest pose if nothing resolves.
fn sample_weighted<'c>(
    weights: &[(usize, f64)],
    skeleton: &Skeleton,
    clip_of: impl Fn(usize) -> Option<&'c AnimClip>,
    t: f64,
) -> Pose {
    // Resolve clips + weights, dropping any that don't resolve.
    let resolved: Vec<(&AnimClip, f64)> = weights
        .iter()
        .filter_map(|&(i, w)| clip_of(i).map(|c| (c, w)))
        .collect();
    if resolved.is_empty() {
        return Pose::rest(skeleton);
    }
    // Weight-blended duration → a single normalized phase → per-clip sample time.
    let blended_dur: f64 = resolved.iter().map(|(c, w)| c.duration as f64 * w).sum();
    let phase = if blended_dur > EXACT_EPS {
        (t / blended_dur).rem_euclid(1.0)
    } else {
        0.0
    };
    let items: Vec<(Pose, f64)> = resolved
        .iter()
        .map(|(c, w)| {
            let ct = (phase * c.duration as f64) as f32;
            (sample_clip(skeleton, c, ct, true), *w)
        })
        .collect();
    weighted_blend(items).unwrap_or_else(|| Pose::rest(skeleton))
}

/// Sample a 1D blend space at parameter `param` and play-head `t` (seconds) into
/// a full [`Pose`]. Clips are resolved through `clips` (a raw-GUID → clip
/// lookup); an unresolved clip drops out of the blend.
pub fn sample_blend_space_1d<'c>(
    space: &BlendSpace1D,
    skeleton: &Skeleton,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    param: f64,
    t: f64,
) -> Pose {
    let weights = blend_weights_1d(space, param);
    sample_weighted(&weights, skeleton, |i| clips(space.entries[i].clip), t)
}

/// Sample a 2D blend space at `params` and play-head `t` (seconds) into a full
/// [`Pose`], via the `k = 3` inverse-distance blend.
pub fn sample_blend_space_2d<'c>(
    space: &BlendSpace2D,
    skeleton: &Skeleton,
    clips: &dyn Fn(ClipRef) -> Option<&'c AnimClip>,
    params: DVec2,
    t: f64,
) -> Pose {
    let weights = blend_weights_2d(space, params);
    sample_weighted(&weights, skeleton, |i| clips(space.entries[i].clip), t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clip::{Interpolation, JointTrack, QuatTrack, Vec3Track};
    use crate::skeleton::{Joint, JointTransform};
    use glam::{Mat4, Quat};

    fn one_joint_skeleton() -> Skeleton {
        Skeleton::new(vec![Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: Mat4::IDENTITY.to_cols_array(),
            local_bind: JointTransform::IDENTITY,
        }])
        .unwrap()
    }

    /// A clip that translates the single joint from `a` to `b` over `dur` s.
    fn slide_clip(a: f32, b: f32, dur: f32) -> AnimClip {
        let mut jt = JointTrack::new(0);
        jt.translation = Some(Vec3Track::new(
            vec![0.0, dur],
            vec![[a, 0.0, 0.0], [b, 0.0, 0.0]],
            Interpolation::Linear,
        ));
        AnimClip::new("slide", vec![jt])
    }

    /// A clip that rotates the single joint from 0° to `deg` over 1 s.
    fn spin_clip(deg: f32) -> AnimClip {
        let mut jt = JointTrack::new(0);
        jt.rotation = Some(QuatTrack::new(
            vec![0.0, 1.0],
            vec![
                Quat::IDENTITY.to_array(),
                Quat::from_rotation_z(deg.to_radians()).to_array(),
            ],
            Interpolation::Linear,
        ));
        AnimClip::new("spin", vec![jt])
    }

    fn id(n: u8) -> ClipRef {
        [n; 16]
    }

    #[test]
    fn weights_1d_are_exact_at_entries() {
        let sp = BlendSpace1D::new(
            "speed",
            vec![
                BlendEntry1D {
                    pos: 0.0,
                    clip: id(1),
                },
                BlendEntry1D {
                    pos: 1.0,
                    clip: id(2),
                },
                BlendEntry1D {
                    pos: 2.0,
                    clip: id(3),
                },
            ],
        );
        assert_eq!(blend_weights_1d(&sp, 0.0), vec![(0, 1.0)]);
        assert_eq!(blend_weights_1d(&sp, 1.0), vec![(1, 1.0)]);
        assert_eq!(blend_weights_1d(&sp, 2.0), vec![(2, 1.0)]);
    }

    #[test]
    fn weights_1d_midpoint_is_half_and_half() {
        let sp = BlendSpace1D::new(
            "speed",
            vec![
                BlendEntry1D {
                    pos: 0.0,
                    clip: id(1),
                },
                BlendEntry1D {
                    pos: 2.0,
                    clip: id(2),
                },
            ],
        );
        let w = blend_weights_1d(&sp, 1.0);
        assert_eq!(w.len(), 2);
        assert!((w[0].1 - 0.5).abs() < 1e-12 && (w[1].1 - 0.5).abs() < 1e-12);
        // A quarter of the way → 0.75 / 0.25.
        let w = blend_weights_1d(&sp, 0.5);
        assert!((w[0].1 - 0.75).abs() < 1e-12 && (w[1].1 - 0.25).abs() < 1e-12);
    }

    #[test]
    fn weights_1d_clamp_past_the_ends() {
        let sp = BlendSpace1D::new(
            "speed",
            vec![
                BlendEntry1D {
                    pos: 0.0,
                    clip: id(1),
                },
                BlendEntry1D {
                    pos: 1.0,
                    clip: id(2),
                },
            ],
        );
        assert_eq!(blend_weights_1d(&sp, -5.0), vec![(0, 1.0)]);
        assert_eq!(blend_weights_1d(&sp, 99.0), vec![(1, 1.0)]);
    }

    #[test]
    fn sample_1d_at_entry_matches_that_clip() {
        let sk = one_joint_skeleton();
        let walk = slide_clip(0.0, 1.0, 1.0);
        let run = slide_clip(0.0, 4.0, 1.0);
        let sp = BlendSpace1D::new(
            "speed",
            vec![
                BlendEntry1D {
                    pos: 0.0,
                    clip: id(1),
                },
                BlendEntry1D {
                    pos: 1.0,
                    clip: id(2),
                },
            ],
        );
        let clips = |r: ClipRef| -> Option<&AnimClip> {
            if r == id(1) {
                Some(&walk)
            } else if r == id(2) {
                Some(&run)
            } else {
                None
            }
        };
        // Exactly at the run entry (param 1.0), t = 0.5 → phase 0.5 → run at x=2.0.
        let pose = sample_blend_space_1d(&sp, &sk, &clips, 1.0, 0.5);
        assert!((pose.locals[0].translation[0] - 2.0).abs() < 1e-4);
        // Midpoint param → halfway between walk(0.5) and run(2.0) = 1.25.
        let mid = sample_blend_space_1d(&sp, &sk, &clips, 0.5, 0.5);
        assert!((mid.locals[0].translation[0] - 1.25).abs() < 1e-4);
    }

    #[test]
    fn sample_1d_phase_syncs_unequal_length_clips() {
        // A 1 s clip and a 2 s clip: at the midpoint blend, both must be sampled
        // at the SAME normalized phase (0.5), i.e. clip A at 0.5 s and clip B at
        // 1.0 s — each reaching its own half-way value.
        let sk = one_joint_skeleton();
        let a = slide_clip(0.0, 2.0, 1.0); // half-way value 1.0
        let b = slide_clip(0.0, 4.0, 2.0); // half-way value 2.0
        let sp = BlendSpace1D::new(
            "speed",
            vec![
                BlendEntry1D {
                    pos: 0.0,
                    clip: id(1),
                },
                BlendEntry1D {
                    pos: 1.0,
                    clip: id(2),
                },
            ],
        );
        let clips = |r: ClipRef| -> Option<&AnimClip> {
            if r == id(1) {
                Some(&a)
            } else if r == id(2) {
                Some(&b)
            } else {
                None
            }
        };
        // blended_dur = 0.5*1 + 0.5*2 = 1.5; at t=0.75 → phase 0.5.
        let pose = sample_blend_space_1d(&sp, &sk, &clips, 0.5, 0.75);
        // Expected blend of a@0.5s (=1.0) and b@1.0s (=2.0) at 0.5 → 1.5.
        assert!(
            (pose.locals[0].translation[0] - 1.5).abs() < 1e-4,
            "{:?}",
            pose.locals[0].translation
        );
    }

    #[test]
    fn weights_2d_are_exact_at_a_sample() {
        let sp = BlendSpace2D::new(
            "x",
            "y",
            vec![
                BlendEntry2D {
                    pos: [0.0, 0.0],
                    clip: id(1),
                },
                BlendEntry2D {
                    pos: [1.0, 0.0],
                    clip: id(2),
                },
                BlendEntry2D {
                    pos: [0.0, 1.0],
                    clip: id(3),
                },
            ],
        );
        assert_eq!(blend_weights_2d(&sp, DVec2::new(1.0, 0.0)), vec![(1, 1.0)]);
    }

    #[test]
    fn weights_2d_nearest_dominates_and_normalizes() {
        let sp = BlendSpace2D::new(
            "x",
            "y",
            vec![
                BlendEntry2D {
                    pos: [0.0, 0.0],
                    clip: id(1),
                },
                BlendEntry2D {
                    pos: [4.0, 0.0],
                    clip: id(2),
                },
                BlendEntry2D {
                    pos: [0.0, 4.0],
                    clip: id(3),
                },
            ],
        );
        let w = blend_weights_2d(&sp, DVec2::new(0.5, 0.5));
        // Sums to 1.
        let total: f64 = w.iter().map(|&(_, x)| x).sum();
        assert!((total - 1.0).abs() < 1e-12);
        // The origin sample (nearest) carries the most weight.
        let origin_w = w.iter().find(|&&(i, _)| i == 0).unwrap().1;
        for &(i, wi) in &w {
            if i != 0 {
                assert!(origin_w > wi, "origin should dominate");
            }
        }
    }

    #[test]
    fn sample_2d_at_entry_matches_that_clip() {
        let sk = one_joint_skeleton();
        let c_left = spin_clip(90.0);
        let sp = BlendSpace2D::new(
            "x",
            "y",
            vec![
                BlendEntry2D {
                    pos: [0.0, 0.0],
                    clip: id(1),
                },
                BlendEntry2D {
                    pos: [1.0, 0.0],
                    clip: id(2),
                },
            ],
        );
        let clips = |r: ClipRef| -> Option<&AnimClip> {
            if r == id(2) {
                Some(&c_left)
            } else {
                None
            }
        };
        // Exactly at entry (1,0), t=1.0 → phase 1.0.rem_euclid=0 → 0° (start).
        // Use t = 0.5 → 45°.
        let pose = sample_blend_space_2d(&sp, &sk, &clips, DVec2::new(1.0, 0.0), 0.5);
        let ang = pose.locals[0]
            .rotation_quat()
            .angle_between(Quat::IDENTITY)
            .to_degrees();
        assert!((ang - 45.0).abs() < 1.0, "{ang}");
    }

    #[test]
    fn unresolved_clips_fall_back_to_rest() {
        let sk = one_joint_skeleton();
        let sp = BlendSpace1D::new(
            "speed",
            vec![BlendEntry1D {
                pos: 0.0,
                clip: id(9),
            }],
        );
        let clips = |_r: ClipRef| -> Option<&AnimClip> { None };
        let pose = sample_blend_space_1d(&sp, &sk, &clips, 0.0, 0.0);
        assert_eq!(pose, Pose::rest(&sk));
    }
}
