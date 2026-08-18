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
//! # Phase synchronization
//!
//! When two locomotion clips of *different* lengths blend (a 1 s walk vs. a
//! 0.6 s run), sampling both at the same absolute second would let their feet
//! drift out of step and slide. v1 blends by **normalized phase**: a single phase
//! `φ ∈ [0,1)` is derived from the play-head `t` against the weight-blended
//! duration of the contributing clips, and each clip is sampled at
//! `φ · clip.duration`. The gait cycles stay locked regardless of clip length.
//!
//! **P29.2 adds the marker scheme that v1's docs deferred.** When every
//! contributing clip carries markers in a common sync group
//! ([`crate::sync`], `.inf_anim` v2), the followers are sampled at the time that
//! puts them at the *same point between the same two named events* as the
//! highest-weight contributor, instead of at the same fraction of their own
//! length. Uniform phase remains the answer for clips without markers, so no
//! committed content moved when this landed.
//!
//! # 2D weighting
//!
//! v1 shipped **inverse-distance weighting of the `k = 3` nearest** samples and
//! named the Delaunay upgrade as the seam to swap. P29.2 swaps it: the samples are
//! triangulated ([`crate::delaunay`]) and the query takes the **barycentric**
//! weights of the triangle it is inside, which is the unique affine weighting and
//! — unlike IDW — gives zero weight to every sample that is not a corner of that
//! triangle. Outside the hull the query projects onto the nearest hull edge (two
//! samples); a set with no area at all (one point, two points, or a collinear row)
//! interpolates along its own line, which is the 1D rule and the only meaningful
//! one there.

use glam::DVec2;
use serde::{Deserialize, Serialize};

use crate::clip::AnimClip;
use crate::delaunay::{barycentric, triangulate};
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

/// The weighted contribution of each 2D sample at `params`: the **barycentric**
/// weights of the Delaunay triangle containing the query, normalized to sum to 1
/// (empty if there are no entries). An exact hit on a sample returns it alone.
///
/// See [`weights_2d`] for the three cases and why each is what it is; this is
/// that function over a [`BlendSpace2D`]'s entries.
pub fn blend_weights_2d(space: &BlendSpace2D, params: DVec2) -> Vec<(usize, f64)> {
    let points: Vec<DVec2> = space
        .entries
        .iter()
        .map(|e| DVec2::new(e.pos[0], e.pos[1]))
        .collect();
    weights_2d(&points, params)
}

/// How far outside a triangle a query may sit and still count as inside it.
///
/// Barycentric coordinates are dimensionless, so this is an absolute tolerance
/// and not a relative one. It exists for the query that lands exactly on a shared
/// edge: without it, floating-point noise can put such a point outside *both*
/// neighbours and send it down the hull-projection path, which answers correctly
/// but is a different code path for a point that is plainly interior.
const INSIDE_EPS: f64 = 1e-9;

/// The barycentric weighting of a 2D sample set at `q`, as `(index, weight)`
/// pairs summing to 1.
///
/// Three cases, in order:
///
/// 1. **Exactly at a sample** (within [`EXACT_EPS`]) — that sample alone. Ties go
///    to the lowest index, so a duplicated position is deterministic.
/// 2. **Inside the hull** — the containing triangle's barycentric weights.
///    Triangles are scanned in the triangulation's canonical order and the first
///    containing one wins, which matters only on a shared edge, where both give
///    the same answer.
/// 3. **Outside the hull, or a set with no area** — the nearest point on the
///    boundary. For a triangulated set that is the nearest hull *edge*, projected
///    and clamped, so a query beyond a corner collapses to that corner. For a set
///    with no triangulation at all (one point, two points, a collinear row) it is
///    the same rule applied to the row itself: project onto the dominant
///    direction and bracket, which is exactly [`blend_weights_1d`] and the only
///    meaningful reading of a blend space with no area.
pub fn weights_2d(points: &[DVec2], q: DVec2) -> Vec<(usize, f64)> {
    let n = points.len();
    if n == 0 {
        return Vec::new();
    }
    // 1. Exact hit — scanned in index order so the lowest index wins a tie.
    let mut nearest = (usize::MAX, f64::INFINITY);
    for (i, p) in points.iter().enumerate() {
        let d2 = (*p - q).length_squared();
        if d2 < nearest.1 {
            nearest = (i, d2);
        }
    }
    if nearest.0 == usize::MAX {
        // Every point was non-finite; nothing can be said.
        return Vec::new();
    }
    if nearest.1 <= EXACT_EPS * EXACT_EPS {
        return vec![(nearest.0, 1.0)];
    }

    // 2. Inside a triangle.
    let tri = triangulate(points);
    for t in &tri.triangles {
        let Some(w) = barycentric(points[t[0]], points[t[1]], points[t[2]], q) else {
            continue;
        };
        if w.iter().all(|x| *x >= -INSIDE_EPS) {
            let clamped = [w[0].max(0.0), w[1].max(0.0), w[2].max(0.0)];
            let total: f64 = clamped.iter().sum();
            if total > 0.0 {
                let mut out: Vec<(usize, f64)> = t
                    .iter()
                    .zip(clamped)
                    .filter(|(_, x)| *x > 0.0)
                    .map(|(i, x)| (*i, x / total))
                    .collect();
                out.sort_by_key(|(i, _)| *i);
                return out;
            }
        }
    }

    // 3. Outside the hull, or no hull at all.
    let edges: Vec<[usize; 2]> = if tri.hull.is_empty() {
        chain_edges(points)
    } else {
        tri.hull.clone()
    };
    let mut best: Option<([usize; 2], f64, f64)> = None; // (edge, t, distance²)
    for e in edges {
        let (a, b) = (points[e[0]], points[e[1]]);
        let ab = b - a;
        let len2 = ab.length_squared();
        let t = if len2 > 0.0 {
            ((q - a).dot(ab) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let d2 = (a + ab * t - q).length_squared();
        if best.as_ref().is_none_or(|(_, _, bd)| d2 < *bd) {
            best = Some((e, t, d2));
        }
    }
    match best {
        Some((e, t, _)) => {
            let mut out = Vec::with_capacity(2);
            if 1.0 - t > 0.0 {
                out.push((e[0], 1.0 - t));
            }
            if t > 0.0 {
                out.push((e[1], t));
            }
            out.sort_by_key(|(i, _)| *i);
            out
        }
        // One usable point, or none but the nearest: it takes everything.
        None => vec![(nearest.0, 1.0)],
    }
}

/// The consecutive pairs of a point set with **no area**, ordered along its own
/// dominant direction — the "hull" of a collinear row.
///
/// Sorting by the wider of the two axes (ties to X, then to the original index)
/// makes this a property of the point set rather than of the order it arrived in,
/// and reduces to [`blend_weights_1d`]'s rule when the row is an axis.
fn chain_edges(points: &[DVec2]) -> Vec<[usize; 2]> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }
    let (mut lo, mut hi) = (points[0], points[0]);
    for p in points {
        if p.is_finite() {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
    }
    let extent = hi - lo;
    let by_x = extent.x >= extent.y;
    let key = |p: DVec2| if by_x { (p.x, p.y) } else { (p.y, p.x) };
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let (ka, kb) = (key(points[a]), key(points[b]));
        ka.0.partial_cmp(&kb.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| ka.1.partial_cmp(&kb.1).unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.cmp(&b))
    });
    order.windows(2).map(|w| [w[0], w[1]]).collect()
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
    let uniform: Vec<f32> = resolved
        .iter()
        .map(|(c, _)| (phase * c.duration as f64) as f32)
        .collect();
    // **Marker warping, when the content supports it** (P29.2). The leader keeps
    // its uniform time — so the cycle's rate still comes from the blended duration
    // — and the followers are moved onto its marker phase. `None` when the clips
    // do not share a usable sync group, which is every clip written before
    // `.inf_anim` v2 and is why nothing committed moved.
    let clips_only: Vec<&AnimClip> = resolved.iter().map(|(c, _)| *c).collect();
    let weights_only: Vec<f64> = resolved.iter().map(|(_, w)| *w).collect();
    let leader = crate::sync::leader_index(&weights_only);
    let times =
        crate::sync::warped_times(&clips_only, &weights_only, uniform[leader]).unwrap_or(uniform);
    let items: Vec<(Pose, f64)> = resolved
        .iter()
        .zip(&times)
        .map(|((c, w), ct)| (sample_clip(skeleton, c, *ct, true), *w))
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

    /// **A quadrant query is a blend of that quadrant's samples and nothing
    /// else.**
    ///
    /// The fixture is the locomotion diamond a wizard actually emits: idle at the
    /// centre with forward / right / back / left around it. The query sits in the
    /// forward-right quadrant, and `back` and `left` take **no** weight, because
    /// they are not corners of the triangle it is inside.
    ///
    /// Honest note on what this arm can and cannot kill: on a *well-formed* grid
    /// `k = 3` inverse-distance weighting usually picks the same three samples, so
    /// this arm is about the **support** and not about IDW. The arm that kills IDW
    /// is its sibling, `the_weights_reproduce_the_query_point` — see there for the
    /// measured error.
    ///
    /// The four outer samples alone (a diamond with no centre) are exactly
    /// **co-circular**, so *some* triangle spanning the middle must exist and
    /// "the opposite clip weighs nothing" is not achievable by any triangulation.
    /// The centre sample is what makes the fan well-formed — which is a fact about
    /// authoring a blend space, and is why the wizard's set has one.
    #[test]
    fn a_quadrant_query_blends_only_its_own_quadrant() {
        let sp = BlendSpace2D::new(
            "vx",
            "vy",
            vec![
                BlendEntry2D {
                    pos: [0.0, 0.0],
                    clip: id(0),
                }, // idle
                BlendEntry2D {
                    pos: [0.0, 1.0],
                    clip: id(1),
                }, // forward
                BlendEntry2D {
                    pos: [1.0, 0.0],
                    clip: id(2),
                }, // right
                BlendEntry2D {
                    pos: [0.0, -1.0],
                    clip: id(3),
                }, // back
                BlendEntry2D {
                    pos: [-1.0, 0.0],
                    clip: id(4),
                }, // left
            ],
        );
        let w = blend_weights_2d(&sp, DVec2::new(0.3, 0.3));
        let of = |i: usize| w.iter().find(|(j, _)| *j == i).map(|(_, x)| *x);
        assert_eq!(of(3), None, "the BACKWARD clip took weight: {w:?}");
        assert_eq!(of(4), None, "the LEFT clip took weight: {w:?}");
        assert!(
            of(0).is_some() && of(1).is_some() && of(2).is_some(),
            "{w:?}"
        );
        let total: f64 = w.iter().map(|(_, x)| x).sum();
        assert!((total - 1.0).abs() < 1e-12, "{w:?}");
        // On the diagonal the two direction samples are symmetric.
        assert!((of(1).unwrap() - of(2).unwrap()).abs() < 1e-12, "{w:?}");
    }

    /// **Barycentric weights are affine, and IDW's are not** — the arm that kills
    /// the thing this clause replaced.
    ///
    /// The claim: weighting the *sample positions* by the returned weights
    /// reproduces the query position exactly. That is what "these are the right
    /// weights" means for a parametric blend — the pose you get at `(0.25, 0.4)`
    /// is the pose the space says lives at `(0.25, 0.4)`.
    ///
    /// Measured against the `k = 3` inverse-distance body this replaced, on this
    /// very fixture: at `(0.25, 0.4)` IDW returns `0.545 / 0.287 / 0.168` over
    /// `(0,0) / (0,1) / (1,0)`, which reconstructs `(0.168, 0.287)` — **0.14 away**
    /// on a grid two units wide, so the character blends as if it were at a
    /// materially different point of its own parameter space. Restoring the IDW
    /// body fails this arm by six orders of magnitude.
    #[test]
    fn the_weights_reproduce_the_query_point() {
        let entries: Vec<BlendEntry2D> = [
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (-1.0, 0.0),
            (0.0, 0.0),
            (1.0, 0.0),
            (-1.0, 1.0),
            (0.0, 1.0),
            (1.0, 1.0),
        ]
        .iter()
        .enumerate()
        .map(|(i, &(x, y))| BlendEntry2D {
            pos: [x, y],
            clip: id(i as u8),
        })
        .collect();
        let sp = BlendSpace2D::new("x", "y", entries.clone());
        for &(qx, qy) in &[
            (0.25, 0.4),
            (-0.7, 0.1),
            (0.9, -0.9),
            (0.0, 0.5),
            (-0.33, -0.66),
        ] {
            let q = DVec2::new(qx, qy);
            let w = blend_weights_2d(&sp, q);
            let total: f64 = w.iter().map(|(_, x)| x).sum();
            assert!((total - 1.0).abs() < 1e-12, "({qx},{qy}) {w:?}");
            let mut p = DVec2::ZERO;
            for &(i, x) in &w {
                p += DVec2::new(entries[i].pos[0], entries[i].pos[1]) * x;
            }
            assert!(
                (p - q).length() < 1e-9,
                "({qx},{qy}) reconstructed as {p:?} from {w:?}"
            );
            // At most three samples ever contribute.
            assert!(w.len() <= 3, "({qx},{qy}) {w:?}");
        }
    }

    /// Outside the hull the query collapses onto the nearest boundary **edge**,
    /// and past a corner onto that corner — never onto an interior sample.
    #[test]
    fn a_query_outside_the_hull_lands_on_the_boundary() {
        let sp = BlendSpace2D::new(
            "x",
            "y",
            vec![
                BlendEntry2D {
                    pos: [0.0, 0.0],
                    clip: id(1),
                },
                BlendEntry2D {
                    pos: [2.0, 0.0],
                    clip: id(2),
                },
                BlendEntry2D {
                    pos: [0.0, 2.0],
                    clip: id(3),
                },
            ],
        );
        // Straight out past the middle of the 0→1 edge.
        let w = blend_weights_2d(&sp, DVec2::new(1.0, -5.0));
        assert_eq!(w.len(), 2, "{w:?}");
        assert!(
            (w[0].1 - 0.5).abs() < 1e-12 && (w[1].1 - 0.5).abs() < 1e-12,
            "{w:?}"
        );
        // Past a corner: that corner alone.
        let w = blend_weights_2d(&sp, DVec2::new(-9.0, -9.0));
        assert_eq!(w, vec![(0, 1.0)]);
    }

    /// A blend space with **no area** is a row, and a row interpolates along
    /// itself — the same rule `blend_weights_1d` applies, reached through the 2D
    /// door.
    #[test]
    fn a_collinear_or_degenerate_space_interpolates_along_its_own_line() {
        let sp = BlendSpace2D::new(
            "x",
            "y",
            vec![
                BlendEntry2D {
                    pos: [0.0, 0.0],
                    clip: id(1),
                },
                BlendEntry2D {
                    pos: [2.0, 0.0],
                    clip: id(2),
                },
                BlendEntry2D {
                    pos: [4.0, 0.0],
                    clip: id(3),
                },
            ],
        );
        let w = blend_weights_2d(&sp, DVec2::new(1.0, 0.0));
        assert_eq!(w.len(), 2, "{w:?}");
        assert!((w[0].1 - 0.5).abs() < 1e-12, "{w:?}");
        // Off the line, it still brackets the nearest span.
        let w = blend_weights_2d(&sp, DVec2::new(3.0, 7.0));
        assert_eq!(w.iter().map(|(i, _)| *i).collect::<Vec<_>>(), vec![1, 2]);
        // A single sample takes everything, whatever is asked.
        let one = BlendSpace2D::new(
            "x",
            "y",
            vec![BlendEntry2D {
                pos: [5.0, 5.0],
                clip: id(1),
            }],
        );
        assert_eq!(
            blend_weights_2d(&one, DVec2::new(-3.0, 8.0)),
            vec![(0, 1.0)]
        );
        // …and an empty one says nothing.
        assert!(blend_weights_2d(&BlendSpace2D::new("x", "y", Vec::new()), DVec2::ZERO).is_empty());
    }

    /// The weights are a property of the point set, not of the order it was
    /// authored in — the determinism law, at the layer a `.inf_sm` actually
    /// carries.
    #[test]
    fn reordering_the_entries_reorders_nothing_else() {
        let pts = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
            (1.6, 0.4),
        ];
        let entry = |i: usize| BlendEntry2D {
            pos: [pts[i].0, pts[i].1],
            clip: id(i as u8),
        };
        let q = DVec2::new(0.62, 0.31);
        let base = BlendSpace2D::new("x", "y", (0..pts.len()).map(entry).collect());
        let want: std::collections::BTreeMap<usize, f64> =
            blend_weights_2d(&base, q).into_iter().collect();
        for shift in 1..pts.len() {
            let order: Vec<usize> = (0..pts.len()).map(|i| (i + shift) % pts.len()).collect();
            let sp = BlendSpace2D::new("x", "y", order.iter().map(|&i| entry(i)).collect());
            let got: std::collections::BTreeMap<usize, f64> = blend_weights_2d(&sp, q)
                .into_iter()
                .map(|(i, w)| (order[i], w))
                .collect();
            assert_eq!(got, want, "shift {shift}");
        }
    }

    /// **Sync markers inside a blend** (P29.2 clause 4), through the sampling door
    /// rather than through `sync`'s own unit arms.
    ///
    /// A 1 s walk and a 0.6 s run whose plants sit at different *fractions* of
    /// their own lengths. With markers, the run is sampled at its own plant when
    /// the walk is at the walk's; without them (the same fixture with the markers
    /// stripped) it is not — which is the foot slide, measured as a pose
    /// difference rather than described.
    #[test]
    fn a_blend_space_warps_its_followers_onto_the_leader_marker_phase() {
        use crate::channels::AnimMarker;
        let sk = one_joint_skeleton();
        // Position along X == clip time, so a sampled pose reads back the time.
        let walk = slide_clip(0.0, 1.0, 1.0);
        let run = slide_clip(0.0, 0.6, 0.6);
        let marked = |c: &AnimClip, a: f32, b: f32| {
            c.clone().with_markers(vec![
                AnimMarker::sync(a, "plant_l", "foot"),
                AnimMarker::sync(b, "plant_r", "foot"),
            ])
        };
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
        // Weight 0.75 on the walk → the walk leads. blended_dur = 0.75*1 + 0.25*0.6
        // = 0.9, so t = 0.63 puts the phase at 0.7 and the walk's play-head at 0.7 —
        // exactly its `plant_r`.
        let run_x = |walk_m: Option<(f32, f32)>, run_m: Option<(f32, f32)>| -> f32 {
            let w = match walk_m {
                Some((a, b)) => marked(&walk, a, b),
                None => walk.clone(),
            };
            let r = match run_m {
                Some((a, b)) => marked(&run, a, b),
                None => run.clone(),
            };
            let clips = |x: ClipRef| -> Option<&AnimClip> {
                if x == id(1) {
                    Some(&w)
                } else if x == id(2) {
                    Some(&r)
                } else {
                    None
                }
            };
            // Weight the run at 1.0 in a second probe would change the leader, so
            // read the run's contribution by sampling the space at the run entry
            // with the walk still leading: instead, read the blended X and solve.
            // Simpler and exact: sample with the walk at weight 1 - eps is not
            // available, so sample the run alone at the warped time by asking the
            // space at param 0.25 and subtracting the walk's known contribution.
            let pose = sample_blend_space_1d(&sp, &sk, &clips, 0.25, 0.63);
            // blended X = 0.75*walk_x + 0.25*run_x, and walk_x is its own play-head.
            let blended = pose.locals[0].translation[0];
            (blended - 0.75 * 0.7) / 0.25
        };
        // Unmarked: uniform phase puts the run at 0.7 * 0.6 = 0.42.
        let plain = run_x(None, None);
        assert!((plain - 0.42).abs() < 1e-3, "unmarked run at {plain}");
        // Marked: the walk's plant_r is at 0.7, the run's is at 0.35, so the run is
        // sampled there — 0.07 s earlier than uniform phase would have it.
        let warped = run_x(Some((0.2, 0.7)), Some((0.05, 0.35)));
        assert!(
            (warped - 0.35).abs() < 1e-3,
            "marked run at {warped}, expected its own plant_r at 0.35"
        );
        assert!(
            (plain - warped).abs() > 0.05,
            "marker warping changed nothing ({plain} vs {warped})"
        );
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
