//! Descriptor matching and the tracks it produces.
//!
//! # The two filters, and why both
//!
//! Brute-force Hamming search finds *a* nearest neighbour for every descriptor,
//! always — including for a feature whose counterpart is not in the other
//! photograph at all. Two independent filters throw those away:
//!
//! * **Lowe's ratio test.** Keep a match only when the best distance is
//!   comfortably below the second best. A descriptor that has no true match
//!   sits at a roughly uniform distance from everything, so its best and second
//!   best are close together and it is rejected. This is the filter that does
//!   most of the work.
//! * **The mutual-best (cross) check.** `a`'s best in `B` must be `b`, and
//!   `b`'s best in `A` must be `a`. This catches the many-to-one collapses the
//!   ratio test lets through, which are exactly what a repetitive texture
//!   produces.
//!
//! Both are cheap and neither subsumes the other. RANSAC downstream removes the
//! rest.
//!
//! # Determinism
//!
//! Nearest-neighbour ties are resolved by **lower index wins**, in both
//! directions, so the match set is a function of the descriptors and their order
//! and nothing else. The per-query scan is dispatched through
//! [`inf_core::job::JobPool::parallel_map_ref`], which is an in-order pure map:
//! the result vector is the same on one worker and on sixty-four.
//!
//! # Pair candidates
//!
//! Below [`MatchConfig::exhaustive_view_cap`] views, every pair is matched —
//! `n(n-1)/2` of them, which is the honest thing to do and is affordable for the
//! capture sizes a wizard will accept. Above the cap, each view is matched
//! against its neighbours in input order only, and an
//! [`Advisory::WindowedPairing`] says so. There is no vocabulary tree, no
//! image-retrieval index, and therefore no way to find an overlap between
//! photograph 3 and photograph 400 of a large unordered set. That is the
//! documented v1 bound; the remedy is a retrieval stage, not a bigger window.

use inf_core::job::JobPool;

use crate::features::{Feature, FeatureSet};
use crate::Advisory;

/// One accepted correspondence between two views' feature lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Match {
    /// Index into the first view's [`FeatureSet`].
    pub a: u32,
    /// Index into the second view's [`FeatureSet`].
    pub b: u32,
    /// Hamming distance in bits.
    pub distance: u32,
}

/// Every accepted correspondence between one ordered pair of views.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairMatches {
    /// The lower view index.
    pub view_a: u32,
    /// The higher view index.
    pub view_b: u32,
    /// Matches, sorted by `(a, b)`.
    pub matches: Vec<Match>,
}

/// How matching is tuned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchConfig {
    /// Lowe's ratio: a match is kept when `best <= ratio * second_best`.
    /// 0.8 is the value the SIFT paper settled on and it transfers to binary
    /// descriptors unchanged.
    pub ratio: f64,
    /// A match further than this many bits is rejected outright, however good
    /// its ratio. 256-bit descriptors of genuinely corresponding patches
    /// typically land under 60.
    pub max_distance: u32,
    /// Require `a`'s best to be `b` and `b`'s best to be `a`.
    pub mutual: bool,
    /// A pair with fewer matches than this is not worth carrying.
    pub min_pair_matches: usize,
    /// Above this many views, pairing switches from exhaustive to windowed.
    pub exhaustive_view_cap: usize,
    /// The windowed strategy's half-window, in input order.
    pub window: usize,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            ratio: 0.8,
            max_distance: 80,
            mutual: true,
            min_pair_matches: 16,
            exhaustive_view_cap: 60,
            window: 8,
        }
    }
}

/// Which view pairs to match, plus an advisory if the strategy degraded.
pub fn candidate_pairs(n_views: usize, cfg: &MatchConfig) -> (Vec<(u32, u32)>, Option<Advisory>) {
    let mut pairs = Vec::new();
    if n_views <= cfg.exhaustive_view_cap {
        for a in 0..n_views {
            for b in (a + 1)..n_views {
                pairs.push((a as u32, b as u32));
            }
        }
        (pairs, None)
    } else {
        let w = cfg.window.max(1);
        for a in 0..n_views {
            for b in (a + 1)..(a + 1 + w).min(n_views) {
                pairs.push((a as u32, b as u32));
            }
        }
        (
            pairs,
            Some(Advisory::WindowedPairing {
                views: n_views,
                cap: cfg.exhaustive_view_cap,
                window: w,
            }),
        )
    }
}

/// The best and second-best Hamming distance from one descriptor into a set,
/// with **lower index winning ties**.
fn best_two(query: &Feature, target: &[Feature]) -> (u32, u32, u32) {
    let mut best = u32::MAX;
    let mut best_idx = u32::MAX;
    let mut second = u32::MAX;
    for (i, t) in target.iter().enumerate() {
        let d = query.distance(t);
        if d < best {
            second = best;
            best = d;
            best_idx = i as u32;
        } else if d < second {
            second = d;
        }
    }
    (best_idx, best, second)
}

/// Match two feature sets, fanning the per-query scan across `pool`. The result
/// is sorted by `(a, b)`.
pub fn match_features(
    a: &FeatureSet,
    b: &FeatureSet,
    cfg: &MatchConfig,
    pool: &JobPool,
) -> Vec<Match> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let forward = pool.parallel_map_ref(&a.features, |f| best_two(f, &b.features));
    let backward = if cfg.mutual {
        pool.parallel_map_ref(&b.features, |f| best_two(f, &a.features))
    } else {
        Vec::new()
    };
    resolve(&forward, &backward, cfg)
}

/// Match two feature sets on the calling thread. [`match_all`] uses this because
/// it is already fanned out **over pairs**, and nesting a second fan-out inside
/// each one buys nothing but scheduler churn.
pub fn match_features_serial(a: &FeatureSet, b: &FeatureSet, cfg: &MatchConfig) -> Vec<Match> {
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let forward: Vec<_> = a
        .features
        .iter()
        .map(|f| best_two(f, &b.features))
        .collect();
    let backward: Vec<_> = if cfg.mutual {
        b.features
            .iter()
            .map(|f| best_two(f, &a.features))
            .collect()
    } else {
        Vec::new()
    };
    resolve(&forward, &backward, cfg)
}

/// Apply the ratio, distance and mutual filters to a pair of nearest-neighbour
/// scans. Split out so the parallel and serial doors cannot drift apart — there
/// is exactly one place the filters live.
fn resolve(
    forward: &[(u32, u32, u32)],
    backward: &[(u32, u32, u32)],
    cfg: &MatchConfig,
) -> Vec<Match> {
    let mut out = Vec::new();
    for (ai, &(bi, best, second)) in forward.iter().enumerate() {
        if bi == u32::MAX || best > cfg.max_distance {
            continue;
        }
        // `second == u32::MAX` means the target set had exactly one feature —
        // there is nothing to ratio against, so the match stands or falls on
        // `max_distance` alone.
        if second != u32::MAX && (best as f64) > cfg.ratio * second as f64 {
            continue;
        }
        if cfg.mutual && backward[bi as usize].0 != ai as u32 {
            continue;
        }
        out.push(Match {
            a: ai as u32,
            b: bi,
            distance: best,
        });
    }
    out.sort_unstable();
    out
}

/// Match every candidate pair. Pairs that fall under
/// [`MatchConfig::min_pair_matches`] are dropped.
pub fn match_all(
    sets: &[FeatureSet],
    cfg: &MatchConfig,
    pool: &JobPool,
) -> (Vec<PairMatches>, Vec<Advisory>) {
    let (pairs, advisory) = candidate_pairs(sets.len(), cfg);
    let matched = pool.parallel_map_ref(&pairs, |&(a, b)| {
        match_features_serial(&sets[a as usize], &sets[b as usize], cfg)
    });
    let mut out = Vec::new();
    for (&(a, b), matches) in pairs.iter().zip(matched) {
        if matches.len() < cfg.min_pair_matches {
            continue;
        }
        out.push(PairMatches {
            view_a: a,
            view_b: b,
            matches,
        });
    }
    (out, advisory.into_iter().collect())
}

/// One feature of one view, as seen by a track.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Observation {
    /// Which view.
    pub view: u32,
    /// Which feature of that view.
    pub feature: u32,
}

/// A feature followed across views: the same surface point, seen more than once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Track {
    /// Observations, sorted by `(view, feature)` and with at most one per view.
    pub observations: Vec<Observation>,
}

impl Track {
    /// The observation of `view`, if this track has one.
    pub fn observation_in(&self, view: u32) -> Option<u32> {
        self.observations
            .iter()
            .find(|o| o.view == view)
            .map(|o| o.feature)
    }

    /// How many views see this track.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Is it empty?
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }
}

/// Union-find with **union-by-smaller-root**, which makes the resulting forest a
/// pure function of the edge *set* rather than of the order edges arrived in.
struct DisjointSet {
    parent: Vec<u32>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).collect(),
        }
    }

    fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let grandparent = self.parent[self.parent[x as usize] as usize];
            self.parent[x as usize] = grandparent;
            x = grandparent;
        }
        x
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        // Always attach the larger root to the smaller one.
        if ra < rb {
            self.parent[rb as usize] = ra;
        } else {
            self.parent[ra as usize] = rb;
        }
    }
}

/// Chain pairwise matches into multi-view tracks.
///
/// A track that ends up seeing one view **twice** is a matching inconsistency —
/// two distinct features of one photograph cannot be the same surface point.
/// Rather than throwing the whole track away (which loses every other view with
/// it), the offending view's observations are dropped and the rest of the track
/// survives, with an [`Advisory::ConflictingTrackViews`] recording how much was
/// lost. Tracks shorter than `min_length` after that are discarded.
///
/// Tracks come back ordered by their first observation, so track indices are
/// canonical.
pub fn build_tracks(
    sets: &[FeatureSet],
    pairs: &[PairMatches],
    min_length: usize,
) -> (Vec<Track>, Vec<Advisory>) {
    let mut offsets = Vec::with_capacity(sets.len() + 1);
    let mut total = 0usize;
    for s in sets {
        offsets.push(total as u32);
        total += s.len();
    }
    offsets.push(total as u32);

    let mut ds = DisjointSet::new(total);
    for p in pairs {
        let base_a = offsets[p.view_a as usize];
        let base_b = offsets[p.view_b as usize];
        for m in &p.matches {
            ds.union(base_a + m.a, base_b + m.b);
        }
    }

    // Group by root. A BTreeMap keyed by root gives a canonical group order for
    // free, since roots are the smallest node id in each group.
    let mut groups: std::collections::BTreeMap<u32, Vec<Observation>> =
        std::collections::BTreeMap::new();
    for view in 0..sets.len() {
        for feature in 0..sets[view].len() {
            let node = offsets[view] + feature as u32;
            let root = ds.find(node);
            groups.entry(root).or_default().push(Observation {
                view: view as u32,
                feature: feature as u32,
            });
        }
    }

    let mut tracks = Vec::new();
    let mut conflicting_tracks = 0usize;
    let mut dropped = 0usize;
    for (_, mut obs) in groups {
        if obs.len() < min_length {
            continue;
        }
        obs.sort_unstable();
        // Find views appearing more than once.
        let mut conflicted: Vec<u32> = Vec::new();
        for w in obs.windows(2) {
            if w[0].view == w[1].view {
                conflicted.push(w[0].view);
            }
        }
        if !conflicted.is_empty() {
            conflicted.dedup();
            let before = obs.len();
            obs.retain(|o| !conflicted.contains(&o.view));
            conflicting_tracks += 1;
            dropped += before - obs.len();
        }
        if obs.len() < min_length {
            continue;
        }
        tracks.push(Track { observations: obs });
    }

    let mut advisories = Vec::new();
    if conflicting_tracks > 0 {
        advisories.push(Advisory::ConflictingTrackViews {
            tracks: conflicting_tracks,
            dropped,
        });
    }
    (tracks, advisories)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{FeatureConfig, DESCRIPTOR_WORDS};

    fn feature_with(desc: [u64; DESCRIPTOR_WORDS], x: f64) -> Feature {
        Feature {
            x,
            y: 0.0,
            level: 0,
            score: 1.0,
            orientation_cos: 1.0,
            orientation_sin: 0.0,
            descriptor: desc,
        }
    }

    fn set_of(descs: &[[u64; DESCRIPTOR_WORDS]]) -> FeatureSet {
        FeatureSet {
            features: descs
                .iter()
                .enumerate()
                .map(|(i, d)| feature_with(*d, i as f64))
                .collect(),
        }
    }

    #[test]
    fn exhaustive_pairing_below_the_cap_and_windowed_above_it() {
        let cfg = MatchConfig {
            exhaustive_view_cap: 5,
            window: 2,
            ..MatchConfig::default()
        };
        let (pairs, adv) = candidate_pairs(5, &cfg);
        assert_eq!(pairs.len(), 10);
        assert!(adv.is_none());
        let (pairs, adv) = candidate_pairs(8, &cfg);
        // 8 views, half-window 2: 2 pairs each except the last two.
        assert_eq!(pairs.len(), 2 * 8 - 3);
        assert!(matches!(
            adv,
            Some(Advisory::WindowedPairing { window: 2, .. })
        ));
        assert!(pairs.iter().all(|&(a, b)| b > a && b - a <= 2));
    }

    #[test]
    fn the_ratio_test_rejects_an_ambiguous_match() {
        let pool = JobPool::new(1);
        // A query equidistant from two targets: nothing should survive.
        let q = set_of(&[[0, 0, 0, 0]]);
        let t = set_of(&[[0xff, 0, 0, 0], [0, 0xff, 0, 0]]);
        let cfg = MatchConfig {
            mutual: false,
            ..MatchConfig::default()
        };
        assert!(match_features(&q, &t, &cfg, &pool).is_empty());
        // Move one target much closer and it survives.
        let t = set_of(&[[0x3, 0, 0, 0], [0, 0xffff_ffff, 0, 0]]);
        let m = match_features(&q, &t, &cfg, &pool);
        assert_eq!(m.len(), 1);
        assert_eq!((m[0].a, m[0].b, m[0].distance), (0, 0, 2));
    }

    #[test]
    fn the_mutual_check_rejects_a_many_to_one_collapse() {
        let pool = JobPool::new(1);
        // Both queries are nearest to target 0, but only one can be mutual.
        let q = set_of(&[[0x1, 0, 0, 0], [0x3, 0, 0, 0]]);
        let t = set_of(&[[0x0, 0, 0, 0], [0xffff_ffff_ffff_ffff, 0, 0, 0]]);
        let loose = MatchConfig {
            mutual: false,
            ..MatchConfig::default()
        };
        assert_eq!(match_features(&q, &t, &loose, &pool).len(), 2);
        let strict = MatchConfig {
            mutual: true,
            ..MatchConfig::default()
        };
        let m = match_features(&q, &t, &strict, &pool);
        assert_eq!(m.len(), 1, "the collapse survived: {m:?}");
        assert_eq!(m[0].a, 0, "the mutual winner should be the closer query");
    }

    #[test]
    fn max_distance_rejects_a_lonely_query() {
        let pool = JobPool::new(1);
        let q = set_of(&[[0, 0, 0, 0]]);
        let t = set_of(&[[u64::MAX, u64::MAX, u64::MAX, u64::MAX]]);
        let cfg = MatchConfig {
            mutual: false,
            max_distance: 80,
            ..MatchConfig::default()
        };
        // One target: no ratio to test, so only max_distance can save us.
        assert!(match_features(&q, &t, &cfg, &pool).is_empty());
    }

    #[test]
    fn matching_is_pool_size_invariant() {
        use crate::gray::GrayImage;
        // Two real feature sets from two different renderings of one texture.
        let mut a = GrayImage::new(128, 128).unwrap();
        let mut b = GrayImage::new(128, 128).unwrap();
        for y in 0..128u32 {
            for x in 0..128u32 {
                let v = (((x * 37 + y * 91) ^ (x * y)) % 251) as u8;
                a.set(x, y, v);
                // A one-pixel shift, so the two are genuinely related.
                b.set(x, y, if x == 0 { v } else { a.at(x - 1, y) });
            }
        }
        let cfg = FeatureConfig::default();
        let fa = crate::features::detect_and_describe(&a, &cfg).unwrap();
        let fb = crate::features::detect_and_describe(&b, &cfg).unwrap();
        let mcfg = MatchConfig::default();
        let one = match_features(&fa, &fb, &mcfg, &JobPool::new(1));
        let eight = match_features(&fa, &fb, &mcfg, &JobPool::new(8));
        assert_eq!(one, eight, "matching depended on the worker count");
    }

    #[test]
    fn tracks_chain_across_three_views() {
        let sets = vec![
            set_of(&[[1, 0, 0, 0]; 2]),
            set_of(&[[1, 0, 0, 0]; 2]),
            set_of(&[[1, 0, 0, 0]; 2]),
        ];
        let pairs = vec![
            PairMatches {
                view_a: 0,
                view_b: 1,
                matches: vec![Match {
                    a: 0,
                    b: 1,
                    distance: 0,
                }],
            },
            PairMatches {
                view_a: 1,
                view_b: 2,
                matches: vec![Match {
                    a: 1,
                    b: 0,
                    distance: 0,
                }],
            },
        ];
        let (tracks, adv) = build_tracks(&sets, &pairs, 2);
        assert!(adv.is_empty());
        assert_eq!(tracks.len(), 1);
        assert_eq!(
            tracks[0].observations,
            vec![
                Observation {
                    view: 0,
                    feature: 0
                },
                Observation {
                    view: 1,
                    feature: 1
                },
                Observation {
                    view: 2,
                    feature: 0
                },
            ]
        );
        assert_eq!(tracks[0].observation_in(2), Some(0));
        assert_eq!(tracks[0].observation_in(7), None);
    }

    #[test]
    fn a_track_that_sees_one_view_twice_drops_that_view_and_says_so() {
        let sets = vec![
            set_of(&[[1, 0, 0, 0]; 2]),
            set_of(&[[1, 0, 0, 0]; 2]),
            set_of(&[[1, 0, 0, 0]; 2]),
        ];
        // View 1's features 0 and 1 both merge with view 0's feature 0.
        let pairs = vec![
            PairMatches {
                view_a: 0,
                view_b: 1,
                matches: vec![
                    Match {
                        a: 0,
                        b: 0,
                        distance: 0,
                    },
                    Match {
                        a: 0,
                        b: 1,
                        distance: 0,
                    },
                ],
            },
            PairMatches {
                view_a: 0,
                view_b: 2,
                matches: vec![Match {
                    a: 0,
                    b: 0,
                    distance: 0,
                }],
            },
        ];
        let (tracks, adv) = build_tracks(&sets, &pairs, 2);
        assert_eq!(
            adv,
            vec![Advisory::ConflictingTrackViews {
                tracks: 1,
                dropped: 2
            }]
        );
        assert_eq!(tracks.len(), 1);
        assert_eq!(
            tracks[0].observations,
            vec![
                Observation {
                    view: 0,
                    feature: 0
                },
                Observation {
                    view: 2,
                    feature: 0
                },
            ],
            "the conflicted view should be gone, the rest kept"
        );
    }

    #[test]
    fn short_tracks_are_discarded() {
        let sets = vec![set_of(&[[1, 0, 0, 0]; 2]), set_of(&[[1, 0, 0, 0]; 2])];
        let (tracks, _) = build_tracks(&sets, &[], 2);
        assert!(
            tracks.is_empty(),
            "isolated features became tracks: {tracks:?}"
        );
    }

    #[test]
    fn track_order_does_not_depend_on_pair_order() {
        let sets = vec![
            set_of(&[[1, 0, 0, 0]; 3]),
            set_of(&[[1, 0, 0, 0]; 3]),
            set_of(&[[1, 0, 0, 0]; 3]),
        ];
        let p01 = PairMatches {
            view_a: 0,
            view_b: 1,
            matches: vec![
                Match {
                    a: 2,
                    b: 0,
                    distance: 0,
                },
                Match {
                    a: 0,
                    b: 2,
                    distance: 0,
                },
            ],
        };
        let p12 = PairMatches {
            view_a: 1,
            view_b: 2,
            matches: vec![Match {
                a: 2,
                b: 1,
                distance: 0,
            }],
        };
        let forward = build_tracks(&sets, &[p01.clone(), p12.clone()], 2).0;
        let backward = build_tracks(&sets, &[p12, p01], 2).0;
        assert_eq!(forward, backward, "track identity depended on pair order");
    }
}
