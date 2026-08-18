//! Sync groups and marker-phase warping (P29.2 clause 4).
//!
//! [`crate::blend_space`]'s v1 docs named this as the follow-up: *"v1 therefore
//! blends by **normalized phase** … (A per-clip sync-marker scheme — aligning
//! specific foot-plant events rather than uniform phase — is the documented
//! follow-up.)"* This is that scheme.
//!
//! # What uniform phase gets wrong
//!
//! Uniform phase says frame 40% of the walk pairs with frame 40% of the run. That
//! is only true if both cycles put their foot-plants at the same *fractions* of
//! their own length, and real clips do not: a run has a flight phase, so its
//! plants sit closer together than a walk's, and the blend slides the feet
//! precisely in the middle of the stride where the contact matters. The fix is to
//! align on the events themselves — the clip says where its plants are
//! ([`crate::AnimMarker`], `.inf_anim` v2) and the blend samples each clip at the
//! time that puts it at the *same point between the same two named markers* as
//! the leader.
//!
//! # The model
//!
//! A [`SyncTrack`] is one clip's markers within one group, ascending. `M` markers
//! make `M` **segments** on a looping clip — `[m₀,m₁) … [m_{M-1}, m₀+duration)` —
//! and a play-head's [`SyncPhase`] is "which segment, and how far through it".
//! Followers match the leader's segment **by the name of the marker that starts
//! it**, so a walk whose cycle is `plant_l → plant_r` and a run whose cycle is
//! `plant_l → plant_r` line up even if one has an extra unnamed marker; a
//! follower that does not carry the leader's name falls back to the segment
//! *index*, modulo its own count, which is the v1-shaped answer and never worse.
//!
//! # Portability
//!
//! Divisions, multiplications and a `rem_euclid`. Nothing transcendental — a
//! warped sample time is a clip time, and a clip time is a pose.

use crate::clip::AnimClip;

/// Where a play-head sits inside a sync cycle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyncPhase {
    /// Index of the segment (equivalently, of the marker that starts it).
    pub segment: usize,
    /// How far through that segment, in `[0,1)`.
    pub frac: f32,
}

/// One clip's markers within one sync group.
#[derive(Clone, Debug, PartialEq)]
pub struct SyncTrack {
    /// The clip's length, seconds — the segment after the last marker wraps
    /// through it.
    pub duration_s: f32,
    /// `(time, name)` ascending. Built by [`SyncTrack::of`], which sorts.
    pub markers: Vec<(f32, String)>,
}

impl SyncTrack {
    /// The clip's markers in `group`, ascending.
    pub fn of(clip: &AnimClip, group: &str) -> Self {
        Self {
            duration_s: clip.duration,
            markers: clip
                .markers_in(group)
                .into_iter()
                .map(|m| (m.time_s, m.name.clone()))
                .collect(),
        }
    }

    /// Whether this track can drive or follow an alignment: at least two markers
    /// (one marker gives one segment, which is uniform phase with extra steps) and
    /// a usable duration.
    pub fn is_usable(&self) -> bool {
        self.markers.len() >= 2 && crate::positive(self.duration_s)
    }

    /// The number of segments — equal to the number of markers on a looping clip.
    pub fn segments(&self) -> usize {
        self.markers.len()
    }

    /// The name of the marker that starts segment `seg`.
    pub fn segment_name(&self, seg: usize) -> &str {
        match self.markers.get(seg) {
            Some((_, n)) => n.as_str(),
            None => "",
        }
    }

    /// The first segment whose start marker is named `name`.
    pub fn segment_named(&self, name: &str) -> Option<usize> {
        self.markers.iter().position(|(_, n)| n == name)
    }

    /// Where `t` seconds sits in the cycle. `None` when the track is unusable.
    pub fn phase_at(&self, t: f32) -> Option<SyncPhase> {
        if !self.is_usable() {
            return None;
        }
        let dur = self.duration_s;
        let t = crate::clip::resolve_time(t, dur, true);
        let m = &self.markers;
        let last = m.len() - 1;
        // The last marker at or before `t`; none means `t` is before the first,
        // which belongs to the wrapped final segment.
        let seg = match m.iter().rposition(|(x, _)| *x <= t) {
            Some(i) => i,
            None => last,
        };
        let start = m[seg].0;
        let (span, into) = if seg == last {
            // Wraps through the clip end: [m_last, m_0 + duration).
            let span = (m[0].0 + dur) - start;
            let into = if t < m[0].0 {
                t + dur - start
            } else {
                t - start
            };
            (span, into)
        } else {
            (m[seg + 1].0 - start, t - start)
        };
        let frac = if crate::positive(span) {
            (into / span).clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some(SyncPhase { segment: seg, frac })
    }

    /// The clip time of a [`SyncPhase`] on **this** track, wrapped into
    /// `[0, duration)`. `None` when the track is unusable.
    pub fn time_at(&self, phase: SyncPhase) -> Option<f32> {
        if !self.is_usable() {
            return None;
        }
        let m = &self.markers;
        let seg = phase.segment % m.len();
        let dur = self.duration_s;
        let start = m[seg].0;
        let end = if seg + 1 < m.len() {
            m[seg + 1].0
        } else {
            m[0].0 + dur
        };
        let frac = if phase.frac.is_finite() {
            phase.frac.clamp(0.0, 1.0)
        } else {
            0.0
        };
        Some(crate::clip::resolve_time(
            start + (end - start) * frac,
            dur,
            true,
        ))
    }

    /// This track's clip time for a leader's phase, matched **by name** with an
    /// index fallback.
    ///
    /// `leader_name` is the name of the marker starting the leader's current
    /// segment. When this track carries that name the segments line up on the
    /// event; when it does not, the leader's segment *index* is used modulo this
    /// track's count — the v1-shaped answer, so an unnamed or mismatched marker
    /// set degrades to uniform-ish phase rather than to a jump.
    pub fn follow(&self, leader: SyncPhase, leader_name: &str) -> Option<f32> {
        if !self.is_usable() {
            return None;
        }
        let segment = self
            .segment_named(leader_name)
            .unwrap_or(leader.segment % self.markers.len());
        self.time_at(SyncPhase {
            segment,
            frac: leader.frac,
        })
    }
}

/// The sync group a set of clips can align on: the lexicographically first group
/// that **every** usable clip names, or `None`.
///
/// Lexicographically first rather than "the biggest" or "the first authored":
/// [`AnimClip::sync_groups`] returns a `BTreeSet`, so this is a property of the
/// content and not of the order the clips were resolved in — the determinism law
/// applied to a choice that would otherwise be invisible.
pub fn common_group<'a>(clips: impl IntoIterator<Item = &'a AnimClip>) -> Option<String> {
    let mut shared: Option<std::collections::BTreeSet<String>> = None;
    for c in clips {
        let mine: std::collections::BTreeSet<String> = c
            .sync_groups()
            .into_iter()
            .filter(|g| SyncTrack::of(c, g).is_usable())
            .map(str::to_owned)
            .collect();
        shared = Some(match shared {
            None => mine,
            Some(s) => s.intersection(&mine).cloned().collect(),
        });
        if shared.as_ref().is_some_and(|s| s.is_empty()) {
            return None;
        }
    }
    shared.and_then(|s| s.into_iter().next())
}

/// The **leader** of a weighted set: the highest weight, ties going to the lowest
/// index.
///
/// One function rather than two copies of three lines: [`warped_times`] needs it
/// to pick whose phase everybody follows, and
/// [`crate::blend_space`] needs the *same* answer to know whose play-head to hand
/// in. Two copies that agree "by construction" is the shape of defect this
/// repository has already paid for more than once.
pub fn leader_index(weights: &[f64]) -> usize {
    let mut leader = 0usize;
    for i in 1..weights.len() {
        // `>` rather than `>=` keeps the tie-break at the lowest index, and puts
        // a NaN weight on the side that never wins.
        if weights[i] > weights[leader] {
            leader = i;
        }
    }
    leader
}

/// The per-clip sample times for one blended set, warped onto the leader's
/// marker phase.
///
/// `clips` and `weights` are index-aligned; `leader_time_s` is the play-head of
/// the **leader** (the highest-weight contributor, ties going to the lowest
/// index). Returns `None` when the set has no common sync group — the caller then
/// keeps uniform phase, which is exactly what v1 did.
///
/// The leader's own time is returned unchanged, so the cycle's *rate* still comes
/// from the caller (a blend-space's weight-blended duration); markers decide only
/// *where in its cycle* each follower is sampled.
pub fn warped_times(clips: &[&AnimClip], weights: &[f64], leader_time_s: f32) -> Option<Vec<f32>> {
    if clips.len() < 2 {
        return None;
    }
    let group = common_group(clips.iter().copied())?;
    let tracks: Vec<SyncTrack> = clips.iter().map(|c| SyncTrack::of(c, &group)).collect();
    let leader = leader_index(weights).min(clips.len() - 1);
    let phase = tracks[leader].phase_at(leader_time_s)?;
    let name = tracks[leader].segment_name(phase.segment).to_owned();
    let mut out = Vec::with_capacity(clips.len());
    for (i, tr) in tracks.iter().enumerate() {
        out.push(if i == leader {
            crate::clip::resolve_time(leader_time_s, tr.duration_s, true)
        } else {
            tr.follow(phase, &name)?
        });
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::AnimMarker;
    use crate::clip::{Interpolation, JointTrack, Vec3Track};

    /// A clip of length `dur` whose single joint slides 0 → 1, with foot-plant
    /// markers at the given times.
    fn gait(name: &str, dur: f32, plants: &[(f32, &str)]) -> AnimClip {
        let mut jt = JointTrack::new(0);
        jt.translation = Some(Vec3Track::new(
            vec![0.0, dur],
            vec![[0.0; 3], [1.0, 0.0, 0.0]],
            Interpolation::Linear,
        ));
        AnimClip::new(name, vec![jt]).with_markers(
            plants
                .iter()
                .map(|&(t, n)| AnimMarker::sync(t, n, "foot"))
                .collect(),
        )
    }

    #[test]
    fn a_phase_locates_its_segment_and_wraps_through_the_end() {
        let c = gait("walk", 1.0, &[(0.2, "plant_l"), (0.7, "plant_r")]);
        let tr = SyncTrack::of(&c, "foot");
        assert!(tr.is_usable());
        assert_eq!(tr.segments(), 2);
        // Inside the first segment [0.2, 0.7): half way is 0.45.
        let p = tr.phase_at(0.45).unwrap();
        assert_eq!(p.segment, 0);
        assert!((p.frac - 0.5).abs() < 1e-6, "{p:?}");
        // Inside the wrapped segment [0.7, 1.2): 0.95 is half way.
        let p = tr.phase_at(0.95).unwrap();
        assert_eq!(p.segment, 1);
        assert!((p.frac - 0.5).abs() < 1e-6, "{p:?}");
        // …and so is 0.0 + wrap: t = 0.0 sits 0.3 into a 0.5-long segment.
        let p = tr.phase_at(0.0).unwrap();
        assert_eq!(p.segment, 1);
        assert!((p.frac - 0.6).abs() < 1e-6, "{p:?}");
        // phase → time → phase is the identity.
        for t in [0.0f32, 0.2, 0.45, 0.7, 0.95, 1.19] {
            let p = tr.phase_at(t).unwrap();
            let back = tr.time_at(p).unwrap();
            let again = tr.phase_at(back).unwrap();
            assert_eq!(again.segment, p.segment, "t={t}");
            assert!((again.frac - p.frac).abs() < 1e-5, "t={t}");
        }
        // A one-marker (or unmarked) track cannot drive anything.
        assert!(!SyncTrack::of(&gait("x", 1.0, &[(0.0, "a")]), "foot").is_usable());
        assert!(!SyncTrack::of(&gait("x", 1.0, &[]), "foot").is_usable());
    }

    /// **The claim the whole clause exists for.** A 1 s walk plants at 0.2/0.7; a
    /// 0.6 s run plants at 0.05/0.35 — very different *fractions* of their own
    /// lengths. Under uniform phase the run is nowhere near its plant when the
    /// walk is at one; under marker warping it is at exactly the same point of the
    /// same named segment.
    #[test]
    fn warping_puts_both_clips_at_the_same_named_event() {
        let walk = gait("walk", 1.0, &[(0.2, "plant_l"), (0.7, "plant_r")]);
        let run = gait("run", 0.6, &[(0.05, "plant_l"), (0.35, "plant_r")]);
        let clips = [&walk, &run];

        // Leader = the walk (weight 0.7). Its play-head is exactly on `plant_r`.
        let times = warped_times(&clips, &[0.7, 0.3], 0.7).unwrap();
        assert!((times[0] - 0.7).abs() < 1e-6, "{times:?}");
        assert!(
            (times[1] - 0.35).abs() < 1e-6,
            "the run should be on its own plant_r, got {times:?}"
        );
        // Uniform phase would have put it at 0.7/1.0 * 0.6 = 0.42 — 0.07 s, more
        // than a fifth of a segment, past the plant. That gap is the foot slide.
        assert!((0.42f32 - times[1]).abs() > 0.05);

        // Leader = the run (weight 0.8), at its plant_l.
        let times = warped_times(&clips, &[0.2, 0.8], 0.05).unwrap();
        assert!((times[1] - 0.05).abs() < 1e-6, "{times:?}");
        assert!((times[0] - 0.2).abs() < 1e-6, "{times:?}");

        // Half way between two plants on the leader is half way between the same
        // two on the follower, whatever the segments' lengths are.
        let times = warped_times(&clips, &[1.0, 0.0], 0.45).unwrap();
        assert!((times[1] - 0.2).abs() < 1e-6, "{times:?}");
    }

    #[test]
    fn a_missing_group_or_a_lone_clip_declines_to_warp() {
        let walk = gait("walk", 1.0, &[(0.2, "plant_l"), (0.7, "plant_r")]);
        let bare = gait("bare", 0.6, &[]);
        // No common group → the caller keeps uniform phase.
        assert!(warped_times(&[&walk, &bare], &[0.5, 0.5], 0.3).is_none());
        // One clip is not a blend.
        assert!(warped_times(&[&walk], &[1.0], 0.3).is_none());
        // A group only one clip names is not common.
        let other = gait("other", 0.6, &[(0.1, "a"), (0.4, "b")]);
        let mut renamed = other.clone();
        for m in &mut renamed.markers {
            m.group = "hand".into();
        }
        assert!(warped_times(&[&walk, &renamed], &[0.5, 0.5], 0.3).is_none());
        assert_eq!(
            common_group([&walk, &other]).as_deref(),
            Some("foot"),
            "two clips that both name `foot` should align on it"
        );
    }

    /// A follower that does not carry the leader's marker **name** falls back to
    /// the segment index, and a follower with a different marker *count* still
    /// lands inside its own cycle rather than off the end.
    #[test]
    fn an_unmatched_name_falls_back_to_the_index_rather_than_refusing() {
        let leader = gait("l", 1.0, &[(0.0, "plant_l"), (0.5, "plant_r")]);
        let odd = gait("o", 0.9, &[(0.1, "a"), (0.4, "b"), (0.7, "c")]);
        let times = warped_times(&[&leader, &odd], &[1.0, 0.0], 0.25).unwrap();
        // Leader is half way through segment 0; the follower has no `plant_l`, so
        // it takes ITS segment 0 — [0.1, 0.4) — half way: 0.25.
        assert!((times[1] - 0.25).abs() < 1e-6, "{times:?}");
        // Every warped time is inside its own clip.
        assert!(times[1] >= 0.0 && times[1] < 0.9);
        // Leader in its last segment: the follower's index 1 is [0.4, 0.7).
        let times = warped_times(&[&leader, &odd], &[1.0, 0.0], 0.75).unwrap();
        assert!((times[1] - 0.55).abs() < 1e-6, "{times:?}");
    }
}
