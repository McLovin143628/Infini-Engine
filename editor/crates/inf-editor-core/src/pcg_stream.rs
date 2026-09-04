//! **The editor camera is a streaming source** (wave EDIT1, clause 1): which PCG
//! volumes the viewport should have evaluated, in what order, and which it should
//! let go of.
//!
//! # Why this exists
//!
//! The showcase island's 172 settlement blocks are authored as one `PcgVolume`
//! each ([`crate::island`]), and a volume's population is a `#[serde(skip)]`
//! cache — derived from the graph and the terrain, never written to the level.
//! The shipped player fills that cache on **cell activation**
//! (`cell_stream::sync_sim`'s phase 6b). The editor had one door and it was
//! manual: the `pcg_evaluate` command, one volume, when a person asked. So the
//! editor drew empty boxes where the player drew a city, and the author of a game
//! could not see the game they were building until they pressed Play. That is the
//! defect this wave exists for, in the user's own words: *"the buildings only
//! appear when the player presses play — how is the user supposed to build the
//! game?"*
//!
//! # Why a policy module and not cell streaming
//!
//! `inf_player::cell_stream::CellStreaming` is Ring 2, inside a binary crate, and
//! the editor cannot link it. It also would not help if it could: cell streaming
//! decides *which entities exist*, and the editor is deliberately single-document
//! — every one of those 172 volumes is already in the doc, already selectable,
//! already saved. The only thing missing is the **evaluation**. So this is not a
//! port of the streamer; it is the smaller thing the editor actually needs, and
//! it follows [`crate::terrain_stream`]'s shape exactly: the policy lives here in
//! Ring 1 (so Linux CI compiles and tests it), and the host only calls it.
//!
//! # The determinism seam
//!
//! Nothing here evaluates anything. It answers *which* and *in what order*, from
//! the camera and the level's own `PartitionSettings` — and the evaluation the
//! caller then runs is the same one `pcg_evaluate` runs, which composes through
//! the same `inf_pcg::compose_volume` and `population_of` the shipped player
//! goes through. A volume's population is a pure function of its graph, its seed,
//! its extent and the terrain under it; **it is not a function of where the
//! camera stands**, and that is what lets a camera drive this at all. The order
//! below is nearest-first tie-broken by GUID, so two sessions that visit a city
//! by different routes end with the same doc.
//!
//! # The budget
//!
//! [`EDITOR_PCG_STEP_BUDGET_MS`] bounds how much work a tick **starts**, not how
//! long one volume takes. A block of grammar buildings can cost tens of
//! milliseconds on its own and there is no way to preempt it half-built, so a
//! tick always evaluates at least one volume — otherwise a level whose blocks are
//! all dearer than the budget would make no progress for ever — and then stops as
//! soon as the clock has passed the ceiling. That is the honest reading of the
//! number and the one [`PcgStreamStats::budget_ms`] reports.

use std::collections::BTreeSet;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use crate::scene::serialize::PartitionSettings;

/// How long one streaming tick may **start** new volume evaluations for, in
/// milliseconds.
///
/// Eight is half a 60 Hz frame. The evaluation holds the scene document's lock
/// (deliberately — `pcg_evaluate` has held it for a whole evaluation since
/// Hardening Wave E, because a half-evaluated document is not a document), so
/// this is time the viewport's projection cannot run, and the ceiling is what
/// keeps "the editor fills the city in" from reading as "the editor froze".
///
/// It is a ceiling on *starting*, not a preemption: see the module doc.
pub const EDITOR_PCG_STEP_BUDGET_MS: f64 = 8.0;

/// How far past the prefetch radius a volume must fall before its population is
/// dropped, as a multiple of that radius.
///
/// Releasing at exactly the radius makes a camera nudged back and forth across
/// the boundary re-evaluate the same block for ever, which is both a stutter and
/// a lie about the budget. A quarter of the radius is wider than any camera
/// jitter and narrower than a deliberate move.
pub const EDITOR_PCG_RELEASE_HYSTERESIS: f64 = 1.25;

/// Hard ceiling on how many volumes may hold a population at once — the memory
/// bound.
///
/// A settlement block's population is thousands of `ScatteredInstance`s and
/// `ScatteredSolid`s; the island carries 172 blocks and a bigger world carries
/// more, so "everything within the radius" needs a second bound that does not
/// depend on how generous an author set the radius. When the set is over the
/// ceiling the **farthest** are released first, which is the same order they
/// would leave by if the camera simply moved away.
pub const EDITOR_PCG_MAX_EVALUATED: usize = 256;

/// The two radii a camera streams by, in metres.
///
/// Derived from the **level's own** [`PartitionSettings`] — the same numbers
/// `inf_player::cell_stream::sync_sim` derives its activation and prefetch sets
/// from — so the editor's working set is the player's, not a second opinion. The
/// editor preference scales both together rather than replacing them, so
/// widening the editor's view can never make it disagree with the player about
/// *what a block contains*, only about *how far away* it bothers to look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcgStreamRadii {
    /// Volumes within this distance of the camera are evaluated first.
    pub activation_m: f64,
    /// Volumes within this distance are evaluated after those; beyond
    /// `prefetch_m × `[`EDITOR_PCG_RELEASE_HYSTERESIS`] they are released.
    pub prefetch_m: f64,
}

impl PcgStreamRadii {
    /// The radii a level's partition block implies, scaled by an editor
    /// preference (`1.0` = exactly the player's).
    ///
    /// The guards mirror `inf_scene::partition::PartitionSettings`'
    /// `effective_activation_radius` / `effective_prefetch_margin`: a
    /// hand-edited or corrupt level must not be able to produce a NaN radius
    /// here and a finite one in the player.
    pub fn from_partition(p: &PartitionSettings, scale: f64) -> Self {
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        let activation = if p.activation_radius_m.is_finite() && p.activation_radius_m >= 0.0 {
            p.activation_radius_m
        } else {
            DEFAULT_ACTIVATION_RADIUS_M
        };
        let margin = if p.prefetch_margin_m.is_finite() && p.prefetch_margin_m >= 0.0 {
            p.prefetch_margin_m
        } else {
            DEFAULT_PREFETCH_MARGIN_M
        };
        Self {
            activation_m: activation * scale,
            prefetch_m: (activation + margin) * scale,
        }
    }
}

/// MIRROR of `inf_scene::partition::DEFAULT_ACTIVATION_RADIUS_M`. This crate does
/// not depend on `inf-scene` — its [`PartitionSettings`] is a hand-written record
/// mirror, pinned by `partition_settings_mirror_matches_the_runtime_defaults` —
/// so the fallback the guard above uses is spelled here beside it.
const DEFAULT_ACTIVATION_RADIUS_M: f64 = 256.0;
/// MIRROR of `inf_scene::partition::DEFAULT_PREFETCH_MARGIN_M`.
const DEFAULT_PREFETCH_MARGIN_M: f64 = 256.0;

/// One PCG volume as the policy sees it: where it is, how big it is, and whether
/// it currently holds a population.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeCandidate {
    pub guid: Uuid,
    /// World-space centre (the entity's `GlobalTransform` translation).
    pub centre: DVec3,
    /// The volume's XZ half-extent — `PcgVolume::extent`.
    pub half_extent: DVec2,
    /// Whether `PcgVolume::evaluated` is non-empty right now.
    pub populated: bool,
}

impl VolumeCandidate {
    /// Distance from `camera` to this volume's XZ box, in metres — **zero when
    /// the camera is inside it**.
    ///
    /// Distance to the box and not to the centre, because a settlement block is
    /// tens of metres across and an author standing in the middle of one is as
    /// close to it as it is possible to be. Measuring to the centre would rank a
    /// large block the camera is inside behind a small one across the street.
    /// Y is ignored for the same reason the partition ignores it: a level
    /// streams on the ground plane.
    pub fn distance_m(&self, camera: DVec3) -> f64 {
        let half = DVec2::new(self.half_extent.x.abs(), self.half_extent.y.abs());
        let d = DVec2::new(
            (camera.x - self.centre.x).abs() - half.x,
            (camera.z - self.centre.z).abs() - half.y,
        );
        DVec2::new(d.x.max(0.0), d.y.max(0.0)).length()
    }
}

/// What one tick should do.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PcgStreamPlan {
    /// Volumes to evaluate, **nearest first, ties broken by GUID**. The caller
    /// walks this until [`EDITOR_PCG_STEP_BUDGET_MS`] is spent, always doing at
    /// least one.
    pub evaluate: Vec<Uuid>,
    /// Volumes whose population should be dropped — beyond the release radius,
    /// or over [`EDITOR_PCG_MAX_EVALUATED`]. **Farthest first**, so a caller that
    /// stops early has released the ones it wanted least.
    pub release: Vec<Uuid>,
    /// How many candidates are inside the prefetch radius, populated or not —
    /// the denominator the "Loading world…" indicator counts against.
    pub in_radius: usize,
    /// How many of those already hold a population.
    pub in_radius_populated: usize,
}

impl PcgStreamPlan {
    /// Whether the editor is still catching up with its own camera — what the
    /// viewport's "Loading world…" indicator is showing.
    pub fn is_loading(&self) -> bool {
        !self.evaluate.is_empty()
    }
}

/// Decide what to evaluate and what to release.
///
/// Pure: same camera, same candidates, same radii ⇒ same plan, on any machine.
/// The sort is `(distance, guid)` rather than distance alone because two blocks
/// of a grid city are frequently equidistant to a metre, and an unstable order
/// there would make the doc depend on `HashMap` iteration.
pub fn plan(
    camera: DVec3,
    candidates: &[VolumeCandidate],
    radii: PcgStreamRadii,
    max_evaluated: usize,
) -> PcgStreamPlan {
    let release_m = radii.prefetch_m * EDITOR_PCG_RELEASE_HYSTERESIS;
    let mut want: Vec<(f64, Uuid)> = Vec::new();
    let mut keep: Vec<(f64, Uuid)> = Vec::new();
    let mut drop: Vec<(f64, Uuid)> = Vec::new();
    let mut in_radius = 0usize;
    let mut in_radius_populated = 0usize;

    for c in candidates {
        let d = c.distance_m(camera);
        if d <= radii.prefetch_m {
            in_radius += 1;
            if c.populated {
                in_radius_populated += 1;
                keep.push((d, c.guid));
            } else {
                want.push((d, c.guid));
            }
        } else if c.populated {
            if d > release_m {
                drop.push((d, c.guid));
            } else {
                // Inside the hysteresis band: already paid for, not yet far
                // enough to be worth paying for again.
                keep.push((d, c.guid));
            }
        }
    }

    want.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    drop.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

    // The memory bound, applied after the radius rule so a generous radius
    // cannot outvote it. `keep` is what would survive the radius rule; anything
    // past the ceiling leaves, farthest first, and the ceiling counts what will
    // be populated once this tick's `evaluate` list has run.
    if max_evaluated > 0 {
        // NEAREST first, so `pop` takes the farthest. Sorting this the other way
        // round and popping is the same two lines and releases the wrong end —
        // it cost two of the tests below on the first cut.
        keep.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let budgeted = max_evaluated.saturating_sub(want.len().min(max_evaluated));
        while keep.len() > budgeted {
            let Some(far) = keep.pop() else { break };
            drop.push(far);
        }
        drop.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        want.truncate(max_evaluated);
    }

    PcgStreamPlan {
        evaluate: want.into_iter().map(|(_, g)| g).collect(),
        release: drop.into_iter().map(|(_, g)| g).collect(),
        in_radius,
        in_radius_populated,
    }
}

/// What the "Streaming" overlay reads (clause 5) and what the throttled
/// `tracing` line prints.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PcgStreamStats {
    /// Volumes in the document that carry a PCG graph at all.
    pub volumes: usize,
    /// Of those, how many are inside the prefetch radius.
    pub in_radius: usize,
    /// Of those, how many hold a population.
    pub in_radius_populated: usize,
    /// How many hold a population anywhere in the document.
    pub populated: usize,
    /// Volumes evaluated by the last tick.
    pub last_tick_evaluated: usize,
    /// Wall time the last tick spent inside evaluation, milliseconds.
    pub last_tick_ms: f64,
    /// The ceiling that tick was measured against — [`EDITOR_PCG_STEP_BUDGET_MS`]
    /// unless a caller overrode it. Reported so the overlay shows the budget
    /// beside the spend rather than a bare number.
    pub budget_ms: f64,
    /// Volumes evaluated and released since the streamer was created.
    pub evaluated_total: u64,
    pub released_total: u64,
    /// The radii in force.
    pub activation_m: f64,
    pub prefetch_m: f64,
}

impl PcgStreamStats {
    /// One line for the `tracing` seam — the [`crate::terrain_stream`] precedent:
    /// no new panel, no new IPC channel.
    pub fn summary(&self) -> String {
        format!(
            "pcg: {}/{} populated in radius ({} of {} in the level), last tick {} vol / \
             {:.2} ms of {:.2}, totals +{} -{}, radii {:.0}/{:.0} m",
            self.in_radius_populated,
            self.in_radius,
            self.populated,
            self.volumes,
            self.last_tick_evaluated,
            self.last_tick_ms,
            self.budget_ms,
            self.evaluated_total,
            self.released_total,
            self.activation_m,
            self.prefetch_m,
        )
    }
}

/// The editor's PCG streaming state — the sibling of
/// [`crate::terrain_stream::EditorTerrainStreams`], and deliberately the same
/// shape: it owns the policy and the counters, and the host owns the calling.
#[derive(Debug, Clone)]
pub struct EditorPcgStreams {
    enabled: bool,
    radius_scale: f64,
    budget_ms: f64,
    max_evaluated: usize,
    /// The guids this streamer has evaluated and not released. Kept beside the
    /// document's own `populated` bit rather than instead of it: the document is
    /// the truth (a person can evaluate a volume by hand, or open a new level),
    /// and this set is only how the streamer knows what *it* is responsible for.
    owned: BTreeSet<Uuid>,
    stats: PcgStreamStats,
}

impl Default for EditorPcgStreams {
    fn default() -> Self {
        Self::new()
    }
}

impl EditorPcgStreams {
    pub fn new() -> Self {
        Self {
            // **On by default.** The whole point of the wave is that an author
            // who opens a level sees it; a feature that has to be switched on
            // first would leave the reported defect exactly where it was.
            enabled: true,
            radius_scale: 1.0,
            budget_ms: EDITOR_PCG_STEP_BUDGET_MS,
            max_evaluated: EDITOR_PCG_MAX_EVALUATED,
            owned: BTreeSet::new(),
            stats: PcgStreamStats {
                budget_ms: EDITOR_PCG_STEP_BUDGET_MS,
                ..PcgStreamStats::default()
            },
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    /// The editor preference: how much wider than the player's own radii this
    /// camera looks. Clamped to a sane band — a scale of zero would stream
    /// nothing while claiming to be on, and an unbounded one would evaluate a
    /// continent on the first tick.
    pub fn set_radius_scale(&mut self, scale: f64) {
        self.radius_scale = if scale.is_finite() {
            scale.clamp(0.1, 8.0)
        } else {
            1.0
        };
    }

    pub fn radius_scale(&self) -> f64 {
        self.radius_scale
    }

    pub fn set_budget_ms(&mut self, ms: f64) {
        self.budget_ms = if ms.is_finite() && ms > 0.0 {
            ms.clamp(0.5, 100.0)
        } else {
            EDITOR_PCG_STEP_BUDGET_MS
        };
        self.stats.budget_ms = self.budget_ms;
    }

    pub fn budget_ms(&self) -> f64 {
        self.budget_ms
    }

    pub fn radii(&self, p: &PartitionSettings) -> PcgStreamRadii {
        PcgStreamRadii::from_partition(p, self.radius_scale)
    }

    /// Plan one tick. Returns an empty plan when streaming is off, so the caller
    /// takes one branch rather than two.
    pub fn plan_tick(
        &mut self,
        camera: DVec3,
        candidates: &[VolumeCandidate],
        partition: &PartitionSettings,
    ) -> PcgStreamPlan {
        let radii = self.radii(partition);
        self.stats.volumes = candidates.len();
        self.stats.populated = candidates.iter().filter(|c| c.populated).count();
        self.stats.activation_m = radii.activation_m;
        self.stats.prefetch_m = radii.prefetch_m;
        if !self.enabled {
            self.stats.in_radius = 0;
            self.stats.in_radius_populated = 0;
            return PcgStreamPlan::default();
        }
        let plan = plan(camera, candidates, radii, self.max_evaluated);
        self.stats.in_radius = plan.in_radius;
        self.stats.in_radius_populated = plan.in_radius_populated;
        plan
    }

    /// Record what a tick actually did. `ms` is the wall time spent inside
    /// evaluation, which is the number the budget is about.
    pub fn note_tick(&mut self, evaluated: &[Uuid], released: &[Uuid], ms: f64) {
        for g in evaluated {
            self.owned.insert(*g);
        }
        for g in released {
            self.owned.remove(g);
        }
        self.stats.last_tick_evaluated = evaluated.len();
        self.stats.last_tick_ms = ms;
        self.stats.evaluated_total += evaluated.len() as u64;
        self.stats.released_total += released.len() as u64;
    }

    /// Forget everything — a new document, or a closed project. Does not touch
    /// the document: the caller owns that.
    pub fn clear(&mut self) {
        self.owned.clear();
        self.stats = PcgStreamStats {
            budget_ms: self.budget_ms,
            ..PcgStreamStats::default()
        };
    }

    pub fn owned(&self) -> &BTreeSet<Uuid> {
        &self.owned
    }

    pub fn stats(&self) -> PcgStreamStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guid(n: u128) -> Uuid {
        Uuid::from_u128(n)
    }

    fn vol(n: u128, x: f64, z: f64, half: f64, populated: bool) -> VolumeCandidate {
        VolumeCandidate {
            guid: guid(n),
            centre: DVec3::new(x, 0.0, z),
            half_extent: DVec2::splat(half),
            populated,
        }
    }

    fn radii(a: f64, p: f64) -> PcgStreamRadii {
        PcgStreamRadii {
            activation_m: a,
            prefetch_m: p,
        }
    }

    #[test]
    fn the_camera_inside_a_block_is_zero_metres_from_it() {
        let v = vol(1, 0.0, 0.0, 40.0, false);
        assert_eq!(v.distance_m(DVec3::new(10.0, 3.0, -20.0)), 0.0);
        // ...and height does not count: a camera 500 m up over the middle of a
        // block is still over it. The partition streams on the ground plane.
        assert_eq!(v.distance_m(DVec3::new(0.0, 500.0, 0.0)), 0.0);
        // Just outside one face: the gap, not the centre distance.
        assert!((v.distance_m(DVec3::new(50.0, 0.0, 0.0)) - 10.0).abs() < 1e-9);
    }

    #[test]
    fn the_nearest_unpopulated_volume_is_evaluated_first_and_ties_break_by_guid() {
        let cands = [
            vol(3, 300.0, 0.0, 10.0, false),
            vol(2, 100.0, 0.0, 10.0, false),
            // Exactly as far as guid 2, on the other axis — the tie.
            vol(1, 0.0, 100.0, 10.0, false),
        ];
        let p = plan(DVec3::ZERO, &cands, radii(256.0, 512.0), 64);
        assert_eq!(p.evaluate, vec![guid(1), guid(2), guid(3)]);
        assert!(p.release.is_empty());
        assert_eq!(p.in_radius, 3);
        assert_eq!(p.in_radius_populated, 0);
        assert!(p.is_loading());
    }

    #[test]
    fn a_populated_volume_is_not_re_evaluated_and_counts_toward_the_indicator() {
        let cands = [
            vol(1, 50.0, 0.0, 10.0, true),
            vol(2, 60.0, 0.0, 10.0, false),
        ];
        let p = plan(DVec3::ZERO, &cands, radii(256.0, 512.0), 64);
        assert_eq!(p.evaluate, vec![guid(2)]);
        assert_eq!(p.in_radius, 2);
        assert_eq!(p.in_radius_populated, 1);
    }

    /// The hysteresis band is the whole reason a camera nudged across the
    /// boundary does not re-evaluate for ever.
    #[test]
    fn a_volume_just_past_the_radius_is_kept_and_one_well_past_it_is_released() {
        let r = radii(256.0, 512.0);
        let just_past = vol(1, 600.0, 0.0, 0.0, true); // 600 m: past 512, inside 640
        let well_past = vol(2, 900.0, 0.0, 0.0, true);
        let p = plan(DVec3::ZERO, &[just_past, well_past], r, 64);
        assert!(p.evaluate.is_empty());
        assert_eq!(p.release, vec![guid(2)]);
        // The kept one is not counted as in-radius: it is paid-for residue, not
        // part of what the indicator is waiting on.
        assert_eq!(p.in_radius, 0);
    }

    #[test]
    fn the_memory_ceiling_releases_the_farthest_first() {
        // Five populated blocks in a line, all well inside the radius, against a
        // ceiling of three.
        let cands: Vec<VolumeCandidate> = (0..5)
            .map(|i| vol(i as u128 + 1, 10.0 * (i as f64 + 1.0), 0.0, 1.0, true))
            .collect();
        let p = plan(DVec3::ZERO, &cands, radii(256.0, 512.0), 3);
        assert!(p.evaluate.is_empty());
        // The two farthest — guids 5 and 4, in that order.
        assert_eq!(p.release, vec![guid(5), guid(4)]);
    }

    /// A ceiling has to bound what a tick is about to *add*, not only what is
    /// already there, or the first tick in a dense city blows straight past it.
    #[test]
    fn the_ceiling_counts_the_volumes_this_tick_is_about_to_evaluate() {
        let mut cands: Vec<VolumeCandidate> = (0..4)
            .map(|i| vol(i as u128 + 1, 10.0 * (i as f64 + 1.0), 0.0, 1.0, true))
            .collect();
        cands.push(vol(9, 5.0, 0.0, 1.0, false)); // the nearest, unevaluated
        let p = plan(DVec3::ZERO, &cands, radii(256.0, 512.0), 3);
        assert_eq!(p.evaluate, vec![guid(9)]);
        // Room for two survivors beside the one arriving: 4 and 3 leave.
        assert_eq!(p.release, vec![guid(4), guid(3)]);
    }

    #[test]
    fn the_radii_come_from_the_levels_own_partition_block_and_scale_together() {
        let p = PartitionSettings {
            enabled: true,
            cell_size_m: 256.0,
            activation_radius_m: 300.0,
            prefetch_margin_m: 100.0,
            ..PartitionSettings::default()
        };
        let r = PcgStreamRadii::from_partition(&p, 1.0);
        assert_eq!(r.activation_m, 300.0);
        assert_eq!(r.prefetch_m, 400.0);
        let wide = PcgStreamRadii::from_partition(&p, 2.0);
        assert_eq!(wide.activation_m, 600.0);
        assert_eq!(wide.prefetch_m, 800.0);
    }

    /// The guard the runtime's `effective_*` helpers apply, applied here — a
    /// corrupt level must not stream differently in the two hosts.
    #[test]
    fn a_nonsense_partition_block_falls_back_to_the_runtime_defaults() {
        let p = PartitionSettings {
            activation_radius_m: f64::NAN,
            prefetch_margin_m: -5.0,
            ..PartitionSettings::default()
        };
        let r = PcgStreamRadii::from_partition(&p, f64::NAN);
        assert_eq!(r.activation_m, DEFAULT_ACTIVATION_RADIUS_M);
        assert_eq!(
            r.prefetch_m,
            DEFAULT_ACTIVATION_RADIUS_M + DEFAULT_PREFETCH_MARGIN_M
        );
    }

    #[test]
    fn a_disabled_streamer_plans_nothing_but_still_counts_the_level() {
        let mut s = EditorPcgStreams::new();
        s.set_enabled(false);
        let cands = [vol(1, 0.0, 0.0, 10.0, false)];
        let p = s.plan_tick(DVec3::ZERO, &cands, &PartitionSettings::default());
        assert!(p.evaluate.is_empty() && !p.is_loading());
        assert_eq!(s.stats().volumes, 1);
        assert_eq!(s.stats().in_radius, 0);
    }

    #[test]
    fn the_preference_is_clamped_rather_than_trusted() {
        let mut s = EditorPcgStreams::new();
        s.set_radius_scale(0.0);
        assert_eq!(s.radius_scale(), 0.1);
        s.set_radius_scale(1000.0);
        assert_eq!(s.radius_scale(), 8.0);
        s.set_radius_scale(f64::NAN);
        assert_eq!(s.radius_scale(), 1.0);
        s.set_budget_ms(0.0);
        assert_eq!(s.budget_ms(), EDITOR_PCG_STEP_BUDGET_MS);
        s.set_budget_ms(4.0);
        assert_eq!(s.budget_ms(), 4.0);
        assert_eq!(s.stats().budget_ms, 4.0);
    }

    #[test]
    fn the_counters_track_what_the_ticks_did() {
        let mut s = EditorPcgStreams::new();
        s.note_tick(&[guid(1), guid(2)], &[], 3.5);
        assert_eq!(s.stats().evaluated_total, 2);
        assert_eq!(s.stats().last_tick_ms, 3.5);
        assert_eq!(s.owned().len(), 2);
        s.note_tick(&[], &[guid(1)], 0.0);
        assert_eq!(s.stats().released_total, 1);
        assert_eq!(s.owned().len(), 1);
        s.clear();
        assert!(s.owned().is_empty());
        assert_eq!(s.stats().evaluated_total, 0);
    }
}
