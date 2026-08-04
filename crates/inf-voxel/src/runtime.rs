//! **Runtime carving** (P21.4): the one rule both hosts run when *gameplay* —
//! not an author — digs.
//!
//! The editor's dig path ([`inf_editor_core::scene::undo::SceneDoc::edit_dig`])
//! is a transaction: it judges size, store, volume and the inline-terrain verdict
//! before a sample moves, cuts into a `CarveStroke`, displaces the spoil and
//! commits one reversible `EditCommand`. **None of that shape survives into a
//! fixed step.** A Blueprint carve has no undo entry, no author to refuse it, no
//! toolbar to report a verdict on, and it runs sixty times a second inside a
//! budget measured in milliseconds. So this module is the smaller, harder rule:
//!
//! * it is a **pure function of the op and the volume** — no camera, no
//!   residency, no session history;
//! * it **cannot fail into an inconsistent world** — every refusal happens before
//!   the first sample moves and returns zero;
//! * it is **idempotent**, because [`VoxelOp`] is (`min`/`max` on a clamped band),
//!   so a replayed step after a rollback lands on the same bytes;
//! * and it lives **here**, in Ring 0, rather than twice in two hosts, because
//!   `terrain.height_at` already taught this repository what happens when a
//!   gameplay answer is written down in two places.
//!
//! # The three refusals, and why each returns zero rather than erroring
//!
//! A blueprint node is not a transaction. Failing the handler on a bad carve
//! would take down the rest of the Tick body — the movement, the animation, the
//! sound — for an op the author can fix by typing a smaller radius. So a refusal
//! is a **zero-volume result plus a report**, and the caller logs it:
//!
//! * [`RuntimeCarveOutcome::NoVolume`] — the entity carries no seeded volume
//!   (no `VoxelVolume`, no `.inf_voxel` asset, or a dangling ref). The world has
//!   no rock there to remove.
//! * [`RuntimeCarveOutcome::NotPermitted`] — the volume's
//!   [`VoxelVolume::runtime_carve`] is `false`. **This is the gate the field was
//!   frozen for**: an author withholding permission from load-bearing geometry
//!   gets a deterministic no-op, identically in preview and in the shipped build,
//!   rather than a carve that depends on which node happened to run.
//! * [`RuntimeCarveOutcome::TooLarge`] — the op would touch more than
//!   [`MAX_RUNTIME_CARVE_SAMPLES`] samples. The bound is checked with
//!   [`VoxelShape::affected_sample_count`], which never *under*-estimates, so the
//!   refusal happens before any allocation.
//!
//! [`VoxelVolume::runtime_carve`]: https://docs.rs/inf-ecs (see `inf_ecs::components::VoxelVolume`)
//! [`inf_editor_core::scene::undo::SceneDoc::edit_dig`]: https://docs.rs/inf-editor-core

use std::collections::BTreeMap;

use crate::chunk::MATERIAL_COUNT;
use crate::data::VoxelData;
use crate::ops::{OpReport, VoxelOp};

/// Upper bound on the samples **one** runtime carve or fill may touch.
///
/// The editor's ceiling is `MAX_DIG_SAMPLES = 2_000_000`, and a dig at it spends
/// ≈1.3 s — which is 78 fixed steps. A gameplay carve runs *inside* a step, so it
/// gets a bound two orders of magnitude smaller: 65 536 samples is a 40-voxel
/// cube, or a sphere of radius 20 voxels (10 m at the default 0.5 m voxel), which
/// is a grenade crater rather than a quarry. A game that wants a quarry issues
/// several carves across several steps, which is also what keeps the frame budget
/// honest.
///
/// Stated as a **sample count, not a radius**, because the same radius is a
/// different bill at a different `voxel_size_m` — and it is the bill, not the
/// radius, that the step has to pay.
pub const MAX_RUNTIME_CARVE_SAMPLES: u64 = 65_536;

/// Why a [`runtime_carve`] did or did not move any rock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeCarveOutcome {
    /// The op ran. (It may still have moved nothing — carving air is legal and
    /// reports `0`.)
    Applied,
    /// The entity has no seeded volume in this sim's map.
    NoVolume,
    /// The volume's `runtime_carve` flag is `false`.
    NotPermitted,
    /// The op would touch more than [`MAX_RUNTIME_CARVE_SAMPLES`] samples.
    TooLarge {
        /// What the op would have touched, for the log line.
        samples: u64,
    },
    /// The shape is degenerate (a non-finite or non-positive extent).
    Invalid,
}

impl RuntimeCarveOutcome {
    /// Whether the op reached the field at all.
    #[inline]
    pub fn applied(&self) -> bool {
        matches!(self, RuntimeCarveOutcome::Applied)
    }

    /// The log line a host emits for a refusal, or `None` when the op ran.
    ///
    /// One text, shared by both hosts, so a refusal reads identically in the
    /// editor's Output Log and in the shipped player's tracing stream — the same
    /// reason the ground query is one function.
    pub fn refusal(&self, op_name: &str) -> Option<String> {
        match self {
            RuntimeCarveOutcome::Applied => None,
            RuntimeCarveOutcome::NoVolume => Some(format!(
                "{op_name}: no voxel volume on that entity — nothing to carve"
            )),
            RuntimeCarveOutcome::NotPermitted => Some(format!(
                "{op_name}: refused — this volume's `runtime_carve` is off"
            )),
            RuntimeCarveOutcome::TooLarge { samples } => Some(format!(
                "{op_name}: refused — {samples} samples exceeds the runtime ceiling \
                 of {MAX_RUNTIME_CARVE_SAMPLES}"
            )),
            RuntimeCarveOutcome::Invalid => Some(format!("{op_name}: refused — degenerate shape")),
        }
    }
}

/// What one runtime carve or fill did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeCarveReport {
    /// Why it did (or did not) run.
    pub outcome: RuntimeCarveOutcome,
    /// Samples that went solid → empty, by the material they carried.
    pub carved: [u64; MATERIAL_COUNT],
    /// Samples that went empty → solid, by the material they now carry.
    pub filled: [u64; MATERIAL_COUNT],
    /// Samples whose distance or material changed at all.
    pub touched: u64,
}

impl RuntimeCarveReport {
    /// A refusal: nothing moved.
    fn refused(outcome: RuntimeCarveOutcome) -> Self {
        Self {
            outcome,
            carved: [0; MATERIAL_COUNT],
            filled: [0; MATERIAL_COUNT],
            touched: 0,
        }
    }

    fn from_op(report: OpReport) -> Self {
        Self {
            outcome: RuntimeCarveOutcome::Applied,
            carved: report.carved,
            filled: report.filled,
            touched: report.touched,
        }
    }

    /// Total samples removed, summed over materials.
    #[inline]
    pub fn total_carved(&self) -> u64 {
        self.carved.iter().sum()
    }

    /// Total samples added, summed over materials.
    #[inline]
    pub fn total_filled(&self) -> u64 {
        self.filled.iter().sum()
    }

    /// **The number a Blueprint sees**: volume removed, in **cubic metres**.
    ///
    /// Exact by construction — an integer sample count times the cube of the
    /// volume's own `voxel_size_m` — so the same trace produces the same `f64`
    /// bits on every host, which is what the PIE == shipping gate compares.
    ///
    /// m³ rather than a raw count because the units doctrine says SI everywhere
    /// (`docs/memos/units-doctrine.md`): a count is a number whose meaning changes
    /// with `voxel_size_m`, and a gameplay counter ("dig out 20 m³ of ore") should
    /// not silently mean something different when an author re-authors the cave at
    /// a finer grid. Divide by `voxel_size_m³` to get back the count.
    #[inline]
    pub fn removed_m3(&self, voxel_size_m: f64) -> f64 {
        self.total_carved() as f64 * voxel_size_m * voxel_size_m * voxel_size_m
    }

    /// The fill mirror of [`removed_m3`](Self::removed_m3).
    #[inline]
    pub fn added_m3(&self, voxel_size_m: f64) -> f64 {
        self.total_filled() as f64 * voxel_size_m * voxel_size_m * voxel_size_m
    }
}

/// Apply one gameplay carve/fill to the **sim's** volume map.
///
/// `permitted` is the entity's `VoxelVolume::runtime_carve`, read by the caller
/// (this crate must not depend on `inf-ecs`). `key` is the **entity** id the map
/// is keyed by — the same key `resolve_voxel_volumes` inserts under, in both
/// hosts.
///
/// # Why there is no store parameter (the paging rule, satisfied structurally)
///
/// P21.3's law is *the dig path pages its footprint before it reads it*, because a
/// non-resident chunk reads as empty and conservation balances anyway. The sim map
/// has **no cold store at all**: [`crate::sim_volume`] decodes a `.inf_voxel`
/// fully resident precisely so a fixed step cannot observe residency, and
/// [`VoxelData::get_or_create_chunk`] materializes any key an op reaches. There is
/// therefore nothing to page and nothing that can be silently missed — the rule is
/// satisfied by there being no cold half, not by a call.
///
/// The caller still owes the **heightfield** half: a cut that crosses the surface
/// must open the terrain's hole mask through [`crate::coupling`], or a game digs a
/// cave with no mouth. That lives in the hosts because the terrain is an ECS
/// component and this crate does not know what an entity is.
pub fn runtime_carve<K: Ord>(
    volumes: &mut BTreeMap<K, VoxelData>,
    key: &K,
    permitted: bool,
    op: &VoxelOp,
) -> RuntimeCarveReport {
    if !op.shape.is_valid() {
        return RuntimeCarveReport::refused(RuntimeCarveOutcome::Invalid);
    }
    // Permission is checked BEFORE the volume lookup so a locked volume reads the
    // same whether or not this sim happened to seed it — a refusal must not depend
    // on load order.
    if !permitted {
        return RuntimeCarveReport::refused(RuntimeCarveOutcome::NotPermitted);
    }
    let Some(data) = volumes.get_mut(key) else {
        return RuntimeCarveReport::refused(RuntimeCarveOutcome::NoVolume);
    };
    let samples = op.shape.affected_sample_count(data.voxel_size_m());
    if samples > MAX_RUNTIME_CARVE_SAMPLES {
        return RuntimeCarveReport::refused(RuntimeCarveOutcome::TooLarge { samples });
    }
    // `apply_op` allocates a `VoxelDelta`; a fixed step has no undo stack to put
    // one on, so the builder-free door is deliberate — the delta would be built
    // and dropped sixty times a second.
    let mut builder = crate::ops::VoxelDeltaBuilder::new();
    RuntimeCarveReport::from_op(data.apply_op_into(op, &mut builder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkKey, VoxelChunk};
    use crate::ops::VoxelShape;

    /// A 1 m-voxel volume of solid rock spanning one chunk at the origin.
    fn rock() -> BTreeMap<u8, VoxelData> {
        let mut v = VoxelData::new(1.0);
        v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(1));
        v.clear_dirty();
        BTreeMap::from([(0u8, v)])
    }

    fn sphere(r: f64) -> VoxelOp {
        VoxelOp::carve(VoxelShape::Sphere {
            center: glam::DVec3::new(8.0, 8.0, 8.0),
            radius_m: r,
        })
    }

    #[test]
    fn a_permitted_carve_removes_rock_and_reports_cubic_metres() {
        let mut vols = rock();
        let r = runtime_carve(&mut vols, &0u8, true, &sphere(3.0));
        assert_eq!(r.outcome, RuntimeCarveOutcome::Applied);
        assert!(
            r.total_carved() > 0,
            "a 3 m ball in solid rock removes rock"
        );
        // Material 1 is what the chunk was made of, so the count is on that index
        // alone — an exact per-material tally, not a lump sum.
        assert_eq!(r.carved[1], r.total_carved());
        assert_eq!(r.carved[0], 0);
        // 1 m voxels ⇒ m³ and count agree numerically, which is the one grid where
        // the conversion cannot hide a factor.
        assert_eq!(r.removed_m3(1.0), r.total_carved() as f64);
        // …and at 0.5 m voxels the same count is an eighth of the volume.
        assert_eq!(r.removed_m3(0.5), r.total_carved() as f64 / 8.0);
    }

    /// **THE GATE the field was frozen for.** `runtime_carve == false` is a
    /// deterministic no-op: nothing moves, and the volume is byte-identical after.
    #[test]
    fn a_locked_volume_is_a_deterministic_no_op() {
        let mut vols = rock();
        let before = vols.clone();
        let r = runtime_carve(&mut vols, &0u8, false, &sphere(3.0));
        assert_eq!(r.outcome, RuntimeCarveOutcome::NotPermitted);
        assert_eq!(r.total_carved(), 0);
        assert_eq!(r.removed_m3(1.0), 0.0);
        assert_eq!(vols, before, "a refused carve moved the field");
        // Non-vacuity: the very same op on the very same volume DOES cut when
        // permitted, so the refusal is the flag and not the fixture.
        let mut vols2 = rock();
        assert!(runtime_carve(&mut vols2, &0u8, true, &sphere(3.0)).total_carved() > 0);
        assert_ne!(vols2, before);
    }

    #[test]
    fn an_unseeded_entity_and_a_degenerate_shape_both_refuse() {
        let mut vols = rock();
        assert_eq!(
            runtime_carve(&mut vols, &7u8, true, &sphere(3.0)).outcome,
            RuntimeCarveOutcome::NoVolume
        );
        assert_eq!(
            runtime_carve(&mut vols, &0u8, true, &sphere(-1.0)).outcome,
            RuntimeCarveOutcome::Invalid
        );
        assert_eq!(
            runtime_carve(&mut vols, &0u8, true, &sphere(f64::NAN)).outcome,
            RuntimeCarveOutcome::Invalid
        );
    }

    /// The ceiling refuses **before** anything moves, and it is a function of the
    /// grid rather than of the radius.
    #[test]
    fn the_sample_ceiling_refuses_before_the_first_sample_moves() {
        let mut vols = rock();
        let before = vols.clone();
        // 65 536 samples is a 40-voxel cube; a 40 m ball at 1 m voxels is far past
        // it. (`affected_sample_count` bounds the AABB, so this is ~82³.)
        let huge = sphere(40.0);
        let n = huge.shape.affected_sample_count(1.0);
        assert!(n > MAX_RUNTIME_CARVE_SAMPLES, "{n}");
        assert_eq!(
            runtime_carve(&mut vols, &0u8, true, &huge).outcome,
            RuntimeCarveOutcome::TooLarge { samples: n }
        );
        assert_eq!(vols, before);

        // The SAME radius at a coarser grid is inside the ceiling — the bound is
        // the bill, not the radius.
        let mut coarse = BTreeMap::from([(0u8, {
            let mut v = VoxelData::new(8.0);
            v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(1));
            v.clear_dirty();
            v
        })]);
        assert!(huge.shape.affected_sample_count(8.0) <= MAX_RUNTIME_CARVE_SAMPLES);
        assert_eq!(
            runtime_carve(&mut coarse, &0u8, true, &huge).outcome,
            RuntimeCarveOutcome::Applied
        );
    }

    /// Idempotence: the second identical carve moves nothing and leaves the same
    /// bytes — which is what makes a replayed step after a rollback safe.
    #[test]
    fn a_repeated_carve_is_idempotent() {
        let mut vols = rock();
        let first = runtime_carve(&mut vols, &0u8, true, &sphere(3.0));
        let after_first = vols.clone();
        let second = runtime_carve(&mut vols, &0u8, true, &sphere(3.0));
        assert!(first.total_carved() > 0);
        assert_eq!(second.total_carved(), 0, "the rock was already gone");
        assert_eq!(vols, after_first);
    }

    /// A fill puts material back and reports it on the material's own index.
    #[test]
    fn a_fill_reports_added_volume_by_material() {
        let mut vols = BTreeMap::from([(0u8, VoxelData::new(1.0))]);
        let op = VoxelOp::fill(
            VoxelShape::Sphere {
                center: glam::DVec3::new(8.0, 8.0, 8.0),
                radius_m: 3.0,
            },
            2,
        );
        let r = runtime_carve(&mut vols, &0u8, true, &op);
        assert_eq!(r.outcome, RuntimeCarveOutcome::Applied);
        assert!(r.total_filled() > 0);
        assert_eq!(r.filled[2], r.total_filled());
        assert_eq!(r.total_carved(), 0);
        assert_eq!(r.added_m3(1.0), r.total_filled() as f64);
    }

    #[test]
    fn every_refusal_has_a_message_and_a_success_has_none() {
        assert!(RuntimeCarveOutcome::Applied
            .refusal("voxel::carve_sphere")
            .is_none());
        for o in [
            RuntimeCarveOutcome::NoVolume,
            RuntimeCarveOutcome::NotPermitted,
            RuntimeCarveOutcome::TooLarge { samples: 1 },
            RuntimeCarveOutcome::Invalid,
        ] {
            let m = o.refusal("voxel::carve_sphere").expect("a refusal reports");
            assert!(m.starts_with("voxel::carve_sphere"), "{m}");
        }
    }
}
