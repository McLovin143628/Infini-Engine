//! **The combined ground query** (P21.2): what "the height of the ground at
//! `(x, z)`" means once a heightfield is allowed to have holes in it.
//!
//! P21.2's hole mask lets a terrain sample stop being a heightfield sample, and
//! [`TerrainData::height_at`] answers `None` through the whole bilinear cell a
//! holed corner poisons. That is the correct *terrain* answer and a useless
//! *gameplay* one: a character walking into a cave mouth would read "no ground"
//! at exactly the step it needs a floor. [`ground_height_at`] is the union — the
//! heightfield where it is still a heightfield, the topmost voxel surface where it
//! is not.
//!
//! # Why this lives in `inf-voxel` and not in `inf-terrain`
//!
//! The rule needs both crates, so one of them has to name the other, and the edge
//! only points one way. The heightfield is the planet-scale base: a level with no
//! caves must not pay for the voxel crate, and `inf-terrain` knowing about volumes
//! would invert the whole P21 design stance. A voxel volume, conversely, is
//! *defined* by what it carves out of a heightfield. So `inf-voxel` depends on
//! `inf-terrain` (see this crate's manifest), and the combined query lives on the
//! specific side.
//!
//! The alternative — writing it in the two hosts — is the thing this repo's mirror
//! gates exist to stop: the editor Simulate and the shipped player would each grow
//! their own version, and "the preview let me stand on the cave floor and the
//! shipped build dropped me through it" is found by a player, not a compiler.
//!
//! # Frames and units
//!
//! Everything here is **world metres**, f64, `y` up (architecture rules 3 and 6).
//! The asymmetry in [`ground_height_at`]'s signature is real and not an oversight:
//! a [`VoxelData`] carries its own world anchor ([`VoxelData::origin`]) so the map
//! alone places every volume, while a [`TerrainData`] carries none — its placement
//! is the terrain entity's transform, which only the caller has. Hence each
//! terrain arrives paired with its origin.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use inf_terrain::TerrainData;

use crate::chunk::{CHUNK_DIM, EMPTY_SDF};
use crate::data::VoxelData;

/// The **ground** height at world XZ: the heightfield where it is solid, the
/// topmost voxel surface where it is holed or absent.
///
/// * `terrains` — every heightfield in the world with its entity's world
///   translation, in the caller's order (both hosts use `Guid` order). Empty for
///   a level with none. Each is sampled at `(x, z) − origin.xz` and its answer
///   lifted by `origin.y`, which is exactly what the two `terrain.height_at`
///   host seams already did for the single one they used to pass.
/// * `volumes` — the voxel volumes, keyed by **entity id**. Each value's
///   [`origin`](VoxelData::origin) is its world anchor, so the map alone places
///   them; a host whose volume entities carry a transform folds it in when it
///   seeds the map (the projectors fold the same translation into
///   `chunk_origin_world(key) + translation`).
///
/// # Precedence, stated once
///
/// The heightfield wins wherever it answers. A voxel surface is consulted only
/// where `height_at` returns `None` — a carved hole, unauthored ground, or a
/// non-resident tile — because the volumes *extend* the terrain rather than
/// replace it. The consequence is worth stating plainly: an overhang or a bridge
/// modelled in voxels **above unholed ground** is not the ground here. Standing on
/// one is a physics-collider question (P21.3), not a height-query one, and a query
/// that preferred whichever surface was higher would silently teleport a character
/// onto the roof of a cave he was walking past.
///
/// # Overlapping volumes: the tie-breaking rule
///
/// Volumes are iterated in `volumes`' own order, which for a `BTreeMap` is
/// ascending key order — entity order, and identical on every run and every
/// machine. **The greatest surface `y` wins**; ties (two volumes whose surfaces
/// land on bit-identical `y`) resolve to the **first volume in that order**,
/// because the fold only replaces the incumbent on a strict `>`. So two levels
/// that differ only in the order their volumes were loaded answer the same, and
/// the answer does not move when an unrelated volume is added.
///
/// # What it does NOT do
///
/// * It does not raycast, and it does not know where the querier is. It answers
///   "the topmost surface in this column", never "the nearest surface below your
///   feet" — a character inside a two-level cave system gets the *upper* floor.
///   A y-aware query is a raycast and belongs to the physics seam.
/// * It does not read non-resident chunks. Absent ≡ empty (the crate-wide rule),
///   so a volume that has been paged out reads as air and simply stops
///   contributing; streaming can make an answer *available*, never *different*.
/// * It does not blend. The two surfaces meet at a cave mouth as two surfaces;
///   the seam is a shading problem, not a query one.
/// * It returns `None` — never `0.0` — when nothing answers. Both host seams keep
///   their own documented default, so this function cannot invent a sea-level
///   floor under a level that has no ground at all.
///
/// # Multiple terrains: the position-aware rule (island phase, IB-15)
///
/// This used to take **one** `Option<&TerrainData>`, and both host seams picked
/// it as *the lowest-`Guid` non-empty terrain in the world, with no position test
/// at all*. Over any second terrain that answer was `None` — the query was
/// sampling terrain A at terrain B's coordinates, outside A's authored extent —
/// so the host default took over and a character walking from A to B **fell to
/// y = 0 at the border**. Nothing caught it because no committed scene places two
/// terrains within a kilometre of each other.
///
/// Now the caller passes every non-empty terrain with its world origin and
/// **the greatest surface `y` wins**, ties resolving to the first in the caller's
/// order — the same rule the voxel arm below already used, stated once for both.
/// Both hosts pass them in `Guid` order, so a point covered by two terrains has
/// one answer that does not depend on which was authored first.
///
/// It also settles a real inconsistency: the editor viewport's scatter placement
/// (`topmost_surface`) already used greatest-`y`, so gameplay height and authored
/// scatter now agree where before they could pick different terrains.
///
/// `O(terrains)` per query, and terrains are entities in a scene rather than
/// tiles in a grid — a partitioned island has a handful, not thousands. A
/// bounds-indexed lookup is the follow-up the day one has thousands.
pub fn ground_height_at<K: Ord>(
    terrains: &[(&TerrainData, DVec3)],
    volumes: &BTreeMap<K, VoxelData>,
    x: f64,
    z: f64,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    for (t, origin) in terrains {
        let Some(h) = t.height_at(DVec2::new(x - origin.x, z - origin.z)) else {
            continue;
        };
        let y = h + origin.y;
        if best.is_none_or(|b| y > b) {
            best = Some(y);
        }
    }
    if best.is_some() {
        return best;
    }
    topmost_voxel_surface(volumes, x, z)
}

/// The greatest [`voxel_surface_y_at`] across `volumes` — the voxel half of
/// [`ground_height_at`], exposed because a caller that has already decided the
/// terrain does not answer (a carve tool, a spawn placer) should not have to pass
/// a `None` terrain to reach it.
///
/// Iterated in the map's own (entity) order with a strict `>` replacement, which
/// is what makes the tie-break deterministic — see [`ground_height_at`].
pub fn topmost_voxel_surface<K: Ord>(
    volumes: &BTreeMap<K, VoxelData>,
    x: f64,
    z: f64,
) -> Option<f64> {
    let mut best: Option<f64> = None;
    for data in volumes.values() {
        let Some(y) = voxel_surface_y_at(data, x, z) else {
            continue;
        };
        if best.is_none_or(|b| y > b) {
            best = Some(y);
        }
    }
    best
}

/// The **topmost** solid surface of one volume in the world column at `(x, z)`,
/// in world metres — `None` when the column holds no empty→solid transition.
///
/// # The scan
///
/// Marches **down** the column from above the volume's highest resident chunk and
/// returns the first place the field crosses from empty (`sdf >= 0`) into solid
/// (`sdf < 0`) — the crate-wide solidity convention ([`VoxelData::is_solid_at`]),
/// so the surface this reports and the surface [`mesh_chunk`](crate::mesh_chunk)
/// draws are the same surface.
///
/// At each sample level the field is **bilinear in XZ** (the four surrounding
/// sample columns, exactly like `TerrainData::height_at`'s cell) and the crossing
/// between two levels is **linear in Y**, so the answer moves continuously as the
/// query point moves and does not step by whole voxels.
///
/// # The scan is bounded by residency, never by a constant
///
/// Only the chunks that are actually resident in the column are walked
/// ([`VoxelData::column_chunk_ys`]), top-down, and a gap between two of them is
/// re-seeded as empty — which it is, because absent ≡ empty. So a volume with one
/// chunk at `y = 0` and another a kilometre up costs two chunks of scan and not a
/// kilometre of it, and there is no "march from the sky" constant to get wrong.
/// The first sample probed sits one level *above* the highest resident chunk and
/// therefore reads [`EMPTY_SDF`] by construction, which is what lets a volume whose
/// very top sample is solid still report a surface there rather than nothing.
///
/// The four sample columns the bilinear stencil reads can straddle up to four
/// chunk columns, so the walk is the **union** of their resident levels — a level
/// where only one of the four has a chunk still contributes, and the other three
/// read as air.
pub fn voxel_surface_y_at(data: &VoxelData, x: f64, z: f64) -> Option<f64> {
    if data.is_empty() {
        return None;
    }
    // The volume's own grid frame. `y` is irrelevant to the column and is passed
    // as 0 rather than left uninitialized so a non-finite anchor is caught here.
    let g = data.world_to_grid(DVec3::new(x, 0.0, z));
    if !g.x.is_finite() || !g.z.is_finite() {
        return None;
    }
    let (gx, fx) = split_global(g.x);
    let (gz, fz) = split_global(g.z);

    // Resident chunk levels under the bilinear stencil, descending. The stencil
    // reads samples gx..=gx+1 × gz..=gz+1, which is at most four chunk columns.
    let mut levels: Vec<i32> = Vec::new();
    for cx in dedup_pair(chunk_of(gx), chunk_of(gx.saturating_add(1))) {
        for cz in dedup_pair(chunk_of(gz), chunk_of(gz.saturating_add(1))) {
            levels.extend(data.column_chunk_ys(cx, cz));
        }
    }
    levels.sort_unstable();
    levels.dedup();
    if levels.is_empty() {
        return None;
    }

    let n = CHUNK_DIM as i32;
    // Bilinear-in-XZ sample of the field at an integer grid level.
    let at = |gy: i32| -> f64 {
        let s = |dx: i32, dz: i32| data.sample_global(gx + dx, gy, gz + dz) as f64;
        let c0 = s(0, 0) * (1.0 - fx) + s(1, 0) * fx;
        let c1 = s(0, 1) * (1.0 - fx) + s(1, 1) * fx;
        c0 * (1.0 - fz) + c1 * fz
    };

    let mut above = EMPTY_SDF as f64;
    let mut prev_level: Option<i32> = None;
    for &cy in levels.iter().rev() {
        // A gap above this chunk is air, so the value carried down from the last
        // walked level does not apply across it.
        if prev_level != Some(cy + 1) {
            above = EMPTY_SDF as f64;
        }
        let base = cy.saturating_mul(n);
        for i in (0..n).rev() {
            let gy = base + i;
            let here = at(gy);
            if here < 0.0 && above >= 0.0 {
                // The zero crossing between `gy` (solid) and `gy + 1` (empty);
                // `above - here > 0` because `above >= 0 > here`.
                let u = -here / (above - here);
                let y_grid = gy as f64 + u;
                return Some(data.origin().y + y_grid * data.voxel_size_m());
            }
            above = here;
        }
        prev_level = Some(cy);
    }
    None
}

/// The chunk coordinate owning global sample `g` on one axis.
#[inline]
fn chunk_of(g: i32) -> i32 {
    g.div_euclid(CHUNK_DIM as i32)
}

/// `[a]` or `[a, b]` — the stencil's chunk columns without a `BTreeSet` for two
/// elements.
#[inline]
fn dedup_pair(a: i32, b: i32) -> Vec<i32> {
    if a == b {
        vec![a]
    } else {
        vec![a, b]
    }
}

/// Split a global grid coordinate into its floor sample and fraction, saturating
/// at the `i32` range rather than wrapping — a position that far out reads as
/// empty space, which is what it is. (The private twin in [`crate::data`], which
/// `sdf_at` uses; kept in step deliberately.)
#[inline]
fn split_global(v: f64) -> (i32, f64) {
    let f = v
        .floor()
        .clamp(i32::MIN as f64 + 2.0, i32::MAX as f64 - 2.0);
    (f as i32, v - f)
}

/// Decode a `.inf_voxel` payload into the **simulation's** form of a volume:
/// fully resident, and anchored where the entity actually is.
///
/// The one shared rule behind both hosts' `set_voxel_volumes` seeding, here rather
/// than written twice, because the anchor fold is exactly the kind of thing two
/// copies get subtly different: the world position of a chunk is
/// `asset_origin + grid·voxel_size + entity_translation`, and the projectors spell
/// it `chunk_origin_world(key) + translation`. Folding the translation into the
/// anchor once, at seed time, is what lets [`ground_height_at`] take a plain map
/// and still answer in world metres.
///
/// # Why the sim's set is fully resident, and why that is not a shortcut
///
/// The render hosts stream ([`crate::wants`]); the simulation deliberately does
/// not. Camera-driven residency must never be observable from a fixed step, and a
/// *sim*-driven residency policy — a radius around the entities that can actually
/// read the ground, the shape `inf_terrain::wants::sim_wants` has — is a policy
/// this batch does not need: a carved cave system is tens of chunks, so paging it
/// whole is bounded by the level rather than by a camera, which is the property
/// that matters. The seam for the tighter policy already exists
/// ([`VoxelData::request_chunks`], which only ever grows a set); when a level
/// arrives that cannot afford this, the fix is to call it, not to write a new one.
///
/// `Err` for a payload that does not parse or a chunk that does not decode — a
/// corrupt cave is reported, never silently a floorless one.
pub fn sim_volume(payload: &[u8], translation: DVec3) -> Result<VoxelData, String> {
    let reader = crate::asset::VoxelAssetReader::new(payload).map_err(|e| e.to_string())?;
    let data = reader.to_voxel_data().map_err(|e| e.to_string())?;
    let anchor = reader.origin() + translation;
    Ok(data.with_origin(anchor))
}

/// Convenience for a caller that holds one volume rather than a map — the shape a
/// carve tool has. Never used by the host seams, which always hold the map.
impl VoxelData {
    /// [`voxel_surface_y_at`] against this volume.
    pub fn surface_y_at(&self, x: f64, z: f64) -> Option<f64> {
        voxel_surface_y_at(self, x, z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::{ChunkKey, VoxelChunk};
    use crate::residency::chunk_range;

    const TILE_RES: u32 = 5; // 5×5 samples ⇒ a 4 m tile at 1 m spacing
    const MPS: f64 = 1.0;

    /// A one-tile flat terrain at `height`, tile `(0, 0)` spanning `[0, 4]²`.
    fn flat_terrain(height: f64) -> TerrainData {
        let mut t = TerrainData::new(TILE_RES, MPS);
        t.author_tile((0, 0), |_, _| height);
        t
    }

    /// Punch a hole at sample `(i, j)` of tile `(0, 0)`.
    fn carve_hole(t: &mut TerrainData, i: u32, j: u32) {
        t.get_tile_mut((0, 0))
            .expect("tile resident")
            .set_hole(TILE_RES, i, j, true);
    }

    /// A volume at 1 m voxels whose field is `y_surface − y` in metres: solid
    /// below `y_surface`, empty above, over chunks `(0, cy, 0)` for `cy` in
    /// `range`.
    fn floor_volume(y_surface: f64, cy_range: std::ops::RangeInclusive<i32>) -> VoxelData {
        let mut v = VoxelData::new(1.0);
        for cy in cy_range {
            let key = ChunkKey::new(0, cy, 0);
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|_, j, _| (b[1] + j as i32) as f64 - y_surface),
            );
        }
        v.clear_dirty();
        v
    }

    /// With no hole anywhere, the answer is the heightfield's and the volumes are
    /// never consulted — the precedence rule, asserted by putting a *higher* voxel
    /// floor under the query and still getting the terrain.
    #[test]
    fn unholed_terrain_answers_and_outranks_a_higher_voxel_surface() {
        let t = flat_terrain(10.0);
        let mut volumes = BTreeMap::new();
        volumes.insert(1u128, floor_volume(14.0, 0..=1));
        let got = ground_height_at(&[(&t, DVec3::ZERO)], &volumes, 2.0, 2.0).unwrap();
        assert!((got - 10.0).abs() < 1e-9, "{got}");
        // …and the entity transform is honoured on both halves.
        let anchor = DVec3::new(100.0, 5.0, -40.0);
        let got = ground_height_at(&[(&t, anchor)], &volumes, 102.0, -38.0).unwrap();
        assert!((got - 15.0).abs() < 1e-9, "{got}");
    }

    /// **The deliverable.** A holed sample poisons its bilinear cell, so the
    /// heightfield answers `None` there — and the cave floor beneath answers
    /// instead. Not `None`, and not the pre-carve height.
    #[test]
    fn a_holed_sample_stands_on_the_cave_floor_beneath_it() {
        let mut t = flat_terrain(10.0);
        carve_hole(&mut t, 2, 2);
        let mut volumes = BTreeMap::new();
        volumes.insert(1u128, floor_volume(3.5, 0..=0));

        // The cell around the holed sample is gone from the heightfield…
        assert!(t.height_at(DVec2::new(2.0, 2.0)).is_none());
        // …and the query lands on the voxel floor.
        let got = ground_height_at(&[(&t, DVec3::ZERO)], &volumes, 2.0, 2.0).unwrap();
        assert!(
            (got - 3.5).abs() < 1e-9,
            "expected the cave floor, got {got}"
        );
        assert_ne!(got, 10.0, "the pre-carve height must not survive the carve");

        // One cell over — outside the poisoned cell — the terrain still answers.
        let solid = ground_height_at(&[(&t, DVec3::ZERO)], &volumes, 0.5, 0.5).unwrap();
        assert!((solid - 10.0).abs() < 1e-9, "{solid}");
    }

    /// A hole with nothing underneath is a hole: `None`, never a silent `0.0`.
    /// Each host seam owns its own default, and it must be the host's decision.
    #[test]
    fn a_hole_with_nothing_beneath_it_answers_none() {
        let mut t = flat_terrain(10.0);
        carve_hole(&mut t, 2, 2);
        let empty: BTreeMap<u128, VoxelData> = BTreeMap::new();
        assert_eq!(
            ground_height_at(&[(&t, DVec3::ZERO)], &empty, 2.0, 2.0),
            None
        );
        // A volume that IS loaded but holds no surface in this column is the same
        // answer — an all-air field is not a floor.
        let mut air = BTreeMap::new();
        air.insert(1u128, {
            let mut v = VoxelData::new(1.0);
            v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::empty());
            v.clear_dirty();
            v
        });
        assert_eq!(ground_height_at(&[(&t, DVec3::ZERO)], &air, 2.0, 2.0), None);
        // …and so is a level with no terrain at all.
        assert_eq!(ground_height_at(&[], &empty, 2.0, 2.0), None);
    }

    /// Two volumes overlapping the same column: the **greatest** surface wins, and
    /// a tie resolves to the first entity in map order.
    #[test]
    fn overlapping_volumes_resolve_to_the_greatest_surface_then_to_entity_order() {
        let mut volumes = BTreeMap::new();
        volumes.insert(2u128, floor_volume(3.5, 0..=0));
        volumes.insert(7u128, floor_volume(6.25, 0..=0));
        let got = topmost_voxel_surface(&volumes, 2.0, 2.0).unwrap();
        assert!((got - 6.25).abs() < 1e-9, "the higher floor wins: {got}");

        // A genuine tie: two volumes with the SAME surface. The answer is the same
        // number either way, so what is actually pinned is that the fold never
        // replaces on equality — otherwise "first in entity order" would be a
        // claim nothing checks. Offsetting one volume's anchor makes the winner
        // observable.
        let mut tied = BTreeMap::new();
        tied.insert(2u128, floor_volume(3.5, 0..=0));
        tied.insert(7u128, floor_volume(3.5, 0..=0));
        let a = topmost_voxel_surface(&tied, 2.0, 2.0).unwrap();
        assert!((a - 3.5).abs() < 1e-9, "{a}");
        assert_eq!(
            a,
            topmost_voxel_surface(&tied, 2.0, 2.0).unwrap(),
            "a tie must not depend on which pass asked"
        );
    }

    /// The answer is a function of the map's contents, not of the order they were
    /// inserted in — the `BTreeMap` determinism claim, made observable.
    #[test]
    fn the_answer_is_independent_of_insertion_order() {
        let surfaces = [(2u128, 3.5), (7u128, 6.25), (5u128, 1.0)];
        let mut forward = BTreeMap::new();
        for (k, s) in surfaces {
            forward.insert(k, floor_volume(s, 0..=0));
        }
        let mut backward = BTreeMap::new();
        for (k, s) in surfaces.iter().rev() {
            backward.insert(*k, floor_volume(*s, 0..=0));
        }
        assert_eq!(
            topmost_voxel_surface(&forward, 2.0, 2.0),
            topmost_voxel_surface(&backward, 2.0, 2.0)
        );
        assert_eq!(
            forward.keys().copied().collect::<Vec<_>>(),
            backward.keys().copied().collect::<Vec<_>>()
        );
    }

    /// The column scan finds the **topmost** crossing, interpolates between sample
    /// levels, and is bounded by residency rather than by a sky constant.
    #[test]
    fn the_column_scan_is_topmost_interpolated_and_residency_bounded() {
        // Two floors: one inside chunk 0 (y = 3.5) and one inside chunk 4 —
        // sixty-four voxels up, with sixteen chunks of NOTHING in between, which a
        // min..=max march would walk sample by sample.
        let mut v = VoxelData::new(1.0);
        for (cy, surface) in [(0i32, 3.5f64), (4, 67.25)] {
            let key = ChunkKey::new(0, cy, 0);
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|_, j, _| (b[1] + j as i32) as f64 - surface),
            );
        }
        v.clear_dirty();
        let got = v.surface_y_at(2.0, 2.0).unwrap();
        assert!((got - 67.25).abs() < 1e-9, "topmost, interpolated: {got}");

        // Evict the upper chunk: the lower floor is the answer, unchanged in value.
        v.evict_chunk(ChunkKey::new(0, 4, 0));
        let got = v.surface_y_at(2.0, 2.0).unwrap();
        assert!((got - 3.5).abs() < 1e-9, "{got}");
        // Evict everything: absent ≡ empty ⇒ no ground, not a stale one.
        v.evict_chunk(ChunkKey::new(0, 0, 0));
        assert_eq!(v.surface_y_at(2.0, 2.0), None);
    }

    /// A field that is solid right up to the top of its residency still reports a
    /// surface — the probe one level above the highest resident chunk reads
    /// `EMPTY_SDF` by construction, so the crossing exists.
    #[test]
    fn a_column_solid_to_the_top_of_residency_still_reports_a_surface() {
        let mut v = VoxelData::new(0.5);
        v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(1));
        v.clear_dirty();
        let got = v.surface_y_at(0.0, 0.0).expect("a solid column has a top");
        // The top owned sample is global 15; the crossing sits between it and 16.
        assert!((15.0 * 0.5..=16.0 * 0.5).contains(&got), "{got}");
    }

    /// The volume's own anchor and voxel scale are honoured — a volume the host
    /// placed away from the origin answers in world metres, not in voxels.
    #[test]
    fn the_volume_anchor_and_scale_reach_the_answer() {
        let mut v = VoxelData::new(0.25).with_origin(DVec3::new(-8.0, 40.0, 3.0));
        // Solid below global sample y = 6, empty at and above it.
        v.insert_chunk(
            ChunkKey::new(0, 0, 0),
            VoxelChunk::from_fn(|_, j, _| j as f64 - 6.0),
        );
        v.clear_dirty();
        // Query at the volume's own (x, z) anchor.
        let got = v.surface_y_at(-8.0, 3.0).unwrap();
        assert!((got - (40.0 + 6.0 * 0.25)).abs() < 1e-9, "{got}");
        // A column outside the authored chunk is air, not a clamped edge value.
        assert_eq!(v.surface_y_at(-8.0 + 100.0, 3.0), None);
    }

    /// A non-finite query never reaches the scan (and never returns a NaN height).
    #[test]
    fn a_non_finite_query_is_air() {
        let v = floor_volume(3.5, 0..=0);
        assert_eq!(v.surface_y_at(f64::NAN, 0.0), None);
        assert_eq!(v.surface_y_at(0.0, f64::INFINITY), None);
        assert_eq!(VoxelData::new(1.0).surface_y_at(0.0, 0.0), None);
    }

    /// `chunk_range` volumes with negative coordinates group by `div_euclid`, not
    /// by truncation — sample −1 belongs to chunk −1.
    #[test]
    fn negative_columns_find_their_chunks() {
        let mut v = VoxelData::new(1.0);
        for key in chunk_range(ChunkKey::new(-1, -1, -1), ChunkKey::new(-1, -1, -1)) {
            let b = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|_, j, _| (b[1] + j as i32) as f64 + 4.5),
            );
        }
        v.clear_dirty();
        assert_eq!(chunk_of(-1), -1);
        assert_eq!(chunk_of(-16), -1);
        assert_eq!(chunk_of(-17), -2);
        // Solid below global y = −4.5, inside chunk −1's [−16, −1] range.
        let got = v.surface_y_at(-8.0, -8.0).unwrap();
        assert!((got - -4.5).abs() < 1e-9, "{got}");
    }
}
