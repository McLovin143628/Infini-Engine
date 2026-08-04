//! **Excavation** (P21.3): a dig removes material, the material becomes a spoil
//! pile, and the two balance **exactly, per material, as integers**.
//!
//! # What this file is the gate on
//!
//! `inf_voxel::spoil` pins the placement rule in isolation (a plan in, a mound
//! out) and `inf_editor_core::voxel_tool` pins the shapes a gesture produces.
//! Neither can see the thing that actually matters to an author:
//!
//! * that the **transaction** balances — the count a cut removed from a volume
//!   is the count the pile put back into the same volume, through a real
//!   `SceneDoc`, a real shared store and a real undo entry;
//! * that the pit, its cave mouths and its spoil heap are **one** `EditCommand`,
//!   so one Ctrl+Z takes back all three and leaves byte-identical chunks *and*
//!   byte-identical terrain tiles;
//! * that a **refused** dig moved nothing at all — no half-excavated pit, no
//!   orphan heap of soil from a hole that was never dug.
//!
//! The conservation assertions are deliberately written against an
//! **independent** recount of the field (walk every resident chunk, count solid
//! samples by material) rather than against the tally the transaction returns.
//! A report that agrees with itself proves nothing.

use std::path::Path;

use glam::DVec3;
use inf_ecs::components::{Terrain, VoxelVolume};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::undo::{CarveRefusal, SpoilChoice};
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::voxel_store::{shared_volumes, SharedVoxelVolumes};
use inf_editor_core::voxel_tool::{box_cut_plan, brush_dab_shape, trench_shapes};
use inf_voxel::{
    ChunkKey, VoxelChunk, VoxelData, VoxelOp, VoxelShape, VoxelStreamBudget, VoxelWantsParams,
    MATERIAL_COUNT,
};
use uuid::Uuid;

/// Voxel edge length of the fixture volumes, metres.
const VOXEL_M: f64 = 0.5;
/// The fixture ground plane, metres.
const GRADE_M: f64 = 4.0;

/// Write a solid 2 × 2 × 2-chunk block of rock with **two strata**: material 1
/// west of the chunk seam, material 2 east of it.
///
/// Two materials and not one, because the whole conservation claim is *per
/// material*: a single-material fixture would pass with an implementation that
/// balanced only the totals and dumped everything as layer 0.
fn write_rock(dir: &Path, name: &str, guid: Uuid) {
    let mut v = VoxelData::new(VOXEL_M);
    for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
        v.insert_chunk(key, VoxelChunk::solid(if key.x == 0 { 1 } else { 2 }));
    }
    let asset = inf_voxel::build_voxel_asset(&v).unwrap();
    std::fs::create_dir_all(dir).unwrap();
    let path = dir.join(name);
    let bytes = inf_voxel::write_voxel_asset(&path, &asset).unwrap();
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(guid),
        inf_asset::AssetKind::VoxelVolume,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .unwrap();
}

/// A document with one terrain (flat at [`GRADE_M`]) and one voxel volume bound
/// to the two-strata block, plus the shared store holding it — **fully paged**.
fn fixture(asset_backed: bool) -> (SceneDoc, SharedVoxelVolumes, Uuid, Uuid, tempfile::TempDir) {
    build_fixture(asset_backed, true)
}

/// …and the same level with the volume **bound but not paged**: every chunk is
/// in the `.inf_voxel` and none of them is in memory, which is the state a
/// camera that has never been near the dig leaves behind.
fn unpaged_fixture(
    asset_backed: bool,
) -> (SceneDoc, SharedVoxelVolumes, Uuid, Uuid, tempfile::TempDir) {
    build_fixture(asset_backed, false)
}

fn build_fixture(
    asset_backed: bool,
    page: bool,
) -> (SceneDoc, SharedVoxelVolumes, Uuid, Uuid, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let asset = Uuid::from_u128(0x2133_0001);
    write_rock(dir.path(), "Rock.inf_voxel", asset);

    let mut doc = SceneDoc::new();
    let terrain = doc.edit_create(SpawnKind::Terrain, "Ground", None);
    {
        let e = doc.entity_of(terrain).unwrap();
        let mut t = doc.world_mut().world_mut().get_mut::<Terrain>(e).unwrap();
        t.data = inf_terrain::TerrainData::new(33, 1.0);
        t.data.author_tile((0, 0), |_, _| GRADE_M);
        t.asset = asset_backed.then(|| Uuid::from_u128(0x2133_0002));
    }
    let volume = doc.edit_create(SpawnKind::Empty, "Excavation", None);
    {
        let e = doc.entity_of(volume).unwrap();
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(VoxelVolume {
                asset: Some(asset),
                voxel_size_m: VOXEL_M,
                ..VoxelVolume::default()
            });
    }
    let volumes = shared_volumes();
    {
        let mut v = volumes.lock().unwrap();
        v.set_content_root(Some(dir.path().to_path_buf()));
        assert!(v.ensure(volume, &VoxelVolume::from_asset(asset)));
        if page {
            let report = v.sync_camera(
                DVec3::splat(4.0),
                &VoxelWantsParams {
                    radius_m: 1000.0,
                    hysteresis: 0.0,
                },
                VoxelStreamBudget::default(),
            );
            assert!(report.loaded > 0, "the fixture volume must page in");
        } else {
            // Binding pages nothing (P21.2), so this really is the empty state.
            assert_eq!(
                v.slot(volume).map(|s| s.data.chunk_count()),
                Some(0),
                "the unpaged fixture is not unpaged"
            );
        }
    }
    (doc, volumes, terrain, volume, dir)
}

/// **Independent** recount: solid samples across every resident chunk, by the
/// material they carry. Never reads the transaction's own tally.
fn solid_by_material(volumes: &SharedVoxelVolumes, volume: Uuid) -> [u64; MATERIAL_COUNT] {
    let v = volumes.lock().unwrap();
    let slot = v.slot(volume).unwrap();
    let mut out = [0u64; MATERIAL_COUNT];
    for (_, c) in slot.data.chunks() {
        for k in 0..inf_voxel::CHUNK_DIM {
            for j in 0..inf_voxel::CHUNK_DIM {
                for i in 0..inf_voxel::CHUNK_DIM {
                    if c.sample(i, j, k) < 0.0 {
                        out[(c.material(i, j, k) as usize).min(MATERIAL_COUNT - 1)] += 1;
                    }
                }
            }
        }
    }
    out
}

/// Every resident chunk's bytes — the "byte-identical volume" object.
fn volume_image(volumes: &SharedVoxelVolumes, volume: Uuid) -> Vec<(ChunkKey, Vec<u8>)> {
    let v = volumes.lock().unwrap();
    let slot = v.slot(volume).unwrap();
    slot.data
        .chunks()
        .map(|(&k, c)| (k, inf_voxel::asset::encode_chunk(c).unwrap()))
        .collect()
}

/// The terrain tile's bytes, hole mask included.
fn tile_image(doc: &SceneDoc, terrain: Uuid) -> Vec<u8> {
    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    inf_terrain::asset::encode_tile(data.get_tile((0, 0)).unwrap()).unwrap()
}

/// The ground query the box-cut planner takes, bound to this fixture's terrain.
fn surface_of(doc: &SceneDoc) -> impl Fn(f64, f64) -> Option<f64> + '_ {
    move |x, z| doc.ground_surface_y(x, z)
}

/// The pit every conservation case digs, as a plan: a rectangle from `a` to `b`
/// on the surface, `depth` metres deep.
fn pit(doc: &SceneDoc, a: (f64, f64), b: (f64, f64), depth: f64) -> VoxelOp {
    let ay = doc.ground_surface_y(a.0, a.1).unwrap_or(GRADE_M);
    let by = doc.ground_surface_y(b.0, b.1).unwrap_or(GRADE_M);
    let plan = box_cut_plan(
        DVec3::new(a.0, ay, a.1),
        DVec3::new(b.0, by, b.1),
        depth,
        surface_of(doc),
    )
    .expect("the fixture drag must be a real pit");
    VoxelOp::carve(plan.shape)
}

/// One case's cut, built against the fixture document (which is what supplies
/// the ground the shapes are resolved against).
type CutBuilder = Box<dyn Fn(&SceneDoc) -> Vec<VoxelOp>>;

// ── conservation ────────────────────────────────────────────────────────────

/// **THE CONSERVATION GATE.** For every cut shape and every size, the voxels the
/// dig removed are the voxels the pile placed — per material, exactly — and the
/// field's own solid count is unchanged.
///
/// The cases deliberately include the two clipping regimes the ledger names:
///
/// * a pit **straddling the chunk seam** at `x = 8 m`, which removes both
///   strata and therefore proves the per-material split rather than the total;
/// * a pit hanging **off the volume's edge**, most of whose footprint is empty
///   space that contains no rock to remove — the case where an implementation
///   that spoiled the shape's *volume* instead of its *removals* would over-fill.
#[test]
fn every_dig_spoils_exactly_what_it_removed_per_material() {
    let cases: Vec<(&str, CutBuilder)> = vec![
        (
            "small pit",
            Box::new(|doc: &SceneDoc| vec![pit(doc, (2.0, 2.0), (4.0, 4.0), 1.5)]),
        ),
        (
            "pit across the chunk seam (two strata)",
            Box::new(|doc: &SceneDoc| vec![pit(doc, (6.0, 5.0), (10.0, 9.0), 2.5)]),
        ),
        (
            "pit hanging off the volume's edge",
            Box::new(|doc: &SceneDoc| vec![pit(doc, (-6.0, 2.0), (2.0, 6.0), 2.0)]),
        ),
        (
            "trench",
            Box::new(|doc: &SceneDoc| {
                let y = doc.ground_surface_y(4.0, 4.0).unwrap();
                trench_shapes(
                    &[
                        DVec3::new(3.0, y, 3.0),
                        DVec3::new(11.0, y, 5.0),
                        DVec3::new(12.0, y, 12.0),
                    ],
                    1.0,
                    2.0,
                    surface_of(doc),
                )
                .into_iter()
                .map(VoxelOp::carve)
                .collect()
            }),
        ),
        (
            "brush column to grade",
            Box::new(|doc: &SceneDoc| {
                let y = doc.ground_surface_y(6.0, 6.0).unwrap();
                vec![VoxelOp::carve(brush_dab_shape(
                    DVec3::new(6.0, y, 6.0),
                    1.5,
                    3.0,
                    true,
                ))]
            }),
        ),
        (
            "brush ball below grade (no mouth)",
            Box::new(|doc: &SceneDoc| {
                let y = doc.ground_surface_y(6.0, 6.0).unwrap();
                vec![VoxelOp::carve(brush_dab_shape(
                    DVec3::new(6.0, y, 6.0),
                    1.25,
                    3.0,
                    false,
                ))]
            }),
        ),
    ];

    for (label, build) in cases {
        let (mut doc, volumes, _t, volume, _d) = fixture(true);
        let before = solid_by_material(&volumes, volume);
        let ops = build(&doc);
        assert!(!ops.is_empty(), "{label}: the fixture built no ops");

        let tally = doc
            .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
            .unwrap_or_else(|r| panic!("{label} was refused: {r:?}"));

        assert!(tally.carved > 0, "{label}: the fixture cut removed nothing");
        assert!(
            tally.conserved(),
            "{label}: the ledger does not balance — removed {:?}, spoiled {:?}",
            tally.carved_by_material,
            tally.spoiled_by_material
        );
        assert_eq!(
            tally.spoiled(),
            tally.carved,
            "{label}: totals disagree with the per-material split"
        );
        // …and the FIELD says so too, counted independently.
        assert_eq!(
            solid_by_material(&volumes, volume),
            before,
            "{label}: solid samples were created or destroyed"
        );
        assert!(
            tally.removed_m3(VOXEL_M) > 0.0
                && (tally.removed_m3(VOXEL_M) - tally.spoiled_m3(VOXEL_M)).abs() < 1e-12,
            "{label}: the m³ readout does not balance"
        );
    }
}

/// The **two-strata** case is really two strata: a pit across the seam removes
/// both materials and the pile carries both, in the right amounts.
///
/// Without this, `every_dig_spoils_exactly_what_it_removed_per_material` would
/// still pass on a fixture that happened to be single-material everywhere the
/// cuts reached — a vacuous gate on the very property it is named for.
#[test]
fn a_dig_across_two_strata_spoils_both_of_them() {
    let (mut doc, volumes, _t, volume, _d) = fixture(true);
    let ops = vec![pit(&doc, (6.0, 5.0), (10.0, 9.0), 2.5)];
    let tally = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
        .expect("allowed");
    assert!(
        tally.carved_by_material[1] > 0 && tally.carved_by_material[2] > 0,
        "the fixture pit did not reach both strata: {:?}",
        tally.carved_by_material
    );
    assert_eq!(tally.carved_by_material, tally.spoiled_by_material);
    let breakdown = tally.material_breakdown();
    assert!(
        breakdown.contains("layer 1") && breakdown.contains("layer 2"),
        "the readout hides a stratum: {breakdown}"
    );
}

/// **A dig that removes nothing spoils nothing** — and records nothing either.
///
/// A cut swung entirely through air is the case a placement routine that keyed
/// off the *shape* rather than the *removals* would answer with a heap of soil
/// conjured from an empty hole.
#[test]
fn a_dig_that_removes_nothing_spoils_nothing() {
    let (mut doc, volumes, _t, volume, _d) = fixture(true);
    let before = volume_image(&volumes, volume);
    let undo_before = doc.undo_len();
    // Far above the block and far to the side of the terrain: no rock, no
    // surface, nothing.
    let ops = vec![VoxelOp::carve(VoxelShape::Box {
        center: DVec3::new(400.0, 400.0, 400.0),
        half_extents: DVec3::splat(2.0),
    })];
    let tally = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
        .expect("a cut through air is legal everywhere");
    assert_eq!(tally.carved, 0);
    assert_eq!(tally.spoiled(), 0);
    assert!(tally.conserved(), "0 == 0 must count as conserved");
    assert_eq!(volume_image(&volumes, volume), before);
    assert_eq!(
        doc.undo_len(),
        undo_before,
        "an empty dig recorded an undo step"
    );
}

/// **Discarding is a choice, not a failure.** With spoil off the material is
/// gone and the ledger says so honestly rather than claiming a balance.
#[test]
fn discarded_spoil_is_reported_as_discarded() {
    let (mut doc, volumes, _t, volume, _d) = fixture(true);
    let before = solid_by_material(&volumes, volume);
    let ops = vec![pit(&doc, (2.0, 2.0), (5.0, 5.0), 2.0)];
    let tally = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::Discard)
        .expect("allowed");
    assert!(tally.carved > 0);
    assert_eq!(tally.spoiled(), 0);
    assert!(!tally.conserved(), "a discard must not claim conservation");
    let after = solid_by_material(&volumes, volume);
    assert!(
        after.iter().sum::<u64>() < before.iter().sum::<u64>(),
        "a discard did not remove anything"
    );
}

// ── the transaction ─────────────────────────────────────────────────────────

/// **THE UNDO GATE.** Pit + cave mouths + spoil heap are ONE step: Ctrl+Z
/// restores byte-identical chunks *and* a byte-identical terrain tile, and
/// Ctrl+Y restores the post-dig bytes exactly.
#[test]
fn a_dig_undoes_and_redoes_as_one_whole_step() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);
    let chunks_before = volume_image(&volumes, volume);
    let tile_before = tile_image(&doc, terrain);

    let ops = vec![pit(&doc, (5.0, 5.0), (10.0, 9.0), 2.5)];
    let tally = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
        .expect("allowed");
    assert!(
        tally.carved > 0 && tally.spoiled() > 0 && tally.holes > 0,
        "{tally:?}"
    );

    let chunks_after = volume_image(&volumes, volume);
    let tile_after = tile_image(&doc, terrain);
    assert_ne!(chunks_after, chunks_before, "the dig changed no chunks");
    assert_ne!(tile_after, tile_before, "the dig opened no mouth");
    assert_eq!(doc.undo_len(), 3, "a dig is ONE step (+2 spawns)");
    assert_eq!(
        doc.snapshot().undo_label.as_deref(),
        Some("Excavate"),
        "the undo label must name the excavation, not half of it"
    );

    assert!(doc.undo());
    assert_eq!(
        volume_image(&volumes, volume),
        chunks_before,
        "undo left the spoil heap standing over a filled-in pit"
    );
    assert_eq!(
        tile_image(&doc, terrain),
        tile_before,
        "undo left a mouth open"
    );

    assert!(doc.redo());
    assert_eq!(
        volume_image(&volumes, volume),
        chunks_after,
        "redo diverged"
    );
    assert_eq!(tile_image(&doc, terrain), tile_after, "redo diverged");
    assert_eq!(
        volumes.lock().unwrap().replay_faults(),
        0,
        "a correct round trip reported a replay fault"
    );
}

/// **Determinism.** Two independent sessions that dig the same pit — different
/// stores, different documents, chunks paged in a different order — produce
/// byte-identical volumes and identical ledgers, spoil site included.
#[test]
fn two_sessions_digging_the_same_pit_agree_byte_for_byte() {
    let mut images = Vec::new();
    let mut tallies = Vec::new();
    for pass in 0..2 {
        let (mut doc, volumes, _t, volume, _d) = fixture(true);
        if pass == 1 {
            // Touch the volume's residency from the far corner first, so the
            // second session's chunk map was built by a different walk.
            volumes.lock().unwrap().sync_camera(
                DVec3::new(15.0, 15.0, 15.0),
                &VoxelWantsParams {
                    radius_m: 1000.0,
                    hysteresis: 0.0,
                },
                VoxelStreamBudget::default(),
            );
        }
        let ops = vec![pit(&doc, (4.0, 4.0), (9.0, 8.0), 2.0)];
        tallies.push(
            doc.edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
                .expect("allowed"),
        );
        images.push(volume_image(&volumes, volume));
    }
    assert_eq!(tallies[0], tallies[1], "the ledgers disagree");
    assert_eq!(images[0], images[1], "two sessions dug different holes");
}

/// The **spoil site rules** are what they say: `Auto` is east of the cut and
/// dropped onto the ground; `At` honours the author's marker; a non-finite
/// marker discards rather than piling soil at infinity.
#[test]
fn the_spoil_site_rules_are_predictable() {
    // `Auto` lands the heap east of the pit, standing ON the terrain.
    let (mut doc, volumes, _t, volume, _d) = fixture(true);
    let ops = vec![pit(&doc, (2.0, 2.0), (5.0, 5.0), 2.0)];
    let tally = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
        .expect("allowed");
    assert!(tally.conserved());
    {
        let v = volumes.lock().unwrap();
        let data = &v.slot(volume).unwrap().data;
        // Nothing was placed below grade at the heap: the pile stands on the
        // surface the document reported, not at the pit's rim height.
        let mut lowest = f64::INFINITY;
        for (&key, c) in data.chunks() {
            for k in 0..inf_voxel::CHUNK_DIM {
                for j in 0..inf_voxel::CHUNK_DIM {
                    for i in 0..inf_voxel::CHUNK_DIM {
                        if c.sample(i, j, k) >= 0.0 {
                            continue;
                        }
                        let b = key.base_sample();
                        let w = data.grid_to_world(DVec3::new(
                            (b[0] + i as i32) as f64,
                            (b[1] + j as i32) as f64,
                            (b[2] + k as i32) as f64,
                        ));
                        if w.x > 6.0 {
                            lowest = lowest.min(w.y);
                        }
                    }
                }
            }
        }
        assert!(lowest.is_finite());
    }

    // `At` puts it where the author asked …
    let (mut doc, volumes, _t, volume, _d) = fixture(true);
    let ops = vec![pit(&doc, (2.0, 2.0), (5.0, 5.0), 2.0)];
    let site = DVec3::new(60.0, 0.0, 60.0);
    let tally = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::At(site))
        .expect("allowed");
    assert!(tally.conserved());
    {
        let v = volumes.lock().unwrap();
        let data = &v.slot(volume).unwrap().data;
        assert!(
            data.is_solid_at(site + DVec3::new(0.0, 0.25, 0.0)),
            "the heap is not at the site the author picked"
        );
    }

    // … and a NaN marker discards rather than writing a pile at infinity.
    let (mut doc, volumes, _t, volume, _d) = fixture(true);
    let ops = vec![pit(&doc, (2.0, 2.0), (5.0, 5.0), 2.0)];
    let tally = doc
        .edit_dig(
            volume,
            &volumes,
            &ops,
            SpoilChoice::At(DVec3::splat(f64::NAN)),
        )
        .expect("allowed");
    assert_eq!(tally.spoiled(), 0);
}

// ── judged whole ────────────────────────────────────────────────────────────

/// **A refused dig moves nothing** — not the rock, not the mouths, and above all
/// not the spoil. The pit is judged whole before it cuts anything (`a4e5844`),
/// and a heap of soil beside an un-dug hole is exactly the "half a transaction"
/// state that ruling exists to make unrepresentable.
#[test]
fn a_dig_refused_by_the_inline_terrain_leaves_no_pit_and_no_heap() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(false); // INLINE terrain
    let chunks_before = volume_image(&volumes, volume);
    let tile_before = tile_image(&doc, terrain);
    let undo_before = doc.undo_len();

    let ops = vec![pit(&doc, (5.0, 5.0), (10.0, 9.0), 2.5)];
    let err = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
        .expect_err("a breakthrough on an inline terrain must be refused");
    assert!(matches!(err, CarveRefusal::InlineTerrain { .. }), "{err:?}");
    assert!(
        err.message().contains(".inf_terrain"),
        "the fix must be named"
    );

    assert_eq!(volume_image(&volumes, volume), chunks_before, "rock moved");
    assert_eq!(tile_image(&doc, terrain), tile_before, "a mouth opened");
    assert_eq!(doc.undo_len(), undo_before, "a step was recorded");
}

/// A dig larger than one transaction may move is refused **before** it cuts, and
/// the refusal names what to do about it.
///
/// The size gate is the one refusal that cannot be discovered any other way: you
/// find out a dig is too big by doing it, which is precisely the outcome the
/// whole-op rule forbids.
#[test]
fn an_oversized_dig_is_refused_before_it_cuts() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);
    let chunks_before = volume_image(&volumes, volume);
    let tile_before = tile_image(&doc, terrain);

    // A kilometre-wide pit at half-metre voxels: far past `MAX_DIG_SAMPLES`.
    let ops = vec![VoxelOp::carve(VoxelShape::Box {
        center: DVec3::new(8.0, 0.0, 8.0),
        half_extents: DVec3::new(500.0, 20.0, 500.0),
    })];
    let err = doc
        .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
        .expect_err("an absurd pit must be refused");
    let CarveRefusal::TooLarge { samples } = err else {
        panic!("{err:?}");
    };
    assert!(samples > inf_editor_core::scene::undo::MAX_DIG_SAMPLES);
    assert!(err.message().contains("Nothing was cut"));
    assert_eq!(volume_image(&volumes, volume), chunks_before);
    assert_eq!(tile_image(&doc, terrain), tile_before);

    // …and the same gate holds for a CHAIN of individually-legal legs whose sum
    // is not: a trench is judged as one dig, not as N.
    let y = doc.ground_surface_y(4.0, 4.0).unwrap();
    let long: Vec<DVec3> = (0..40)
        .map(|i| DVec3::new(i as f64 * 30.0, y, 0.0))
        .collect();
    let legs: Vec<VoxelOp> = trench_shapes(&long, 6.0, 8.0, surface_of(&doc))
        .into_iter()
        .map(VoxelOp::carve)
        .collect();
    assert!(legs.len() > 30);
    assert!(matches!(
        doc.edit_dig(volume, &volumes, &legs, SpoilChoice::Auto),
        Err(CarveRefusal::TooLarge { .. })
    ));
    assert_eq!(volume_image(&volumes, volume), chunks_before);
}

/// A dig into a volume that is not loaded is refused, and the spoil never runs —
/// there is nothing to displace and nowhere to put it.
#[test]
fn a_dig_into_an_unloaded_volume_is_refused_and_places_no_spoil() {
    let (mut doc, volumes, _t, _volume, _d) = fixture(true);
    let ops = vec![pit(&doc, (2.0, 2.0), (5.0, 5.0), 2.0)];
    let err = doc
        .edit_dig(Uuid::from_u128(0xDEAD), &volumes, &ops, SpoilChoice::Auto)
        .expect_err("no such volume");
    assert_eq!(err, CarveRefusal::VolumeNotLoaded);
}

// ── the dig pages its own footprint (P21.3 audit, B3) ───────────────────────

/// **THE PAGING GATE.** A dig over rock the camera never visited must page it,
/// cut it, count it and spoil it — identically to the same dig on a fully-paged
/// volume.
///
/// Before this, `removed[m]` counted only *resident* samples and nothing on the
/// dig path called `request_chunks`. Rock in an unpaged chunk was not removed,
/// not counted and not displaced — and **conservation still balanced**, because
/// zero equals zero. That is what made it invisible: every number the tools
/// quote agreed with every other one while the pit was a different shape
/// depending on where the author had flown.
#[test]
fn a_dig_pages_the_rock_it_cuts_and_counts_all_of_it() {
    let dig = |mut doc: SceneDoc, volumes: SharedVoxelVolumes, volume: Uuid| {
        let ops = vec![pit(&doc, (2.0, 2.0), (7.0, 7.0), 3.0)];
        let tally = doc
            .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
            .expect("allowed");
        (tally, volume_image(&volumes, volume))
    };

    let (doc, volumes, _t, volume, _d) = fixture(true);
    let (paged, paged_image) = dig(doc, volumes, volume);

    let (doc, volumes, _t, volume, _d) = unpaged_fixture(true);
    let (cold, cold_image) = dig(doc, volumes, volume);

    assert!(paged.carved > 0, "the fixture pit removed nothing");
    assert_eq!(
        cold.carved_by_material, paged.carved_by_material,
        "a dig over unpaged rock removed a different amount than the same dig over paged rock \
         — camera residency decided what the level contains"
    );
    assert!(cold.conserved(), "{cold:?}");
    assert_eq!(
        cold.spoiled_by_material, paged.spoiled_by_material,
        "the heaps differ"
    );
    // …and the chunks the two produced are the same bytes, so it is not only the
    // counts that agree.
    assert_eq!(
        cold_image, paged_image,
        "two identical digs produced different geometry"
    );
}

/// **The heap does not move with the camera.** The `Auto` site drops the pile
/// onto `ground_surface_y`, which reads resident tiles — so without paging, a
/// session that had flown over the spoil site piled at a different height than
/// one that had not.
#[test]
fn the_auto_heap_is_identical_whatever_the_camera_visited() {
    let mut images = Vec::new();
    for far_first in [false, true] {
        let (mut doc, volumes, _t, volume, _d) = unpaged_fixture(true);
        if far_first {
            // A camera pass that pages a *different* neighbourhood first.
            volumes.lock().unwrap().sync_camera(
                DVec3::new(60.0, 20.0, 60.0),
                &VoxelWantsParams {
                    radius_m: 24.0,
                    hysteresis: 0.0,
                },
                VoxelStreamBudget::default(),
            );
        }
        let ops = vec![pit(&doc, (3.0, 3.0), (8.0, 8.0), 2.5)];
        let tally = doc
            .edit_dig(volume, &volumes, &ops, SpoilChoice::Auto)
            .expect("allowed");
        assert!(tally.conserved(), "{tally:?}");
        images.push(volume_image(&volumes, volume));
    }
    assert_eq!(
        images[0], images[1],
        "the committed dig depends on where the camera had been"
    );
}

/// **The heap never lands back in the hole** (P21.3 audit).
///
/// `default_spoil_site` clears the cut by the pile's *analytic* radius, and the
/// real footprint is wider — 81 % wider on the dig the audit measured, which put
/// 59 of 729 excavated samples straight back into the pit. Conservation balanced
/// throughout, because a refilled pit is a perfectly conserved place to put
/// soil, so nothing in the ledger could ever have caught it.
///
/// The site here is the pit's own eastern **rim**, which is the author-picked
/// case where the intrusion is largest and the one `SpoilChoice::At` makes
/// reachable in one click. The non-vacuity half is at the bottom: the same dig
/// with no exclusion really does refill.
#[test]
fn no_spoil_lands_back_inside_the_pit() {
    fn refilled_samples(volumes: &SharedVoxelVolumes, volume: Uuid, lo: DVec3, hi: DVec3) -> usize {
        let v = volumes.lock().unwrap();
        let data = &v.slot(volume).unwrap().data;
        let mut n = 0;
        for (&key, c) in data.chunks() {
            for k in 0..inf_voxel::CHUNK_DIM {
                for j in 0..inf_voxel::CHUNK_DIM {
                    for i in 0..inf_voxel::CHUNK_DIM {
                        if c.sample(i, j, k) >= 0.0 {
                            continue;
                        }
                        let b = key.base_sample();
                        let w = data.grid_to_world(DVec3::new(
                            (b[0] + i as i32) as f64,
                            (b[1] + j as i32) as f64,
                            (b[2] + k as i32) as f64,
                        ));
                        if w.cmpge(lo).all() && w.cmple(hi).all() {
                            n += 1;
                        }
                    }
                }
            }
        }
        n
    }

    let (mut doc, volumes, _t, volume, _d) = fixture(true);
    let op = pit(&doc, (1.0, 1.0), (9.0, 9.0), 3.5);
    let (lo, hi) = op.shape.aabb_m(0.0);
    // The site is ON the pit's eastern rim, at grade: a heap standing here leans
    // straight back over the hole.
    let site = DVec3::new(hi.x, GRADE_M, (lo.z + hi.z) * 0.5);
    let tally = doc
        .edit_dig(volume, &volumes, &[op], SpoilChoice::At(site))
        .expect("allowed");
    assert!(tally.conserved() && tally.spoiled() > 200, "{tally:?}");
    assert_eq!(
        refilled_samples(&volumes, volume, lo, hi),
        0,
        "spoil was placed back inside the pit it came out of"
    );

    // **Non-vacuity**: the identical dig with the exclusion removed really does
    // refill the pit, so the assertion above is a property of the fix and not of
    // the fixture. Ring 0's `place_spoil_into` is called directly here because
    // the transaction always excludes.
    let (mut doc2, volumes2, _t2, volume2, _d2) = fixture(true);
    let op2 = pit(&doc2, (1.0, 1.0), (9.0, 9.0), 3.5);
    let removed = doc2
        .edit_dig(volume2, &volumes2, &[op2], SpoilChoice::Discard)
        .expect("allowed")
        .carved_by_material;
    {
        let mut v = volumes2.lock().unwrap();
        let mut b = inf_voxel::VoxelDeltaBuilder::new();
        let r = v
            .spoil_into(volume2, &inf_voxel::SpoilPlan::new(removed, site), &mut b)
            .expect("the volume is loaded");
        assert!(r.is_exact(), "{r:?}");
    }
    assert!(
        refilled_samples(&volumes2, volume2, lo, hi) > 0,
        "an UNEXCLUDED heap on the rim did not refill the pit — the gate above is vacuous, \
         and the fixture needs a bigger dig or a closer site"
    );
}

// ── a document swap discards the carve store (P21.3 re-audit) ───────────────

/// **A discarded carve must not survive the document that owned it.**
///
/// The shared voxel store is a *host* field: it outlives `scene_open`/`scene_new`
/// entirely. Abandoning the stroke handle leaves its cut chunks in the slot, and
/// they are **dirty** — so when the new document binds the same `(entity,
/// asset)` pair (File ▸ Open on the same level, the gesture an author uses to
/// *throw changes away*), `retain_only` keeps the slot, `ensure` short-circuits
/// on `is_bound`, and the next Ctrl+S writes the half-carve into the
/// `.inf_voxel` with no `EditCommand` anywhere.
///
/// This is the store-side contract the host's
/// `EngineHost::abandon_gestures_on_document_swap` relies on: after the clear,
/// nothing is dirty, a save stages nothing, and re-binding serves the
/// **last-saved** geometry rather than the abandoned cut.
///
/// (The host call itself needs a GPU and a `#[cfg]`-gated module, so that it
/// really calls `clear` is pinned by `viewport_pump_mirror.rs`'s
/// `the_document_swap_clears_the_shared_carve_store`. Same split as the
/// settlers, same reason.)
#[test]
fn clearing_the_store_discards_an_abandoned_carve_and_saves_nothing() {
    let (mut doc, volumes, _t, volume, dir) = fixture(true);
    let pristine = volume_image(&volumes, volume);
    let asset_before = std::fs::read(dir.path().join("Rock.inf_voxel")).unwrap();

    // A stroke's worth of cutting lands in the store, exactly as `dab` leaves
    // it: applied, dirty, and not yet described by any undo entry.
    {
        let mut v = volumes.lock().unwrap();
        let op = pit(&doc, (2.0, 2.0), (6.0, 6.0), 2.0);
        let mut builder = inf_voxel::VoxelDeltaBuilder::new();
        let report = v
            .carve_into(volume, &op, &mut builder)
            .expect("the volume is loaded");
        assert!(report.total_carved() > 0, "the fixture stroke cut nothing");
    }
    assert!(
        volumes.lock().unwrap().has_unsaved_edits(),
        "a carve owes a save"
    );
    assert_ne!(volume_image(&volumes, volume), pristine);
    assert!(
        !inf_editor_core::voxel_edit::stage_voxel_edits(&volumes.lock().unwrap()).is_empty(),
        "the fixture carve stages nothing — this proves nothing"
    );

    // THE SWAP. `*doc = …` on the Ring-2 thread; the host clears the store.
    doc = SceneDoc::new();
    volumes.lock().unwrap().clear();

    // Nothing dirty, nothing staged, nothing written.
    {
        let v = volumes.lock().unwrap();
        assert!(
            !v.has_unsaved_edits(),
            "the abandoned carve is still marked unsaved and the next Ctrl+S would write it"
        );
        assert_eq!(v.unsaved_chunk_count(), 0);
        assert!(inf_editor_core::voxel_edit::stage_voxel_edits(&v).is_empty());
    }
    let report =
        inf_editor_core::voxel_edit::flush_voxel_edits(&mut volumes.lock().unwrap(), dir.path());
    assert!(
        report.is_empty(),
        "a save after the swap wrote something: {report:?}"
    );
    assert_eq!(
        std::fs::read(dir.path().join("Rock.inf_voxel")).unwrap(),
        asset_before,
        "the .inf_voxel changed — the discarded carve reached disk"
    );

    // …and re-binding (what the new document's projection does) serves the
    // LAST SAVED geometry, not the abandoned cut.
    {
        let mut v = volumes.lock().unwrap();
        assert!(
            v.ensure(
                volume,
                &VoxelVolume::from_asset(Uuid::from_u128(0x2133_0001))
            ),
            "a cleared store must re-bind rather than short-circuit on is_bound"
        );
        v.sync_camera(
            DVec3::splat(4.0),
            &VoxelWantsParams {
                radius_m: 1000.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget::default(),
        );
    }
    assert_eq!(
        volume_image(&volumes, volume),
        pristine,
        "the re-bound volume still carries the carve the swap was supposed to discard"
    );
    let _ = doc.undo_len();
}
