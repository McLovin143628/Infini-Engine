//! **The coupled carve** (P21.2 deliverable 6): one gesture cuts rock and opens
//! the ground above it, one Ctrl+Z puts both back, and a cut that would break
//! through a terrain that cannot save the hole is refused outright.
//!
//! # Why this is an integration test and not a unit test
//!
//! The transaction spans three owners: `inf_voxel`'s SDF chunks, `inf_terrain`'s
//! hole mask, and `SceneDoc`'s history. Each half is unit-tested where it lives
//! (`inf_voxel::coupling` pins the rule and its invertibility; `inf_voxel::ops`
//! pins the stroke merge). What no unit can see is whether they stay **one**
//! step — and "the rock came back but the ground did not" is a bug an author
//! finds by looking at a cave with no floor.

use std::path::Path;

use glam::{DVec2, DVec3};
use inf_ecs::components::{Terrain, VoxelVolume};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::undo::{CarveStroke, CarveVerdict};
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::voxel_store::{shared_volumes, SharedVoxelVolumes};
use inf_voxel::{
    ChunkKey, VoxelChunk, VoxelData, VoxelOp, VoxelShape, VoxelStreamBudget, VoxelWantsParams,
};
use uuid::Uuid;

/// Voxel edge length of the fixture volumes, metres. Coarse enough that one
/// brush reaches many samples without the test walking millions.
const VOXEL_M: f64 = 0.5;

/// Write a solid 2 × 2 × 2-chunk block of rock as a loose `.inf_voxel` with a
/// sidecar — a hillside interior, which is what a carve brush is for.
fn write_rock(dir: &Path, name: &str, guid: Uuid) {
    let mut v = VoxelData::new(VOXEL_M);
    for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
        v.insert_chunk(key, VoxelChunk::solid(1));
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

/// A document with one terrain (flat at `y = 4`, `asset_backed` or not) and one
/// voxel volume bound to a solid block, plus the shared store holding it.
///
/// Returns `(doc, volumes, terrain_guid, volume_guid)`.
fn fixture(asset_backed: bool) -> (SceneDoc, SharedVoxelVolumes, Uuid, Uuid, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let asset = Uuid::from_u128(0x212C_0001);
    write_rock(dir.path(), "Rock.inf_voxel", asset);

    let mut doc = SceneDoc::new();

    // ── the terrain ──
    let terrain = doc.edit_create(SpawnKind::Terrain, "Ground", None);
    {
        let e = doc.entity_of(terrain).unwrap();
        let mut t = doc.world_mut().world_mut().get_mut::<Terrain>(e).unwrap();
        // Replace the starter hill with a flat sheet at y = 4, one 33² tile at
        // 1 m spacing, so the surface sits inside the rock block (which spans
        // y ∈ [0, 16] voxels = [0, 8] m).
        t.data = inf_terrain::TerrainData::new(33, 1.0);
        t.data.author_tile((0, 0), |_, _| 4.0);
        // Asset-backed-ness is the whole subject of the refusal test. There is no
        // `edit_*` for `Terrain.asset` (it is an `#[reflect(ignore)]` asset
        // reference the import wizard writes), so the fixture writes it directly.
        t.asset = asset_backed.then(|| Uuid::from_u128(0x212C_0002));
    }

    // ── the volume ──
    let volume = doc.edit_create(SpawnKind::Empty, "Cave", None);
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
        assert!(
            v.ensure(volume, &VoxelVolume::from_asset(asset)),
            "the fixture volume must bind"
        );
        // Binding pages nothing (P21.2) — page the whole block in.
        let report = v.sync_camera(
            DVec3::splat(4.0),
            &VoxelWantsParams {
                radius_m: 1000.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget::default(),
        );
        assert!(report.loaded > 0, "the fixture volume must page in");
    }
    (doc, volumes, terrain, volume, dir)
}

/// The cut every test below uses: a ball straddling the ground plane at
/// `y = 4`, so it takes rock **and** breaks the surface.
fn breakthrough() -> VoxelOp {
    VoxelOp::carve(VoxelShape::Sphere {
        center: DVec3::new(8.0, 4.0, 8.0),
        radius_m: 2.5,
    })
}

/// The hole mask of the fixture's single tile, as raw bytes — what the
/// byte-identity assertions compare.
fn tile_image(doc: &SceneDoc, terrain: Uuid) -> Vec<u8> {
    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    inf_terrain::asset::encode_tile(data.get_tile((0, 0)).unwrap()).unwrap()
}

/// Solid voxel samples across the volume — the cheap "is the rock back?" probe.
fn solid_count(volumes: &SharedVoxelVolumes, volume: Uuid) -> usize {
    let v = volumes.lock().unwrap();
    let slot = v.slot(volume).unwrap();
    slot.data
        .chunks()
        .map(|(_, c)| {
            (0..inf_voxel::CHUNK_DIM)
                .flat_map(|k| {
                    (0..inf_voxel::CHUNK_DIM)
                        .flat_map(move |j| (0..inf_voxel::CHUNK_DIM).map(move |i| (i, j, k)))
                })
                .filter(|&(i, j, k)| c.sample(i, j, k) < 0.0)
                .count()
        })
        .sum()
}

/// Every resident chunk's **mesh stamp** — the key the viewport's GPU cache
/// re-uploads off. `slot.meshes.version(key)`, ascending by key.
fn mesh_stamps(volumes: &SharedVoxelVolumes, volume: Uuid) -> Vec<(ChunkKey, u64)> {
    let v = volumes.lock().unwrap();
    let slot = v.slot(volume).unwrap();
    slot.data
        .chunks()
        .map(|(&k, _)| (k, slot.meshes.version(k)))
        .collect()
}

/// **The re-upload gate.** A carve must move the mesh stamps of the chunks it
/// reached, and bump the document version — those two are the *only* things that
/// make the cave appear on screen.
///
/// The viewport's GPU cache is keyed on `VoxelMeshCache::version`, and its
/// projection is document-version-gated. A carve that moved neither would mesh
/// correctly in the store and be invisible in the editor — the failure mode that
/// looks like "the brush does nothing" and has no rendering explanation.
///
/// Both halves are asserted for a **deep** cut as well, which touches no terrain
/// tile at all: that is the case where a version bump driven only by the hole
/// mask would silently stop happening.
#[test]
fn a_carve_moves_the_mesh_stamps_and_the_document_version() {
    for (label, op, expect_holes) in [
        ("breakthrough", breakthrough(), true),
        (
            "deep",
            VoxelOp::carve(VoxelShape::Sphere {
                center: DVec3::new(8.0, 1.0, 8.0),
                radius_m: 1.5,
            }),
            false,
        ),
    ] {
        let (mut doc, volumes, _t, volume, _d) = fixture(true);
        let before = mesh_stamps(&volumes, volume);
        let version = doc.version();
        assert!(!before.is_empty(), "{label}: the fixture has no chunks");

        let tally = doc
            .edit_carve(volume, &volumes, &op)
            .unwrap_or_else(|| panic!("{label} was refused"));
        assert!(tally.carved > 0, "{label}");
        assert_eq!(tally.holes > 0, expect_holes, "{label}");

        let after = mesh_stamps(&volumes, volume);
        let moved = before
            .iter()
            .zip(&after)
            .filter(|((ka, a), (kb, b))| {
                assert_eq!(ka, kb, "{label}: the chunk set changed");
                b > a
            })
            .count();
        assert!(
            moved > 0,
            "{label}: no mesh stamp moved, so the viewport's GPU cache would keep \
             drawing the rock the carve removed"
        );
        assert!(
            after.iter().zip(&before).all(|((_, a), (_, b))| a >= b),
            "{label}: a mesh stamp went BACKWARDS — the cache would treat stale \
             geometry as fresh"
        );
        assert!(
            doc.version() > version,
            "{label}: the document version did not move, so the version-gated \
             projection would never re-read the store"
        );
        assert!(doc.is_dirty(), "{label}: a carve is an unsaved edit");
    }
}

/// **The verdict gate.** A cut that breaks the surface of an *inline* terrain is
/// refused; the same cut over an asset-backed one is allowed; a cut that never
/// reaches a surface is allowed either way.
#[test]
fn the_verdict_refuses_only_a_breakthrough_on_an_inline_terrain() {
    let deep = VoxelOp::carve(VoxelShape::Sphere {
        center: DVec3::new(8.0, 1.0, 8.0),
        radius_m: 1.0,
    });

    let (doc, _v, terrain, _g, _d) = fixture(false);
    assert_eq!(
        doc.carve_verdict(&breakthrough().shape),
        CarveVerdict::RefusedInline { terrain },
        "an inline terrain cannot persist a mouth, so the cut must be refused"
    );
    assert!(!doc.carve_verdict(&breakthrough().shape).allowed());
    assert_eq!(
        doc.carve_verdict(&deep.shape),
        CarveVerdict::NoSurface,
        "a cave BELOW the surface needs no hole and is legal anywhere"
    );

    let (doc, _v, terrain, _g, _d) = fixture(true);
    assert_eq!(
        doc.carve_verdict(&breakthrough().shape),
        CarveVerdict::Opens {
            terrains: vec![terrain]
        }
    );
    assert!(doc.carve_verdict(&breakthrough().shape).allowed());
}

/// **The whole-op refusal.** A refused carve leaves *both* containers untouched —
/// zero holes AND zero voxel samples changed — and records nothing.
///
/// The voxel half is the load-bearing assertion. "Carve the rock, skip the holes"
/// is the tempting partial success, and it leaves the author with a cave that has
/// no mouth: geometry they cannot see, cannot reach, and have no reason to
/// suspect exists.
#[test]
fn a_refused_carve_changes_nothing_at_all() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(false);
    let before_tile = tile_image(&doc, terrain);
    let before_solid = solid_count(&volumes, volume);
    let before_undos = doc.undo_len();
    let before_version = doc.version();
    assert!(before_solid > 0, "the fixture must start as solid rock");

    assert_eq!(
        doc.edit_carve(volume, &volumes, &breakthrough()),
        None,
        "the refused carve reported success"
    );

    assert_eq!(
        solid_count(&volumes, volume),
        before_solid,
        "a refused carve removed voxels — the partial success this gate exists to stop"
    );
    assert_eq!(tile_image(&doc, terrain), before_tile, "it punched holes");
    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    assert!(!data.has_holes());
    assert_eq!(doc.undo_len(), before_undos, "it recorded an undo step");
    assert_eq!(doc.version(), before_version, "it bumped the version");
}

/// The same cut over an **asset-backed** terrain succeeds, and it really does
/// both halves: rock removed, mouth opened.
#[test]
fn an_asset_backed_carve_opens_the_ground_and_takes_the_rock() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);
    let before_solid = solid_count(&volumes, volume);

    let tally = doc
        .edit_carve(volume, &volumes, &breakthrough())
        .expect("the carve must be allowed");
    assert!(tally.carved > 0, "no rock was removed");
    assert!(tally.holes > 0, "no cave mouth was opened");
    assert!(!tally.is_noop());
    assert!(tally.net_removed_m3(VOXEL_M) > 0.0);

    assert!(solid_count(&volumes, volume) < before_solid);
    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    assert!(data.has_holes(), "the heightfield kept its surface");
    // The poisoned bilinear cell really removes the surface from a query — this
    // is what the character walking into the mouth reads.
    assert_eq!(data.height_at(DVec2::new(8.0, 8.0)), None);
    assert!(data.is_hole_at(DVec2::new(8.0, 8.0)));
    // …and well outside the cut the ground is untouched.
    assert_eq!(data.height_at(DVec2::new(28.0, 28.0)), Some(4.0));
}

/// **THE TRANSACTION GATE: three carves, three undos, three redos.**
///
/// Each Ctrl+Z must take back a whole cut — rock *and* mouth — and land the
/// document on the exact bytes it had after the previous one. A history that
/// reverted the halves separately (or dropped one) shows up here as a tile image
/// that does not match, or as rock that comes back over a hole that stays open.
#[test]
fn three_carves_undo_and_redo_as_three_whole_steps() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);

    // Snapshots after each step, starting from the untouched document.
    let mut tiles = vec![tile_image(&doc, terrain)];
    let mut solids = vec![solid_count(&volumes, volume)];
    let undos0 = doc.undo_len();

    for (step, x) in [6.0f64, 8.0, 10.0].into_iter().enumerate() {
        let op = VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(x, 4.0, 8.0),
            radius_m: 2.0,
        });
        let tally = doc
            .edit_carve(volume, &volumes, &op)
            .unwrap_or_else(|| panic!("carve {step} was refused"));
        assert!(tally.carved > 0 && tally.holes > 0, "carve {step}");
        assert_eq!(
            doc.undo_len(),
            undos0 + step + 1,
            "carve {step} is ONE step"
        );
        tiles.push(tile_image(&doc, terrain));
        solids.push(solid_count(&volumes, volume));
    }
    assert!(
        solids[3] < solids[2] && solids[2] < solids[1] && solids[1] < solids[0],
        "each carve must remove more rock: {solids:?}"
    );

    // Three undos, walking back down the snapshots.
    for step in (0..3).rev() {
        assert!(doc.undo(), "undo {step}");
        assert_eq!(
            tile_image(&doc, terrain),
            tiles[step],
            "undo {step} left the heightfield in a state no carve produced — the \
             two halves came apart"
        );
        assert_eq!(solid_count(&volumes, volume), solids[step], "undo {step}");
    }
    // Back to the start: the mask is GONE, not merely zeroed (`heal_holes`), so
    // the tile serializes byte-identically to one nothing ever carved.
    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    assert!(!data.has_holes());
    assert!(
        data.get_tile((0, 0)).unwrap().holes_are_default(),
        "undo left a healed-but-materialized mask — a permanent res²/8 scar"
    );
    assert_eq!(data.height_at(DVec2::new(8.0, 8.0)), Some(4.0));

    // …and three redos walk back up, byte for byte.
    for step in 0..3 {
        assert!(doc.redo(), "redo {step}");
        assert_eq!(tile_image(&doc, terrain), tiles[step + 1], "redo {step}");
        assert_eq!(
            solid_count(&volumes, volume),
            solids[step + 1],
            "redo {step}"
        );
    }
    assert!(!doc.can_redo());
}

/// A stroke — many overlapping dabs under one mouse-down — is **one** undo entry
/// that lands back on the pre-stroke bytes, not one entry per dab.
#[test]
fn a_carve_stroke_is_one_undo_step() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);
    let before_tile = tile_image(&doc, terrain);
    let before_solid = solid_count(&volumes, volume);
    let undos = doc.undo_len();

    let terrains = doc.carve_verdict(&breakthrough().shape).terrains().to_vec();
    assert_eq!(terrains, vec![terrain]);

    let mut stroke = CarveStroke::begin(volume, volumes.clone());
    for i in 0..5 {
        let op = VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(5.0 + i as f64, 4.0, 8.0),
            radius_m: 1.5,
        });
        stroke.dab(&mut doc, &op, &terrains);
    }
    let tally = stroke.tally();
    assert!(tally.carved > 0 && tally.holes > 0);
    assert!(doc.edit_commit_carve(stroke));
    assert_eq!(doc.undo_len(), undos + 1, "five dabs, one undo entry");
    assert!(solid_count(&volumes, volume) < before_solid);

    assert!(doc.undo());
    assert_eq!(
        tile_image(&doc, terrain),
        before_tile,
        "the stroke undid to a mid-drag state rather than to where it started"
    );
    assert_eq!(solid_count(&volumes, volume), before_solid);
}

/// Voxel samples lying **exactly** on `shape`'s surface (`distance == 0`).
///
/// The one place a carve and its own fill do not cancel: the carve writes
/// `max(old, −0) = 0` and `0` is not solid, while the fill writes
/// `min(0, 0) = 0` and cannot make it solid again. A measure-zero shell of
/// lattice points, counted here rather than fudged with a tolerance.
fn shell_count(volumes: &SharedVoxelVolumes, volume: Uuid, shape: &VoxelShape) -> usize {
    let v = volumes.lock().unwrap();
    let data = &v.slot(volume).unwrap().data;
    let mut n = 0;
    for (&key, _) in data.chunks() {
        let b = key.base_sample();
        for k in 0..inf_voxel::CHUNK_DIM as i32 {
            for j in 0..inf_voxel::CHUNK_DIM as i32 {
                for i in 0..inf_voxel::CHUNK_DIM as i32 {
                    let p = data.grid_to_world(DVec3::new(
                        (b[0] + i) as f64,
                        (b[1] + j) as f64,
                        (b[2] + k) as f64,
                    ));
                    n += usize::from(shape.distance_m(p) == 0.0);
                }
            }
        }
    }
    n
}

/// **The spline tunnel's atomicity gate.** A chain of capsules is ONE
/// transaction: one undo entry for the whole tube, and — if any segment is
/// refused — nothing written at all, not even the segments before it.
///
/// The partial case is the load-bearing one. A chain that judged as it went would
/// leave the first half of a tunnel in the volume with **no undo entry describing
/// it**: a partial success that is not even reversible, which is strictly worse
/// than the sealed-cave one `RefusedInline` exists to prevent. The fixture makes
/// the *last* segment the offending one, so a "judge as you go" implementation
/// gets two clean segments in before it notices.
#[test]
fn a_carve_chain_is_all_or_nothing_and_one_undo_step() {
    let leg = |x: f64, y: f64| VoxelShape::Capsule {
        a: DVec3::new(x, y, 8.0),
        b: DVec3::new(x + 3.0, y, 8.0),
        radius_m: 1.5,
    };
    // Three legs: two buried, then one that surfaces.
    let ops: Vec<VoxelOp> = [leg(2.0, 1.5), leg(5.0, 1.5), leg(8.0, 4.0)]
        .into_iter()
        .map(VoxelOp::carve)
        .collect();

    // ── over an INLINE terrain: refused whole ──
    let (mut doc, volumes, terrain, volume, _d) = fixture(false);
    let before_tile = tile_image(&doc, terrain);
    let before_solid = solid_count(&volumes, volume);
    let undos = doc.undo_len();
    assert_eq!(doc.edit_carve_path(volume, &volumes, &ops), None);
    assert_eq!(
        solid_count(&volumes, volume),
        before_solid,
        "the two buried segments were cut before the third was judged — a partial \
         tunnel with no undo entry"
    );
    assert_eq!(tile_image(&doc, terrain), before_tile);
    assert_eq!(doc.undo_len(), undos);

    // ── over an ASSET-BACKED terrain: cut whole, in one step ──
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);
    let before_tile = tile_image(&doc, terrain);
    let before_solid = solid_count(&volumes, volume);
    let undos = doc.undo_len();
    let tally = doc
        .edit_carve_path(volume, &volumes, &ops)
        .expect("allowed over an asset-backed terrain");
    assert!(tally.carved > 0 && tally.holes > 0, "{tally:?}");
    assert_eq!(doc.undo_len(), undos + 1, "three legs, ONE undo entry");
    assert!(solid_count(&volumes, volume) < before_solid);

    assert!(doc.undo());
    assert_eq!(
        tile_image(&doc, terrain),
        before_tile,
        "one Ctrl+Z must take the whole tunnel's mouths back"
    );
    assert_eq!(
        solid_count(&volumes, volume),
        before_solid,
        "one Ctrl+Z must take the whole tunnel's rock back"
    );
}

/// A **fill** over the same region closes exactly the mouths the carve opened,
/// leaving the tile byte-identical — the deliverable's symmetric-healing gate,
/// asserted through the editor's own doors and not just through the Ring-0 rule.
///
/// The voxel half is checked too, and is deliberately **not** claimed to be an
/// exact identity: the CSG ops leave the lattice points sitting exactly on the
/// brush surface empty (see [`shell_count`]). Counting that shell rather than
/// asserting "close enough" is what keeps this a gate — if a future op change
/// made the fill lose a sample anywhere else, the arithmetic stops balancing.
#[test]
fn a_fill_over_the_carve_restores_byte_identical_ground() {
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);
    let before_tile = tile_image(&doc, terrain);
    let before_solid = solid_count(&volumes, volume);

    let shape = breakthrough().shape;
    let shell = shell_count(&volumes, volume, &shape);
    assert!(doc
        .edit_carve(volume, &volumes, &VoxelOp::carve(shape))
        .is_some());
    assert_ne!(tile_image(&doc, terrain), before_tile);

    // Material 1 — the fixture's rock — so the refilled samples carry the block's
    // own material and nothing but the surface shell is left hollow.
    assert!(doc
        .edit_carve(volume, &volumes, &VoxelOp::fill(shape, 1))
        .is_some());
    assert_eq!(
        tile_image(&doc, terrain),
        before_tile,
        "the fill left a scar in the hole mask — carve → fill is not invertible"
    );
    assert_eq!(
        solid_count(&volumes, volume),
        before_solid - shell,
        "the fill lost solid samples somewhere other than the brush surface"
    );
    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    assert!(!data.has_holes());
    assert!(data.get_tile((0, 0)).unwrap().holes_are_default());
}

/// The defensive advisory: a document that arrived carrying holes on an inline
/// terrain — an older session, a hand-edited file, a terrain converted back to
/// inline after a carve — is detectable, with the fix named.
///
/// The tools refuse to *create* that state (above); this is what finds a document
/// already in it, before a save quietly seals every mouth in the level.
#[test]
fn holes_on_an_inline_terrain_are_detected_as_a_lossy_document() {
    // Carve legally against an asset-backed terrain …
    let (mut doc, volumes, terrain, volume, _d) = fixture(true);
    assert!(doc.edit_carve(volume, &volumes, &breakthrough()).is_some());

    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    assert!(inf_voxel::inline_hole_advisory(data, true).is_none());

    // … then make the terrain inline behind the tools' back, which is exactly the
    // state a stale document can be in.
    {
        let e = doc.entity_of(terrain).unwrap();
        doc.world_mut()
            .world_mut()
            .get_mut::<Terrain>(e)
            .unwrap()
            .asset = None;
    }
    let (data, _) = doc.terrain_data_and_origin(terrain).unwrap();
    let note = inf_voxel::inline_hole_advisory(data, false).expect("the advisory must fire");
    assert!(note.tiles > 0);
    assert!(note.message().contains(".inf_terrain"));
}
