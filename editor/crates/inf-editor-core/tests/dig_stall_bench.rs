//! **The measurement behind the P21.4 off-lock re-mesh numbers.** Not a gate:
//! `#[ignore]`d, because a timing assertion on a shared CI runner is a flake
//! generator, and because the claim it supports is a *ratio* an author can
//! reproduce rather than a ceiling anyone should ratchet.
//!
//! Run it with:
//!
//! ```sh
//! cargo test -p inf-editor-core --test dig_stall_bench --release -- --ignored --nocapture
//! ```
//!
//! What it prints, and what the ROADMAP's P21.4 block quotes (release, one
//! machine, 108 chunks at 0.5 m, a 405 000-sample box cut):
//!
//! | | under the volumes guard | re-meshed after |
//! |---|---|---|
//! | spoil discarded, re-mesh inline (pre-P21.4) | 86.1 ms | — |
//! | spoil discarded, re-mesh deferred | 72.1 ms | 11.3 ms |
//! | **Auto spoil**, re-mesh deferred | 197.8 ms | 26.8 ms |
//!
//! So the re-mesh is **12–16 %** of a big dig's lock time, and it is the only
//! part with no reason to be there: the remaining 84–88 % is the cut and the
//! spoil search, which *are* the edit. Deferral moves a real slice off the lock
//! and does not make a big dig interactive — the honest conclusion, recorded in
//! the ROADMAP rather than rounded up.

use glam::DVec3;
use inf_ecs::components::{Terrain, VoxelVolume};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::undo::SpoilChoice;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::voxel_store::{shared_volumes, SharedVoxelVolumes};
use inf_voxel::{
    ChunkKey, VoxelChunk, VoxelData, VoxelOp, VoxelShape, VoxelStreamBudget, VoxelWantsParams,
};
use uuid::Uuid;

const VOXEL_M: f64 = 0.5;
const ASSET: u128 = 0x9001;

fn fixture() -> (SceneDoc, SharedVoxelVolumes, Uuid, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut v = VoxelData::new(VOXEL_M);
    for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(5, 2, 5)) {
        v.insert_chunk(key, VoxelChunk::solid(1));
    }
    eprintln!("fixture chunks = {}", v.chunk_count());
    let asset = inf_voxel::build_voxel_asset(&v).unwrap();
    let path = dir.path().join("Rock.inf_voxel");
    let bytes = inf_voxel::write_voxel_asset(&path, &asset).unwrap();
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(Uuid::from_u128(ASSET)),
        inf_asset::AssetKind::VoxelVolume,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .unwrap();

    let mut doc = SceneDoc::new();
    let terrain = doc.edit_create(SpawnKind::Terrain, "Ground", None);
    {
        let e = doc.entity_of(terrain).unwrap();
        let mut t = doc.world_mut().world_mut().get_mut::<Terrain>(e).unwrap();
        t.data = inf_terrain::TerrainData::new(65, 1.0);
        t.data.author_tile((0, 0), |_, _| 40.0);
        t.asset = Some(Uuid::from_u128(0x9002));
    }
    let volume = doc.edit_create(SpawnKind::Empty, "Excavation", None);
    {
        let e = doc.entity_of(volume).unwrap();
        doc.world_mut()
            .world_mut()
            .entity_mut(e)
            .insert(VoxelVolume {
                asset: Some(Uuid::from_u128(ASSET)),
                voxel_size_m: VOXEL_M,
                ..VoxelVolume::default()
            });
    }
    let volumes = shared_volumes();
    {
        let mut store = volumes.lock().unwrap();
        store.set_content_root(Some(dir.path().to_path_buf()));
        assert!(store.ensure(volume, &VoxelVolume::from_asset(Uuid::from_u128(ASSET))));
        let report = store.sync_camera(
            DVec3::splat(24.0),
            &VoxelWantsParams {
                radius_m: 4000.0,
                hysteresis: 0.0,
            },
            VoxelStreamBudget {
                max_resident_chunks: 100_000,
                max_loads_per_sync: 100_000,
            },
        );
        eprintln!("paged {} chunk(s)", report.loaded);
    }
    (doc, volumes, volume, dir)
}

#[test]
#[ignore]
fn measure_dig_stall() {
    let (mut doc, volumes, volume, _dir) = fixture();
    let op = VoxelOp::carve(VoxelShape::Box {
        center: DVec3::new(24.0, 12.0, 24.0),
        half_extents: DVec3::new(20.0, 10.0, 20.0),
    });
    eprintln!("samples = {}", op.shape.affected_sample_count(VOXEL_M));
    let t0 = std::time::Instant::now();
    let tally = doc.edit_dig(volume, &volumes, &[op], SpoilChoice::Auto);
    eprintln!(
        "edit_dig = {:.1} ms (carved {:?})",
        t0.elapsed().as_secs_f64() * 1000.0,
        tally.map(|t| t.carved)
    );
    let t1 = std::time::Instant::now();
    {
        let mut store = volumes.lock().unwrap();
        store.resync(volume);
    }
    eprintln!(
        "follow-up resync = {:.1} ms",
        t1.elapsed().as_secs_f64() * 1000.0
    );
}
