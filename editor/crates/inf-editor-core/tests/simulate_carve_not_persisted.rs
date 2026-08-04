//! **A gameplay carve is never written into the author's `.inf_voxel`** (P21.4).
//!
//! The editor's Simulate fold mirrors a session's runtime carves into the
//! viewport's voxel store, so an author watching Play sees what the Blueprint
//! digs. That store is *also* the one a save stages from — `SceneDoc::save` →
//! `voxel_edit::stage_voxel_edits` → `write_voxel_edits`, reached from Ring 2's
//! `commands/scene.rs::{stage_voxels, voxels_still_dirty}`, which are thin
//! wrappers over the two functions this file drives directly.
//!
//! The first version of the fold copied through `VoxelData::insert_chunk`, which
//! marks the chunk **dirty**. So:
//!
//! > Simulate → the borer digs → Stop → Ctrl+S
//!
//! wrote a player's runtime craters into the authored asset — flatly
//! contradicting the rule the phase recorded, that nothing a game digs is
//! persisted. And `has_unsaved_edits` stayed true after a clean save, so the
//! crash-recovery file was never released and the author was told they had
//! unsaved caves they had never carved.
//!
//! The Ring-0 rule has its own tests (`inf-voxel/tests/overlay_sim.rs`); this is
//! the one at the seam whose answer deletes files.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_ecs::components::VoxelVolume;
use inf_editor_core::voxel_edit::{stage_voxel_edits, write_voxel_edits};
use inf_editor_core::voxel_store::shared_volumes;
use inf_voxel::{ChunkKey, VoxelChunk, VoxelData, VoxelDeltaBuilder, VoxelOp, VoxelShape};
use uuid::Uuid;

const ENTITY: Uuid = Uuid::from_u128(0x2104_0F01);
const ASSET: Uuid = Uuid::from_u128(0x2104_0F02);

/// One chunk of solid rock, written to disk with its sidecar — the authored
/// `.inf_voxel` the store binds and the save would overwrite.
fn author(dir: &std::path::Path) -> Vec<u8> {
    let mut v = VoxelData::new(1.0);
    v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(1));
    v.clear_dirty();
    let asset = inf_voxel::build_voxel_asset(&v).expect("the fixture builds");
    let path = dir.join("Rock.inf_voxel");
    let bytes = inf_voxel::write_voxel_asset(&path, &asset).expect("write");
    inf_asset::AssetSidecar::new(
        inf_asset::AssetId(ASSET),
        inf_asset::AssetKind::VoxelVolume,
        inf_asset::ContentHash::of(bytes),
    )
    .save(&path)
    .expect("sidecar");
    std::fs::read(&path).expect("read back")
}

/// One gameplay carve, through the Ring-0 rule a Blueprint node runs.
fn gameplay_carve(sim: VoxelData) -> VoxelData {
    let mut map = BTreeMap::from([(ENTITY, sim)]);
    let report = inf_voxel::runtime_carve(
        &mut map,
        &ENTITY,
        true,
        &VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 4.0,
        }),
    );
    assert!(
        report.total_carved() > 0,
        "the fixture carve removed nothing"
    );
    map.remove(&ENTITY).expect("the volume comes back")
}

/// **THE GATE.** Dig in Simulate, stop, save: nothing is staged, the file is
/// byte-unchanged, and the unsaved-edits flag is clear.
#[test]
fn a_simulate_carve_stages_nothing_and_leaves_the_file_untouched() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("Rock.inf_voxel");
    let on_disk = author(dir.path());

    let volumes = shared_volumes();
    {
        let mut store = volumes.lock().unwrap();
        store.set_content_root(Some(dir.path().to_path_buf()));
        assert!(store.ensure(ENTITY, &VoxelVolume::from_asset(ASSET)));
        store.page_region(ENTITY, DVec3::splat(-1.0), DVec3::splat(17.0));
        assert!(!store.has_unsaved_edits(), "the fixture starts clean");
    }

    // The SIM digs, and the fold mirrors it into the store the save reads.
    let sim = gameplay_carve(inf_voxel::sim_volume(&on_disk, DVec3::ZERO).unwrap());
    let copied = {
        let mut store = volumes.lock().unwrap();
        store.overlay_sim(ENTITY, ASSET, &sim)
    };
    assert!(
        copied > 0,
        "the fold copied nothing, so this proves nothing"
    );

    // THE CLAIM.
    {
        let store = volumes.lock().unwrap();
        assert!(
            stage_voxel_edits(&store).is_empty(),
            "a gameplay carve was staged for write-back — Ctrl+S would write the \
             player's craters into the author's .inf_voxel"
        );
        assert!(
            !store.has_unsaved_edits(),
            "after a clean save the author is told they have unsaved carves they \
             never made, and the crash-recovery file is never released"
        );
    }

    // Belt and braces: run the writer over what staging produced. The file is
    // byte-identical, which is the only form of "not persisted" worth asserting.
    let report = write_voxel_edits(&[], dir.path());
    assert!(report.written.is_empty());
    assert_eq!(std::fs::read(&path).unwrap(), on_disk);

    // ANTI-VACUITY: an AUTHORING carve into the same store DOES stage, so
    // "nothing staged" is a statement about gameplay rather than about a seam
    // that stopped working.
    {
        let mut store = volumes.lock().unwrap();
        let mut builder = VoxelDeltaBuilder::new();
        store
            .carve_into(
                ENTITY,
                &VoxelOp::carve(VoxelShape::Sphere {
                    center: DVec3::new(4.0, 4.0, 4.0),
                    radius_m: 2.0,
                }),
                &mut builder,
            )
            .expect("the volume is loaded");
    }
    {
        let store = volumes.lock().unwrap();
        assert!(
            !stage_voxel_edits(&store).is_empty(),
            "an authoring carve no longer stages — the save seam is broken, and the \
             assertion above passes for the wrong reason"
        );
        assert!(store.has_unsaved_edits());
    }
}
