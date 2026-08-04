//! **The sim → render overlay** (P21.4): the seam that lets a shipped player see
//! what its Blueprints carve.
//!
//! `VoxelVolumes::overlay_sim` shipped with **no unit tests at all**, and the two
//! defects that found is exactly the pair a unit test would have caught in a line
//! each:
//!
//! * it copied through `insert_chunk`, which marks the chunk **dirty** — and in
//!   the editor this store is the one `SceneDoc::save` stages from, so a player's
//!   runtime craters were about to be written into the author's `.inf_voxel`;
//! * it **baselined away the first carve on every chunk**, because `overlaid` is
//!   empty when a volume binds and both hosts bind after the first step. A
//!   one-shot dig on the first Tick — the pattern the roadmap prescribes — was
//!   invisible for the rest of the session, and a *continuous* borer hid it
//!   because tick 2 repaired tick 1.
//!
//! Both host orderings are modelled here by hand, because neither host can be
//! constructed in CI (one needs a window, the other a GPU) and the ordering is the
//! whole bug.

use std::collections::BTreeMap;

use glam::DVec3;
use inf_voxel::{
    build_voxel_asset, ChunkKey, VoxelChunk, VoxelData, VoxelOp, VoxelShape, VoxelStreamBudget,
    VoxelVolumes, VoxelWantsParams,
};

const ENTITY: u128 = 0x2104_0E00;
const ASSET: u128 = 0x2104_0EAA;
/// 1 m voxels, so every coordinate below is metres.
const MPS: f64 = 1.0;

/// The authored volume: a 2 × 1 × 2 block of solid rock, as it lives on disk.
fn authored() -> VoxelData {
    let mut v = VoxelData::new(MPS);
    for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 0, 1)) {
        v.insert_chunk(key, VoxelChunk::solid(1));
    }
    v.clear_dirty();
    v
}

fn payload() -> Vec<u8> {
    build_voxel_asset(&authored())
        .expect("the fixture builds")
        .into_bytes()
}

/// A bound render store, paged against a camera standing in the middle of it —
/// the state the player's `sync_voxel_store` leaves behind.
fn render_store() -> VoxelVolumes {
    let mut store = VoxelVolumes::new();
    store
        .ensure(ENTITY, ASSET, &payload())
        .expect("the payload binds");
    store.place(ENTITY, DVec3::ZERO);
    let report = store.sync_camera(
        DVec3::new(16.0, 8.0, 16.0),
        &VoxelWantsParams {
            radius_m: 1000.0,
            hysteresis: 0.0,
        },
        VoxelStreamBudget::default(),
    );
    assert!(report.loaded > 0, "the fixture store paged nothing");
    store
}

/// The **simulation's** copy of the same volume, as `sim_volume` produces it:
/// fully resident, dirty set cleared.
fn sim_volume() -> VoxelData {
    let v = inf_voxel::sim_volume(&payload(), DVec3::ZERO).expect("the payload loads");
    assert!(
        !v.has_dirty_chunks(),
        "a freshly loaded sim volume must not look carved"
    );
    v
}

/// One gameplay carve, the way a Blueprint node makes it.
fn carve(sim: &mut VoxelData) -> u64 {
    let mut map = BTreeMap::from([(ENTITY, std::mem::replace(sim, VoxelData::new(MPS)))]);
    let report = inf_voxel::runtime_carve(
        &mut map,
        &ENTITY,
        true,
        &VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 4.0,
        }),
    );
    *sim = map.remove(&ENTITY).expect("the volume comes back");
    assert!(
        report.total_carved() > 0,
        "the fixture carve removed nothing"
    );
    report.total_carved()
}

/// The chunk the carve above is centred in.
fn carved_key() -> ChunkKey {
    ChunkKey::new(0, 0, 0)
}

// ── the two blockers ────────────────────────────────────────────────────────

/// **BLOCKER 1 — the overlay must not DIRTY what it copies.**
///
/// In the editor this store is the one the save path stages from
/// (`SceneDoc::save` → `VoxelEdits::from_dirty` → `write_voxel_edits`), so a
/// dirty overlay writes a player's runtime craters into the author's `.inf_voxel`
/// on the next Ctrl+S — flatly contradicting the rule this phase recorded, that
/// runtime carves are not persisted. It also leaves "you have unsaved carves"
/// true for ever, because a clean save never clears a mark it never staged.
#[test]
fn an_overlaid_carve_is_never_staged_for_write_back() {
    let mut store = render_store();
    let mut sim = sim_volume();
    carve(&mut sim);

    assert!(
        !store.get(ENTITY).unwrap().data.has_dirty_chunks(),
        "the fixture store starts clean"
    );
    assert_eq!(store.overlay_sim(ENTITY, ASSET, &sim), 1);

    let data = &store.get(ENTITY).unwrap().data;
    assert!(
        !data.has_dirty_chunks(),
        "the overlay marked chunks dirty — a save would write gameplay's craters \
         into the authored .inf_voxel, and the 'unsaved carves' flag would never \
         clear"
    );
    assert!(data.dirty_chunks().is_empty(), "{:?}", data.dirty_chunks());
    // …and it really did copy: the store's field now matches the sim's, so this is
    // not "clean because nothing happened".
    assert_eq!(
        data.get_chunk(carved_key()).unwrap().sdf(),
        sim.get_chunk(carved_key()).unwrap().sdf(),
        "the overlay reported a copy and copied nothing"
    );
}

/// **BLOCKER 2 — the FIRST carve on a chunk must copy, in BOTH host orderings.**
///
/// `overlaid` is empty when a volume binds, so a baseline-on-first-sight rule
/// records whatever stamps the sim happens to have — *including ones a carve has
/// already moved*. Both hosts bind after the first step, so a one-shot dig on the
/// first Tick was baselined away entirely.
#[test]
fn a_one_shot_dig_on_the_first_tick_reaches_the_render_store() {
    // ── the PLAYER's ordering: the sim steps, and only then does the render
    //    host bind and sync (`sync_voxel_store` runs after the first frame).
    {
        let mut sim = sim_volume();
        carve(&mut sim); // tick 1: the only carve there will ever be
        let mut store = render_store(); // …and only now does the store exist
        assert_eq!(
            store.overlay_sim(ENTITY, ASSET, &sim),
            1,
            "the player's first overlay baselined the only carve away"
        );
        assert_eq!(
            store
                .get(ENTITY)
                .unwrap()
                .data
                .get_chunk(carved_key())
                .unwrap()
                .sdf(),
            sim.get_chunk(carved_key()).unwrap().sdf()
        );
    }

    // ── the EDITOR's ordering: the store exists first (the viewport binds it
    //    when the document loads), and the fold runs after `session.tick`.
    {
        let mut store = render_store();
        let mut sim = sim_volume();
        carve(&mut sim);
        assert_eq!(
            store.overlay_sim(ENTITY, ASSET, &sim),
            1,
            "the editor's first overlay baselined the only carve away"
        );
        assert_eq!(
            store
                .get(ENTITY)
                .unwrap()
                .data
                .get_chunk(carved_key())
                .unwrap()
                .sdf(),
            sim.get_chunk(carved_key()).unwrap().sdf()
        );
    }
}

/// The budget property the baseline rule existed to protect, kept: a level nobody
/// digs copies **nothing**, however many times the overlay runs.
#[test]
fn an_undug_level_copies_nothing_ever() {
    let mut store = render_store();
    let sim = sim_volume();
    for _ in 0..8 {
        assert_eq!(store.overlay_sim(ENTITY, ASSET, &sim), 0);
    }
    assert_eq!(store.overlaid_len(ENTITY), 0);
}

/// A carve copies **once**, not once per frame.
#[test]
fn a_carve_copies_once_and_then_settles() {
    let mut store = render_store();
    let mut sim = sim_volume();
    carve(&mut sim);
    assert_eq!(store.overlay_sim(ENTITY, ASSET, &sim), 1);
    for _ in 0..8 {
        assert_eq!(
            store.overlay_sim(ENTITY, ASSET, &sim),
            0,
            "the overlay re-copies an unchanged carve every frame"
        );
    }
    // A second carve moves it again.
    carve_elsewhere(&mut sim);
    assert!(store.overlay_sim(ENTITY, ASSET, &sim) > 0);
}

fn carve_elsewhere(sim: &mut VoxelData) {
    let mut map = BTreeMap::from([(ENTITY, std::mem::replace(sim, VoxelData::new(MPS)))]);
    let r = inf_voxel::runtime_carve(
        &mut map,
        &ENTITY,
        true,
        &VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(20.0, 8.0, 20.0),
            radius_m: 3.0,
        }),
    );
    assert!(r.total_carved() > 0);
    *sim = map.remove(&ENTITY).unwrap();
}

// ── the eviction round trip (the replacement for the pin) ───────────────────

/// **The carve survives a page-out and a page-in, without a pin.**
///
/// The overlay used to hold carved chunks resident for ever by marking them
/// dirty, which grew a session's resident set without bound — a camera a thousand
/// kilometres away still kept every carved chunk meshed and uploaded. Instead the
/// overlay runs *after* the camera pass and re-copies whatever residency undid:
/// a chunk paged back in from the `.inf_voxel` arrives as pre-carve rock with a
/// fresh stamp, which is exactly the condition it re-copies on.
#[test]
fn a_carve_is_re_applied_after_the_camera_pages_it_out_and_back_in() {
    let mut store = render_store();
    let mut sim = sim_volume();
    carve(&mut sim);
    assert_eq!(store.overlay_sim(ENTITY, ASSET, &sim), 1);
    let carved_sdf = sim.get_chunk(carved_key()).unwrap().sdf().to_vec();

    // The camera leaves. Nothing pins the chunk, so residency drops it — which is
    // the whole point: a session cannot grow without bound.
    let away = VoxelWantsParams {
        radius_m: 1.0,
        hysteresis: 0.0,
    };
    store.sync_camera(
        DVec3::new(100_000.0, 0.0, 100_000.0),
        &away,
        VoxelStreamBudget::default(),
    );
    assert!(
        !store.get(ENTITY).unwrap().data.is_resident(carved_key()),
        "the carved chunk was pinned resident — the unbounded case is back"
    );

    // The camera returns. Residency pages the ASSET's pre-carve rock back in…
    store.sync_camera(
        DVec3::new(8.0, 8.0, 8.0),
        &VoxelWantsParams {
            radius_m: 1000.0,
            hysteresis: 0.0,
        },
        VoxelStreamBudget::default(),
    );
    assert!(store.get(ENTITY).unwrap().data.is_resident(carved_key()));

    // …and the overlay puts the carve back on top of it.
    assert_eq!(
        store.overlay_sim(ENTITY, ASSET, &sim),
        1,
        "the overlay believed it had already done this chunk, so the store is \
         drawing the rock the player removed"
    );
    assert_eq!(
        store
            .get(ENTITY)
            .unwrap()
            .data
            .get_chunk(carved_key())
            .unwrap()
            .sdf(),
        &carved_sdf[..]
    );
}

// ── identity ────────────────────────────────────────────────────────────────

/// **A mismatched lattice or a mismatched ASSET copies nothing.**
///
/// An author can re-point `VoxelVolume.asset` in the Details panel mid-Simulate.
/// Without the asset check, asset A's chunks are copied into a slot bound to
/// asset B — and, before Blocker 1 was fixed, written into B's file.
#[test]
fn a_foreign_asset_or_a_foreign_lattice_is_refused() {
    let mut store = render_store();
    let mut sim = sim_volume();
    carve(&mut sim);

    assert_eq!(
        store.overlay_sim(ENTITY, ASSET ^ 1, &sim),
        0,
        "chunks were copied into a slot bound to a DIFFERENT asset"
    );
    assert_eq!(
        store.overlay_sim(ENTITY + 1, ASSET, &sim),
        0,
        "unknown entity"
    );

    // A different voxel size is a different lattice.
    let mut coarse = VoxelData::new(MPS * 2.0);
    coarse.insert_chunk(carved_key(), VoxelChunk::solid(1));
    assert_eq!(store.overlay_sim(ENTITY, ASSET, &coarse), 0);

    // …and so is the same lattice at a different anchor. `sim_volume` folds the
    // entity's translation into the anchor, so a volume the level moved is a
    // volume whose chunks land somewhere else.
    let moved = inf_voxel::sim_volume(&payload(), DVec3::new(3.0, 0.0, 0.0)).unwrap();
    assert_eq!(store.overlay_sim(ENTITY, ASSET, &moved), 0);

    // ANTI-VACUITY: the matching call still works.
    assert_eq!(store.overlay_sim(ENTITY, ASSET, &sim), 1);
}
