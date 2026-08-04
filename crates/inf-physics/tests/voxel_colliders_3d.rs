//! **P21.4: a voxel volume's chunks really do become rapier colliders — and a
//! runtime carve rebuilds them.**
//!
//! Before this batch a cave was categorically walk-through in the same way P19.5's
//! scattered buildings were: `VoxelData` is not an entity, its chunks have no
//! `Guid`, and `PhysicsBridge3D::sync_from_world`'s walk keys on exactly that. The
//! P21.2 combined ground query let a *Blueprint* stand on a cave floor by asking
//! `terrain.height_at`; a rigid body dropped into the same cave fell forever.
//!
//! These tests are about the **bridge**, not about caves. The mesher has its own
//! suite; what is unfalsifiable without these is that the mesh reaches the
//! simulation, that a carve moves it, and that the change stamp does not quietly
//! make "moves it" mean "once, at load".

use std::collections::BTreeMap;

use glam::DVec3;
use inf_ecs::EcsWorld;
use inf_physics::d3::voxel_chunk_guid;
use inf_physics::PhysicsBridge3D;
use inf_voxel::{ChunkKey, VoxelChunk, VoxelData, VoxelOp, VoxelShape};
use uuid::Uuid;

const CAVE: Uuid = Uuid::from_u128(0x2104_0001);
/// 1 m voxels, so a chunk is a 16 m cube and every coordinate below is metres.
const MPS: f64 = 1.0;

/// One chunk of solid rock at the origin, fully resident.
fn rock() -> BTreeMap<Uuid, VoxelData> {
    let mut v = VoxelData::new(MPS);
    v.insert_chunk(ChunkKey::new(0, 0, 0), VoxelChunk::solid(1));
    v.clear_dirty();
    BTreeMap::from([(CAVE, v)])
}

/// The bridge needs a world; nothing in it carries a body, so every collider the
/// tests find came from the voxel path.
fn empty_world() -> EcsWorld {
    let mut w = EcsWorld::new();
    w.mark_dirty();
    w
}

fn key0() -> Uuid {
    voxel_chunk_guid(CAVE, ChunkKey::new(0, 0, 0))
}

// ── the headline ────────────────────────────────────────────────────────────

/// A resident chunk with a surface becomes one **static** collider, anchored at
/// the chunk's own world origin.
#[test]
fn a_resident_chunk_becomes_a_static_collider() {
    let world = empty_world();
    let mut volumes = rock();
    // A fully solid chunk has no surface INSIDE it — Surface Nets emits vertices
    // at sign changes — so carve a ball to give it one. This is also the shape of
    // the real fixture: a cave is a hole in rock.
    volumes
        .get_mut(&CAVE)
        .unwrap()
        .apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 4.0,
        }));

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world_with_voxels(&world, &volumes);

    let guid = key0();
    let body = bridge.body_of(guid).expect("the chunk has a body");
    assert!(
        bridge.collider_of(guid).is_some(),
        "the chunk has no collider"
    );
    let at = bridge
        .world()
        .body_translation(body)
        .expect("the body is live");
    assert_eq!(
        at,
        volumes[&CAVE].chunk_origin_world(ChunkKey::new(0, 0, 0)),
        "the chunk collider is not at its own world origin"
    );

    // Rock does not fall: the body is static, so stepping leaves it put.
    for _ in 0..30 {
        bridge.step(1.0 / 60.0);
    }
    assert_eq!(bridge.world().body_translation(body).unwrap(), at);
}

/// **ANTI-VACUITY, and the reason the test above carves first**: a chunk that
/// meshes to nothing gets no collider at all, so "there is a collider" is a
/// statement about geometry rather than about residency.
///
/// The fixture is a 3×3×3 block of solid chunks: the **centre** one is buried, its
/// padded gather sees rock in every direction, and it has no sign change anywhere
/// — so it costs nothing. Its neighbours on the outside of the block *do* have a
/// surface (the mesher reads an absent chunk as air) and do get colliders, which
/// is what makes this a claim about the interior rather than about the volume.
#[test]
fn a_buried_chunk_gets_no_collider_while_its_shell_does() {
    let world = empty_world();
    let mut v = VoxelData::new(MPS);
    for key in inf_voxel::chunk_range(ChunkKey::new(-1, -1, -1), ChunkKey::new(1, 1, 1)) {
        v.insert_chunk(key, VoxelChunk::solid(1));
    }
    v.clear_dirty();
    let solid = BTreeMap::from([(CAVE, v)]);
    assert_eq!(
        inf_voxel::mesh_chunk(&solid[&CAVE], ChunkKey::new(0, 0, 0)).triangle_count(),
        0,
        "the buried chunk is meant to have no surface"
    );

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world_with_voxels(&world, &solid);
    assert!(
        bridge.body_of(key0()).is_none(),
        "a buried chunk produced a collider — every interior chunk of a mountain \
         would be paying for a trimesh nothing can reach"
    );
    // …and the shell around it did not vanish along with it.
    assert!(
        bridge
            .body_of(voxel_chunk_guid(CAVE, ChunkKey::new(1, 0, 0)))
            .is_some(),
        "the shell chunk lost its collider too — the fixture proves nothing"
    );
}

// ── the carve ───────────────────────────────────────────────────────────────

/// **THE POINT OF THE BATCH.** A runtime carve changes the collider: the same
/// chunk key is re-described because its `chunk_version` moved, and the new
/// surface is a different triangle count from the old one.
#[test]
fn a_runtime_carve_rebuilds_the_chunk_collider() {
    let world = empty_world();
    let mut volumes = rock();
    volumes
        .get_mut(&CAVE)
        .unwrap()
        .apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 3.0,
        }));

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world_with_voxels(&world, &volumes);
    let first = bridge.collider_of(key0()).expect("a collider");
    let small = inf_voxel::mesh_chunk(&volumes[&CAVE], ChunkKey::new(0, 0, 0)).triangle_count();
    assert!(small > 0);

    // Dig it wider — through the Ring-0 runtime rule, which is what a Blueprint
    // node runs.
    let report = inf_voxel::runtime_carve(
        &mut volumes,
        &CAVE,
        true,
        &VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 6.0,
        }),
    );
    assert!(report.total_carved() > 0, "the second dig removed nothing");

    bridge.sync_from_world_with_voxels(&world, &volumes);
    let second = bridge.collider_of(key0()).expect("still a collider");
    let big = inf_voxel::mesh_chunk(&volumes[&CAVE], ChunkKey::new(0, 0, 0)).triangle_count();
    assert_ne!(big, small, "the carve did not change the surface");
    assert_ne!(
        first, second,
        "the collider handle was reused — the trimesh was NOT rebuilt, so a body \
         would still be standing on the pre-carve wall"
    );
}

/// **The change stamp really is a stamp.** A sync that changes nothing must not
/// rebuild the collider — otherwise a cave system re-meshes every chunk at 60 Hz
/// and the pattern's whole reason for existing is gone.
#[test]
fn an_unchanged_volume_keeps_its_collider_handle() {
    let world = empty_world();
    let mut volumes = rock();
    volumes
        .get_mut(&CAVE)
        .unwrap()
        .apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 4.0,
        }));

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world_with_voxels(&world, &volumes);
    let first = bridge.collider_of(key0()).expect("a collider");
    for _ in 0..8 {
        bridge.sync_from_world_with_voxels(&world, &volumes);
    }
    assert_eq!(
        bridge.collider_of(key0()),
        Some(first),
        "an unchanged chunk was re-described"
    );
    // And the retained key survived the despawn sweep — the trap the
    // `structure_stamps` pattern documents.
    assert!(bridge.body_of(key0()).is_some());
}

/// A volume that leaves the sim map drops its colliders, and its stamps, so a key
/// that comes back is re-described rather than inheriting a stale version.
#[test]
fn a_dropped_volume_despawns_its_chunk_colliders() {
    let world = empty_world();
    let mut volumes = rock();
    volumes
        .get_mut(&CAVE)
        .unwrap()
        .apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 4.0,
        }));
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world_with_voxels(&world, &volumes);
    assert!(bridge.body_of(key0()).is_some());

    bridge.sync_from_world_with_voxels(&world, &BTreeMap::new());
    assert!(
        bridge.body_of(key0()).is_none(),
        "a removed volume leaked its chunk colliders"
    );

    // Back again: re-described, not silently absent because the stamp survived.
    bridge.sync_from_world_with_voxels(&world, &volumes);
    assert!(
        bridge.body_of(key0()).is_some(),
        "a returning volume inherited a stale stamp and got no collider"
    );
}

/// The synthetic identity is a pure function of `(entity, chunk key)` and cannot
/// alias — the `pcg_structure_guid` reasoning, with three signed coordinates
/// instead of an index.
#[test]
fn chunk_identities_are_distinct_and_deterministic() {
    let a = Uuid::from_u128(0x1000);
    let b = Uuid::from_u128(0x1001);
    // Same call twice is the same id.
    assert_eq!(
        voxel_chunk_guid(a, ChunkKey::new(1, 2, 3)),
        voxel_chunk_guid(a, ChunkKey::new(1, 2, 3))
    );
    // Neighbouring volume ids do not alias.
    assert_ne!(
        voxel_chunk_guid(a, ChunkKey::new(1, 2, 3)),
        voxel_chunk_guid(b, ChunkKey::new(1, 2, 3))
    );
    // Coordinate permutations are distinct — the per-lane tag is what stops
    // (1,2,3) and (3,2,1) folding to the same word.
    let mut seen = std::collections::BTreeSet::new();
    for k in [
        ChunkKey::new(1, 2, 3),
        ChunkKey::new(3, 2, 1),
        ChunkKey::new(2, 1, 3),
        ChunkKey::new(-1, 2, 3),
        ChunkKey::new(1, -2, 3),
        ChunkKey::new(1, 2, -3),
        ChunkKey::new(0, 0, 0),
    ] {
        assert!(seen.insert(voxel_chunk_guid(a, k)), "{k:?} aliased");
    }
}

// ── B3 / B4: the mesher's key set, and the mesher's stamp ────────────────────

/// Total collidable triangles across every chunk collider the bridge described.
fn collider_triangles(volumes: &BTreeMap<Uuid, VoxelData>) -> usize {
    let mut n = 0;
    for data in volumes.values() {
        for key in inf_voxel::mesh_keys_for(data) {
            n += inf_voxel::mesh_chunk(data, key).triangle_count();
        }
    }
    n
}

/// Triangles the bridge would have described under the **residency** key set —
/// the first cut of P21.4, kept here as the measurement the fix is against.
fn resident_only_triangles(volumes: &BTreeMap<Uuid, VoxelData>) -> usize {
    let mut n = 0;
    for data in volumes.values() {
        for key in data.resident_keys() {
            n += inf_voxel::mesh_chunk(data, key).triangle_count();
        }
    }
    n
}

/// **B4 — the collider set is the MESHER's key set, not the resident set.**
///
/// A cell owned by chunk `K` has corners in `K + 1`, so the surface of a volume
/// lives on `mesh_keys_for` — the resident set closed downward by one chunk on
/// each axis, which `inf_voxel::mesh` calls "a correctness requirement, not
/// defensive padding". Iterating `resident_keys()` instead leaves every −X/−Y/−Z
/// face drawn and walk-through.
///
/// Asserted as an inequality *and* an equality: the two key sets really do differ
/// on this fixture (so the test is not vacuous), and the bridge's set is the
/// renderer's.
#[test]
fn the_collider_set_covers_every_triangle_the_renderer_draws() {
    let world = empty_world();
    let mut volumes = rock();
    volumes
        .get_mut(&CAVE)
        .unwrap()
        .apply_op(&VoxelOp::carve(VoxelShape::Sphere {
            center: DVec3::new(8.0, 8.0, 8.0),
            radius_m: 4.0,
        }));

    let drawn = collider_triangles(&volumes);
    let resident_only = resident_only_triangles(&volumes);
    assert!(
        drawn > resident_only,
        "the fixture cannot tell the two key sets apart ({drawn} vs {resident_only}) — \
         this test would pass with the bug in place"
    );

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world_with_voxels(&world, &volumes);

    // Every mesh key with geometry got a collider; the closure keys are exactly
    // the ones the resident walk would have missed.
    let mut described = 0usize;
    let mut missed_by_residency = 0usize;
    let data = &volumes[&CAVE];
    let resident: std::collections::BTreeSet<ChunkKey> = data.resident_keys().into_iter().collect();
    for key in inf_voxel::mesh_keys_for(data) {
        if inf_voxel::mesh_chunk(data, key).triangle_count() == 0 {
            continue;
        }
        assert!(
            bridge.collider_of(voxel_chunk_guid(CAVE, key)).is_some(),
            "chunk {key:?} is drawn and has no collider"
        );
        described += 1;
        if !resident.contains(&key) {
            missed_by_residency += 1;
        }
    }
    assert!(described > 0);
    assert!(
        missed_by_residency > 0,
        "no closure key carried geometry here, so this fixture proves nothing"
    );
}

/// **B3 — the stamp is the MESHER's key, not the chunk's own version.**
///
/// A mesh is a function of its 3×3×3 neighbourhood, so a chunk's own
/// `chunk_version` cannot see a neighbour change. The sharpest case is an
/// **eviction**, which `MeshSourceKey`'s own docs name: a non-resident neighbour
/// reads as empty space, so chunk `K`'s surface really does move — and no stamp
/// on `K` will ever record it.
///
/// Keyed on `chunk_version` this test fails with a *retained* collider: the wall
/// against the evicted neighbour survives for ever.
#[test]
fn evicting_a_neighbour_re_describes_the_chunk_that_meshed_against_it() {
    let world = empty_world();
    // Two chunks of rock side by side, with a surface inside each.
    let mut v = VoxelData::new(MPS);
    for key in [ChunkKey::new(0, 0, 0), ChunkKey::new(1, 0, 0)] {
        v.insert_chunk(key, VoxelChunk::solid(1));
    }
    v.clear_dirty();
    let mut volumes = BTreeMap::from([(CAVE, v)]);

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world_with_voxels(&world, &volumes);
    let before = bridge
        .collider_of(key0())
        .expect("chunk 0 has a surface against the outside air");
    let version_before = volumes[&CAVE].chunk_version(ChunkKey::new(0, 0, 0));

    // Evict the NEIGHBOUR. Chunk 0's own field does not move at all…
    volumes
        .get_mut(&CAVE)
        .unwrap()
        .evict_chunk(ChunkKey::new(1, 0, 0));
    assert_eq!(
        volumes[&CAVE].chunk_version(ChunkKey::new(0, 0, 0)),
        version_before,
        "the fixture moved chunk 0's own version, so this proves nothing"
    );

    // …but its SURFACE does: the seam it meshed against is now open air.
    bridge.sync_from_world_with_voxels(&world, &volumes);
    let after = bridge
        .collider_of(key0())
        .expect("chunk 0 still has a surface");
    assert_ne!(
        before, after,
        "chunk 0 kept its collider after its neighbour was evicted — the stamp is \
         the chunk's own version rather than the mesher's source key, which is the \
         M3 defect P21.1 paid for, re-introduced in the physics bridge"
    );
}
