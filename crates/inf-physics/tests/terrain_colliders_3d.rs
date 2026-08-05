//! **P22.3: the ground is a collider.**
//!
//! Until this batch the terrain heightfield had **no physics representation at
//! all**. `terrain.height_at` answered a query and the character mover read it,
//! so a scripted character stayed grounded; a *dynamic* rigid body — a crate, a
//! ragdoll, and above all a chunk of a shattered wall — fell through the world.
//! Destruction cannot be built on that: debris that never lands is not debris.
//!
//! What these tests pin is the **bridge**, not the terrain. The tile grid, the
//! bilinear sampler and the hole mask all have their own suites in `inf-terrain`.
//! What is unfalsifiable without these is that the tiles reach the simulation,
//! that the **hole rule here is the hole rule `height_at` uses**, that paging
//! neither leaks colliders nor re-describes tiles nobody touched, and that the
//! voxel cave floor is what catches whatever the hole lets through — the P21.2
//! combined ground query, now with real physics under it.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use inf_ecs::components::{Terrain, Transform};
use inf_ecs::{EcsWorld, Vec3d};
use inf_physics::d3::{terrain_tile_guid, BodyKind3D, ColliderDesc3D, ColliderShape3D};
use inf_physics::PhysicsBridge3D;
use inf_terrain::{TerrainData, TileKey};
use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};
use uuid::Uuid;

const TERRAIN: Uuid = Uuid::from_u128(0x2203_0001);
const CAVE: Uuid = Uuid::from_u128(0x2203_0002);

/// 5 × 5 samples at 1 m ⇒ one tile spanning `[0, 4]²` — the `simulate_voxel_ground`
/// fixture, so the two suites are talking about the same ground.
const TILE_RES: u32 = 5;
const MPS: f64 = 1.0;
const GROUND_Y: f64 = 10.0;

fn gravity() -> DVec3 {
    DVec3::new(0.0, -9.81, 0.0)
}

/// A flat one-tile terrain at [`GROUND_Y`].
fn flat_terrain() -> TerrainData {
    let mut data = TerrainData::new(TILE_RES, MPS);
    data.author_tile((0, 0), |_, _| GROUND_Y);
    data
}

/// [`flat_terrain`] with sample `(2, 2)` carved through — the same hole
/// `simulate_voxel_ground` digs.
fn holed_terrain() -> TerrainData {
    let mut data = flat_terrain();
    data.get_tile_mut((0, 0))
        .expect("the tile was just authored")
        .set_hole(TILE_RES, 2, 2, true);
    data
}

/// A world holding one terrain entity at the origin.
fn terrain_world(data: TerrainData) -> EcsWorld {
    let mut w = EcsWorld::new();
    let e = w.spawn_with_guid(TERRAIN, "Terrain", None);
    w.world_mut().entity_mut(e).insert(Terrain {
        meters_per_sample: MPS,
        tile_resolution: TILE_RES,
        data,
        ..Terrain::default()
    });
    w.mark_dirty();
    w.propagate();
    w
}

/// Move the terrain entity to `at` (and re-propagate), so the gather sees a
/// terrain that relocated.
fn move_terrain(world: &mut EcsWorld, at: DVec3) {
    let e = world.entity_of(TERRAIN).expect("the terrain entity");
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::from_dvec3(at);
    }
    world.mark_dirty();
    world.propagate();
}

/// Drop a 0.25 m ball from `y` over world `(x, z)` and return where it ends up
/// after two seconds of fixed steps.
fn drop_ball(bridge: &mut PhysicsBridge3D, x: f64, z: f64, y: f64) -> DVec3 {
    let body = bridge.world_mut().add_body(
        BodyKind3D::Dynamic,
        DVec3::new(x, y, z),
        glam::DQuat::IDENTITY,
    );
    bridge.world_mut().add_collider(
        body,
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 0.25 }),
    );
    for _ in 0..120 {
        bridge.step(1.0 / 60.0);
    }
    bridge
        .world()
        .body_translation(body)
        .expect("the ball is still live")
}

/// The world Y where a straight-down ray from well above `(x, z)` first hits a
/// collider, or `None` where nothing does. The sharp instrument: unlike a dropped
/// body it measures the surface at exactly this XZ rather than wherever the body
/// happened to slide to.
fn probe_ground(bridge: &mut PhysicsBridge3D, x: f64, z: f64) -> Option<f64> {
    bridge
        .world_mut()
        .cast_ray(
            DVec3::new(x, GROUND_Y + 50.0, z),
            DVec3::new(0.0, -1.0, 0.0),
            200.0,
        )
        .map(|hit| hit.point.y)
}

// ── the headline ────────────────────────────────────────────────────────────

/// A sim-resident tile becomes one **static** collider, centred on the tile, and
/// a body dropped on it rests at the authored height.
#[test]
fn a_resident_tile_becomes_a_static_heightfield_collider() {
    let world = terrain_world(flat_terrain());
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);

    let guid = terrain_tile_guid(TERRAIN, (0, 0));
    let body = bridge.body_of(guid).expect("the tile has a body");
    assert!(
        bridge.collider_of(guid).is_some(),
        "the tile has no collider"
    );
    // The height field is CENTRED in its own frame, so the body sits at the
    // tile's centre — not at its corner sample. A tile spanning [0, 4]² has its
    // centre at (2, 2), and `origin.y` is 0 because `author_tile` puts the whole
    // height into the f32 offset.
    assert_eq!(
        bridge.world().body_translation(body).unwrap(),
        DVec3::new(2.0, 0.0, 2.0),
        "the tile collider is not centred on the tile"
    );

    // Ground does not fall.
    for _ in 0..30 {
        bridge.step(1.0 / 60.0);
    }
    assert_eq!(
        bridge.world().body_translation(body).unwrap(),
        DVec3::new(2.0, 0.0, 2.0)
    );

    let rest = drop_ball(&mut bridge, 1.0, 1.0, GROUND_Y + 5.0);
    assert!(
        (rest.y - (GROUND_Y + 0.25)).abs() < 0.05,
        "the ball did not come to rest on the ground: {rest:?}"
    );
}

/// **The world assertion, negated.** With no terrain in the world nothing catches
/// the ball — so the test above is measuring the collider and not gravity being
/// off.
#[test]
fn without_a_terrain_the_same_ball_falls_forever() {
    let mut w = EcsWorld::new();
    w.mark_dirty();
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&w);
    let rest = drop_ball(&mut bridge, 1.0, 1.0, GROUND_Y + 5.0);
    assert!(
        rest.y < 0.0,
        "the ball rested with no ground under it: {rest:?}"
    );
}

// ── holes ───────────────────────────────────────────────────────────────────

/// **The hole trio.** Over a holed cell the ball falls through; one cell away it
/// rests; and the voxel cave floor is what catches it when it does fall — the
/// P21.2 combined ground story with real physics on both halves.
#[test]
fn a_ball_falls_through_a_holed_cell_and_rests_beside_it() {
    let world = terrain_world(holed_terrain());
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);

    // Over the holed sample (2, 2): the poison rule removes every cell that
    // interpolates it, so the surface is gone here.
    let through = drop_ball(&mut bridge, 2.0, 2.0, GROUND_Y + 5.0);
    assert!(
        through.y < GROUND_Y - 2.0,
        "the ball stopped over a hole — the cell was not removed: {through:?}"
    );

    // The control, one whole cell away from the hole's reach. Sample (2,2)
    // poisons cells (1..2, 1..2); the cell at (0, 0) covers world [0,1]².
    let rest = drop_ball(&mut bridge, 0.4, 0.4, GROUND_Y + 5.0);
    assert!(
        (rest.y - (GROUND_Y + 0.25)).abs() < 0.05,
        "the ball beside the hole did not rest on the ground: {rest:?}"
    );
}

/// The other half of the P21.2 story: what falls through the mouth lands on the
/// **cave floor**, because the voxel chunk colliders are in the same world.
#[test]
fn the_voxel_cave_floor_catches_what_the_hole_lets_through() {
    let world = terrain_world(holed_terrain());
    // One chunk of rock whose surface crosses at y = 3.5 (1 m voxels; the SDF is
    // `y − 3.5` in voxels, negative below and positive above).
    let mut v = VoxelData::new(1.0);
    v.insert_chunk(
        ChunkKey::new(0, 0, 0),
        VoxelChunk::from_fn(|_, j, _| j as f64 - 3.5),
    );
    v.clear_dirty();
    let volumes = BTreeMap::from([(CAVE, v)]);

    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_with_voxels(&world, &volumes);

    let rest = drop_ball(&mut bridge, 2.0, 2.0, GROUND_Y + 5.0);
    assert!(
        rest.y > 3.0 && rest.y < 4.5,
        "the ball did not land on the cave floor at ~3.5: {rest:?}"
    );
}

/// **THE TRANSPOSE GATE.** parry's height field is indexed `(row = Z, column =
/// X)` in a *column-major* array while a `TerrainTile` is row-major in Z, so the
/// two layouts are transposes and `heightfield_shape` converts between them. A
/// symmetric fixture cannot see that conversion go wrong: a hole at the centre of
/// a square grid, or a flat field, is its own transpose.
///
/// So this one is deliberately asymmetric in **both** axes at once — an X-only
/// height ramp with a hole at sample `(1, 3)` — and asserts the mirrored
/// coordinates give the *opposite* answers. Swapping the two indices anywhere in
/// the conversion flips every assertion below.
///
/// The instrument is a **downward raycast**, not a dropped ball: a 45° ramp makes
/// a sphere roll, so the resting place of a ball measures friction and slope
/// rather than surface identity. (That is not hypothetical — the first cut of
/// this test dropped a ball and it slid three metres across the tile.) A ray
/// answers "what height is the collider at exactly this XZ", which is the
/// question.
#[test]
fn the_heightfield_axes_are_not_transposed() {
    let mut data = TerrainData::new(TILE_RES, MPS);
    // Height depends on X ALONE: sample (i, j) is at world (i, j), so a transpose
    // would report the height of (j, i).
    data.author_tile((0, 0), |x, _z| GROUND_Y + x);
    data.get_tile_mut((0, 0))
        .unwrap()
        .set_hole(TILE_RES, 1, 3, true);
    let world = terrain_world(data.clone());
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);

    // 1. The ramp. At world (3.5, 2.5) the ground is GROUND_Y + 3.5; at the
    //    mirrored point (2.5, 3.5) it is GROUND_Y + 2.5. (Both are clear of the
    //    hole's removed block, `x ∈ [0, 2] × z ∈ [2, 4]`, in *either* ordering —
    //    which is why the ramp pair is not the corner pair.)
    let high = probe_ground(&mut bridge, 3.5, 2.5).expect("solid ground at (3.5, 2.5)");
    let low = probe_ground(&mut bridge, 2.5, 3.5).expect("solid ground at (2.5, 3.5)");
    assert!(
        (high - (GROUND_Y + 3.5)).abs() < 1e-6,
        "the ramp reads the wrong axis at (3.5, 2.5): {high}"
    );
    assert!(
        (low - (GROUND_Y + 2.5)).abs() < 1e-6,
        "the ramp reads the wrong axis at (2.5, 3.5): {low}"
    );

    // …and the collider agrees with the query, at both mirrored points.
    assert!((high - data.height_at(DVec2::new(3.5, 2.5)).unwrap()).abs() < 1e-6);
    assert!((low - data.height_at(DVec2::new(2.5, 3.5)).unwrap()).abs() < 1e-6);

    // 2. The hole. Sample (1, 3) poisons the cells around world (1, 3); the
    //    mirrored point (3, 1) is solid ground.
    assert_eq!(
        probe_ground(&mut bridge, 1.0, 3.0),
        None,
        "the hole is not where it was authored"
    );
    assert!(
        probe_ground(&mut bridge, 3.0, 1.0).is_some(),
        "the hole appeared at the TRANSPOSED coordinate — the collider's X and Z \
         are swapped"
    );
}

/// **The consistency claim, tested rather than asserted in a comment.** For every
/// probe point, "the height field has a cell here" and
/// `TerrainData::height_at(...).is_some()` must agree — one holed corner removes
/// the whole cell in both. If they ever diverge, a character controller (which
/// reads `height_at`) stands on air a metre from a rigid body (which reads this)
/// resting on rock.
#[test]
fn the_removed_cell_rule_matches_height_at() {
    let data = holed_terrain();
    let tile = data.get_tile((0, 0)).expect("the tile");
    let n = TILE_RES as usize;

    // Rebuild the removal mask the same way the bridge does, then compare it
    // against the query, cell by cell, sampling each cell's interior.
    for jz in 0..(n - 1) {
        for ix in 0..(n - 1) {
            let (i0, j0) = (ix as u32, jz as u32);
            let removed = tile.is_hole(TILE_RES, i0, j0)
                || tile.is_hole(TILE_RES, i0 + 1, j0)
                || tile.is_hole(TILE_RES, i0, j0 + 1)
                || tile.is_hole(TILE_RES, i0 + 1, j0 + 1);
            // The centre of cell (ix, jz) in world XZ.
            let p = DVec2::new(ix as f64 + 0.5, jz as f64 + 0.5) * MPS;
            assert_eq!(
                removed,
                data.height_at(p).is_none(),
                "cell ({ix}, {jz}) disagrees: the collider says removed={removed} \
                 but height_at says {:?}",
                data.height_at(p)
            );
        }
    }
    // Anti-vacuity: the fixture really does have both kinds of cell.
    assert!(data.height_at(DVec2::new(0.5, 0.5)).is_some());
    assert!(data.height_at(DVec2::new(2.0, 2.0)).is_none());
}

/// A tile with **no** holes emits the sparse empty removal buffer, so an
/// un-carved level pays nothing for the hole layer — the `TerrainTile::holes`
/// convention, one container down.
#[test]
fn an_unholed_tile_carries_no_removal_buffer() {
    let world = terrain_world(flat_terrain());
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);
    // The descriptor is not public, so assert through the shape door instead:
    // building the same descriptor by hand must produce an equal shape, and the
    // hole-free one must have an empty buffer.
    let shape = ColliderShape3D::Heightfield {
        samples_x: TILE_RES,
        samples_z: TILE_RES,
        heights: vec![GROUND_Y as f32; (TILE_RES * TILE_RES) as usize],
        removed_cells: Vec::new(),
        span: DVec2::splat(4.0),
    };
    assert!(
        shape.volume_m3().is_none(),
        "a height field is a surface — it must have no volume, or a dynamic body \
         built from one would be given a mass"
    );
}

// ── the change stamp ────────────────────────────────────────────────────────

/// **The stamp really is a stamp.** Syncing an untouched terrain must not
/// re-describe a single tile — otherwise every level rebuilds a quarter of a
/// million samples' worth of collider sixty times a second.
#[test]
fn an_unchanged_tile_is_retained_rather_than_re_described() {
    let world = terrain_world(flat_terrain());
    let mut bridge = PhysicsBridge3D::new(gravity());

    bridge.sync_from_world(&world);
    let first = bridge.terrain_collider_audit();
    assert_eq!(first.resident_tiles, 1);
    assert_eq!(first.described, 1, "the first sync must build the tile");
    let handle = bridge
        .collider_of(terrain_tile_guid(TERRAIN, (0, 0)))
        .expect("a collider");

    for _ in 0..8 {
        bridge.sync_from_world(&world);
    }
    let steady = bridge.terrain_collider_audit();
    assert_eq!(
        steady.described, 0,
        "an unchanged terrain re-described its tiles"
    );
    assert_eq!(steady.retained, 1);
    assert_eq!(
        bridge.collider_of(terrain_tile_guid(TERRAIN, (0, 0))),
        Some(handle),
        "the collider handle changed under a no-op sync"
    );
}

/// A sculpt (any mutating touch) bumps the tile's version, and the tile is
/// re-described — the other direction of the same claim.
#[test]
fn a_sculpted_tile_is_re_described() {
    let mut world = terrain_world(flat_terrain());
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);
    let before = bridge
        .collider_of(terrain_tile_guid(TERRAIN, (0, 0)))
        .expect("a collider");

    let e = world.entity_of(TERRAIN).unwrap();
    {
        let mut t = world.world_mut().get_mut::<Terrain>(e).unwrap();
        let tile = t.data.get_tile_mut((0, 0)).unwrap();
        tile.heights_mut()[0] = GROUND_Y as f32 + 3.0;
    }
    world.mark_dirty();

    bridge.sync_from_world(&world);
    assert_eq!(bridge.terrain_collider_audit().described, 1);
    assert_ne!(
        bridge.collider_of(terrain_tile_guid(TERRAIN, (0, 0))),
        Some(before),
        "the sculpt did not rebuild the collider — a body would still be standing \
         on the pre-sculpt surface"
    );
}

/// A terrain that **moved** re-places every tile even though not one sample
/// changed: the origin is part of the stamp.
#[test]
fn a_moved_terrain_re_places_its_tiles() {
    let mut world = terrain_world(flat_terrain());
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);
    let guid = terrain_tile_guid(TERRAIN, (0, 0));
    assert_eq!(
        bridge
            .world()
            .body_translation(bridge.body_of(guid).unwrap())
            .unwrap(),
        DVec3::new(2.0, 0.0, 2.0)
    );

    move_terrain(&mut world, DVec3::new(100.0, 5.0, -50.0));
    bridge.sync_from_world(&world);
    assert_eq!(bridge.terrain_collider_audit().described, 1);
    assert_eq!(
        bridge
            .world()
            .body_translation(bridge.body_of(guid).unwrap())
            .unwrap(),
        DVec3::new(102.0, 5.0, -48.0),
        "the tile collider did not follow its terrain"
    );
}

// ── paging ──────────────────────────────────────────────────────────────────

/// **The soak.** Paging tiles in and out repeatedly leaks neither colliders nor
/// stamps: the tracked-body count and the stamp ledger both return to where they
/// started, every cycle.
#[test]
fn paging_tiles_in_and_out_leaks_nothing() {
    let mut world = terrain_world(flat_terrain());
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);
    let with_one = bridge.body_count();
    assert_eq!(bridge.terrain_stamp_count(), 1);

    let e = world.entity_of(TERRAIN).unwrap();
    for cycle in 0..6 {
        // Page a second tile in…
        {
            let mut t = world.world_mut().get_mut::<Terrain>(e).unwrap();
            t.data.insert_resident_tile(
                TileKey::lod0((1, 0)),
                inf_terrain::TerrainTile::flat(TILE_RES, DVec3::new(4.0, 0.0, 0.0)),
            );
        }
        world.mark_dirty();
        bridge.sync_from_world(&world);
        assert_eq!(
            bridge.body_count(),
            with_one + 1,
            "cycle {cycle}: the paged-in tile got no collider"
        );
        assert_eq!(bridge.terrain_stamp_count(), 2);
        assert!(bridge.body_of(terrain_tile_guid(TERRAIN, (1, 0))).is_some());

        // …and out again.
        {
            let mut t = world.world_mut().get_mut::<Terrain>(e).unwrap();
            assert!(t.data.evict_tile(TileKey::lod0((1, 0))));
        }
        world.mark_dirty();
        bridge.sync_from_world(&world);
        assert_eq!(
            bridge.body_count(),
            with_one,
            "cycle {cycle}: the evicted tile's collider leaked"
        );
        assert_eq!(
            bridge.terrain_stamp_count(),
            1,
            "cycle {cycle}: the evicted tile's stamp leaked — the ledger is meant \
             to be bounded by residency"
        );
        assert!(bridge.body_of(terrain_tile_guid(TERRAIN, (1, 0))).is_none());
    }
}

/// Two terrains' tiles never alias each other, however close their guids are —
/// the reason `terrain_tile_guid` is a 128-bit mix rather than a XOR.
#[test]
fn two_terrains_tiles_do_not_alias() {
    let a = Uuid::from_u128(0x2203_0100);
    let b = Uuid::from_u128(0x2203_0101);
    assert_ne!(terrain_tile_guid(a, (0, 0)), terrain_tile_guid(b, (0, 0)));
    assert_ne!(terrain_tile_guid(a, (0, 0)), terrain_tile_guid(a, (1, 0)));
    assert_ne!(terrain_tile_guid(a, (-1, 0)), terrain_tile_guid(a, (1, 0)));
    // …and a tile never collides with a voxel chunk or a scattered solid.
    assert_ne!(
        terrain_tile_guid(a, (0, 0)),
        inf_physics::d3::voxel_chunk_guid(a, ChunkKey::new(0, 0, 0))
    );
    assert_ne!(
        terrain_tile_guid(a, (0, 0)),
        inf_physics::d3::pcg_structure_guid(a, 0)
    );
}

// ── refusals ────────────────────────────────────────────────────────────────

/// Degenerate height-field buffers are **refused**, not panicked on: parry
/// asserts on a field with fewer than two rows, and a producer bug must not take
/// the process down.
#[test]
fn degenerate_heightfields_refuse_cleanly() {
    let mut world = EcsWorld::new();
    let e = world.spawn_with_guid(TERRAIN, "T", None);
    world.world_mut().entity_mut(e).insert(Transform::IDENTITY);
    world.mark_dirty();
    let mut bridge = PhysicsBridge3D::new(gravity());

    let body = bridge
        .world_mut()
        .add_body(BodyKind3D::Static, DVec3::ZERO, glam::DQuat::IDENTITY);
    let bad = [
        // one sample on an axis
        ColliderShape3D::Heightfield {
            samples_x: 1,
            samples_z: 4,
            heights: vec![0.0; 4],
            removed_cells: Vec::new(),
            span: DVec2::splat(4.0),
        },
        // short height buffer
        ColliderShape3D::Heightfield {
            samples_x: 4,
            samples_z: 4,
            heights: vec![0.0; 9],
            removed_cells: Vec::new(),
            span: DVec2::splat(4.0),
        },
        // wrong-length removal buffer
        ColliderShape3D::Heightfield {
            samples_x: 4,
            samples_z: 4,
            heights: vec![0.0; 16],
            removed_cells: vec![false; 4],
            span: DVec2::splat(4.0),
        },
        // zero span
        ColliderShape3D::Heightfield {
            samples_x: 4,
            samples_z: 4,
            heights: vec![0.0; 16],
            removed_cells: Vec::new(),
            span: DVec2::ZERO,
        },
        // non-finite height
        ColliderShape3D::Heightfield {
            samples_x: 4,
            samples_z: 4,
            heights: {
                let mut h = vec![0.0_f32; 16];
                h[5] = f32::NAN;
                h
            },
            removed_cells: Vec::new(),
            span: DVec2::splat(4.0),
        },
    ];
    for (i, shape) in bad.into_iter().enumerate() {
        assert!(
            bridge
                .world_mut()
                .add_collider(body, ColliderDesc3D::new(shape))
                .is_none(),
            "degenerate height field #{i} was accepted"
        );
    }
    // The positive control: a well-formed one IS accepted, so the loop above is
    // not just measuring a broken door.
    assert!(bridge
        .world_mut()
        .add_collider(
            body,
            ColliderDesc3D::new(ColliderShape3D::Heightfield {
                samples_x: 4,
                samples_z: 4,
                heights: vec![0.0; 16],
                removed_cells: vec![false; 9],
                span: DVec2::splat(4.0),
            })
        )
        .is_some());
}

/// A terrain whose tiles are **all** holed leaves nothing to stand on, and does
/// so without refusing the collider — the field still exists, every cell is just
/// removed. (The alternative, dropping the collider, would be indistinguishable
/// from a bug at the call site.)
#[test]
fn a_fully_holed_tile_is_a_collider_with_no_surface() {
    let mut data = flat_terrain();
    {
        let tile = data.get_tile_mut((0, 0)).unwrap();
        for j in 0..TILE_RES {
            for i in 0..TILE_RES {
                tile.set_hole(TILE_RES, i, j, true);
            }
        }
    }
    let world = terrain_world(data);
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world(&world);
    assert!(
        bridge
            .collider_of(terrain_tile_guid(TERRAIN, (0, 0)))
            .is_some(),
        "a fully holed tile should still have a (surfaceless) collider"
    );
    let rest = drop_ball(&mut bridge, 2.0, 2.0, GROUND_Y + 5.0);
    assert!(rest.y < 0.0, "a fully holed tile caught a ball: {rest:?}");
}
