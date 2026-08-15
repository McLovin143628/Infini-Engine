//! **The caches release what they cache** (Hardening D).
//!
//! Every pass in this tree that materializes GPU resources per unit of content
//! reconciles its cache with the projection *inside* the body its early-out
//! guards. That is correct on every frame the node runs — and invisible on the
//! one transition that matters: the frame the last unit of content leaves the
//! scene, which is exactly when the guard fires and the reconciliation never
//! runs. `VoxelNode` and `FractureNode` have carried the release on the early-out
//! since P21/P22 ("the one transition the early-out would otherwise hide: the
//! last volume leaving the scene"); `ClassicVgeomNode` has carried `retain_live`
//! *before* its early-out since P18.3. The terrain and meshlet nodes did not.
//!
//! # Why these arms exist at all
//!
//! **A stranded cache renders identically to a released one.** There is no pixel,
//! no golden and no command-stream counter that can tell the two apart — which is
//! why the defect survived every gate in the tree. So the assertions here read the
//! *maps*: after the content leaves, the cache count is zero and the streamer's
//! resident bytes are zero. Assert the WORLD, not the report (P21).
//!
//! Skips cleanly with no GPU adapter, like every GPU path in this repo.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, RenderScene, RenderSettings, RenderTerrain,
    RenderTerrainTile, RenderView, TerrainTileKey, VgeomAsset, VgeomInstance, VgeomMesh,
    VgeomSettings, HEADLESS_FORMAT,
};
use inf_vgeom::test_support::dense_grid_mesh;

const W: u32 = 160;
const H: u32 = 120;
const RES: u32 = 16;
const MPS: f64 = 1.0;
const ASSET: u128 = 0x00CA_0FEE_0001;

fn gpu_or_skip(name: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("{name}: no GPU adapter ({e}) — skipping");
            None
        }
    }
}

fn view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 12.0, 24.0),
        forward: Vec3::new(0.0, -0.4, -1.0).normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// One flat level-0 tile at `coord`.
fn tile_at(coord: (i32, i32)) -> RenderTerrainTile {
    let span = (RES - 1) as f64 * MPS;
    RenderTerrainTile {
        key: TerrainTileKey::lod0(coord),
        origin: DVec3::new(coord.0 as f64 * span, 0.0, coord.1 as f64 * span),
        heights: vec![0.0; (RES * RES) as usize],
        weights: Vec::new(),
        biomes: Vec::new(),
        height_bounds: (0.0, 0.0),
        holes: Vec::new(),
        version: 1,
    }
}

fn terrain_scene(tiles: usize) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    if tiles > 0 {
        scene.terrains.push(RenderTerrain {
            id: 7,
            tile_resolution: RES,
            meters_per_sample: MPS,
            tiles: (0..tiles as i32).map(|x| tile_at((x, 0))).collect(),
            ..Default::default()
        });
    }
    scene.mark_dirty();
    scene
}

fn vgeom_scene(mesh: &Arc<VgeomMesh>, instances: bool) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        vgeom_assets: vec![VgeomAsset::from_mesh(ASSET, mesh).expect("index the vmesh")],
        ..Default::default()
    };
    if instances {
        scene.vgeom_instances.push(VgeomInstance::lit(
            ASSET,
            DVec3::ZERO,
            Quat::IDENTITY,
            Vec3::splat(4.0),
            [0.7, 0.5, 0.3, 1.0],
            1,
        ));
    }
    scene.mark_dirty();
    scene
}

fn vgeom_settings(enabled: bool) -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled,
            occlusion: false,
            two_pass: false,
            visbuffer: false,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// **The terrain finding (lens 2 H1).** `TerrainNode::run` returns before
/// `sync_textures` when no terrain has a resident tile, and `sync_textures` owns
/// the only two eviction paths there are. So a level switch (or every tile
/// streaming out) used to strand the whole per-tile texture cache — four textures
/// per tile — plus one splat-material slot per terrain, for the renderer's life.
#[test]
fn terrain_releases_its_tile_cache_when_the_last_tile_leaves() {
    let Some(gpu) = gpu_or_skip("terrain_releases_its_tile_cache_when_the_last_tile_leaves") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let view = view();

    let populated = terrain_scene(4);
    renderer.render(&gpu, &populated, &view, &target.view, (W, H));
    let (tiles, materials) = renderer.terrain_cache_counts();
    assert_eq!(
        (tiles, materials),
        (4, 1),
        "a four-tile terrain should cache four tiles and one material slot"
    );

    // The transition: the level's terrain content is gone. A scene with the
    // terrain still listed but nothing resident takes the same branch, which is
    // the streamed-out case.
    let empty = terrain_scene(0);
    renderer.render(&gpu, &empty, &view, &target.view, (W, H));
    assert_eq!(
        renderer.terrain_cache_counts(),
        (0, 0),
        "the tile textures and the splat-material slot must not outlive the content"
    );

    // And it comes back: the release is not a one-way door.
    renderer.render(&gpu, &populated, &view, &target.view, (W, H));
    assert_eq!(
        renderer.terrain_cache_counts(),
        (4, 1),
        "a terrain that returns is re-cached"
    );
}

/// The other half of the same branch: a terrain that is *listed* but has streamed
/// every tile out is the transition the guard reads as "no terrain".
#[test]
fn a_terrain_that_streams_every_tile_out_releases_too() {
    let Some(gpu) = gpu_or_skip("a_terrain_that_streams_every_tile_out_releases_too") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let view = view();

    renderer.render(&gpu, &terrain_scene(2), &view, &target.view, (W, H));
    assert_eq!(renderer.terrain_cache_counts().0, 2);

    // Same terrain id, zero resident tiles — `tiles.is_empty()` for every terrain.
    let mut streamed_out = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    streamed_out.terrains.push(RenderTerrain {
        id: 7,
        tile_resolution: RES,
        meters_per_sample: MPS,
        tiles: Vec::new(),
        ..Default::default()
    });
    streamed_out.mark_dirty();
    renderer.render(&gpu, &streamed_out, &view, &target.view, (W, H));
    assert_eq!(
        renderer.terrain_cache_counts(),
        (0, 0),
        "residency reaching zero must release, not park"
    );
}

/// **The meshlet finding (lens 2 H2).** `plan_cluster_pages` returns before the
/// streamer's plan when vgeom is off or the scene carries none, and `plan` →
/// `plan.dropped` → `draws.remove` is the only eviction chain there is. Two
/// transitions were therefore invisible: the content leaving, and the setting
/// being switched off. `stream_report` kept publishing the stale floor either way,
/// which is what the unified arbiter reserves against.
#[test]
fn vgeom_releases_residency_when_the_content_leaves() {
    let Some(gpu) = gpu_or_skip("vgeom_releases_residency_when_the_content_leaves") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(vgeom_settings(true));
    let view = view();
    let mesh = Arc::new(dense_grid_mesh(24));

    let populated = vgeom_scene(&mesh, true);
    for _ in 0..3 {
        renderer.render(&gpu, &populated, &view, &target.view, (W, H));
    }
    let hot = renderer.vgeom_stream_report();
    assert!(
        hot.stats.resident_bytes > 0 && hot.stats.assets == 1,
        "the fixture must actually be resident before a release can mean anything \
         (assets {}, bytes {})",
        hot.stats.assets,
        hot.stats.resident_bytes
    );
    assert!(
        hot.floor_bytes > 0,
        "the arbiter floor must be non-zero first"
    );

    // The transition: the scene stops carrying vgeom.
    let empty = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    renderer.render(&gpu, &empty, &view, &target.view, (W, H));
    let cold = renderer.vgeom_stream_report();
    assert_eq!(cold.stats.assets, 0, "every asset's residency is released");
    assert_eq!(cold.stats.resident_bytes, 0, "and its pool blocks with it");
    assert_eq!(
        cold.floor_bytes, 0,
        "and the arbiter stops reserving a floor for content nothing draws"
    );
    assert!(
        cold.pages.is_empty() && cold.floor_lod.is_empty(),
        "the per-asset report is emptied, not frozen at its last value"
    );
    assert!(
        cold.stats.evictions >= hot.stats.resident_pages as u64,
        "the release is an eviction of every resident page, counted"
    );
}

/// The setting's own off-transition — the second door `plan_cluster_pages`'s guard
/// hides. The scene still carries the asset; nothing is allowed to draw it.
#[test]
fn vgeom_releases_residency_when_the_setting_goes_off() {
    let Some(gpu) = gpu_or_skip("vgeom_releases_residency_when_the_setting_goes_off") else {
        return;
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(vgeom_settings(true));
    let view = view();
    let mesh = Arc::new(dense_grid_mesh(24));
    let scene = vgeom_scene(&mesh, true);

    for _ in 0..3 {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    }
    assert!(renderer.vgeom_stream_report().stats.resident_bytes > 0);

    renderer.set_settings(vgeom_settings(false));
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let cold = renderer.vgeom_stream_report();
    assert_eq!(
        (
            cold.stats.assets,
            cold.stats.resident_bytes,
            cold.floor_bytes
        ),
        (0, 0, 0),
        "switching the meshlet path off releases what it was streaming"
    );

    // Back on: the streamer re-pages rather than staying dead.
    renderer.set_settings(vgeom_settings(true));
    for _ in 0..3 {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    }
    assert!(
        renderer.vgeom_stream_report().stats.resident_bytes > 0,
        "the release must not be a one-way door"
    );
}
