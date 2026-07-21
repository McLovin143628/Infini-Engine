//! Golden-image harness (P2.5). For each scene we:
//!   1. render it twice and assert the two frames match — the renderer is
//!      deterministic on a fixed adapter (catches nondeterminism/races);
//!   2. assert scene-specific structural properties (sky is sky, the cube is
//!      where it should be) — adapter-independent;
//!   3. compare against the committed golden PNG. This runs when
//!      `INF_GOLDEN_STRICT=1` (a matched-adapter run) and always writes the
//!      golden when it's missing or `INF_BLESS_GOLDENS=1`.
//!
//! Exact cross-GPU pixels differ (AA/rasterization), so strict pixel diffing
//! is opt-in; CI relies on the determinism + structural gates, which are
//! adapter-robust. Regenerate goldens with `INF_BLESS_GOLDENS=1 cargo test -p
//! inf-render --test golden`. The harness skips entirely with no GPU adapter
//! (headless CI without lavapipe/WARP).

use std::path::PathBuf;

use glam::{DVec3, Quat, Vec2, Vec3};
use inf_math::FloatingOrigin;
use inf_render::gizmo::{self, GizmoAxis, GizmoMode};
use inf_render::golden::{image_diff, within_tolerance};
use inf_render::{
    Ambient2D, EngineRenderer, GpuContext, HeadlessTarget, LightKind, MeshInstance, RenderChunk,
    RenderLight, RenderLight2D, RenderScene, RenderTilemap, RenderView, SpriteInstance,
    SpriteTextureUpload, TilemapParams, HEADLESS_FORMAT, TILE_CHUNK_DIM,
};

const W: u32 = 320;
const H: u32 = 180;

fn goldens_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("goldens")
}

fn gpu_or_skip() -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP golden: no GPU adapter ({e})");
            None
        }
    }
}

fn overlook_view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(6.0, 4.5, 9.0),
        forward: (DVec3::ZERO - DVec3::new(6.0, 4.5, 9.0))
            .as_vec3()
            .normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
    }
}

fn render(gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.render(gpu, scene, view, &target.view, (W, H));
    target.read_rgba(gpu).expect("readback")
}

fn write_png(path: &std::path::Path, rgba: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let file = std::fs::File::create(path).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), W, H);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header().unwrap().write_image_data(rgba).unwrap();
}

fn read_png(path: &std::path::Path) -> Option<Vec<u8>> {
    let file = std::fs::File::open(path).ok()?;
    let dec = png::Decoder::new(std::io::BufReader::new(file));
    let mut reader = dec.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    buf.truncate(info.buffer_size());
    Some(buf)
}

fn px(rgba: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

/// The shared gate: determinism, then golden write/compare.
fn check_golden(gpu: &GpuContext, name: &str, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    let a = render(gpu, scene, view);
    let b = render(gpu, scene, view);
    let (mean, max) = image_diff(&a, &b, W, H);
    assert!(
        mean < 0.005 && max < 0.05,
        "{name}: renderer not deterministic (mean {mean}, max {max})"
    );

    let path = goldens_dir().join(format!("{name}.png"));
    let bless = std::env::var("INF_BLESS_GOLDENS").is_ok();
    let strict = std::env::var("INF_GOLDEN_STRICT").is_ok();

    if bless || read_png(&path).is_none() {
        write_png(&path, &a);
        eprintln!("golden {name}: wrote {}", path.display());
    } else if strict {
        let golden = read_png(&path).expect("golden png");
        let (mean, max) = image_diff(&a, &golden, W, H);
        assert!(
            within_tolerance(mean, max),
            "{name}: differs from golden (mean {mean}, max {max})"
        );
    }
    a
}

#[test]
fn golden_grid_and_sky() {
    let Some(gpu) = gpu_or_skip() else { return };
    let scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    let img = check_golden(&gpu, "grid_sky", &scene, &overlook_view());

    // Structural: top rows are sky (not black, leaning blue); bottom rows show
    // the lit grid on the ground plane.
    let sky = px(&img, W / 2, 3);
    assert!(
        sky[2] as u16 + 4 >= sky[0] as u16,
        "sky not bluish: {sky:?}"
    );
    assert!(sky[2] > 3, "sky too dark: {sky:?}");
    // The frame is not uniform (grid present).
    let ground = px(&img, W / 2, H - 20);
    assert_ne!(sky, ground);
}

#[test]
fn golden_cubes() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    for (i, (x, z, c)) in [
        (0.0, 0.0, [0.80, 0.20, 0.20]),
        (2.5, -1.0, [0.20, 0.70, 0.30]),
        (-2.0, 1.5, [0.25, 0.45, 0.95]),
    ]
    .into_iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.5, z),
            Quat::from_rotation_y(0.3),
            Vec3::ONE,
            [c[0], c[1], c[2], 1.0],
            i as u32 + 1,
        ));
    }
    scene.mark_dirty();
    let img = check_golden(&gpu, "cubes", &scene, &overlook_view());

    // The central red cube dominates the middle of the frame.
    let center = px(&img, W / 2, H / 2);
    assert!(
        center[0] > center[2] && center[0] > 40,
        "expected the red cube at center: {center:?}"
    );
}

#[test]
fn golden_selection_gizmo() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        selected: vec![1],
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 0.5, 0.0),
        Quat::IDENTITY,
        Vec3::ONE,
        [0.30, 0.55, 0.65, 1.0],
        1,
    ));
    scene.mark_dirty();
    let view = overlook_view();
    // Translate gizmo at the cube, screen-constant size.
    let origin_local = view.origin.to_render(DVec3::new(0.0, 0.5, 0.0));
    let size = gizmo::gizmo_world_size(origin_local, view.eye_local(), view.fov_y);
    gizmo::build_geometry(
        &mut scene.debug,
        GizmoMode::Translate,
        origin_local,
        size,
        Some(GizmoAxis::X),
    );

    let img = check_golden(&gpu, "selection_gizmo", &scene, &view);

    // The selection outline paints the composite's orange edge (linear
    // (1.0, 0.42, 0.05) → sRGB ≈ [255, 171, 63]) somewhere in the central
    // band: red-dominant, mid green, low blue.
    let mut found_outline = false;
    'scan: for y in (H / 4)..(3 * H / 4) {
        for x in (W / 4)..(3 * W / 4) {
            let p = px(&img, x, y);
            if p[0] > 200 && (100..=210).contains(&p[1]) && p[2] < 120 && p[0] > p[2] + 90 {
                found_outline = true;
                break 'scan;
            }
        }
    }
    assert!(found_outline, "selection outline (orange) not found");
}

/// PBR material scene (P7.4 golden): metallic/roughness/emissive variation lit by
/// a directional key + a coloured point light. Exercises the P7.1 shading path
/// in CI (determinism gate via `check_golden`; strict pixel diff is opt-in).
#[test]
fn golden_pbr_materials() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    // A row of cubes sweeping roughness at metallic = 1.
    for (i, &(x, rough, metallic, emissive)) in [
        (-3.0f64, 0.1f32, 1.0f32, [0.0f32, 0.0, 0.0]),
        (-1.0, 0.4, 1.0, [0.0, 0.0, 0.0]),
        (1.0, 0.7, 0.0, [0.0, 0.0, 0.0]),
        (3.0, 0.5, 0.0, [0.6, 0.15, 0.0]), // emissive
    ]
    .iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance {
            translation: DVec3::new(x, 0.5, 0.0),
            rotation: Quat::from_rotation_y(0.4),
            scale: Vec3::ONE,
            color: [0.85, 0.78, 0.55, 1.0],
            metallic,
            roughness: rough,
            emissive,
            id: i as u32 + 1,
        });
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.4, 0.8, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
    });
    scene.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [0.3, 0.5, 1.0],
        intensity: 30.0,
        direction: Vec3::ZERO,
        position: DVec3::new(0.0, 2.5, 2.0),
        range: 12.0,
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "pbr_materials", &scene, &overlook_view());
    // The scene is lit: some pixel is clearly brighter than the dark backdrop.
    let lit = img.chunks(4).any(|p| p[0] > 90 || p[1] > 90 || p[2] > 90);
    assert!(lit, "expected a lit PBR pixel");
}

/// A view looking straight down -Z at the world XY plane, so sprites (which lie
/// in that plane facing +Z) face the camera head-on.
fn front_view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 6.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
    }
}

/// Procedural checkerboard: `cells×cells` grid over `size×size` px alternating
/// `color` and white; `alpha` sets the color cells' opacity.
fn checkerboard(size: u32, cells: u32, color: [u8; 3], alpha: u8) -> Vec<u8> {
    let cell = (size / cells).max(1);
    let mut v = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let on = ((x / cell) + (y / cell)).is_multiple_of(2);
            if on {
                v.extend_from_slice(&[color[0], color[1], color[2], alpha]);
            } else {
                v.extend_from_slice(&[235, 235, 235, 255]);
            }
        }
    }
    v
}

/// 2D sprite golden (P8.1a): two textured, alpha-blended sprites on distinct
/// sorting layers, backed by in-test procedural checkerboards (no binary
/// fixtures). Exercises the batcher → texture cache → sprite pass path in CI
/// (determinism gate via `check_golden`; strict pixel diff is opt-in).
#[test]
fn golden_sprites_2d() {
    let Some(gpu) = gpu_or_skip() else { return };

    const TEX_A: u64 = 0xA1;
    const TEX_B: u64 = 0xB2;
    let mut scene = RenderScene {
        grid_enabled: true,
        pending_texture_uploads: vec![
            SpriteTextureUpload {
                handle: TEX_A,
                width: 64,
                height: 64,
                rgba8: checkerboard(64, 8, [220, 40, 40], 255),
            },
            SpriteTextureUpload {
                handle: TEX_B,
                width: 64,
                height: 64,
                rgba8: checkerboard(64, 8, [40, 90, 220], 255),
            },
        ],
        ..Default::default()
    };

    // Two overlapping sprites: the blue one (higher layer) draws over the red.
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(-0.6, 0.0, 0.0),
        size: Vec2::new(2.4, 2.4),
        color: [1.0, 1.0, 1.0, 1.0],
        texture: TEX_A,
        sorting_layer: 0,
        ..Default::default()
    });
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.6, 0.0, 0.0),
        size: Vec2::new(2.4, 2.4),
        color: [1.0, 1.0, 1.0, 1.0],
        texture: TEX_B,
        sorting_layer: 1,
        ..Default::default()
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "sprites_2d", &scene, &front_view());

    // Both sprites are visible: a red checker cell (texture A, its non-overlapped
    // left half) and a blue one (texture B, drawn on top on the higher layer),
    // plus the shared bright checker cells.
    let mut red = false;
    let mut blue = false;
    let mut bright = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 140 && r - b > 60 && r - g > 60 {
            red = true;
        }
        if b > 120 && b - r > 40 {
            blue = true;
        }
        if r > 200 && g > 200 && b > 200 {
            bright = true;
        }
    }
    assert!(red, "expected a red sprite checker cell");
    assert!(blue, "expected a blue sprite checker cell");
    assert!(bright, "expected bright sprite checker cells");
}

/// A `size×size` RGBA atlas of four solid-color quadrants (2×2 grid), laid out
/// so 1-based tile indices 1..=4 map to top-left, top-right, bottom-left,
/// bottom-right respectively (row-major, row 0 = the atlas top row).
fn quad_atlas(size: u32, cells: [[u8; 3]; 4]) -> Vec<u8> {
    let half = size / 2;
    let mut v = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let q = (y >= half) as usize * 2 + (x >= half) as usize; // 0=TL,1=TR,2=BL,3=BR
            let c = cells[q];
            v.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    v
}

/// 2D tilemap golden (P8.1b): an in-test procedural 4-cell atlas painted across
/// a patch of tiles that straddles **two** chunks, with one loose sprite on a
/// higher sorting layer drawn over the tiles (an ordering proof). Exercises the
/// chunk cull → expansion → prebatched-run → sprite-pass path headlessly
/// (determinism gate via `check_golden`; strict pixel diff opt-in).
#[test]
fn golden_tilemap_2d() {
    let Some(gpu) = gpu_or_skip() else { return };

    const ATLAS: u64 = 0x71;
    // 1→red(TL), 2→green(TR), 3→blue(BL), 4→yellow(BR).
    let atlas = quad_atlas(
        64,
        [[220, 40, 40], [40, 200, 60], [60, 90, 220], [230, 210, 40]],
    );

    let tile = 0.3_f64;
    let dim = TILE_CHUNK_DIM as f64;
    let params = TilemapParams {
        // Place the vertical chunk boundary (global tile x=32) at world x=0 and
        // the row gy=16 at world y=0, so the painted patch centers on screen.
        origin: DVec3::new(-dim * tile, -dim * 0.5 * tile, 0.0),
        tile_size: Vec2::new(tile as f32, tile as f32),
        atlas_cols: 2,
        atlas_rows: 2,
        texture: ATLAS,
        color: [1.0, 1.0, 1.0, 1.0],
        sorting_layer: 0,
        order: 0,
    };

    // Paint tiles gx∈[28,36) gy∈[14,18): 8×4 = 32 tiles spanning chunk (0,0)
    // (gx 28..31) and chunk (1,0) (gx 32..35). Index cycles 1..=4.
    let n = (TILE_CHUNK_DIM * TILE_CHUNK_DIM) as usize;
    let mut chunk0 = vec![0u32; n];
    let mut chunk1 = vec![0u32; n];
    for gy in 14..18i32 {
        for gx in 28..36i32 {
            let idx = (((gx + gy).rem_euclid(4)) + 1) as u32;
            let (cx, lx) = (gx.div_euclid(TILE_CHUNK_DIM), gx.rem_euclid(TILE_CHUNK_DIM));
            let ly = gy.rem_euclid(TILE_CHUNK_DIM);
            let slot = (ly * TILE_CHUNK_DIM + lx) as usize;
            match cx {
                0 => chunk0[slot] = idx,
                1 => chunk1[slot] = idx,
                _ => unreachable!(),
            }
        }
    }

    let mut scene = RenderScene {
        grid_enabled: true,
        pending_texture_uploads: vec![SpriteTextureUpload {
            handle: ATLAS,
            width: 64,
            height: 64,
            rgba8: atlas,
        }],
        tilemaps: vec![RenderTilemap {
            params,
            chunks: vec![
                RenderChunk {
                    coord: (0, 0),
                    tiles: chunk0,
                },
                RenderChunk {
                    coord: (1, 0),
                    tiles: chunk1,
                },
            ],
        }],
        ..Default::default()
    };

    // A loose magenta sprite on a HIGHER sorting layer, centered over the tiles:
    // it must draw on top (proving loose-vs-prebatched ordering).
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.0, 0.0, 0.0),
        size: Vec2::new(0.8, 0.8),
        color: [0.95, 0.15, 0.95, 1.0],
        sorting_layer: 1,
        ..Default::default()
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "tilemap_2d", &scene, &front_view());

    // At least two distinct atlas cells are visible (proves 1-based indexing +
    // ≥2 chunks expanded), and the loose magenta sprite paints over the center.
    let mut red = false;
    let mut blue = false;
    let mut magenta = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 140 && r - g > 80 && r - b > 80 {
            red = true;
        }
        if b > 140 && b - r > 60 && b - g > 40 {
            blue = true;
        }
        if r > 150 && b > 150 && r - g > 60 && b - g > 60 {
            magenta = true;
        }
    }
    assert!(red, "expected a red tile (atlas cell 1)");
    assert!(blue, "expected a blue tile (atlas cell 3)");
    assert!(magenta, "expected the loose magenta sprite over the tiles");
}

/// 2D lighting golden (P8.1c): a dark scene ambient with two colored 2D lights
/// (red on the left, blue on the right) over a big white sprite patch. The
/// `smoothstep` falloff paints a red glow on the left half, a blue glow on the
/// right, and near-black between/outside the radii — proving the sprite shader's
/// 2D-light path (determinism gate via `check_golden`; strict pixel diff opt-in).
#[test]
fn golden_2d_lit() {
    let Some(gpu) = gpu_or_skip() else { return };

    let mut scene = RenderScene {
        // Grid off so the sprite lighting reads without the grid underneath.
        grid_enabled: false,
        // Fully dark ambient: the two lights alone shape the image, and the
        // sprite's far corners stay black (the "dark region" assertion below).
        // (sRGB encoding lifts small linear values a lot, so a truly dark region
        // needs ~0 linear ambient.)
        ambient_2d: Ambient2D([0.0, 0.0, 0.0]),
        ..Default::default()
    };

    // One big white quad (the untextured white fallback) covering the frame, so
    // every sampled pixel is the lit sprite (no sky leaking into the readback).
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.0, 0.0, 0.0),
        size: Vec2::new(40.0, 24.0),
        color: [1.0, 1.0, 1.0, 1.0],
        sorting_layer: 0,
        ..Default::default()
    });

    // Red light on the left, blue on the right.
    scene.lights_2d.push(RenderLight2D {
        color: [1.0, 0.1, 0.1],
        intensity: 1.5,
        radius: 2.2,
        position: DVec3::new(-1.4, 0.0, 0.0),
    });
    scene.lights_2d.push(RenderLight2D {
        color: [0.1, 0.2, 1.0],
        intensity: 1.5,
        radius: 2.2,
        position: DVec3::new(1.4, 0.0, 0.0),
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "2d_lit", &scene, &front_view());

    // The left glow is red-dominant, the right glow blue-dominant, and some
    // pixel is near-black (outside both radii / dark ambient).
    let mut red = false;
    let mut blue = false;
    let mut dark = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 90 && r - g > 50 && r - b > 50 {
            red = true;
        }
        if b > 90 && b - r > 50 && b - g > 30 {
            blue = true;
        }
        if r < 20 && g < 20 && b < 20 {
            dark = true;
        }
    }
    assert!(red, "expected the red 2D light glow");
    assert!(blue, "expected the blue 2D light glow");
    assert!(dark, "expected a dark region outside the light radii");
}
