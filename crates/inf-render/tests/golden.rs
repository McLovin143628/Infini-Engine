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
use std::sync::Arc;

use glam::{DVec3, Mat3, Mat4, Quat, Vec2, Vec3};
use inf_math::FloatingOrigin;
use inf_render::gizmo::{self, GizmoAxis, GizmoMode};
use inf_render::golden::{image_diff, within_tolerance};
use inf_render::passes::vgeom::{cpu_visible_set, cull_flags, frustum_planes, lod_threshold};
use inf_render::{
    assemble_patches, cull_visible, expand_text, Ambient2D, BloomSettings, EngineRenderer,
    GpuContext, HAlign, HeadlessTarget, LightKind, MeshInstance, PrebatchedRun, RenderChunk,
    RenderLight, RenderLight2D, RenderScene, RenderSettings, RenderTerrain, RenderTerrainLayer,
    RenderTerrainTile, RenderTilemap, RenderView, SkinnedInstance, SkinnedMeshData, SkinnedVertex,
    SpriteInstance, SpriteTextureUpload, SsaoSettings, TextParams, TilemapParams, VgeomAsset,
    VgeomInstance, VgeomMesh, VgeomSettings, BILLBOARD_CYLINDRICAL, BILLBOARD_NONE,
    BILLBOARD_SPHERICAL, BUILTIN_FONT_COLS, BUILTIN_FONT_FIRST_CP, BUILTIN_FONT_ROWS,
    BUILTIN_FONT_TEXTURE, HEADLESS_FORMAT, TILE_CHUNK_DIM,
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
        ortho: None,
    }
}

fn render(gpu: &GpuContext, scene: &RenderScene, view: &RenderView) -> Vec<u8> {
    render_with(gpu, scene, view, RenderSettings::default())
}

/// Render one frame with explicit HDR/post settings (bloom/SSAO/exposure). TAA is
/// intentionally not exercised here — single-frame determinism goldens keep it
/// off; the multi-frame convergence is covered by `taa_multiframe_stable`.
fn render_with(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
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
    check_golden_with(gpu, name, scene, view, RenderSettings::default())
}

/// [`check_golden`] with explicit post settings (bloom/SSAO goldens).
fn check_golden_with(
    gpu: &GpuContext,
    name: &str,
    scene: &RenderScene,
    view: &RenderView,
    settings: RenderSettings,
) -> Vec<u8> {
    let a = render_with(gpu, scene, view, settings);
    let b = render_with(gpu, scene, view, settings);
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
        false,
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

/// 2.5D billboard golden (P8.4a): an **angled perspective** camera over a row of
/// sprites — one flat (planar, in the world XY plane), one spherical billboard,
/// one cylindrical billboard — plus a ground grid for depth context. Under the
/// oblique view the planar sprite is seen edge-on/foreshortened while the two
/// billboards turn to face the camera, proving the vertex-shader orientation
/// (determinism gate via `check_golden`; strict pixel diff opt-in). The camera
/// basis rides in the view uniform (`cam_right`/`cam_up`).
#[test]
fn golden_billboards() {
    let Some(gpu) = gpu_or_skip() else { return };

    const TEX: u64 = 0xB1;
    let mut scene = RenderScene {
        grid_enabled: true,
        pending_texture_uploads: vec![SpriteTextureUpload {
            handle: TEX,
            width: 64,
            height: 64,
            rgba8: checkerboard(64, 4, [230, 80, 40], 255),
        }],
        ..Default::default()
    };

    // Three cards standing on the ground plane (pivot bottom-centre), spread
    // along X, each with a distinct billboard mode + tint.
    for (x, mode, tint) in [
        (-2.2f64, BILLBOARD_NONE, [1.0f32, 0.3, 0.3, 1.0]),
        (0.0, BILLBOARD_SPHERICAL, [0.3, 1.0, 0.4, 1.0]),
        (2.2, BILLBOARD_CYLINDRICAL, [0.4, 0.5, 1.0, 1.0]),
    ] {
        scene.sprites.push(SpriteInstance {
            position: DVec3::new(x, 1.0, 0.0),
            size: Vec2::new(1.6, 2.0),
            pivot: Vec2::new(0.5, 0.5),
            color: tint,
            texture: TEX,
            sorting_layer: 1,
            billboard: mode,
            ..Default::default()
        });
    }
    scene.mark_dirty();

    // The angled overlook view (perspective, camera at (6,4.5,9) → origin).
    let img = check_golden(&gpu, "billboards", &scene, &overlook_view());

    // Each tinted billboard shows up: a red, a green and a blue sprite region.
    let (mut red, mut green, mut blue) = (false, false, false);
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 130 && r - g > 60 && r - b > 60 {
            red = true;
        }
        if g > 130 && g - r > 50 && g - b > 40 {
            green = true;
        }
        if b > 130 && b - r > 40 && b - g > 40 {
            blue = true;
        }
    }
    assert!(red, "expected the planar (red) sprite");
    assert!(green, "expected the spherical (green) billboard");
    assert!(blue, "expected the cylindrical (blue) billboard");
}

/// The default four-layer splat palette (grass / rock / dirt / snow), mirroring
/// `inf_ecs::components::default_terrain_layers` (kept inline so inf-render stays
/// free of an inf-ecs dep). Used by the terrain goldens so the layer-blended
/// shading is exercised.
fn default_layers() -> [RenderTerrainLayer; 4] {
    [
        RenderTerrainLayer {
            albedo: [0.20, 0.34, 0.14, 1.0], // grass
            roughness: 0.92,
            tex_scale: 6.0,
        },
        RenderTerrainLayer {
            albedo: [0.33, 0.30, 0.27, 1.0], // rock
            roughness: 0.85,
            tex_scale: 4.0,
        },
        RenderTerrainLayer {
            albedo: [0.42, 0.30, 0.18, 1.0], // dirt
            roughness: 0.95,
            tex_scale: 5.0,
        },
        RenderTerrainLayer {
            albedo: [0.86, 0.89, 0.94, 1.0], // snow
            roughness: 0.65,
            tex_scale: 10.0,
        },
    ]
}

/// A procedural sine-hills terrain across `ntx × ntz` tiles, authored from one
/// global height function so tile edges are seamless. `res` samples/tile, `mps`
/// metres/sample. Tiles are pushed in `(i32,i32)`-sorted order (matching the
/// host's BTreeMap projection). Unpainted (uniform layer 0 = grass) — the splat
/// golden authors real weight gradients.
fn hill_terrain(res: u32, mps: f64, ntx: i32, ntz: i32) -> RenderTerrain {
    let span = (res as f64 - 1.0) * mps;
    let f = |x: f64, z: f64| 4.0 * (x * 0.15).sin() * (z * 0.15).cos() + 3.5;
    let mut tiles = Vec::new();
    for tx in 0..ntx {
        for tz in 0..ntz {
            let (ox, oz) = (tx as f64 * span, tz as f64 * span);
            let mut heights = vec![0f32; (res * res) as usize];
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for j in 0..res {
                for i in 0..res {
                    let h = f(ox + i as f64 * mps, oz + j as f64 * mps) as f32;
                    heights[(j * res + i) as usize] = h;
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            tiles.push(RenderTerrainTile {
                coord: (tx, tz),
                origin: DVec3::new(ox, 0.0, oz),
                heights,
                weights: Vec::new(),
                height_bounds: (lo, hi),
            });
        }
    }
    RenderTerrain {
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
        version: 1,
    }
}

/// A perspective view from `eye` looking at `target`.
fn look_view(eye: DVec3, target: DVec3) -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (target - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Terrain golden (P10.1 geometry, P10.4 shading): a sine-hills heightfield
/// across 2×2 tiles under an angled perspective camera, showing the terrain
/// silhouette + **splat-blended layer shading** against the sky. Unpainted, so
/// weights are uniform layer 0 (grass) — this golden was **regenerated for
/// P10.4** because the shading changed from the old slope/altitude debug ramp to
/// the layer-based blend (albedo + triplanar grain + macro variation). Exercises
/// the clipmap patch assembly → height/weight texture → vertex displacement path
/// headlessly (determinism gate via `check_golden`; strict pixel diff opt-in).
#[test]
fn golden_terrain() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let terrain = hill_terrain(res, 1.0, 2, 2); // 2×2 tiles, ~64 m square
    let scene = RenderScene {
        grid_enabled: true,
        terrain: Some(terrain),
        ..Default::default()
    };
    // Angled overlook of the terrain centre (~(32, ·, 32)).
    let view = look_view(DVec3::new(32.0, 24.0, -12.0), DVec3::new(32.0, 3.0, 32.0));
    let img = check_golden(&gpu, "terrain", &scene, &view);

    // The lower band (terrain) is lit and clearly differs from the sky band above.
    let sky = px(&img, W / 2, 6);
    let ground = px(&img, W / 2, H - 12);
    assert_ne!(sky, ground, "terrain band should differ from sky");
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit terrain pixel");
}

/// Terrain LOD golden (P10.1): the same hills across a long 6×2 strip with the
/// camera at the near end, so tiles resolve to ≥2 distinct clipmap LOD rings by
/// distance. Structural gate: assembly yields multiple LODs + the frame renders
/// deterministically.
#[test]
fn golden_terrain_lod() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let mps = 1.0;
    let terrain = hill_terrain(res, mps, 6, 2); // long strip along +X
                                                // Camera at the near (-X) end looking down the strip: near tiles LOD 0, far
                                                // tiles coarsen → concentric rings.
    let view = look_view(DVec3::new(-6.0, 40.0, 32.0), DVec3::new(140.0, 0.0, 32.0));

    // The pure assembly must produce ≥2 distinct LOD levels (the "≥2 rings" gate).
    let patches = assemble_patches(&terrain, &view, &view.origin);
    let mut lods: Vec<u32> = patches.iter().map(|p| p.lod).collect();
    lods.sort_unstable();
    lods.dedup();
    assert!(
        lods.len() >= 2,
        "expected ≥2 LOD rings, got LODs {lods:?} from {} patches",
        patches.len()
    );

    let scene = RenderScene {
        grid_enabled: true,
        terrain: Some(terrain),
        ..Default::default()
    };
    let img = check_golden(&gpu, "terrain_lod", &scene, &view);
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit terrain pixel");
}

/// A splat-painted terrain (P10.4): 2×2 tiles with hand-authored weight gradients
/// banding all four layers across +X (grass → dirt → rock → snow), plus a **steep
/// cliff** wall so the triplanar detail path is exercised on near-vertical faces.
/// Seamless across tile edges (weights authored from one global world function).
fn splat_terrain(res: u32, mps: f64, ntx: i32, ntz: i32) -> RenderTerrain {
    let span = (res as f64 - 1.0) * mps;
    let total_w = ntx as f64 * span;
    // A steep cliff wall at ~63% of the width (6 m rise over a ~4% band) over a
    // gently rolling base — the wall's near-vertical normals drive triplanar.
    let smoothstep = |e0: f64, e1: f64, x: f64| {
        let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    let height = |x: f64, z: f64| {
        2.0 + 6.0 * smoothstep(0.60 * total_w, 0.64 * total_w, x) + 0.6 * (z * 0.2).sin()
    };
    // Four tent bands across the normalized width → four distinct, blended layers.
    let weight = |x: f64| -> [u8; 4] {
        let u = (x / total_w).clamp(0.0, 1.0);
        let tent = |c: f64| (1.0 - (u - c).abs() * 3.0).max(0.0);
        let raw = [tent(0.0), tent(1.0 / 3.0), tent(2.0 / 3.0), tent(1.0)];
        let s: f64 = raw.iter().sum::<f64>().max(1e-6);
        let mut out = [0u8; 4];
        let mut acc = 0i32;
        for k in 0..4 {
            out[k] = (raw[k] / s * 255.0).round() as u8;
            acc += out[k] as i32;
        }
        out[0] = (out[0] as i32 + (255 - acc)).clamp(0, 255) as u8; // exact sum 255
        out
    };
    let mut tiles = Vec::new();
    for tx in 0..ntx {
        for tz in 0..ntz {
            let (ox, oz) = (tx as f64 * span, tz as f64 * span);
            let mut heights = vec![0f32; (res * res) as usize];
            let mut weights = vec![[0u8; 4]; (res * res) as usize];
            let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
            for j in 0..res {
                for i in 0..res {
                    let (wx, wz) = (ox + i as f64 * mps, oz + j as f64 * mps);
                    let h = height(wx, wz) as f32;
                    heights[(j * res + i) as usize] = h;
                    weights[(j * res + i) as usize] = weight(wx);
                    lo = lo.min(h);
                    hi = hi.max(h);
                }
            }
            tiles.push(RenderTerrainTile {
                coord: (tx, tz),
                origin: DVec3::new(ox, 0.0, oz),
                heights,
                weights,
                height_bounds: (lo, hi),
            });
        }
    }
    RenderTerrain {
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
        version: 1,
    }
}

/// Terrain splat golden (P10.4): a heightfield with hand-authored weight gradients
/// banding all four material layers across +X plus a steep cliff, proving the
/// splat blend + triplanar path headlessly (determinism gate via `check_golden`;
/// strict pixel diff opt-in).
#[test]
fn golden_terrain_splat() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let terrain = splat_terrain(res, 1.0, 2, 2); // ~64 m square, banded layers
    let scene = RenderScene {
        grid_enabled: true,
        terrain: Some(terrain),
        ..Default::default()
    };
    // Angled overlook of the banded terrain, side-on to the cliff.
    let view = look_view(DVec3::new(4.0, 22.0, -10.0), DVec3::new(40.0, 3.0, 32.0));
    let img = check_golden(&gpu, "terrain_splat", &scene, &view);

    // The four layers span from a green (grass) low band to a bright (snow) high
    // band — assert both a greenish and a bright near-white terrain pixel exist.
    let mut green = false;
    let mut snow = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if g > 60 && g - r > 20 && g - b > 20 {
            green = true;
        }
        if r > 180 && g > 180 && b > 180 {
            snow = true;
        }
    }
    assert!(green, "expected the grass (green) layer band");
    assert!(snow, "expected the snow (bright) layer band");
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
        ortho: None,
    }
}

/// A top-down orthographic view over the world XY plane (2D editor mode): eye at
/// +Z looking down -Z, up = +Y. Half-height 4 world units frames a small patch.
fn ortho_view() -> RenderView {
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 100.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 1.0,
        width: W,
        height: H,
        ortho: Some(inf_render::OrthoParams {
            half_height: 4.0,
            near: 1.0,
            far: 200.0,
        }),
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

/// Orthographic 2D-editor golden (P8.2c): the ortho camera over a tile patch, a
/// loose sprite, and a built-in-font text run, with the XY grid enabled. Proves
/// the ortho projection + XY-grid shader path + the 2D content passes render
/// coherently under a parallel projection (determinism gate via `check_golden`;
/// strict pixel diff opt-in, blessed on a GPU host).
#[test]
fn golden_ortho_2d() {
    let Some(gpu) = gpu_or_skip() else { return };

    const ATLAS: u64 = 0x51;
    // 1→red(TL), 2→green(TR), 3→blue(BL), 4→yellow(BR).
    let atlas = quad_atlas(
        64,
        [[220, 40, 40], [40, 200, 60], [60, 90, 220], [230, 210, 40]],
    );
    let tile = 0.5_f64;
    let params = TilemapParams {
        origin: DVec3::new(-1.5, -1.5, 0.0),
        tile_size: Vec2::new(tile as f32, tile as f32),
        atlas_cols: 2,
        atlas_rows: 2,
        texture: ATLAS,
        color: [1.0, 1.0, 1.0, 1.0],
        sorting_layer: 0,
        order: 0,
    };
    // A 6×6 patch of tiles in chunk (0,0), index cycling 1..=4.
    let n = (TILE_CHUNK_DIM * TILE_CHUNK_DIM) as usize;
    let mut chunk = vec![0u32; n];
    for gy in 0..6i32 {
        for gx in 0..6i32 {
            let idx = (((gx + gy).rem_euclid(4)) + 1) as u32;
            chunk[(gy * TILE_CHUNK_DIM + gx) as usize] = idx;
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
            chunks: vec![RenderChunk {
                coord: (0, 0),
                tiles: chunk,
            }],
        }],
        ..Default::default()
    };

    // A loose magenta sprite over the tiles (higher sorting layer).
    scene.sprites.push(SpriteInstance {
        position: DVec3::new(0.4, 0.4, 0.0),
        size: Vec2::new(1.0, 1.0),
        color: [0.95, 0.15, 0.95, 1.0],
        sorting_layer: 2,
        ..Default::default()
    });

    // A short text run in the built-in 8×8 bitmap font (exercises the text path
    // under ortho).
    let text_params = TextParams {
        position: DVec3::new(-2.4, 1.8, 0.0),
        text: "2D",
        glyph_cols: BUILTIN_FONT_COLS,
        glyph_rows: BUILTIN_FONT_ROWS,
        first_codepoint: BUILTIN_FONT_FIRST_CP,
        glyph_size: Vec2::new(0.6, 0.6),
        tracking: 0.1,
        color: [1.0, 1.0, 1.0, 1.0],
        texture: BUILTIN_FONT_TEXTURE,
        sorting_layer: 1,
        order: 0,
        halign: HAlign::Left,
    };
    let glyphs = expand_text(&text_params);
    if !glyphs.is_empty() {
        scene.prebatched.push(PrebatchedRun {
            texture: BUILTIN_FONT_TEXTURE,
            sorting_layer: 1,
            order: 0,
            instances: glyphs,
        });
    }
    scene.mark_dirty();

    let img = check_golden(&gpu, "ortho_2d", &scene, &ortho_view());

    // Under the ortho camera: a red tile cell and the magenta sprite are both
    // visible (proving the parallel projection places 2D content correctly).
    let mut red = false;
    let mut magenta = false;
    for chunk in img.chunks(4) {
        let (r, g, b) = (chunk[0] as i32, chunk[1] as i32, chunk[2] as i32);
        if r > 140 && r - g > 80 && r - b > 80 {
            red = true;
        }
        if r > 150 && b > 150 && r - g > 60 && b - g > 60 {
            magenta = true;
        }
    }
    assert!(red, "expected a red tile under the ortho camera");
    assert!(magenta, "expected the magenta sprite over the tiles");
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A procedural skinned cylinder (P11.1) + a 2-joint bend skeleton + a rotation
/// clip. The lower half is weighted to the root joint, the upper half to the
/// child joint (blended across the middle band), so rotating the child bends the
/// top of the cylinder. Returns the skeleton, the clip, and the bind-space mesh.
fn skinned_cylinder() -> (inf_anim::Skeleton, inf_anim::AnimClip, SkinnedMeshData) {
    use inf_anim::{
        AnimClip, Interpolation, Joint, JointTrack, JointTransform, QuatTrack, Skeleton,
    };

    // Skeleton: root at the origin, child 1 unit up (+Y). Inverse binds are the
    // inverse of each joint's global bind, so the rest pose is undeformed.
    let j0 = JointTransform::IDENTITY;
    let j1 = JointTransform::from_trs(Vec3::Y, Quat::IDENTITY, Vec3::ONE);
    let g0 = j0.to_mat4();
    let g1 = g0 * j1.to_mat4();
    let skeleton = Skeleton::new(vec![
        Joint {
            name: "root".into(),
            parent: None,
            inverse_bind: g0.inverse().to_cols_array(),
            local_bind: j0,
        },
        Joint {
            name: "upper".into(),
            parent: Some(0),
            inverse_bind: g1.inverse().to_cols_array(),
            local_bind: j1,
        },
    ])
    .unwrap();

    // Clip: rotate the child joint 0° → 60° about +Z over 1 second.
    let mut jt = JointTrack::new(1);
    jt.rotation = Some(QuatTrack::new(
        vec![0.0, 1.0],
        vec![
            Quat::IDENTITY.to_array(),
            Quat::from_rotation_z(60f32.to_radians()).to_array(),
        ],
        Interpolation::Linear,
    ));
    let clip = AnimClip::new("bend", vec![jt]);

    // A radial cylinder along +Y, height 2, radius 0.35.
    let (radial, rings, radius, height) = (16usize, 8usize, 0.35f32, 2.0f32);
    let mut vertices = Vec::new();
    for r in 0..=rings {
        let y = height * r as f32 / rings as f32;
        let w1 = smoothstep(0.5, 1.5, y);
        let w0 = 1.0 - w1;
        for s in 0..radial {
            let a = std::f32::consts::TAU * s as f32 / radial as f32;
            let (c, sn) = (a.cos(), a.sin());
            vertices.push(SkinnedVertex {
                pos: [radius * c, y, radius * sn],
                normal: [c, 0.0, sn],
                joints: [0, 1, 0, 0],
                weights: [w0, w1, 0.0, 0.0],
            });
        }
    }
    let mut indices = Vec::new();
    for r in 0..rings {
        for s in 0..radial {
            let s1 = (s + 1) % radial;
            let a = (r * radial + s) as u32;
            let b = (r * radial + s1) as u32;
            let c = ((r + 1) * radial + s) as u32;
            let d = ((r + 1) * radial + s1) as u32;
            // Outward-facing winding (CCW seen from outside the tube).
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    (skeleton, clip, SkinnedMeshData { vertices, indices })
}

/// The skinning palette (`global · inverse_bind` per joint) for a clip at time `t`.
fn palette_at(sk: &inf_anim::Skeleton, clip: &inf_anim::AnimClip, t: f32) -> Vec<Mat4> {
    let pose = inf_anim::sample_clip(sk, clip, t, false);
    inf_anim::skinning_matrices(sk, &pose)
}

/// Skinned-mesh golden (P11.1): a procedural skinned cylinder driven by a real
/// `inf-anim` clip, rendered at `t=0` (rest, straight) vs `t=mid` (bent). The
/// committed golden is the bent pose; the structural gate proves **deformation**
/// — the two poses render meaningfully differently — and that the skinned pixels
/// are lit (the GPU skinning path actually ran). Determinism gate via
/// `check_golden`; strict pixel diff opt-in. The unskinned pipeline is untouched,
/// so every other golden stays byte-stable.
#[test]
fn golden_skinned_mesh() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (sk, clip, mesh) = skinned_cylinder();

    let make = |palette: Vec<Mat4>| SkinnedInstance {
        translation: DVec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
        color: [0.75, 0.55, 0.35, 1.0],
        metallic: 0.0,
        roughness: 0.6,
        emissive: [0.0; 3],
        id: 1,
        mesh: 0,
        palette,
    };

    let mut rest = RenderScene {
        grid_enabled: true,
        skinned_meshes: vec![mesh.clone()],
        ..Default::default()
    };
    rest.skinned.push(make(palette_at(&sk, &clip, 0.0)));
    rest.mark_dirty();

    let mut bent = RenderScene {
        grid_enabled: true,
        skinned_meshes: vec![mesh],
        ..Default::default()
    };
    bent.skinned.push(make(palette_at(&sk, &clip, 0.5)));
    bent.mark_dirty();

    // Angled view framing the ~2 m tall cylinder around its middle.
    let view = look_view(DVec3::new(3.2, 1.6, 3.6), DVec3::new(0.0, 1.0, 0.0));
    let rest_img = render(&gpu, &rest, &view);
    let bent_img = check_golden(&gpu, "skinned_mesh", &bent, &view);

    // Deformation: the bent pose differs meaningfully from the rest pose.
    let (mean, max) = image_diff(&rest_img, &bent_img, W, H);
    assert!(
        mean > 0.002,
        "expected visible skinning deformation between t=0 and t=mid (mean {mean}, max {max})"
    );
    // The skinned cylinder is actually lit (the GPU skinning path ran).
    let lit = bent_img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit skinned pixel");
}

// ── P13.3a: HDR post pipeline (bloom, SSAO, TAA) ─────────────────────────────

/// HDR bloom golden (P13.3a): a dark scene with a few **strongly emissive** cubes
/// (linear emissive ≫ 1) so the bloom threshold prefilter + blur mip chain lights
/// up a soft glow the tonemap adds back. Structural gate: with bloom ON the frame
/// carries more total energy than with bloom OFF (the additive blurred glow),
/// while the emitters stay bright and coloured (no NaN blowout). Determinism via
/// `check_golden_with`; strict pixel diff opt-in.
#[test]
fn golden_hdr_bloom() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    for (i, (x, emissive)) in [
        (-2.6f64, [8.0f32, 1.0, 0.4]),
        (0.0, [0.5, 7.0, 1.0]),
        (2.6, [0.6, 1.2, 9.0]),
    ]
    .into_iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance {
            translation: DVec3::new(x, 0.5, 0.0),
            rotation: Quat::from_rotation_y(0.3),
            scale: Vec3::splat(0.6),
            color: [0.02, 0.02, 0.02, 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive,
            id: i as u32 + 1,
        });
    }
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 1.5, 7.0), DVec3::new(0.0, 0.5, 0.0));

    let bloom_on = RenderSettings {
        bloom: BloomSettings {
            enabled: true,
            threshold: 1.0,
            knee: 0.6,
            intensity: 0.5,
        },
        ..RenderSettings::default()
    };

    let img = check_golden_with(&gpu, "hdr_bloom", &scene, &view, bloom_on);
    let img_off = render_with(&gpu, &scene, &view, RenderSettings::default());

    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum()
    };
    let (sum_on, sum_off) = (sum(&img), sum(&img_off));
    assert!(
        sum_on > sum_off + (img.len() as u64 / 4),
        "bloom should add glow energy (on {sum_on} vs off {sum_off})"
    );
    let bright = img
        .chunks(4)
        .any(|p| p[0] > 180 || p[1] > 180 || p[2] > 180);
    assert!(bright, "expected the emissive cubes to stay bright");
}

/// SSAO golden (P13.3a): a cluster of boxes forming **crevices** (a floor slab
/// with blocks pressed together and one stacked), lit by a single soft
/// directional key, SSAO ON. Structural gate: SSAO **darkens** the frame overall
/// (ambient occluded in the contact creases) while the scene stays lit — proving
/// the depth-prepass → half-res AO → ambient-multiply path ran. Determinism via
/// `check_golden_with`; strict pixel diff opt-in.
#[test]
fn golden_ssao() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(8.0, 0.5, 8.0),
        [0.6, 0.6, 0.62, 1.0],
        1,
    ));
    for (p, id) in [
        (DVec3::new(-0.55, 0.5, 0.0), 2u32),
        (DVec3::new(0.55, 0.5, 0.0), 3),
        (DVec3::new(0.0, 0.5, -0.9), 4),
        (DVec3::new(0.0, 1.5, 0.0), 5),
    ] {
        scene.instances.push(MeshInstance::lit(
            p,
            Quat::IDENTITY,
            Vec3::ONE,
            [0.7, 0.65, 0.6, 1.0],
            id,
        ));
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 1.2,
        direction: Vec3::new(0.3, 0.9, 0.3).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(3.0, 2.6, 4.2), DVec3::new(0.0, 0.6, 0.0));

    let ssao_on = RenderSettings {
        ssao: SsaoSettings {
            enabled: true,
            radius: 0.7,
            intensity: 1.0,
            bias: 0.03,
        },
        ..RenderSettings::default()
    };

    let img = check_golden_with(&gpu, "ssao", &scene, &view, ssao_on);
    let img_off = render_with(&gpu, &scene, &view, RenderSettings::default());

    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum()
    };
    let (sum_on, sum_off) = (sum(&img), sum(&img_off));
    assert!(
        sum_on < sum_off,
        "SSAO should darken the ambient term (on {sum_on} vs off {sum_off})"
    );
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 200);
    assert!(lit, "expected the SSAO scene to stay lit");
}

/// TAA multi-frame stability smoke (P13.3a): with TAA ON and a **static** camera,
/// render N frames on one renderer (the history accumulates). Asserts (1) no NaN /
/// out-of-range garbage ever appears, and (2) after convergence consecutive frames
/// differ by a small, bounded amount (jitter+history settle to a steady image) —
/// not a pixel golden, since TAA is intentionally non-deterministic frame to
/// frame. Skips with no GPU adapter.
#[test]
fn taa_multiframe_stable() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    for (i, (x, z, c)) in [
        (0.0, 0.0, [0.80, 0.20, 0.20]),
        (2.0, -1.0, [0.20, 0.70, 0.30]),
        (-1.8, 1.2, [0.25, 0.45, 0.95]),
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
    let view = overlook_view();

    let settings = RenderSettings {
        taa: true,
        ..RenderSettings::default()
    };
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);

    let mut prev: Option<Vec<u8>> = None;
    let mut last_delta = (1.0f32, 1.0f32);
    for f in 0..12 {
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let img = target.read_rgba(&gpu).expect("readback");
        // A NaN blowout would clamp the whole buffer to black or white.
        let nonblack = img.chunks(4).any(|p| p[0] > 5 || p[1] > 5 || p[2] > 5);
        let nonwhite = img
            .chunks(4)
            .any(|p| p[0] < 250 || p[1] < 250 || p[2] < 250);
        assert!(nonblack && nonwhite, "frame {f} degenerate (NaN blowout?)");
        if let Some(p) = &prev {
            last_delta = image_diff(p, &img, W, H);
        }
        prev = Some(img);
    }
    let (mean, max) = last_delta;
    assert!(
        mean < 0.02 && max < 0.35,
        "TAA did not converge: last frame delta mean {mean}, max {max}"
    );
}

// ── P13.1b: GPU-driven virtualized-geometry (meshlet) path ───────────────────

/// A dense procedural mesh: an `n×n` grid quad-plane displaced by a smooth
/// bi-sinusoid (so it has real curvature → nontrivial normal cones + multiple LOD
/// levels), spanning x,z ∈ [-1, 1]. `2·n·n` triangles → the vgeom builder produces
/// several meshlets and coarser LOD levels.
fn dense_grid_mesh(n: usize) -> VgeomMesh {
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let u = i as f32 / n as f32;
            let v = j as f32 / n as f32;
            let x = (u - 0.5) * 2.0;
            let z = (v - 0.5) * 2.0;
            let y = 0.3 * (x * 3.0).sin() * (z * 3.0).cos();
            let dydx = 0.3 * 3.0 * (x * 3.0).cos() * (z * 3.0).cos();
            let dydz = -0.3 * 3.0 * (x * 3.0).sin() * (z * 3.0).sin();
            let nrm = Vec3::new(-dydx, 1.0, -dydz).normalize();
            positions.push([x, y, z]);
            normals.push(nrm.to_array());
            uvs.push([u, v]);
        }
    }
    let stride = (n + 1) as u32;
    let mut indices = Vec::new();
    for j in 0..n as u32 {
        for i in 0..n as u32 {
            let a = j * stride + i;
            let b = a + 1;
            let c = a + stride;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    inf_vgeom::build_vgeom(
        &positions,
        &normals,
        &uvs,
        &indices,
        inf_vgeom::BuildParams::default(),
    )
}

const VGEOM_ASSET: u128 = 0x1313_1b00_dead_beef;

fn vgeom_scene(mesh: Arc<VgeomMesh>, scale: f32) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: true,
        vgeom_assets: vec![VgeomAsset {
            id: VGEOM_ASSET,
            mesh,
        }],
        ..Default::default()
    };
    scene.vgeom_instances.push(VgeomInstance::lit(
        VGEOM_ASSET,
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(scale),
        [0.72, 0.52, 0.30, 1.0],
        1,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.35, 0.85, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
    });
    scene.mark_dirty();
    scene
}

fn vgeom_settings() -> RenderSettings {
    RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    }
}

/// vgeom **dense** golden (P13.1b): the dense mesh at close range under an angled
/// key light, drawn entirely through the GPU meshlet path (cull+LOD compute →
/// vertex-pulled indirect draw). Structural gate: the meshlet surface is lit (the
/// path actually rasterized geometry) and the visible-meshlet count read back from
/// the cull compute is > 0. Determinism via `check_golden_with`; strict pixel diff
/// opt-in. The classic path is untouched (no `MeshInstance`s), so every other
/// golden stays byte-identical.
#[test]
fn golden_vgeom_dense() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mesh = Arc::new(dense_grid_mesh(40));
    let scene = vgeom_scene(mesh.clone(), 2.0);
    // Close overlook of the ~4 m mesh.
    let view = look_view(DVec3::new(0.0, 3.2, 4.6), DVec3::new(0.0, 0.0, 0.0));

    let img = check_golden_with(&gpu, "vgeom_dense", &scene, &view, vgeom_settings());
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected the meshlet surface to be lit");

    // The cull compute selected some meshlets for this frame.
    let visible = cull_visible(
        &gpu,
        &mesh,
        &scene.vgeom_instances,
        &view,
        &vgeom_settings().vgeom,
    );
    assert!(
        !visible.is_empty(),
        "expected visible meshlets at close range"
    );
}

/// vgeom **far** golden (P13.1b) + the **LOD proof**: the same dense mesh viewed
/// from far away resolves to a COARSER cut — the cull compute selects strictly
/// FEWER meshlets than at close range (larger projected screen-error threshold ⇒
/// coarser LOD). Determinism via `check_golden_with`; strict pixel diff opt-in.
#[test]
fn golden_vgeom_far() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mesh = Arc::new(dense_grid_mesh(40));
    let scene = vgeom_scene(mesh.clone(), 2.0);

    let close = look_view(DVec3::new(0.0, 3.2, 4.6), DVec3::new(0.0, 0.0, 0.0));
    let far = look_view(DVec3::new(0.0, 26.0, 38.0), DVec3::new(0.0, 0.0, 0.0));

    let img = check_golden_with(&gpu, "vgeom_far", &scene, &far, vgeom_settings());
    // Something rendered (the mesh is small but present).
    let any = img.chunks(4).any(|p| p[0] > 8 || p[1] > 8 || p[2] > 8);
    assert!(any, "expected the far meshlet mesh to render");

    let s = vgeom_settings().vgeom;
    let n_close = cull_visible(&gpu, &mesh, &scene.vgeom_instances, &close, &s).len();
    let n_far = cull_visible(&gpu, &mesh, &scene.vgeom_instances, &far, &s).len();
    eprintln!(
        "vgeom LOD proof: {} total meshlets, close cut = {n_close}, far cut = {n_far}",
        mesh.meshlet_count()
    );
    assert!(
        n_close > 0 && n_far > 0,
        "both cuts non-empty (close {n_close}, far {n_far})"
    );
    assert!(
        n_far < n_close,
        "LOD proof: far cut should select fewer meshlets (close {n_close}, far {n_far})"
    );
}

/// CPU-vs-GPU cut parity (P13.1b — the strongest gate): for a fixed camera with
/// the whole mesh comfortably in-frustum and cone culling off, the GPU cull
/// compute's visible meshlet set (read back) must **exactly** equal the CPU
/// reference `cpu_visible_set` (the identical LOD cut + frustum filter), which in
/// turn equals `VgeomMesh::select(t)` (the offline reference rule). The
/// per-instance threshold `t` is a single scalar uploaded verbatim, so the
/// branchless cut is bit-identical on both sides.
#[test]
fn vgeom_cpu_gpu_cut_parity() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mesh = dense_grid_mesh(40);

    // The whole mesh (spans ±1, scale 1) sits well inside a 60° frustum at this
    // distance — no meshlet is near a frustum boundary, so float divergence can't
    // flip a cull. Cone culling is off (its per-normal boundary is the only place
    // CPU/GPU could disagree); it is exercised by the pure `cpu_visible_set` unit
    // tests instead.
    let view = look_view(DVec3::new(0.0, 2.2, 4.2), DVec3::new(0.0, 0.0, 0.0));
    let inst = VgeomInstance::lit(
        VGEOM_ASSET,
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::ONE,
        [0.7, 0.7, 0.7, 1.0],
        1,
    );
    let settings = VgeomSettings {
        enabled: true,
        cone_cull: false,
        frustum_cull: true,
        occlusion: false,
        pixel_error: 1.0,
        debug_meshlets: false,
    };

    let gpu_pairs = cull_visible(&gpu, &mesh, std::slice::from_ref(&inst), &view, &settings);
    // Single instance ⇒ instance index is always 0; extract the meshlet ids.
    let gpu_meshlets: Vec<u32> = gpu_pairs.iter().map(|e| e[1]).collect();

    // CPU reference (same math as the shader).
    let origin = view.origin;
    let model = origin.model_matrix(inst.translation, inst.rotation, inst.scale);
    let max_scale = inst.scale.abs().max_element().max(1e-6);
    let inv_scale = inst.scale.max(Vec3::splat(1e-6)).recip();
    let normal_mat = Mat3::from_quat(inst.rotation) * Mat3::from_diagonal(inv_scale);
    let eye = view.eye_local();
    let center_world = model.transform_point3(Vec3::from(mesh.center));
    let radius = mesh.radius * max_scale;
    let t = lod_threshold(
        eye,
        center_world,
        radius,
        max_scale,
        &view,
        settings.pixel_error,
    );
    let planes = frustum_planes(view.view_proj());
    let cpu_meshlets = cpu_visible_set(
        &mesh,
        model,
        normal_mat,
        eye,
        t,
        max_scale,
        &planes,
        cull_flags(&settings),
    );

    assert!(!cpu_meshlets.is_empty(), "reference cut is empty");
    assert_eq!(
        gpu_meshlets, cpu_meshlets,
        "GPU visible set must equal the CPU reference (frustum + LOD)"
    );

    // And the CPU reference (frustum passes everything here) equals the offline
    // rule VgeomMesh::select(t) — the meshlet DAG cut.
    let select_ids: Vec<u32> = mesh.select(t).map(|(i, _)| i as u32).collect();
    assert_eq!(
        cpu_meshlets, select_ids,
        "frustum passes all in-view meshlets ⇒ cut == VgeomMesh::select(t)"
    );
}
