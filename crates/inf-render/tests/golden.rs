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
    assemble_patches, cull_visible, detail_texel, expand_text, shape_texel, Ambient2D,
    AtmosphereParams, AtmosphereQuality, BloomSettings, CloudParams, CloudQuality, CloudVolumes,
    EngineRenderer, GiSettings, GpuContext, HAlign, HeadlessTarget, HeightFog, LightKind,
    MeshInstance, PrebatchedRun, PrecipParams, PrecipQuality, PrimMesh, RenderChunk, RenderLight,
    RenderLight2D, RenderScene, RenderSettings, RenderTerrain, RenderTerrainLayer,
    RenderTerrainTile, RenderTilemap, RenderView, ShadowSettings, SkinnedInstance, SkinnedMeshData,
    SkinnedVertex, SpriteInstance, SpriteTextureUpload, SsaoSettings, SunParams, TerrainTileKey,
    TextParams, TilemapParams, VgeomAsset, VgeomInstance, VgeomMesh, VgeomSettings, ViewMode,
    BILLBOARD_CYLINDRICAL, BILLBOARD_NONE, BILLBOARD_SPHERICAL, BUILTIN_FONT_COLS,
    BUILTIN_FONT_FIRST_CP, BUILTIN_FONT_ROWS, BUILTIN_FONT_TEXTURE, CPU_GPU_EXACT_FRACTION,
    CPU_GPU_SHADOW_TOLERANCE, CPU_GPU_TEXEL_TOLERANCE, HEADLESS_FORMAT, TILE_CHUNK_DIM,
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

/// Render one frame in a given [`ViewMode`] (R-P2), default post settings.
fn render_view_mode(
    gpu: &GpuContext,
    scene: &RenderScene,
    view: &RenderView,
    mode: ViewMode,
) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_view_mode(mode);
    renderer.render(gpu, scene, view, &target.view, (W, H));
    target.read_rgba(gpu).expect("readback")
}

/// Unlit view-mode golden (R-P2): the same three cubes as [`golden_cubes`], but
/// rendered with `set_view_mode(Unlit)` so the lit passes short-circuit to
/// albedo+emissive (no lighting). Determinism gate (render twice), a new committed
/// golden `unlit.png` (bless with `INF_BLESS_GOLDENS=1`), and a structural gate:
/// the unlit frame must differ from the lit one (proving the flag actually flipped
/// the shading), each cube still shows its flat base colour, and — crucially — the
/// *lit* frame stays byte-identical to `golden_cubes` (view mode never perturbs the
/// default Lit path; every pre-R-P2 golden is unaffected). Wireframe is NOT
/// goldened — line raster is adapter-fragile (feature-gated + AA-dependent) — so it
/// is covered by the naga compose test + the caps/degrade unit tests instead.
#[test]
fn golden_unlit() {
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
    let view = overlook_view();

    // Determinism + golden write/compare for the Unlit render.
    let a = render_view_mode(&gpu, &scene, &view, ViewMode::Unlit);
    let b = render_view_mode(&gpu, &scene, &view, ViewMode::Unlit);
    let (mean, max) = image_diff(&a, &b, W, H);
    assert!(
        mean < 0.005 && max < 0.05,
        "unlit: renderer not deterministic (mean {mean}, max {max})"
    );
    let path = goldens_dir().join("unlit.png");
    if std::env::var("INF_BLESS_GOLDENS").is_ok() || read_png(&path).is_none() {
        write_png(&path, &a);
        eprintln!("golden unlit: wrote {}", path.display());
    } else if std::env::var("INF_GOLDEN_STRICT").is_ok() {
        let golden = read_png(&path).expect("golden png");
        let (m, mx) = image_diff(&a, &golden, W, H);
        assert!(
            within_tolerance(m, mx),
            "unlit: differs from golden (mean {m}, max {mx})"
        );
    }

    // The unlit render differs from the lit one (the flag genuinely changed the
    // shading — unlit is flatter/brighter, no GGX/ambient/haze).
    let lit = render_view_mode(&gpu, &scene, &view, ViewMode::Lit);
    let (dmean, _dmax) = image_diff(&a, &lit, W, H);
    assert!(dmean > 0.002, "unlit should differ from lit (mean {dmean})");

    // The default Lit path is byte-stable vs the plain-renderer `golden_cubes`
    // frame — view mode never perturbs Lit (the byte-identical guarantee).
    let plain = render(&gpu, &scene, &view);
    let (lmean, lmax) = image_diff(&lit, &plain, W, H);
    assert!(
        lmean < 1e-6 && lmax < 1e-6,
        "Lit view mode must match the default renderer exactly (mean {lmean}, max {lmax})"
    );

    // The central red cube still reads as red under unlit shading.
    let center = px(&a, W / 2, H / 2);
    assert!(
        center[0] > center[2] && center[0] > 40,
        "expected the red cube at center (unlit): {center:?}"
    );
}

/// Primitive-geometry golden (R-P1): one of each of the five built-in kinds
/// (Cube, Sphere, Plane, Cylinder, Cone) in a row on the ground grid, each a
/// distinct colour. Proves every kind renders as its real shape through the whole
/// mesh path. Structural gate: swapping all kinds to Cube changes the frame (so
/// the per-kind geometry genuinely varies) and the row is lit. Determinism gate
/// via `check_golden`; strict pixel diff opt-in.
#[test]
fn golden_primitives() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    let kinds = [
        PrimMesh::Cube,
        PrimMesh::Sphere,
        PrimMesh::Plane,
        PrimMesh::Cylinder,
        PrimMesh::Cone,
    ];
    let colors = [
        [0.85, 0.25, 0.25],
        [0.25, 0.75, 0.35],
        [0.30, 0.45, 0.95],
        [0.85, 0.75, 0.30],
        [0.75, 0.35, 0.85],
    ];
    for (i, (&kind, c)) in kinds.iter().zip(colors).enumerate() {
        scene.instances.push(MeshInstance {
            translation: DVec3::new(-4.0 + i as f64 * 2.0, 0.5, 0.0),
            rotation: Quat::from_rotation_y(0.3),
            scale: Vec3::ONE,
            color: [c[0], c[1], c[2], 1.0],
            metallic: 0.0,
            roughness: 0.5,
            emissive: [0.0; 3],
            id: i as u32 + 1,
            mesh: kind,
            blend: 0,
            cutoff: 0.5,
        });
    }
    scene.mark_dirty();

    let img = check_golden(&gpu, "primitives", &scene, &overlook_view());

    // Swapping every kind to Cube must change the image — proof the per-kind
    // geometry (not just the cube) actually reaches the rasterizer.
    let mut cubes = scene.clone();
    for inst in &mut cubes.instances {
        inst.mesh = PrimMesh::Cube;
    }
    cubes.mark_dirty();
    let cube_img = render(&gpu, &cubes, &overlook_view());
    let (mean, _max) = image_diff(&img, &cube_img, W, H);
    assert!(
        mean > 0.002,
        "primitive kinds should differ from an all-cube row (mean {mean})"
    );

    let lit = img.chunks(4).any(|p| p[0] > 60 || p[1] > 60 || p[2] > 60);
    assert!(lit, "expected a lit primitive pixel");
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
        glam::Quat::IDENTITY,
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
            mesh: PrimMesh::Cube,
            blend: 0,
            cutoff: 0.5,
        });
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.4, 0.8, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [0.3, 0.5, 1.0],
        intensity: 30.0,
        direction: Vec3::ZERO,
        position: DVec3::new(0.0, 2.5, 2.0),
        range: 12.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "pbr_materials", &scene, &overlook_view());
    // The scene is lit: some pixel is clearly brighter than the dark backdrop.
    let lit = img.chunks(4).any(|p| p[0] > 90 || p[1] > 90 || p[2] > 90);
    assert!(lit, "expected a lit PBR pixel");
}

/// Translucency golden (R-P5): opaque cubes behind, TWO overlapping tinted
/// translucent panes (`blend == 2`, 50% alpha) in front — proving alpha blending +
/// the deterministic back-to-front sort — plus one **masked** cube (`blend == 1`)
/// whose uniform alpha is below its cutoff, so the alpha-test discards it entirely
/// (a "cutout" hole; per-fragment texture opacity is deferred). Determinism gate
/// via `check_golden`; strict pixel diff opt-in, blessed on a GPU host. Structural
/// gates prove that (a) the panes genuinely blend (vs the same scene made opaque)
/// and (b) the masked instance is genuinely alpha-tested away (vs made opaque).
#[test]
fn golden_translucency() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    // A directional key so the panes + cubes are lit.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.4, 0.8, 0.4).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    // Two opaque cubes at the back (−z).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(-1.4, 0.5, -0.6),
        Quat::from_rotation_y(0.3),
        Vec3::ONE,
        [0.85, 0.22, 0.22, 1.0],
        1,
    ));
    scene.instances.push(MeshInstance::lit(
        DVec3::new(1.4, 0.5, -0.6),
        Quat::from_rotation_y(0.3),
        Vec3::ONE,
        [0.22, 0.72, 0.32, 1.0],
        2,
    ));
    // A masked cube up front whose uniform alpha (0.3) is below its cutoff (0.5)
    // → the mesh fs discards every fragment (the visible cutout: when drawn opaque
    // it would occlude the panes + cubes behind it; masked, it vanishes).
    scene.instances.push(MeshInstance {
        translation: DVec3::new(0.0, 0.9, 3.2),
        rotation: Quat::from_rotation_y(0.3),
        scale: Vec3::splat(1.3),
        color: [0.90, 0.85, 0.20, 0.30],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 3,
        mesh: PrimMesh::Cube,
        blend: 1,
        cutoff: 0.5,
    });
    // Two overlapping translucent panes (thin cubes) in front (+z, toward the
    // camera), tinted blue then orange at 50% alpha. The farther one draws first.
    scene.instances.push(MeshInstance {
        translation: DVec3::new(-0.4, 0.9, 1.2),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(2.4, 2.4, 0.06),
        color: [0.25, 0.45, 1.0, 0.5],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 4,
        mesh: PrimMesh::Cube,
        blend: 2,
        cutoff: 0.5,
    });
    scene.instances.push(MeshInstance {
        translation: DVec3::new(0.5, 0.7, 2.1),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(2.4, 2.4, 0.06),
        color: [1.0, 0.5, 0.15, 0.5],
        metallic: 0.0,
        roughness: 0.5,
        emissive: [0.0; 3],
        id: 5,
        mesh: PrimMesh::Cube,
        blend: 2,
        cutoff: 0.5,
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "translucency", &scene, &overlook_view());

    // The scene renders lit + blended content.
    let lit = img.chunks(4).any(|p| p[0] > 60 || p[1] > 60 || p[2] > 60);
    assert!(lit, "expected a lit/blended pixel");

    // (a) The translucent panes genuinely blend: making them opaque changes the
    // frame (an opaque pane would fully hide what's behind it).
    let mut solid_panes = scene.clone();
    for inst in &mut solid_panes.instances {
        if inst.blend == 2 {
            inst.blend = 0;
            inst.color[3] = 1.0;
        }
    }
    solid_panes.mark_dirty();
    let solid_panes_img = render(&gpu, &solid_panes, &overlook_view());
    let (mean, _max) = image_diff(&img, &solid_panes_img, W, H);
    assert!(
        mean > 0.002,
        "translucent blending should differ from opaque panes (mean {mean})"
    );

    // (b) The masked instance is genuinely alpha-tested away: making it opaque
    // (fully drawn) changes the frame.
    let mut solid_mask = scene.clone();
    for inst in &mut solid_mask.instances {
        if inst.blend == 1 {
            inst.blend = 0;
            inst.color[3] = 1.0;
        }
    }
    solid_mask.mark_dirty();
    let solid_mask_img = render(&gpu, &solid_mask, &overlook_view());
    let (m2, _max) = image_diff(&img, &solid_mask_img, W, H);
    assert!(
        m2 > 0.001,
        "masked discard should differ from an opaque instance (mean {m2})"
    );
}

/// Spot-light golden (R-P3): a ground plane + a few cubes lit by a single spot
/// aimed obliquely, so the cone's lit ellipse and its soft outer-cone falloff are
/// both on screen. No directional light — the spot shapes the frame. Exercises
/// the shaders' `w == 2` branch (cone `smoothstep` × windowed inverse-square)
/// through the mesh path headlessly (determinism gate via `check_golden`; strict
/// pixel diff opt-in, blessed on a GPU host). Structural gate: the spot frame
/// differs from the same scene lit by a plain point light — proving the cone mask
/// actually clips the illumination (a point light would light the plane broadly).
#[test]
fn golden_spot_lights() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };
    // A large ground plane to catch the cone.
    scene.instances.push(MeshInstance {
        translation: DVec3::new(0.0, 0.0, 0.0),
        rotation: Quat::IDENTITY,
        scale: Vec3::new(20.0, 1.0, 20.0),
        color: [0.60, 0.60, 0.62, 1.0],
        metallic: 0.0,
        roughness: 0.7,
        emissive: [0.0; 3],
        id: 1,
        mesh: PrimMesh::Plane,
        blend: 0,
        cutoff: 0.5,
    });
    // A few cubes standing in and around the beam.
    for (i, (x, z)) in [(1.5, 1.5), (3.0, 0.5), (-1.0, 2.5)]
        .into_iter()
        .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.5, z),
            Quat::from_rotation_y(0.3),
            Vec3::ONE,
            [0.80, 0.75, 0.70, 1.0],
            i as u32 + 2,
        ));
    }
    // One spot high above, aimed obliquely toward (2, 0, 2). `direction` is the
    // toward-the-light vector; the emission axis is its negation.
    let emit = Vec3::new(2.0, -5.0, 2.0).normalize();
    scene.lights.push(RenderLight {
        kind: LightKind::Spot,
        color: [1.0, 0.95, 0.8],
        intensity: 60.0,
        direction: -emit,
        position: DVec3::new(0.0, 5.0, 0.0),
        range: 20.0,
        inner_cos: 15f32.to_radians().cos(),
        outer_cos: 25f32.to_radians().cos(),
        cast_shadows: false,
    });
    scene.mark_dirty();

    let img = check_golden(&gpu, "spot_lights", &scene, &overlook_view());

    // The cone lights a bright patch.
    let bright = img.chunks(4).any(|p| p[0] > 90 || p[1] > 90 || p[2] > 90);
    assert!(bright, "expected a lit spot pixel");

    // Swapping the spot for a plain point light (same position/intensity) lights
    // the ground far more broadly — the frames must differ, proving the cone mask.
    let mut point = scene.clone();
    point.lights[0].kind = LightKind::Point;
    point.mark_dirty();
    let point_img = render(&gpu, &point, &overlook_view());
    let (mean, _max) = image_diff(&img, &point_img, W, H);
    assert!(
        mean > 0.002,
        "spot cone should differ from a point light (mean {mean})"
    );
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
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(ox, 0.0, oz),
                heights,
                weights: Vec::new(),
                height_bounds: (lo, hi),
                version: 1,
            });
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
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
        terrains: vec![terrain],
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
    let mut lods: Vec<u32> = patches.iter().map(|p| p.ring).collect();
    lods.sort_unstable();
    lods.dedup();
    assert!(
        lods.len() >= 2,
        "expected ≥2 LOD rings, got LODs {lods:?} from {} patches",
        patches.len()
    );

    let scene = RenderScene {
        grid_enabled: true,
        terrains: vec![terrain],
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
                key: TerrainTileKey::lod0((tx, tz)),
                origin: DVec3::new(ox, 0.0, oz),
                heights,
                weights,
                height_bounds: (lo, hi),
                version: 1,
            });
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
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
        terrains: vec![terrain],
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

/// A synthetic **streamed** terrain (P16.3b1): three asset LOD levels over one
/// global height function, handed to the renderer as a quadtree cut instead of a
/// fully-resident heightfield.
///
/// * level 0 — only the 2 × 2 block at the origin is resident (a deliberately
///   *partial* level-0 residency, the streaming shape);
/// * level 1 — the 2 × 2 block covering 4 × that area (the mid ring);
/// * level 2 — the 2 × 2 block covering 16 × it, **minus `(1,1)`** so one far
///   quadrant is genuinely uncovered (the renderer must render the hole, not
///   invent coverage).
///
/// Coarse pages sample the same global function at `2^lod ·` the spacing, so —
/// exactly like the real `inf_terrain::pyramid` decimation — every coarse sample
/// *is* one of the fine samples, and the shared edges agree bit-for-bit.
fn streamed_terrain(res: u32, mps: f64) -> RenderTerrain {
    let f = |x: f64, z: f64| 5.0 * (x * 0.04).sin() * (z * 0.04).cos() + 4.0;
    let span0 = (res as f64 - 1.0) * mps;
    let page = |lod: u32, coord: (i32, i32), version: u64| {
        let step = mps * (1u64 << lod) as f64;
        let span = span0 * (1u64 << lod) as f64;
        let (ox, oz) = (coord.0 as f64 * span, coord.1 as f64 * span);
        let mut heights = vec![0f32; (res * res) as usize];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for j in 0..res {
            for i in 0..res {
                let h = f(ox + i as f64 * step, oz + j as f64 * step) as f32;
                heights[(j * res + i) as usize] = h;
                lo = lo.min(h);
                hi = hi.max(h);
            }
        }
        RenderTerrainTile {
            key: TerrainTileKey::new(lod, coord),
            origin: DVec3::new(ox, 0.0, oz),
            heights,
            weights: Vec::new(),
            height_bounds: (lo, hi),
            version: 1 + version,
        }
    };
    let block = [(0, 0), (0, 1), (1, 0), (1, 1)];
    // Key-ascending (level 0, then level 1, then level 2) — the projection order.
    let mut tiles = Vec::new();
    for (lod, coords) in [
        (0u32, &block[..]),
        (1, &block[..]),
        (2, &block[..3]), // (1,1) deliberately absent
    ] {
        for &c in coords {
            tiles.push(page(lod, c, tiles.len() as u64));
        }
    }
    RenderTerrain {
        id: 0,
        tile_resolution: res,
        meters_per_sample: mps,
        tiles,
        layers: default_layers(),
        macro_variation: 0.15,
    }
}

/// Streamed-terrain headless gate (P16.3b1). A partially-resident level 0 with
/// coarse pyramid pages covering the outer rings must render deterministically —
/// **including across frames that share one renderer**, where the second frame's
/// per-tile version gate finds every stamp unchanged and uploads nothing. A
/// regression that dropped or half-refreshed a cached page would show up here as
/// a differing frame.
///
/// Deliberately **not** a committed golden PNG: the pixels exercise the same
/// shading path the three terrain goldens already pin, while everything this
/// batch adds (which page sources which patch) is asserted structurally, which is
/// adapter-robust — the harness's stated bar for what CI can actually check.
#[test]
fn streamed_terrain_renders_partial_residency() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 33;
    let terrain = streamed_terrain(res, 1.0);
    // Overlook from the near corner down the +X/+Z diagonal, so the resident
    // level-0 block is close and the coarse pages recede into the outer rings.
    let view = look_view(
        DVec3::new(-20.0, 60.0, -20.0),
        DVec3::new(120.0, 0.0, 120.0),
    );

    // ── structural: source selection over the residency set ──────────────────
    let patches = assemble_patches(&terrain, &view, &view.origin);
    assert_eq!(
        patches,
        assemble_patches(&terrain, &view, &view.origin),
        "assembly must be a pure function of (residency set, view)"
    );
    let drawn: Vec<TerrainTileKey> = patches.iter().map(|p| p.key).collect();
    // Fine wins: every resident level-0 page draws …
    for c in [(0, 0), (0, 1), (1, 0), (1, 1)] {
        assert!(
            drawn.contains(&TerrainTileKey::lod0(c)),
            "level-0 page {c:?} must draw (fine wins)"
        );
    }
    // … and the coarse pages whose whole footprint they cover stand down.
    assert!(
        !drawn.contains(&TerrainTileKey::new(1, (0, 0))),
        "the fully-subdivided level-1 page must not double-draw over level 0"
    );
    assert!(
        !drawn.contains(&TerrainTileKey::new(2, (0, 0))),
        "the fully-subdivided level-2 page must not double-draw"
    );
    // Coarse pages serve the outer coverage, at ≥2 distinct asset levels.
    let mut levels: Vec<u32> = drawn.iter().map(|k| k.lod).collect();
    levels.sort_unstable();
    levels.dedup();
    assert!(
        levels.len() >= 2 && levels.contains(&0),
        "expected fine + coarse sources, got asset levels {levels:?}"
    );
    // The absent page is simply not drawn — a hole, faithfully rendered.
    assert!(!drawn.contains(&TerrainTileKey::new(2, (1, 1))));
    // Nothing is drawn twice, and a coarse patch keeps the full-density grid its
    // level already decimated for (ring − lod).
    let mut unique = drawn.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), drawn.len(), "a page was assembled twice");
    for p in &patches {
        assert_eq!(p.mesh_lod, inf_render::patch_mesh_lod(p.ring, p.key.lod));
        assert_eq!(terrain.tiles[p.tile].key, p.key);
    }

    // ── determinism: two fresh renderers, then two frames on a warm cache ─────
    let scene = RenderScene {
        grid_enabled: true,
        terrains: vec![terrain],
        ..Default::default()
    };
    let cold_a = render(&gpu, &scene, &view);
    let cold_b = render(&gpu, &scene, &view);
    assert_eq!(
        cold_a, cold_b,
        "streamed terrain must render deterministically"
    );

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let warm_a = target.read_rgba(&gpu).expect("readback");
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let warm_b = target.read_rgba(&gpu).expect("readback");
    assert_eq!(
        warm_a, warm_b,
        "a second frame over an unchanged residency set uploads nothing and must \
         be byte-identical"
    );
    assert_eq!(
        cold_a, warm_a,
        "a warm tile cache must render the cold frame"
    );

    let lit = warm_a
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 150);
    assert!(lit, "expected a lit terrain pixel");
}

/// Shift every tile of `t` by `offset` and stamp it with `id` — a second terrain
/// placed elsewhere in the world while keeping the *same* tile coordinates, which
/// is the collision case the P16.6 cache key exists for.
fn placed_terrain(mut t: RenderTerrain, id: u64, offset: DVec3) -> RenderTerrain {
    t.id = id;
    for tile in &mut t.tiles {
        tile.origin += offset;
    }
    t
}

/// **P16.6 multi-terrain headless gate.** Two independent terrains — same tile
/// coordinates, different world anchors, different splat layers — render in one
/// frame, deterministically, and both are actually drawn.
///
/// Deliberately **not** a committed golden PNG, following the streamed-terrain
/// precedent above: the shading path is already pinned by the three terrain
/// goldens, and everything this batch adds (per-terrain cache slots, per-terrain
/// material uniforms, one instance buffer across both patch lists) is asserted
/// structurally, which is adapter-robust — the harness's stated bar for what CI
/// can actually check. The single-terrain goldens are what pin byte-identity.
#[test]
fn two_terrains_render_independently_in_one_frame() {
    let Some(gpu) = gpu_or_skip() else { return };
    let res = 17;
    let a = placed_terrain(hill_terrain(res, 1.0, 2, 2), 1, DVec3::ZERO);
    // B sits 40 m down +X, so both are in frame at once, and it paints from a
    // different layer set so a shared material uniform would be visible.
    let mut b = placed_terrain(hill_terrain(res, 1.0, 2, 2), 2, DVec3::new(40.0, 6.0, 0.0));
    b.layers[0].albedo = [0.75, 0.18, 0.12, 1.0];
    b.macro_variation = 0.0;

    // The fixture only bites while the two terrains share tile coordinates.
    let keys_a: Vec<_> = a.tiles.iter().map(|t| t.key).collect();
    let keys_b: Vec<_> = b.tiles.iter().map(|t| t.key).collect();
    assert_eq!(keys_a, keys_b, "the two terrains must share tile keys");

    let view = look_view(DVec3::new(16.0, 34.0, -26.0), DVec3::new(36.0, 4.0, 16.0));

    // Both terrains assemble patches under this view (each against its OWN grid).
    let pa = assemble_patches(&a, &view, &view.origin);
    let pb = assemble_patches(&b, &view, &view.origin);
    assert!(!pa.is_empty() && !pb.is_empty(), "both must be visible");

    let one = RenderScene {
        grid_enabled: true,
        terrains: vec![a.clone()],
        ..Default::default()
    };
    let both = RenderScene {
        grid_enabled: true,
        terrains: vec![a, b],
        ..Default::default()
    };

    // Determinism: two fresh renderers over the two-terrain scene agree…
    let cold_a = render(&gpu, &both, &view);
    let cold_b = render(&gpu, &both, &view);
    assert_eq!(cold_a, cold_b, "two terrains must render deterministically");

    // …and a warm cache (second frame, every stamp unchanged, nothing uploaded)
    // reproduces the cold frame — the per-terrain cache slots stay in step.
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &both, &view, &target.view, (W, H));
    let warm_a = target.read_rgba(&gpu).expect("readback");
    renderer.render(&gpu, &both, &view, &target.view, (W, H));
    let warm_b = target.read_rgba(&gpu).expect("readback");
    assert_eq!(warm_a, warm_b, "a warm two-terrain frame must be stable");
    assert_eq!(cold_a, warm_a, "warm != cold for two terrains");

    // The second terrain really contributed pixels.
    let solo = render(&gpu, &one, &view);
    assert_ne!(
        solo, cold_a,
        "adding the second terrain changed nothing — it never drew"
    );

    // Dropping a terrain from the scene mid-session frees its cache and still
    // renders the survivor exactly as a fresh renderer would.
    renderer.render(&gpu, &one, &view, &target.view, (W, H));
    let after_drop = target.read_rgba(&gpu).expect("readback");
    assert_eq!(
        after_drop, solo,
        "evicting terrain B's pages perturbed terrain A"
    );
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
            mesh: PrimMesh::Cube,
            blend: 0,
            cutoff: 0.5,
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
        ..RenderLight::default()
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
        ..RenderLight::default()
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
        two_pass: false,
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

// ── P13.3b: cascaded shadow maps + dynamic GI ────────────────────────────────

/// CSM golden (P13.3b): a caster/receiver scene — three boxes standing on a large
/// white floor slab, lit by a single **low** directional sun so they cast long
/// shadows across the floor. Shadows ON. Structural gate: the cascaded shadows
/// **darken** the frame overall (direct light removed in the occluded floor
/// regions) while the scene stays lit — proving the cascade render → PCF sample
/// path ran. Determinism via `check_golden_with`; strict pixel diff opt-in. With
/// shadows off every other golden stays byte-stable (verified).
#[test]
fn golden_csm() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // A large receiver floor slab (top surface at y = 0).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.0),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 12.0),
        [0.78, 0.78, 0.80, 1.0],
        1,
    ));
    // Three caster boxes.
    for (i, (x, z)) in [(-2.2, 0.5), (1.0, -1.5), (2.6, 1.2)]
        .into_iter()
        .enumerate()
    {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(x, 0.9, z),
            Quat::from_rotation_y(0.3),
            Vec3::new(0.9, 1.8, 0.9),
            [0.80, 0.42, 0.32, 1.0],
            i as u32 + 2,
        ));
    }
    // A low directional sun (grazing → long shadows).
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.97, 0.9],
        intensity: 3.0,
        direction: Vec3::new(0.55, 0.32, 0.45).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(5.0, 5.5, 8.5), DVec3::new(0.0, 0.5, 0.0));

    let shadows_on = RenderSettings {
        shadows: ShadowSettings {
            enabled: true,
            ..ShadowSettings::default()
        },
        ..RenderSettings::default()
    };

    let img = check_golden_with(&gpu, "csm", &scene, &view, shadows_on);
    let img_off = render_with(&gpu, &scene, &view, RenderSettings::default());

    let sum = |img: &[u8]| -> u64 {
        img.chunks(4)
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum()
    };
    let (sum_on, sum_off) = (sum(&img), sum(&img_off));
    assert!(
        sum_on < sum_off,
        "CSM should darken shadowed regions (on {sum_on} vs off {sum_off})"
    );
    let lit = img
        .chunks(4)
        .any(|p| p[0] as u16 + p[1] as u16 + p[2] as u16 > 200);
    assert!(lit, "expected the CSM scene to stay lit");
}

/// Mean red/green ratio of the floor pixels in a screen band (rows `y0..y1`,
/// central columns), skipping near-black (unlit / off-floor) pixels. The proof
/// metric for `golden_gi_bleed`.
fn band_red_ratio(img: &[u8], y0: u32, y1: u32) -> f32 {
    let (mut r, mut g, mut n) = (0.0f64, 0.0f64, 0u32);
    for y in y0..y1 {
        for x in (W * 30 / 100)..(W * 70 / 100) {
            let p = px(img, x, y);
            // Skip near-black pixels (not lit floor).
            if (p[0] as u16 + p[1] as u16 + p[2] as u16) < 30 {
                continue;
            }
            r += p[0] as f64;
            g += p[1] as f64;
            n += 1;
        }
    }
    if n == 0 || g == 0.0 {
        return 0.0;
    }
    (r / g.max(1.0)) as f32
}

/// The GI proof golden (P13.3b) — **`golden_gi_bleed`**: a white floor and a tall
/// **red** wall, with the sun angled so the wall's front face is lit and the floor
/// receives grazing light. With dynamic GI ON, the floor **near the wall** picks up
/// a red single-bounce, so its mean red/green ratio exceeds the far floor's by a
/// clear margin — asserted structurally over two screen bands (the region assert,
/// not a pixel compare). Also asserts determinism (two renders byte-identical).
/// GI off keeps the hemispheric ambient path byte-stable (verified).
#[test]
fn golden_gi_bleed() {
    let Some(gpu) = gpu_or_skip() else { return };
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    // White floor slab (top surface at y = 0), extending toward the wall (−Z).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.25, 0.5),
        Quat::IDENTITY,
        Vec3::new(12.0, 0.5, 11.0),
        [0.90, 0.90, 0.90, 1.0],
        1,
    ));
    // A tall RED wall along the far (−Z) edge, front face toward +Z (the floor).
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, 1.5, -4.0),
        Quat::IDENTITY,
        Vec3::new(11.0, 3.0, 0.5),
        [0.90, 0.05, 0.05, 1.0],
        2,
    ));
    // Sun from +Z and above: lights the wall's +Z face and grazes the floor. Kept
    // moderate so the single-bounce GI (not the direct white light) shapes the
    // floor's near-wall colour.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 2.0,
        direction: Vec3::new(0.0, 0.5, 1.0).normalize(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    // Look toward the wall base: the wall sits high in the frame, the floor fills
    // the lower two thirds (near-wall floor above, far floor below).
    let view = look_view(DVec3::new(0.0, 4.5, 7.0), DVec3::new(0.0, 0.0, -1.5));

    let gi_on = RenderSettings {
        gi: GiSettings {
            enabled: true,
            extent: 40.0,
            rays: 48,
            intensity: 2.5,
        },
        ..RenderSettings::default()
    };

    let img = check_golden_with(&gpu, "gi_bleed", &scene, &view, gi_on);

    // Determinism: a second render is byte-identical to the golden render.
    let img2 = render_with(&gpu, &scene, &view, gi_on);
    let (mean, max) = image_diff(&img, &img2, W, H);
    assert!(
        mean == 0.0 && max == 0.0,
        "GI must be deterministic (mean {mean}, max {max})"
    );

    // Region assert: the near-wall floor band is redder than the far floor band.
    let near = band_red_ratio(&img, H * 40 / 100, H * 52 / 100);
    let far = band_red_ratio(&img, H * 74 / 100, H * 90 / 100);
    eprintln!("gi_bleed red/green ratio: near-wall {near:.3}, far {far:.3}");
    assert!(
        near > far + 0.05,
        "expected red colour bleed near the wall (near {near:.3} vs far {far:.3})"
    );

    // Sanity: the near-wall floor actually picks up red (ratio clearly > 1).
    assert!(
        near > 1.03,
        "near-wall floor not reddened (ratio {near:.3})"
    );
}

// ── P17.2 physical atmosphere ────────────────────────────────────────────────
//
// The time-of-day sweep. These are the FIRST goldens in the suite that carry an
// atmosphere at all — every scene above renders with `AtmosphereParams::default()`
// (disabled), which is what keeps all 23 pre-P17.2 goldens byte-identical.
//
// The scenes are built from `inf_math::solar` exactly as both scene projectors
// build them, at `SkyAtmosphere`'s defaults, so what these images show is what a
// new level actually looks like — not a hand-tuned demo of the shader.

/// The default sky authority a new level gets: day 172 (June solstice), 48.9° N,
/// prime meridian — `TimeOfDay::default()`'s place, at `seconds` UTC.
fn tod_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    let bodies = inf_math::solar::bodies(&inf_math::solar::SolarInput {
        seconds,
        day_of_year: 172,
        latitude_deg: 48.9,
        longitude_deg: 0.0,
    });
    // The `SkyAtmosphere::default()` values, mapped the way `project_sky` maps
    // them in both hosts.
    let scene = RenderScene {
        sun: SunParams {
            direction: bodies.sun.as_vec3(),
            color: [1.0, 0.98, 0.95],
            intensity: 3.0,
            moon_direction: bodies.moon.as_vec3(),
            moon_color: [0.62, 0.72, 1.0],
            moon_intensity: 0.15,
            moon_phase: bodies.moon_phase as f32,
        },
        atmosphere: AtmosphereParams {
            enabled: true,
            moon_phase: bodies.moon_phase as f32,
            ..AtmosphereParams::default()
        },
        ..Default::default()
    };
    (scene, bodies)
}

/// A ground-level camera looking along the horizontal azimuth of `toward`,
/// pitched up by `pitch_deg` so the horizon sits low in frame. Aiming at a body's
/// azimuth (rather than at a fixed compass point) keeps the disc in shot whatever
/// the date and latitude, so these goldens do not silently become pictures of
/// empty sky if the solar model is ever refined.
fn horizon_view(toward: DVec3, pitch_deg: f64) -> RenderView {
    let flat = DVec3::new(toward.x, 0.0, toward.z);
    let flat = if flat.length_squared() > 1e-9 {
        flat.normalize()
    } else {
        DVec3::X
    };
    let p = pitch_deg.to_radians();
    let forward = (flat * p.cos() + DVec3::Y * p.sin()).normalize();
    RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 2.0, 0.0),
        forward: forward.as_vec3(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// Mean sRGB-encoded RGB of a screen rectangle (0..1 per channel). Ratios between
/// two such means are what the structural assertions below compare, which is
/// adapter-robust in a way absolute pixel values are not.
fn mean_rgb(img: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> [f32; 3] {
    let mut acc = [0.0f32; 3];
    let mut n = 0.0;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = px(img, x, y);
            for c in 0..3 {
                acc[c] += p[c] as f32 / 255.0;
            }
            n += 1.0;
        }
    }
    [acc[0] / n, acc[1] / n, acc[2] / n]
}

fn luma(c: [f32; 3]) -> f32 {
    c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722
}

/// The brightest single pixel in a screen rectangle, as a 0..1 mean of channels.
fn brightest(img: &[u8], x0: u32, y0: u32, x1: u32, y1: u32) -> f32 {
    let mut best = 0.0f32;
    for y in y0..y1 {
        for x in x0..x1 {
            let p = px(img, x, y);
            let v = (p[0] as f32 + p[1] as f32 + p[2] as f32) / (3.0 * 255.0);
            best = best.max(v);
        }
    }
    best
}

/// Sky brightness gradient + colour at high noon: deep blue overhead, brighter
/// and less saturated toward the horizon (Rayleigh optical depth grows with the
/// path length). This is the shape a three-colour gradient cannot fake — it falls
/// out of the LUT parameterization, and is wrong the moment that is.
#[test]
fn golden_sky_noon() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(43_200.0); // 12:00 UTC
    assert!(bodies.sun.y > 0.85, "12:00 at the solstice should be high");
    let view = horizon_view(bodies.sun, 25.0);
    let img = check_golden(&gpu, "sky_noon", &scene, &view);

    // The camera is pitched +25° with a 60° vertical FOV, so the horizon LINE
    // sits ~75 px below centre (y ≈ 165) and everything below it is the sky
    // pass's ground. Both bands are sampled in sky, above that line.
    let top = mean_rgb(&img, 0, 0, W, H / 8);
    let horizon = mean_rgb(&img, 0, H * 80 / 100, W, H * 90 / 100);
    eprintln!("sky_noon top {top:?} horizon {horizon:?}");
    assert!(top[2] > top[0] + 0.08, "zenith not blue: {top:?}");
    // A real daytime sky, not a dim one.
    assert!(top[2] > 0.35, "zenith too dark for noon: {top:?}");
    assert!(
        luma(horizon) > luma(top),
        "horizon should out-brighten the zenith: {horizon:?} vs {top:?}"
    );
    assert!(
        horizon[0] / horizon[2] > top[0] / top[2] + 0.05,
        "horizon should be less blue than the zenith: {horizon:?} vs {top:?}"
    );
}

/// Dawn: the sun is a few degrees up, so its light has crossed a long slab of
/// air. The band around it must be markedly redder than the zenith — the single
/// assertion that catches a swapped Rayleigh triple, and the GPU sibling of the
/// CPU `sunset_is_redder_than_noon` unit test.
#[test]
fn golden_sky_dawn() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(16_200.0); // 04:30 UTC
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "04:30 should be just after sunrise, got y {}",
        bodies.sun.y
    );
    let view = horizon_view(bodies.sun, 6.0);
    let img = check_golden(&gpu, "sky_dawn", &scene, &view);

    // The band just above the horizon line (the camera is pitched +6°, so the
    // horizon sits ~18 px below centre) — not the bottom of the frame, which is
    // the sky pass's ground.
    let top = mean_rgb(&img, 0, 0, W, H / 6);
    let low = mean_rgb(&img, 0, H * 50 / 100, H, H * 58 / 100);
    eprintln!("sky_dawn top {top:?} low {low:?}");
    assert!(
        low[0] / low[2].max(1e-4) > top[0] / top[2].max(1e-4) + 0.15,
        "the horizon band should be redder than the zenith: {low:?} vs {top:?}"
    );
    // The sun disc is in frame and clips to (near) white.
    let peak = brightest(&img, 0, 0, W, H);
    assert!(peak > 0.94, "no sun disc in frame (brightest {peak:.3})");
}

/// Dusk, on the other side of the sky. A separate golden from dawn because the
/// sun's azimuth differs by ~100° and so does the ozone-shaped blue of the
/// opposite horizon — a sweep with only one twilight would not notice a model
/// that made both ends identical.
#[test]
fn golden_sky_dusk() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(71_100.0); // 19:45 UTC
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "19:45 should be just before sunset, got y {}",
        bodies.sun.y
    );
    // Dawn and dusk must not be the same picture.
    let (_, dawn) = tod_scene(16_200.0);
    assert!(
        bodies.sun.x * dawn.sun.x + bodies.sun.z * dawn.sun.z < 0.5,
        "dawn and dusk azimuths are too close to be distinct goldens"
    );
    let view = horizon_view(bodies.sun, 6.0);
    let img = check_golden(&gpu, "sky_dusk", &scene, &view);

    let top = mean_rgb(&img, 0, 0, W, H / 6);
    let low = mean_rgb(&img, 0, H * 50 / 100, H, H * 58 / 100);
    eprintln!("sky_dusk top {top:?} low {low:?}");
    assert!(
        low[0] / low[2].max(1e-4) > top[0] / top[2].max(1e-4) + 0.15,
        "the horizon band should be redder than the zenith: {low:?} vs {top:?}"
    );
}

/// Night: the sky collapses to the multiple-scattering floor and the procedural
/// starfield appears. The star assertion is a *local-contrast* one (a bright
/// isolated texel against a dark field) rather than a mean, because a mean would
/// also pass for a uniformly-raised black level.
#[test]
fn golden_sky_night() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(84_600.0); // 23:30 UTC
    assert!(bodies.sun.y < -0.2, "23:30 should be deep night");
    // Look away from the sun and well up, where the stars are.
    let view = horizon_view(-bodies.sun, 35.0);
    let img = check_golden(&gpu, "sky_night", &scene, &view);

    let sky = mean_rgb(&img, 0, 0, W, H / 2);
    eprintln!("sky_night mean {sky:?}");
    assert!(sky[2] < 0.30, "night sky is not dark: {sky:?}");
    let field = (sky[0] + sky[1] + sky[2]) / 3.0;
    let peak = brightest(&img, 0, 0, W, H / 2);
    assert!(
        peak > field + 0.12,
        "no starfield contrast (brightest {peak:.3} vs field {field:.3})"
    );
}

/// The starfield is a pure function of the view direction: two renders of the
/// same night sky must be byte-identical (the hash is integer-only, per the
/// psin/pcos law's spirit — no trig anywhere in it), and a *rotated* camera must
/// see a different patch of sky rather than a field pinned to the screen.
#[test]
fn stars_are_deterministic_and_world_locked() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(84_600.0);
    let view = horizon_view(-bodies.sun, 35.0);
    let a = render(&gpu, &scene, &view);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the starfield is not deterministic");

    // Yaw the camera 40°: the same screen pixels must now show different sky.
    let f = view.forward;
    let (s, c) = 40f32.to_radians().sin_cos();
    let rotated = RenderView {
        forward: Vec3::new(f.x * c + f.z * s, f.y, -f.x * s + f.z * c).normalize(),
        ..view
    };
    let r = render(&gpu, &scene, &rotated);
    assert_ne!(a, r, "the starfield followed the camera instead of the sky");
}

/// Aerial perspective + height fog on lit geometry, as a **controlled
/// experiment** rather than a pretty picture: two walls with identical albedo,
/// identical orientation and identical screen size, one at 50 m and one at
/// 1500 m — the far one is the near one scaled 30× about the eye, so it projects
/// to the same rectangle mirrored across the frame. Every pixel-level difference
/// between them is therefore the atmosphere and nothing else.
///
/// Also carries the off-path proof: the same scene with the atmosphere disabled
/// is the pre-P17.2 render.
#[test]
fn golden_aerial_fog() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = tod_scene(43_200.0);
    scene.atmosphere.fog = HeightFog {
        density: 1.5e-3, // ≈ 2 km visibility — a properly foggy morning
        falloff: 0.002,  // 500 m e-folding height
        height: 0.0,
        color: [1.0, 1.0, 1.0],
    };
    // Ground, deliberately DARK: a bright albedo is already near white before any
    // scattering touches it, so the wash toward the sky would be invisible.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, -6000.0),
        Quat::IDENTITY,
        Vec3::new(8000.0, 1.0, 16000.0),
        [0.10, 0.11, 0.12, 1.0],
        1,
    ));
    // The matched pair. `NEAR` is at −14 m across at 50 m out; `FAR` is that
    // exact vector × 30, mirrored in x, so both subtend the same angle.
    const WALL: [f32; 4] = [0.16, 0.17, 0.18, 1.0];
    scene.instances.push(MeshInstance::lit(
        DVec3::new(-14.0, 8.0, -50.0),
        Quat::IDENTITY,
        Vec3::new(16.0, 16.0, 1.0),
        WALL,
        2,
    ));
    scene.instances.push(MeshInstance::lit(
        DVec3::new(420.0, 8.0, -1500.0),
        Quat::IDENTITY,
        Vec3::new(480.0, 480.0, 30.0),
        WALL,
        3,
    ));
    // A few pillars down the centre — not measured, but they are what makes the
    // golden readable as a picture of depth rather than two grey squares.
    for (i, d) in [15.0f64, 45.0, 140.0, 420.0].into_iter().enumerate() {
        scene.instances.push(MeshInstance::lit(
            DVec3::new(0.0, 6.0, -d),
            Quat::IDENTITY,
            Vec3::new(2.0, 12.0, 2.0),
            [0.13, 0.14, 0.15, 1.0],
            i as u32 + 4,
        ));
    }
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 3.0,
        direction: bodies.sun.as_vec3(),
        position: DVec3::ZERO,
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    // Eye at the walls' centre height looking dead ahead, so the two rectangles
    // land symmetrically about the frame centre.
    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 8.0, 0.0),
        forward: Vec3::NEG_Z,
        up: Vec3::Y,
        fov_y: 45f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    };
    let img = check_golden(&gpu, "aerial_fog", &scene, &view);

    // Both walls are sampled 20 px above centre — inside each rectangle, and
    // above the horizon line so no ground creeps into either box.
    let near = mean_rgb(&img, 91, 62, 107, 78);
    let far = mean_rgb(&img, 213, 62, 229, 78);
    // The sky is sampled at the SAME screen height as the walls, off to the right
    // of the far one: the in-scattered light a horizontal ray picks up is the
    // horizon's air column, which is markedly whiter than the deep blue overhead.
    let sky = mean_rgb(&img, 276, 62, 316, 78);
    let gap = |a: [f32; 3], b: [f32; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    eprintln!(
        "aerial_fog near {near:?} far {far:?} sky {sky:?} gap {:.3} / {:.3}",
        gap(near, sky),
        gap(far, sky)
    );
    assert!(
        luma(far) > luma(near) + 0.10,
        "the far wall did not lighten: near {near:?} far {far:?}"
    );
    // ...and it converges ON THE SKY, which is the assertion that actually pins
    // the model. "Gets bluer" would be wrong here and would rightly fail: a hazy
    // noon horizon is whiter than the blue hemispheric ambient, so the far wall
    // gets *less* blue while still becoming more sky-coloured. Distance in RGB
    // says what was meant.
    assert!(
        gap(far, sky) < gap(near, sky) * 0.6,
        "the far wall did not converge on the sky: near {near:?} far {far:?} sky {sky:?}"
    );

    // Off path, in its strongest form: the SAME geometry with no atmosphere is
    // the pre-P17.2 render, and it must be deterministic and different.
    let mut plain = scene.clone();
    plain.atmosphere = AtmosphereParams::default();
    plain.sun = SunParams::default();
    let a = render(&gpu, &plain, &view);
    let b = render(&gpu, &plain, &view);
    assert_eq!(a, b, "the no-atmosphere path must stay deterministic");
    assert_ne!(a, img, "the atmosphere changed nothing about the scene");
    // With no atmosphere the two walls are the SAME colour (only the old fixed
    // distance haze separates them) — which is exactly what P17.2 replaced.
    let plain_near = mean_rgb(&a, 91, 62, 107, 78);
    let plain_far = mean_rgb(&a, 213, 62, 229, 78);
    assert!(
        (luma(plain_far) - luma(plain_near)).abs() < luma(far) - luma(near),
        "the old haze separated the walls more than the atmosphere does: \
         plain {plain_near:?}/{plain_far:?} vs atmos {near:?}/{far:?}"
    );
}

/// The LUT determinism gate: two independent renderers bake the same LUTs from
/// the same inputs, and the texels must match **byte for byte**. This is the
/// atmosphere's version of the double-render gate every golden runs, one level
/// lower — on the intermediate the sky is a lookup into — so a nondeterministic
/// march surfaces here rather than as a flaky pixel three passes downstream.
#[test]
fn atmosphere_luts_are_deterministic() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = tod_scene(43_200.0);
    let view = horizon_view(bodies.sun, 20.0);

    let bake = || {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let a = renderer.atmosphere();
        (
            a.read_transmittance(&gpu).expect("transmittance readback"),
            a.read_sky_view(&gpu).expect("sky-view readback"),
        )
    };
    let (t1, s1) = bake();
    let (t2, s2) = bake();
    assert_eq!(t1, t2, "transmittance LUT is not deterministic");
    assert_eq!(s1, s2, "sky-view LUT is not deterministic");
    // ...and not trivially empty (a bake that never dispatched would also match).
    assert!(
        t1.iter().any(|&b| b != 0),
        "transmittance LUT is empty — did the bake dispatch?"
    );
    assert!(
        s1.iter().any(|&b| b != 0),
        "sky-view LUT is empty — did the bake dispatch?"
    );
    let (tw, th) = AtmosphereQuality::High.transmittance_size();
    let (sw, sh) = AtmosphereQuality::High.skyview_size();
    assert_eq!(t1.len(), (tw * th * 8) as usize);
    assert_eq!(s1.len(), (sw * sh * 8) as usize);
}

/// A quality change RESIZES the LUTs — exactly the case the `EnvBinding` cache
/// invariant guards. A stale key does **not** validate-error and does not blank
/// the frame: wgpu keeps the old texture alive as long as a bind group references
/// it, so the pass just silently samples the previous quality's LUT. So the
/// assertions here are frame **differences**, not liveness:
///
/// * `frames[1] != frames[0]` — Low really does produce a different image from
///   High (if it did not, every other assertion would be vacuous);
/// * `frames[3] == frames[0]` — coming back to High reproduces High **exactly**.
///   With a stale bind group the last frame would keep Medium's LUTs and this is
///   the assertion that catches it.
///
/// The cube is placed along the view direction, so the lit pass — the one that
/// binds `EnvBinding` — actually covers pixels; that it does is asserted rather
/// than assumed. The adapter-free half of this gate is
/// `passes::gen_cache_tests::pointer_identity_changes_only_when_the_key_does`.
#[test]
fn atmosphere_quality_switch_rebuilds_the_env_bind() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = tod_scene(43_200.0);
    let view = horizon_view(bodies.sun, 10.0);
    // Down the view axis, not down −Z: `horizon_view` aims at the sun's azimuth,
    // which at noon is due south (+Z), so a cube at −Z would be behind the camera
    // and the env bind group would never be sampled at all.
    let ahead = DVec3::new(bodies.sun.x, 0.0, bodies.sun.z).normalize();
    // A DISTANT wall filling the middle of the frame, under thick fog. NEAR
    // geometry is not enough: at a few metres the aerial/fog term is ~nothing, so
    // the lit pass barely reads the LUT and a stale `EnvBinding` is invisible
    // (verified by mutation — with a near cube, dropping the atmosphere
    // generation from the key left this test green). At 1500 m under 2 km
    // visibility the wall's colour is mostly in-scattered sky sampled *through
    // the env bind*, so a stale LUT moves it.
    scene.atmosphere.fog = HeightFog {
        density: 1.5e-3,
        falloff: 0.0,
        height: 0.0,
        color: [1.0, 1.0, 1.0],
    };
    scene.instances.push(MeshInstance::lit(
        ahead * 1500.0 + DVec3::new(0.0, 266.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(1400.0, 540.0, 5.0),
        [0.06, 0.06, 0.07, 1.0],
        1,
    ));
    scene.mark_dirty();

    // Prove the cube covers pixels: the same frame without it must differ.
    let mut sky_only = scene.clone();
    sky_only.instances.clear();
    sky_only.mark_dirty();
    let bare = render(&gpu, &sky_only, &view);

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let mut settings = RenderSettings::default();
    let mut frames = Vec::new();
    let mut seen = Vec::new();
    for q in [
        AtmosphereQuality::High,
        AtmosphereQuality::Low,
        AtmosphereQuality::Medium,
        AtmosphereQuality::High,
    ] {
        settings.atmosphere.quality = q;
        renderer.set_settings(settings);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        frames.push(target.read_rgba(&gpu).expect("readback"));
        seen.push(renderer.atmosphere().quality);
    }

    let differing = |a: &[u8], b: &[u8]| a.iter().zip(b).filter(|(x, y)| x != y).count();
    // The wall's interior, well inside its screen rectangle: LIT pixels, which is
    // the only place the env bind group is read at all.
    let wall = |img: &[u8]| -> Vec<u8> {
        (70..110)
            .flat_map(|y| (100..220).map(move |x| (x, y)))
            .flat_map(|(x, y)| px(img, x, y))
            .collect()
    };
    let covered = differing(&frames[0], &bare);
    eprintln!(
        "quality switch: wall covers {covered} bytes; whole frame Low-vs-High {};          wall region Low-vs-High {}",
        differing(&frames[1], &frames[0]),
        differing(&wall(&frames[1]), &wall(&frames[0]))
    );
    assert!(
        covered > 20_000,
        "the lit wall covers almost nothing ({covered} bytes) — the env bind group          is not being sampled, so this test would pass with a stale key"
    );

    // The env-bind assertion: the LIT region must change with the LUT. A stale
    // `EnvBinding` keeps High's views for every frame, so this region would come
    // back byte-identical while the (separately keyed) sky around it changed —
    // which is exactly what a whole-frame comparison would fail to notice.
    assert_ne!(
        wall(&frames[1]),
        wall(&frames[0]),
        "the lit wall is byte-identical at Low and High — the env bind group is          still holding the previous quality's LUT views"
    );
    // And the whole frame differs too (the sky path, separately keyed).
    assert_ne!(
        frames[1], frames[0],
        "Low and High rendered identically — the LUT resize had no visible effect"
    );
    // Round trip: back at High, the frame must reproduce the first High frame
    // byte for byte.
    assert_eq!(
        frames[3], frames[0],
        "returning to High did not reproduce the High frame"
    );

    assert_eq!(
        seen,
        vec![
            AtmosphereQuality::High,
            AtmosphereQuality::Low,
            AtmosphereQuality::Medium,
            AtmosphereQuality::High,
        ],
        "the resources did not follow the settings"
    );
}

/// **The editor default look** (P17.2's "gorgeous default sky" deliverable): the
/// `TimeOfDay::default()` clock — 10:00 UTC on day 172 at 48.9° N — over a
/// primitive scene of the same shape a new level is built with.
///
/// This is deliberately *not* a mirror of `inf_editor_core::scene::demo::build`,
/// which lives in another ring and would silently drift from a copy here. It is
/// the same **sky** over representative geometry: what this golden pins is the
/// default clock's look, which is the thing a change to the defaults would move.
///
/// Why 10:00 rather than noon: the sun lands ≈ 55° up, which keeps a real
/// direction — long enough shadows and a clear light/shade split to read shape —
/// while still giving a saturated blue zenith. A noon sun lights everything from
/// straight overhead and flattens exactly the geometry a default scene exists to
/// show off.
#[test]
fn golden_editor_default() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = tod_scene(36_000.0); // TimeOfDay::default(): 10:00 UTC
    let elevation = bodies.sun.y.asin().to_degrees();
    assert!(
        (50.0..60.0).contains(&elevation),
        "the default clock should put the sun ~55° up, got {elevation:.1}°"
    );
    scene.grid_enabled = true;
    // P17.3: a new level now boots with CLOUDS. This is the one golden P17.3
    // re-blessed, and this line is the reason — `inf_editor_core::scene::demo`
    // sets `clouds_enabled = true` on the default scene's `SkyAtmosphere` while
    // the *component* default stays false (which is what every existing v12 level
    // lifts to). Everything else stays at `CloudParams::default()`, so what this
    // pictures is the documented defaults rather than a private tuning.
    scene.atmosphere.clouds = CloudParams {
        enabled: true,
        ..CloudParams::default()
    };

    // Ground plane + the three props, at the default scene's placements/colours.
    let mut push = |mesh, t: DVec3, s: Vec3, c: [f32; 3], id| {
        scene.instances.push(MeshInstance {
            translation: t,
            rotation: Quat::IDENTITY,
            scale: s,
            color: [c[0], c[1], c[2], 1.0],
            metallic: 0.0,
            roughness: 0.6,
            emissive: [0.0; 3],
            id,
            mesh,
            blend: 0,
            cutoff: 0.5,
        })
    };
    push(
        PrimMesh::Plane,
        DVec3::ZERO,
        Vec3::new(20.0, 1.0, 20.0),
        [0.30, 0.32, 0.35],
        1,
    );
    push(
        PrimMesh::Cube,
        DVec3::new(-2.0, 0.5, 0.0),
        Vec3::ONE,
        [0.80, 0.25, 0.22],
        2,
    );
    push(
        PrimMesh::Sphere,
        DVec3::new(0.0, 0.6, -1.5),
        Vec3::ONE,
        [0.25, 0.55, 0.85],
        3,
    );
    push(
        PrimMesh::Cylinder,
        DVec3::new(2.0, 0.75, 0.5),
        Vec3::ONE,
        [0.30, 0.70, 0.35],
        4,
    );
    // The sky's own key light, exactly as `project_sky` pushes it, plus the
    // default scene's point fill.
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        color: [1.0, 0.98, 0.95],
        intensity: 3.0,
        direction: bodies.sun.as_vec3(),
        position: DVec3::ZERO,
        range: 0.0,
        cast_shadows: true,
        ..RenderLight::default()
    });
    // The default scene's point fill, at `Light::default()`'s intensity 1.0 —
    // not a number invented here.
    scene.lights.push(RenderLight {
        kind: LightKind::Point,
        color: [1.0, 1.0, 1.0],
        intensity: 1.0,
        direction: Vec3::Y,
        position: DVec3::new(4.0, 3.0, 4.0),
        range: 0.0,
        ..RenderLight::default()
    });
    scene.mark_dirty();

    // A near-horizontal camera rather than the suite's usual look-down overlook:
    // the point of this golden is the SKY, and an overlook shows almost none.
    let view = look_view(DVec3::new(7.0, 2.4, 9.5), DVec3::new(0.0, 1.6, 0.0));
    let img = check_golden(&gpu, "editor_default", &scene, &view);

    // The sky above the horizon is a believable daytime blue: bright, clearly
    // blue-dominant, and NOT the near-black editor gradient this replaced (whose
    // zenith was linear 0.038 — about 0.07 sRGB).
    let sky = mean_rgb(&img, 0, 0, W, H / 6);
    eprintln!("editor_default sky {sky:?}");
    assert!(sky[2] > 0.45, "the default sky is not bright: {sky:?}");
    assert!(
        sky[2] > sky[0] + 0.08,
        "the default sky is not blue: {sky:?}"
    );
    assert!(sky[2] < 0.98, "the default sky is blown out: {sky:?}");
    // The props are lit and readable against it.
    assert!(
        luma(mean_rgb(
            &img,
            W / 2 - 60,
            H * 70 / 100,
            W / 2 + 60,
            H * 92 / 100
        )) > 0.15,
        "the ground and props are too dark under the default sun"
    );

    // P17.3: and the default level really does have clouds in it. Without this,
    // the golden's one re-bless would be justified by a code comment alone — this
    // is what makes "the new-level look changed" a measured claim. (Declared
    // after the cloud helpers below, which is fine: Rust does not care, and
    // keeping this assertion beside the rest of `editor_default` does.)
    let mut cloudless = scene.clone();
    cloudless.atmosphere.clouds = CloudParams::default();
    cloudless.mark_dirty();
    let bare = render(&gpu, &cloudless, &view);
    let covered = changed_fraction(&img, &bare, H / 2);
    eprintln!("editor_default cloud cover {covered:.3}");
    assert!(
        covered > 0.05,
        "the default level's sky has no clouds in it ({covered:.3}) — the \
         `editor_default` re-bless would then have no reason"
    );
}

// ── P17.3 volumetric clouds ──────────────────────────────────────────────────
//
// The cloud goldens extend the P17.2 time-of-day sweep rather than inventing a
// scene: `tod_scene` builds the sun from `inf_math::solar` at the component
// defaults, and each test flips ONLY the cloud fields. What the images show is
// therefore what a level actually gets when it ticks the Clouds box, not a
// hand-tuned demo of the raymarch.
//
// None of the 29 pre-P17.3 goldens moves: clouds default to disabled, the bake
// and raymarch nodes dispatch nothing, and the lit shaders' cloud-shadow multiply
// sits inside a guarded branch. That was verified the P17.2 way — running the
// whole suite under `INF_BLESS_GOLDENS=1` and confirming `git status` reports
// zero changed PNGs — not merely asserted.

/// A cloud layer over the P17.2 sky. `coverage`/`cloud_type` are the two knobs a
/// level actually reaches for; everything else stays at `CloudParams::default()`.
fn cloud_scene(
    seconds: f64,
    coverage: f32,
    cloud_type: f32,
) -> (RenderScene, inf_math::solar::SkyBodies) {
    let (mut scene, bodies) = tod_scene(seconds);
    scene.atmosphere.clouds = CloudParams {
        enabled: true,
        coverage,
        cloud_type,
        ..CloudParams::default()
    };
    (scene, bodies)
}

/// The same scene with clouds switched back off — the off-path control every
/// cloud golden compares against, so "the feature drew something" is measured
/// rather than assumed.
fn without_clouds(scene: &RenderScene) -> RenderScene {
    let mut s = scene.clone();
    s.atmosphere.clouds = CloudParams::default();
    s.mark_dirty();
    s
}

/// Fraction of the given screen band that a cloud layer **perceptibly** changed.
///
/// Perceptibly, not at all: with premultiplied compositing plus aerial
/// perspective, an alpha of a thousandth still moves the low bit of a pixel, so
/// an exact-inequality count reports 100 % coverage for any sky that has a wisp
/// anywhere. The threshold (8/255 summed over RGB, ~1 % of range) is what makes
/// "covered" mean what an author means by it.
fn changed_fraction(a: &[u8], b: &[u8], rows: u32) -> f32 {
    let mut n = 0u32;
    for y in 0..rows {
        for x in 0..W {
            let p = px(a, x, y);
            let q = px(b, x, y);
            let d: i32 = (0..3).map(|c| (p[c] as i32 - q[c] as i32).abs()).sum();
            if d > 8 {
                n += 1;
            }
        }
    }
    n as f32 / (W * rows) as f32
}

/// Standard deviation of luma over the top `rows` of the frame — the measure that
/// tells broken cloud apart from a flat wash.
fn luma_spread(img: &[u8], rows: u32) -> f32 {
    let n = (W * rows) as f32;
    let mut sum = 0.0;
    let mut sum2 = 0.0;
    for y in 0..rows {
        for x in 0..W {
            let p = px(img, x, y);
            let l = luma([
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]);
            sum += l;
            sum2 += l * l;
        }
    }
    (sum2 / n - (sum / n) * (sum / n)).max(0.0).sqrt()
}

/// **Overcast noon.** Solid coverage of a low stratus sheet: the sky must be
/// mostly cloud, and the cloud must be *bright* — an overcast sky is the single
/// hardest case for a single-scattering march, which renders it as soot. The
/// assertion on absolute luminance is the one that catches that.
#[test]
fn golden_clouds_overcast() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 1.0, 0.15);
    // Stratus geometry: a thinner sheet, lower down.
    scene.atmosphere.clouds.bottom = 900.0;
    scene.atmosphere.clouds.top = 2200.0;
    scene.mark_dirty();
    let view = horizon_view(bodies.sun, 30.0);
    let img = check_golden(&gpu, "clouds_overcast", &scene, &view);

    let bare = render(&gpu, &without_clouds(&scene), &view);
    let sky = mean_rgb(&img, 0, 0, W, H / 2);
    let clear = mean_rgb(&bare, 0, 0, W, H / 2);
    let covered = changed_fraction(&img, &bare, H / 2);
    eprintln!("clouds_overcast sky {sky:?} vs clear {clear:?}; covered {covered:.3}");

    // An overcast sky is bright, and grey rather than blue: the droplets' albedo
    // is neutral, so the sky's blue excess must collapse.
    assert!(luma(sky) > 0.30, "overcast sky is soot: {sky:?}");
    assert!(
        sky[2] - sky[0] < (clear[2] - clear[0]) * 0.7,
        "overcast did not de-blue the sky: {sky:?} vs {clear:?}"
    );
    assert!(
        covered > 0.9,
        "coverage 1.0 left {:.1}% of the sky untouched",
        100.0 * (1.0 - covered)
    );
}

/// **Scattered cumulus at noon** — the default look, and the one that proves the
/// field has *structure*: broken cloud with real gaps, not a uniform haze. The
/// assertion is on the spread of luma across the sky band, which a flat wash
/// cannot pass, plus a floor on how much clear sky survives.
#[test]
fn golden_clouds_scattered() {
    let Some(gpu) = gpu_or_skip() else { return };
    // The *component default* coverage, so this golden pictures what a level
    // actually gets rather than a tuned demo — the P17.2 doctrine.
    let (scene, bodies) = cloud_scene(43_200.0, CloudParams::default().coverage, 0.9);
    let view = horizon_view(bodies.sun, 28.0);
    let img = check_golden(&gpu, "clouds_scattered", &scene, &view);

    let bare = render(&gpu, &without_clouds(&scene), &view);
    let rows = H / 3;
    let cloudy = luma_spread(&img, rows);
    let clear = luma_spread(&bare, rows);
    let covered = changed_fraction(&img, &bare, rows);
    eprintln!("clouds_scattered luma sd {cloudy:.4} vs clear {clear:.4}; covered {covered:.3}");
    assert!(
        cloudy > clear * 1.8,
        "scattered clouds have no structure: sd {cloudy:.4} vs clear sky {clear:.4}"
    );
    // Both ends, which is the whole meaning of "scattered": the clouds are really
    // there, and so are the gaps.
    assert!(
        covered > 0.2,
        "the default coverage drew almost nothing ({covered:.3})"
    );
    assert!(
        covered < 0.9,
        "no clear sky survives at the default coverage ({covered:.3}) — that is          overcast, and the default is meant to be broken cumulus"
    );
}

/// **Dusk clouds.** A cloud's lit top is lit by *the sun's transmittance through
/// the atmosphere*, so at 19:45 it must be measurably warmer than the same cloud
/// at noon. This is the single assertion that would catch clouds being lit by a
/// hard-coded white sun instead of by the transmittance LUT.
#[test]
fn golden_clouds_dusk() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(71_100.0, 0.6, 0.85);
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "19:45 should put the sun just above the horizon"
    );
    let view = horizon_view(bodies.sun, 14.0);
    let img = check_golden(&gpu, "clouds_dusk", &scene, &view);

    // The same clouds under a noon sun, from the same relative viewpoint.
    let (noon_scene, noon_bodies) = cloud_scene(43_200.0, 0.6, 0.85);
    let noon = render(&gpu, &noon_scene, &horizon_view(noon_bodies.sun, 14.0));

    let dusk_rgb = mean_rgb(&img, 0, 0, W, H / 2);
    let noon_rgb = mean_rgb(&noon, 0, 0, W, H / 2);
    let warm = |c: [f32; 3]| c[0] / c[2].max(1e-4);
    eprintln!(
        "clouds_dusk {dusk_rgb:?} (r/b {:.3}) vs noon {noon_rgb:?} (r/b {:.3})",
        warm(dusk_rgb),
        warm(noon_rgb)
    );
    assert!(
        warm(dusk_rgb) > warm(noon_rgb) + 0.15,
        "dusk clouds are not warmer than noon clouds: {dusk_rgb:?} vs {noon_rgb:?}"
    );
}

/// **Night clouds.** Stars stay visible through the gaps while being occluded
/// where a cloud is. The cloud pass composites over the sky pass, so this is the
/// test that the premultiplied alpha is doing its job rather than the clouds
/// being drawn behind everything.
#[test]
fn golden_clouds_night() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(84_600.0, 0.45, 0.8);
    assert!(bodies.sun.y < -0.2, "23:30 should be deep night");
    let view = horizon_view(-bodies.sun, 35.0);
    let img = check_golden(&gpu, "clouds_night", &scene, &view);

    let bare = render(&gpu, &without_clouds(&scene), &view);

    // Stars survive the gaps. Asserted by REMOVING them: a contrast-against-the-
    // mean test would also be satisfied by a bright cloud edge, whereas the only
    // thing that can make the peak drop when `star_intensity` goes to zero is a
    // star that was visible. This is the same reasoning `sky_night` uses, taken
    // one step further because there is now something else bright in frame.
    let mut starless = scene.clone();
    starless.atmosphere.star_intensity = 0.0;
    starless.mark_dirty();
    let no_stars = render(&gpu, &starless, &view);

    let m = mean_rgb(&img, 0, 0, W, H / 2);
    let field = (m[0] + m[1] + m[2]) / 3.0;
    let peak = brightest(&img, 0, 0, W, H / 2);
    let peak_starless = brightest(&no_stars, 0, 0, W, H / 2);
    eprintln!("clouds_night field {field:.3} peak {peak:.3} (starless {peak_starless:.3})");
    assert!(
        peak > peak_starless + 0.04,
        "no stars survive the gaps: the brightest pixel barely moves when the          starfield is switched off ({peak:.3} vs {peak_starless:.3})"
    );
    assert!(peak > field + 0.05, "no star contrast at all");
    assert_ne!(img, bare, "night clouds drew nothing");

    // ...and the clouds really do occlude: somewhere the frame got DARKER than
    // the starfield alone, which only happens where a dim night cloud covered a
    // star.
    let mut occluded = 0u32;
    for y in 0..H / 2 {
        for x in 0..W {
            let a = px(&img, x, y);
            let b = px(&bare, x, y);
            let sa = a[0] as i16 + a[1] as i16 + a[2] as i16;
            let sb = b[0] as i16 + b[1] as i16 + b[2] as i16;
            if sa < sb - 12 {
                occluded += 1;
            }
        }
    }
    eprintln!("clouds_night occluded {occluded} px");
    assert!(
        occluded > 0,
        "clouds never occluded a single star — is the alpha compositing backwards?"
    );
}

/// The **depth** contract: geometry in front of the cloud layer occludes it.
/// Without it the raymarch would hang in front of the world, which is exactly
/// what drawing clouds inside the sky pass would have produced (that pass clears
/// depth, so there is nothing to test against yet).
#[test]
fn clouds_are_occluded_by_geometry() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 1.0, 0.4);
    let view = horizon_view(bodies.sun, 20.0);
    let sky_only = render(&gpu, &scene, &view);

    // A wall a few metres in front of the camera, filling the middle of the
    // frame. Everything behind it — including a kilometre of overcast — must go.
    let ahead = DVec3::new(bodies.sun.x, 0.36, bodies.sun.z).normalize();
    scene.instances.push(MeshInstance::lit(
        ahead * 12.0,
        Quat::IDENTITY,
        Vec3::new(60.0, 60.0, 0.5),
        [0.5, 0.1, 0.1, 1.0],
        1,
    ));
    scene.mark_dirty();
    let walled = render(&gpu, &scene, &view);
    let walled_clear = render(&gpu, &without_clouds(&scene), &view);

    let centre = |img: &[u8]| -> Vec<u8> {
        (H * 42 / 100..H * 58 / 100)
            .flat_map(|y| (W * 42 / 100..W * 58 / 100).map(move |x| (x, y)))
            .flat_map(|(x, y)| px(img, x, y))
            .collect()
    };
    assert_ne!(
        centre(&sky_only),
        centre(&walled),
        "the wall covers nothing in the sampled region — this test would pass vacuously"
    );
    assert_eq!(
        centre(&walled),
        centre(&walled_clear),
        "clouds bled through solid geometry — the depth test is not rejecting them"
    );
}

/// The **intersecting**-geometry contract, which the entry-depth test alone
/// cannot satisfy: a summit that pokes *into* the cloud deck must not be veiled
/// by the cloud physically behind it.
///
/// This is the case `clouds_are_occluded_by_geometry` does not reach. There the
/// wall is entirely in front of the slab, so its fragments' depth beats the
/// slab's entry plane and the hardware test rejects the cloud outright. A mesa
/// whose top is 500 m inside a 1.5–4 km deck sits *beyond* that entry plane, so
/// it passes a `Greater` test — and without the `t_far` clamp the shader would
/// composite the whole marched span over it, including the five kilometres of
/// cloud behind the mountain. On an 8 km terrain that is not an exotic case, it
/// is Tuesday.
///
/// Measured as a **reduction**, not as an absence, because the correct answer is
/// not zero: there really is ~1 km of cloud between the eye and the mesa's face,
/// and it should still be visible. The reference for "the whole veil" is the same
/// scene with the mesa removed, so the two numbers are produced by the same
/// shader on the same pixels and the comparison needs no second build.
///
/// Mutation-verified: disabling the `t_far` clamp in `cloud.wgsl` moves the
/// measured alpha over the mesa from **0.275 to 0.588** and fails the assertion.
/// (It does not go to 1.0 because only ~1.4 km of this thin deck lies behind the
/// mesa along the ray, and ACES compresses the top end — the veil is a doubling,
/// not a wipe, which is exactly the sort of wrongness that ships unnoticed.)
#[test]
fn clouds_do_not_veil_geometry_inside_the_slab() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut open, bodies) = cloud_scene(43_200.0, 1.0, 0.3);
    // A LOW, THIN deck rather than the default 1.5–4 km one, for a reason worth
    // stating: at the default extinction a 2.5 km column saturates within the
    // first kilometre, so the cloud *behind* a mountain contributes almost
    // nothing and the bug hides itself. A 100–900 m deck at an optical depth
    // around 1 is both real content (a valley stratus deck) and the regime where
    // the veil is actually visible — which is exactly where a player would see it.
    open.atmosphere.clouds.bottom = 100.0;
    open.atmosphere.clouds.top = 900.0;
    open.atmosphere.clouds.density = 0.0012;
    open.mark_dirty();
    // Looking up at ~10°, so the ray enters the deck ~580 m out and the mesa's
    // face sits just past that — the correct span is short and the naive one is
    // the rest of the deck.
    let view = horizon_view(bodies.sun, 10.0);

    // A mesa 800 m away rising to 250 m: its top 150 m are inside the deck.
    let ahead = DVec3::new(bodies.sun.x, 0.0, bodies.sun.z).normalize();
    let mut walled = open.clone();
    walled.instances.push(MeshInstance::lit(
        ahead * 800.0 + DVec3::new(0.0, 125.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(3000.0, 250.0, 200.0),
        [0.42, 0.36, 0.30, 1.0],
        1,
    ));
    walled.mark_dirty();

    let open_clouds = render(&gpu, &open, &view);
    let open_clear = render(&gpu, &without_clouds(&open), &view);
    let walled_clouds = render(&gpu, &walled, &view);
    let walled_clear = render(&gpu, &without_clouds(&walled), &view);

    // The mesa's silhouette, derived rather than hard-coded: the pixels the mesa
    // changed in the cloudless pair are exactly the ones it covers.
    let mask: Vec<(u32, u32)> = (0..H)
        .flat_map(|y| (0..W).map(move |x| (x, y)))
        .filter(|&(x, y)| px(&walled_clear, x, y) != px(&open_clear, x, y))
        .collect();
    assert!(
        mask.len() > (W * H) as usize / 20,
        "the mesa covers only {} px — this test would prove nothing",
        mask.len()
    );

    // Mean luminance over the mesa's pixels. Comparing RGB *deltas* would be a
    // mistake here, and was the first thing tried: the cloud sits over a dark
    // mesa in one frame and over a bright sky in the other, so the same alpha
    // produces wildly different deltas and the metric reported 97 % either way.
    // What is comparable is the composited **alpha**, and over a near-black
    // surface that is directly recoverable.
    let lum = |img: &[u8]| -> f32 {
        let sum: f32 = mask
            .iter()
            .map(|&(x, y)| {
                let p = px(img, x, y);
                luma([
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ])
            })
            .sum();
        sum / mask.len() as f32
    };
    // `open_clouds` at these pixels is the same cloud seen at ~full alpha against
    // the sky, so it stands in for the cloud's own radiance, and the dark mesa
    // stands in for zero:
    //   alpha = (L_composited − L_background) / (L_cloud − L_background)
    let bg = lum(&walled_clear);
    let cloud = lum(&open_clouds);
    let composited = lum(&walled_clouds);
    let alpha = (composited - bg) / (cloud - bg).max(1e-4);
    eprintln!(
        "veil over {} mesa px: background {bg:.4}, cloud {cloud:.4}, composited \
         {composited:.4} => alpha {alpha:.3}",
        mask.len()
    );

    assert!(
        cloud > bg + 0.05,
        "the cloud is not brighter than the mesa ({cloud:.4} vs {bg:.4}) — the \
         alpha estimate below would be meaningless"
    );
    assert!(
        alpha < 0.45,
        "geometry inside the slab is still veiled at alpha {alpha:.3} — the whole \
         deck behind the mesa is being composited over it, so the `t_far` depth \
         clamp is not doing its job"
    );
    // ...and the correct answer is not zero: ~120 m of real cloud sits between the
    // eye and the mesa's face and must still be visible. An over-eager clamp (one
    // that stopped at the slab entry, say) would fail here.
    assert!(
        alpha > 0.02,
        "the mesa shows no cloud at all (alpha {alpha:.3}) — the clamp went too far \
         and removed the cloud that is genuinely in front of it"
    );
}

/// Cloud **shadows on the world**: the layer darkens lit geometry softly and at a
/// large scale, and is byte-neutral when off.
#[test]
fn cloud_shadows_darken_lit_geometry() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 1.0, 0.2);
    // A big bright ground plane plus the sky's own key light, so what the ground
    // band measures is dominated by the DIRECT term rather than by the sky.
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -1.0, 0.0),
        Quat::IDENTITY,
        Vec3::new(4000.0, 1.0, 4000.0),
        [0.8, 0.8, 0.8, 1.0],
        1,
    ));
    scene.lights.push(RenderLight {
        kind: LightKind::Directional,
        position: DVec3::ZERO,
        direction: bodies.sun.as_vec3(),
        color: [1.0, 0.98, 0.95],
        intensity: 3.0,
        range: 0.0,
        inner_cos: 1.0,
        outer_cos: 0.0,
        cast_shadows: false,
    });
    scene.mark_dirty();
    let view = look_view(DVec3::new(0.0, 3.0, 0.0), DVec3::new(0.0, 0.5, 60.0));

    let shaded = render(&gpu, &scene, &view);
    let mut unshadowed = scene.clone();
    unshadowed.atmosphere.clouds.shadow_strength = 0.0;
    unshadowed.mark_dirty();
    let lit = render(&gpu, &unshadowed, &view);

    let ground = |img: &[u8]| mean_rgb(img, 0, H * 78 / 100, W, H * 96 / 100);
    let a = ground(&shaded);
    let b = ground(&lit);
    eprintln!("cloud shadow: ground {a:?} vs unshadowed {b:?}");
    assert!(
        luma(a) < luma(b) - 0.01,
        "a solid overcast layer did not darken the ground: {a:?} vs {b:?}"
    );

    // Off ⇒ byte-identical to the same scene with clouds entirely absent, over
    // the GROUND band (the sky above still has clouds in it either way). This is
    // what the lit shaders' guarded branch exists for.
    let mut no_clouds = scene.clone();
    no_clouds.atmosphere.clouds = CloudParams::default();
    no_clouds.mark_dirty();
    let bare = render(&gpu, &no_clouds, &view);
    let band = |img: &[u8]| -> Vec<u8> {
        (H * 78 / 100..H * 96 / 100)
            .flat_map(|y| (0..W).map(move |x| (x, y)))
            .flat_map(|(x, y)| px(img, x, y))
            .collect()
    };
    assert_eq!(
        band(&lit),
        band(&bare),
        "shadow_strength = 0 is not byte-identical to no clouds at all — the \
         off path is not off"
    );
}

/// The bake-determinism gate, one level below the frame: two independent
/// renderers must write byte-identical noise volumes and shadow maps. A
/// nondeterministic bake surfaces here rather than as a flaky pixel three passes
/// downstream — the same shape as `atmosphere_luts_are_deterministic`.
#[test]
fn cloud_bakes_are_deterministic() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.7, 0.8);
    let view = horizon_view(bodies.sun, 20.0);

    let bake = || {
        let target = HeadlessTarget::new(&gpu, W, H);
        let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        let a = renderer.atmosphere();
        (
            a.read_cloud_shape(&gpu).expect("shape readback"),
            a.read_cloud_detail(&gpu).expect("detail readback"),
            a.read_cloud_shadow(&gpu).expect("shadow readback"),
        )
    };
    let (s1, d1, h1) = bake();
    let (s2, d2, h2) = bake();
    assert_eq!(s1, s2, "cloud shape volume is not deterministic");
    assert_eq!(d1, d2, "cloud detail volume is not deterministic");
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&h1),
        bits(&h2),
        "cloud shadow map is not deterministic"
    );

    // ...and not trivially empty (an undispatched bake would also compare equal).
    assert!(s1.iter().any(|&b| b != 0), "shape volume is empty");
    assert!(d1.iter().any(|&b| b != 0), "detail volume is empty");
    assert!(
        h1.iter().any(|&v| v < 0.99),
        "the shadow map is uniformly transparent — did the bake dispatch?"
    );
    let q = CloudQuality::High;
    let r = q.shape_res() as usize;
    assert_eq!(s1.len(), r * r * r * 4);
    let r = q.detail_res() as usize;
    assert_eq!(d1.len(), r * r * r * 4);
    let r = q.shadow_res() as usize;
    assert_eq!(h1.len(), r * r);
}

/// **CPU/GPU parity of the noise bake.** The GPU volumes must reproduce
/// `inf_render::shape_texel` / `detail_texel` to within the documented envelope:
/// at most `CPU_GPU_TEXEL_TOLERANCE` LSBs anywhere, and exactly equal for at
/// least `CPU_GPU_EXACT_FRACTION` of texels.
///
/// The envelope exists because WGSL permits an implementation to contract
/// `a*b + c` into an FMA, which shifts a result by ~1 ULP and, after the
/// `* 255 + 0.5` quantization, by at most one LSB. Everything the field is built
/// on that *could* diverge structurally — the hash, the gradient table, the
/// lattice wrap — is pure integer arithmetic, and a mistake in any of those moves
/// whole texels rather than last places, failing both halves of the gate at once.
#[test]
fn cloud_noise_bake_matches_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.7, 0.8);
    let view = horizon_view(bodies.sun, 20.0);
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let res = renderer.atmosphere();
    let q = res.cloud_quality;
    let seed = scene.atmosphere.clouds.seed;

    let compare =
        |what: &str, data: &[u8], edge: u32, reference: &dyn Fn(u32, u32, u32) -> [u8; 4]| {
            let mut exact = 0u64;
            let mut total = 0u64;
            let mut worst = 0u8;
            let mut worst_at = (0, 0, 0);
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        let i = (((z * edge + y) * edge + x) * 4) as usize;
                        let got = [data[i], data[i + 1], data[i + 2], data[i + 3]];
                        let want = reference(x, y, z);
                        total += 1;
                        if got == want {
                            exact += 1;
                        }
                        for c in 0..4 {
                            let d = got[c].abs_diff(want[c]);
                            if d > worst {
                                worst = d;
                                worst_at = (x, y, z);
                            }
                        }
                    }
                }
            }
            let frac = exact as f64 / total as f64;
            eprintln!(
            "{what} parity: {:.4}% exact, worst |d| = {worst} LSB at {worst_at:?} ({total} texels)",
            frac * 100.0
        );
            assert!(
                worst <= CPU_GPU_TEXEL_TOLERANCE,
                "{what}: worst |d| = {worst} LSB at {worst_at:?} exceeds the \
             {CPU_GPU_TEXEL_TOLERANCE}-LSB envelope — that is a port error, not FMA contraction"
            );
            assert!(
                frac >= CPU_GPU_EXACT_FRACTION,
                "{what}: only {:.2}% of texels are bit-exact (the envelope requires {:.0}%)",
                frac * 100.0,
                CPU_GPU_EXACT_FRACTION * 100.0
            );
        };

    let shape = res.read_cloud_shape(&gpu).expect("shape readback");
    compare("cloud shape", &shape, q.shape_res(), &|x, y, z| {
        shape_texel(seed, x, y, z, q.shape_res())
    });
    let detail = res.read_cloud_detail(&gpu).expect("detail readback");
    compare("cloud detail", &detail, q.detail_res(), &|x, y, z| {
        detail_texel(seed, x, y, z, q.detail_res())
    });
}

/// **CPU/GPU parity of the density function**, measured end-to-end through the
/// cloud-shadow map.
///
/// The shadow map is the right probe: every texel is a Beer–Lambert march of
/// `cloud_density` along the sun, so agreeing on it means agreeing on the whole
/// density function — the weather bias, the height gradient, the Perlin–Worley
/// remap, the coverage dissolve and the erosion, in the right order. The CPU
/// reference evaluates against the **read-back** volumes rather than re-baking
/// them, so any disagreement is attributable to the density function itself and
/// not to the (separately gated) bake.
///
/// The envelope is relative and much looser than the bake's, for a stated reason:
/// hardware trilinear filtering carries only ~8 bits of sub-texel precision while
/// the reference filters in full f32, so exact agreement is not available at any
/// price. `CPU_GPU_SHADOW_TOLERANCE` is far tighter than what a genuinely wrong
/// march produces.
#[test]
fn cloud_density_matches_the_cpu_reference() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.75, 0.8);
    let view = horizon_view(bodies.sun, 20.0);
    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    renderer.render(&gpu, &scene, &view, &target.view, (W, H));
    let res = renderer.atmosphere();
    let q = res.cloud_quality;

    let volumes = CloudVolumes {
        shape: res.read_cloud_shape(&gpu).expect("shape readback"),
        shape_res: q.shape_res(),
        detail: res.read_cloud_detail(&gpu).expect("detail readback"),
        detail_res: q.detail_res(),
    };
    let gpu_map = res.read_cloud_shadow(&gpu).expect("shadow readback");
    let params = scene.atmosphere.clouds;
    let sun = bodies.sun.as_vec3().normalize();

    // The map's parameterization, mirrored from `cs_cloud_shadow`.
    let edge = q.shadow_res();
    let extent = inf_render::passes::sky_lut::CLOUD_SHADOW_EXTENT_M;
    let centre = inf_render::passes::sky_lut::AtmosphereGpu::cloud_shadow_centre(
        [view.eye_world.x as f32, view.eye_world.z as f32],
        q,
    );

    // A deterministic scatter of taps rather than every texel: the CPU march is
    // orders of magnitude slower than the GPU's, and a few thousand taps is
    // plenty to catch a structural disagreement.
    let stride = (edge / 48).max(1);
    let mut worst = 0.0f32;
    let mut worst_at = (0, 0);
    let mut sum = 0.0f64;
    let mut n = 0u64;
    let mut shadowed = 0u64;
    for iy in (0..edge).step_by(stride as usize) {
        for ix in (0..edge).step_by(stride as usize) {
            let u = (ix as f32 + 0.5) / edge as f32;
            let v = (iy as f32 + 0.5) / edge as f32;
            let p = [
                centre[0] + (u - 0.5) * extent,
                params.bottom,
                centre[1] + (v - 0.5) * extent,
            ];
            let want =
                volumes.sun_transmittance(&params, p, [sun.x, sun.y, sun.z], q.shadow_steps());
            let got = gpu_map[(iy * edge + ix) as usize];
            let d = (got - want).abs();
            if d > worst {
                worst = d;
                worst_at = (ix, iy);
            }
            sum += d as f64;
            n += 1;
            if want < 0.99 {
                shadowed += 1;
            }
        }
    }
    let mean = sum / n as f64;
    eprintln!(
        "cloud density parity: {n} taps, mean |d| = {mean:.5}, worst = {worst:.5} at \
         {worst_at:?}, {shadowed} taps genuinely shadowed"
    );
    // The gate must not pass by both sides finding an empty sky.
    assert!(
        shadowed > n / 10,
        "only {shadowed}/{n} taps are shadowed at all — the fixture is too clear to \
         test anything"
    );
    assert!(
        worst <= CPU_GPU_SHADOW_TOLERANCE,
        "worst |d| = {worst:.5} at {worst_at:?} exceeds the documented \
         {CPU_GPU_SHADOW_TOLERANCE} envelope"
    );
    assert!(
        (mean as f32) < CPU_GPU_SHADOW_TOLERANCE * 0.25,
        "mean |d| = {mean:.5} is too large even if the worst case fits"
    );
}

/// The cloud field drifts with the **level's clock**, not with a wall clock — the
/// deterministic-wind law. Two renders at the same time of day are byte-identical;
/// advancing the clock moves the sky; and a whole number of tile wraps is a no-op,
/// which is what keeps an all-day session from quantizing into stair-steps.
#[test]
fn cloud_wind_follows_the_level_clock() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = cloud_scene(43_200.0, 0.6, 0.8);
    let view = horizon_view(bodies.sun, 25.0);

    scene.atmosphere.clouds.time_s = 600.0;
    scene.mark_dirty();
    let a = render(&gpu, &scene, &view);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the same clock rendered two different skies");

    scene.atmosphere.clouds.time_s = 1200.0;
    scene.mark_dirty();
    let later = render(&gpu, &scene, &view);
    assert_ne!(a, later, "ten minutes of wind moved nothing");

    // One whole tile of drift is exactly a no-op, because the volumes tile. Both
    // wind components are set to the same speed so they wrap in the same breath.
    scene.atmosphere.clouds.wind_x = 8.0;
    scene.atmosphere.clouds.wind_z = 8.0;
    scene.atmosphere.clouds.time_s = 0.0;
    scene.mark_dirty();
    let t0 = render(&gpu, &scene, &view);
    scene.atmosphere.clouds.time_s = (inf_render::clouds::SHAPE_TILE_M / 8.0) as f64;
    scene.mark_dirty();
    let wrapped = render(&gpu, &scene, &view);
    let (mean, max) = image_diff(&t0, &wrapped, W, H);
    eprintln!("one-tile wrap: mean {mean:.5} max {max:.5}");
    assert!(
        mean < 0.01 && max < 0.08,
        "a whole tile of wind drift was not a no-op (mean {mean}, max {max}) — the \
         field is not tiling"
    );
}

/// The quality-switch seam, extended to the cloud resources. The three cloud
/// textures live in `AtmosphereResources` and are recreated with the LUTs, so a
/// bind group that missed the generation would keep sampling the previous tier's
/// volumes — silently, exactly as the P17.2 `EnvBinding` comment warns.
#[test]
fn cloud_quality_switch_rebuilds_the_cloud_binds() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (scene, bodies) = cloud_scene(43_200.0, 0.7, 0.85);
    let view = horizon_view(bodies.sun, 25.0);

    let target = HeadlessTarget::new(&gpu, W, H);
    let mut renderer = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
    let mut settings = RenderSettings::default();
    let mut frames = Vec::new();
    let mut seen = Vec::new();
    let mut sizes = Vec::new();
    let seed = scene.atmosphere.clouds.seed;
    for q in [
        AtmosphereQuality::High,
        AtmosphereQuality::Low,
        AtmosphereQuality::Medium,
        AtmosphereQuality::High,
    ] {
        settings.atmosphere.quality = q;
        renderer.set_settings(settings);
        renderer.render(&gpu, &scene, &view, &target.view, (W, H));
        frames.push(target.read_rgba(&gpu).expect("readback"));
        let a = renderer.atmosphere();
        seen.push(a.cloud_quality);

        // ── the assertion that actually bites ──
        //
        // Every whole-frame comparison below can be satisfied by a STALE bind
        // group, and that was the original version of this test's flaw: with the
        // generation dropped from the bake's `GenCache` key, the bake keeps
        // writing into the *previous* tier's texture views, the newly created
        // ones stay at their zeroed initial contents, and the frames still differ
        // by tier because the march step counts come from the uniform rather than
        // from the bind group. So: read the freshly-created volume back and
        // require it to carry the field. Zeros mean the bake wrote somewhere else.
        //
        // Mutation-verified: dropping `res.generation` from `CloudBakeNode`'s
        // `noise_bg` key makes this fail on the second tier with an all-zero
        // volume, while every other assertion in this test still passes.
        let cq = a.cloud_quality;
        let shape = a.read_cloud_shape(&gpu).expect("shape readback");
        assert!(
            shape.iter().any(|&b| b != 0),
            "{q:?}: the volume is all zeros after the switch — the bake wrote into              a previous tier's texture, so a cloud bind group is stale"
        );
        // Stronger than "not zero": it must be the field this tier should hold, at
        // this tier's resolution. A stale *render* bind group cannot be caught by
        // a readback, but a stale bake one cannot survive this.
        let res = cq.shape_res();
        for &(x, y, z) in &[
            (0u32, 0u32, 0u32),
            (res / 3, res / 2, res / 5),
            (res - 1, res - 1, res - 1),
        ] {
            let i = (((z * res + y) * res + x) * 4) as usize;
            let got = [shape[i], shape[i + 1], shape[i + 2], shape[i + 3]];
            let want = shape_texel(seed, x, y, z, res);
            for c in 0..4 {
                assert!(
                    got[c].abs_diff(want[c]) <= CPU_GPU_TEXEL_TOLERANCE,
                    "{q:?}: texel ({x},{y},{z}) channel {c} is {} not {} — the                      volume does not hold this tier's field",
                    got[c],
                    want[c]
                );
            }
        }
        sizes.push(shape.len());
    }

    // The tier followed the setting and the volumes really did resize.
    assert_eq!(
        seen,
        vec![
            CloudQuality::High,
            CloudQuality::Low,
            CloudQuality::Medium,
            CloudQuality::High,
        ],
        "the cloud resources did not follow the atmosphere quality"
    );
    assert!(
        sizes[1] < sizes[0],
        "the Low volume is not smaller than High's"
    );

    assert_ne!(
        frames[1], frames[0],
        "Low and High rendered identically — the cloud tier had no visible effect"
    );
    assert_eq!(
        frames[3], frames[0],
        "returning to High did not reproduce the High frame — a cloud bind group is \
         still holding a previous tier's volume"
    );
}

// ── P17.4 weather states + precipitation ─────────────────────────────────────
//
// Three more sweep goldens, built the same way the P17.2/P17.3 ones are: from
// `inf_math::solar` at the component defaults, with the weather block's values
// taken straight from `inf_ecs::components::WeatherPreset::params()`. So what
// these picture is what a level gets when it clicks a preset button, not a tuned
// demo of the shader.
//
// The preset numbers are LITERALS here rather than reached for through `inf-ecs`,
// for the reason `tod_scene` spells the `SkyAtmosphere` defaults out: `inf-render`
// does not depend on `inf-ecs` and must not start doing so for a test. The Ring-0
// side pins the same table (`preset_names_round_trip_and_reject_typos` asserts the
// presets are distinct; the phase-17 gate asserts the projected values), and the
// frontend mirror is pinned by `todModel.test.ts` — three copies, each with a test
// naming the other.
//
// None of the 33 pre-P17.4 goldens moves: weather is disabled by default, the
// precipitation node dispatches nothing at all, and the projection's
// `precip.enabled` is false unless a weather block asks for rain. Verified the
// P17.2/P17.3 way — running the whole suite under `INF_BLESS_GOLDENS=1` and
// confirming `git status` reports zero changed PNGs — not merely asserted.

/// A weather state over the P17.3 clouds and the P17.2 sky.
///
/// `(coverage, cloud_type, wind_x, wind_z, fog_density, precipitation, snowiness)`
/// — the seven fields of `WeatherParams`, in declaration order.
#[allow(clippy::too_many_arguments)]
fn weather_scene(
    seconds: f64,
    coverage: f32,
    cloud_type: f32,
    wind_x: f32,
    wind_z: f32,
    fog_density: f32,
    precipitation: f32,
    snowiness: f32,
) -> (RenderScene, inf_math::solar::SkyBodies) {
    let (mut scene, bodies) = tod_scene(seconds);
    scene.atmosphere.clouds = CloudParams {
        enabled: true,
        coverage,
        cloud_type,
        wind_x,
        wind_z,
        ..CloudParams::default()
    };
    scene.atmosphere.fog = HeightFog {
        density: fog_density,
        ..HeightFog::default()
    };
    scene.atmosphere.precip = PrecipParams {
        enabled: precipitation > 0.0,
        intensity: precipitation,
        snowiness,
        wind_x,
        wind_z,
        // A fixed clock reading: the golden must picture ONE instant, and the
        // drift is a pure function of this number (`PrecipParams::offsets`), so
        // pinning it is what makes the image reproducible at all.
        time_s: 1_234.5,
        ..PrecipParams::default()
    };
    (scene, bodies)
}

/// The Storm preset — `WeatherPreset::Storm.params()`, field for field — over a
/// low, thick storm deck. The slab geometry is *not* part of a preset (a preset
/// drives coverage and type, not altitude), so it is authored here the way a
/// level would author it, and for the reason `clouds_overcast` lowers its stratus
/// sheet: a 1.5-4 km fair-weather base seen from the ground is mostly horizon,
/// and a storm is a ceiling.
fn storm_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    let (mut scene, bodies) = weather_scene(seconds, 1.0, 0.35, 22.0, 9.0, 6.0e-4, 1.0, 0.0);
    scene.atmosphere.clouds.bottom = 600.0;
    scene.atmosphere.clouds.top = 2800.0;
    scene.mark_dirty();
    (scene, bodies)
}
/// The Fog preset.
fn fog_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    weather_scene(seconds, 0.5, 0.1, 1.5, 0.5, 6.0e-3, 0.0, 0.0)
}
/// The Snow preset.
fn snow_scene(seconds: f64) -> (RenderScene, inf_math::solar::SkyBodies) {
    weather_scene(seconds, 0.9, 0.3, 5.0, 2.0, 1.2e-3, 0.7, 1.0)
}

/// The same scene with the precipitation switched off — the control every
/// precipitation assertion compares against, so "it drew something" is measured
/// rather than assumed.
fn without_precip(scene: &RenderScene) -> RenderScene {
    let mut s = scene.clone();
    s.atmosphere.precip = PrecipParams::default();
    s.mark_dirty();
    s
}

/// A few metres of ground under the camera, so the frame is not all sky and the
/// depth buffer has something in it for the precipitation to be tested against.
fn ground_plane(scene: &mut RenderScene) {
    scene.instances.push(MeshInstance::lit(
        DVec3::new(0.0, -0.5, 0.0),
        Quat::IDENTITY,
        Vec3::new(400.0, 1.0, 400.0),
        [0.22, 0.24, 0.26, 1.0],
        1,
    ));
    scene.mark_dirty();
}

/// **Storm at noon.** Full coverage, a hard wind and heavy rain. The assertions
/// are the two things a weather state has to get right at once: the *sky* went
/// overcast (the cloud half), and the *air* filled with rain (the precipitation
/// half), measured against the same frame with each switched off in turn.
#[test]
fn golden_weather_storm_noon() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 14.0);
    let img = check_golden(&gpu, "weather_storm_noon", &scene, &view);

    let dry = render(&gpu, &without_precip(&scene), &view);
    let clear = render(&gpu, &without_clouds(&without_precip(&scene)), &view);

    // The cloud half. `clouds_overcast` already owns the de-blueing claim; what a
    // *weather preset* has to show is that the whole coherent state took effect,
    // so this measures the two things a storm ceiling does to the sky: it covers
    // it, and it darkens it.
    let covered = changed_fraction(&dry, &clear, H / 3);
    let sky = mean_rgb(&dry, 0, 0, W, H / 3);
    let bare = mean_rgb(&clear, 0, 0, W, H / 3);
    eprintln!("storm sky {sky:?} vs clear {bare:?}; covered {covered:.3}");
    assert!(
        covered > 0.9,
        "the storm deck left {:.1}% of the sky open",
        100.0 * (1.0 - covered)
    );
    assert!(
        luma(sky) < luma(bare) * 0.9,
        "the storm deck did not darken the sky: {sky:?} vs {bare:?}"
    );

    // The precipitation half: the rain perceptibly changed a real fraction of the
    // frame. The threshold is low on purpose — a drop is a faint mark by design
    // (see PRECIP_ALPHA), and a sheet of rain is a *density* of them, so demanding
    // heavy per-pixel deltas would be demanding the wrong look.
    let wet = changed_fraction(&img, &dry, H);
    eprintln!("storm rain covered {wet:.4} of the frame");
    assert!(wet > 0.01, "heavy rain changed almost nothing ({wet:.4})");

    // …and it is DISTRIBUTED rather than a blob: split the frame into eight
    // vertical bands and require rain in most of them. A single bright artefact
    // (a degenerate quad, a NaN centre) would pass the fraction test above and
    // fail this one.
    let mut bands = 0;
    for b in 0..8u32 {
        let (x0, x1) = (b * W / 8, (b + 1) * W / 8);
        let mut n = 0u32;
        for y in 0..H {
            for x in x0..x1 {
                let a = px(&img, x, y);
                let c = px(&dry, x, y);
                let d: i32 = (0..3).map(|i| (a[i] as i32 - c[i] as i32).abs()).sum();
                if d > 4 {
                    n += 1;
                }
            }
        }
        if n > 4 {
            bands += 1;
        }
    }
    eprintln!("storm rain reached {bands}/8 bands");
    assert!(bands >= 6, "the rain is not distributed across the frame");
}

/// **Fog at dawn.** The Fog preset's 6e-3 m⁻¹ is a Koschmieder visibility of
/// ~500 m, so the assertion is the one thing fog must do: a distant wall loses
/// its contrast against the sky while a near one keeps it.
#[test]
fn golden_weather_fog_dawn() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = fog_scene(18_000.0);
    assert!(
        bodies.sun.y > -0.1 && bodies.sun.y < 0.25,
        "05:00 should put the sun near the horizon"
    );
    ground_plane(&mut scene);
    // Two dark walls of identical albedo and identical SCREEN size — the far one
    // is the near one scaled 30x about the eye — so every pixel of difference
    // between them is the fog. (The `aerial_fog` construction, reused because it
    // is the only way to compare two distances without also comparing two sizes.)
    let dir = DVec3::new(bodies.sun.x, 0.0, bodies.sun.z).normalize();
    for (dist, scale) in [(30.0f64, 1.0f32), (900.0, 30.0)] {
        let across = DVec3::new(-dir.z, 0.0, dir.x);
        scene.instances.push(MeshInstance::lit(
            dir * dist + across * (dist * 0.25),
            Quat::IDENTITY,
            Vec3::new(12.0 * scale, 12.0 * scale, 0.5 * scale),
            [0.08, 0.08, 0.09, 1.0],
            1,
        ));
    }
    scene.mark_dirty();
    let view = horizon_view(bodies.sun, 2.0);
    let img = check_golden(&gpu, "weather_fog_dawn", &scene, &view);

    // The control: the same scene with the fog density back at zero.
    let mut clear = scene.clone();
    clear.atmosphere.fog = HeightFog::default();
    clear.mark_dirty();
    let dry = render(&gpu, &clear, &view);

    // Fog raises the darkest thing in frame toward the sky: the walls are much
    // darker than the air, so the frame's minimum luma must climb.
    let darkest = |img: &[u8]| {
        let mut lo = 1.0f32;
        for y in 0..H {
            for x in 0..W {
                let c = px(img, x, y);
                let l = luma([
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                ]);
                lo = lo.min(l);
            }
        }
        lo
    };
    let foggy = darkest(&img);
    let clean = darkest(&dry);
    eprintln!("fog_dawn darkest {foggy:.4} vs clear {clean:.4}");
    assert!(
        foggy > clean + 0.01,
        "the fog preset did not lift the shadows: {foggy:.4} vs {clean:.4}"
    );
    // …and it did it *with distance*: the whole frame is not merely brighter.
    assert!(
        changed_fraction(&img, &dry, H) > 0.1,
        "the fog changed almost nothing"
    );
}

/// **Snow at dusk.** Two claims, and the second is the interesting one: the
/// flakes are lit by the *sky* rather than by a hard-coded white, so the same
/// snow is measurably warmer at dusk than at noon. That is the single assertion
/// that would catch precipitation being shaded by a constant — the P17.3
/// `clouds_dusk` argument, one layer down.
#[test]
fn golden_weather_snow_dusk() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = snow_scene(71_100.0);
    assert!(
        bodies.sun.y > 0.0 && bodies.sun.y < 0.2,
        "19:45 should put the sun just above the horizon"
    );
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 12.0);
    let img = check_golden(&gpu, "weather_snow_dusk", &scene, &view);

    let dry = render(&gpu, &without_precip(&scene), &view);
    let snowing = changed_fraction(&img, &dry, H);
    eprintln!("snow_dusk covered {snowing:.4}");
    assert!(
        snowing > 0.01,
        "the snow drew almost nothing ({snowing:.4})"
    );

    // Snow is not rain: the same intensity at `snowiness = 0` produces a
    // different image, because the fall speed, the streak length and the flake
    // radius all move with the phase.
    let mut as_rain = scene.clone();
    as_rain.atmosphere.precip.snowiness = 0.0;
    as_rain.mark_dirty();
    let rain = render(&gpu, &as_rain, &view);
    assert_ne!(
        img, rain,
        "snowiness changed nothing — the phase is ignored"
    );

    // The colour claim. Measure only where the precipitation actually IS (pixels
    // the dry control differs from), so the sky behind it cannot carry the test.
    let warmth = |img: &[u8], base: &[u8]| {
        let (mut r, mut b, mut n) = (0.0f32, 0.0f32, 0.0f32);
        for y in 0..H {
            for x in 0..W {
                let a = px(img, x, y);
                let c = px(base, x, y);
                let d: i32 = (0..3).map(|i| (a[i] as i32 - c[i] as i32).abs()).sum();
                if d > 4 {
                    r += a[0] as f32;
                    b += a[2] as f32;
                    n += 1.0;
                }
            }
        }
        (r / n.max(1.0)) / (b / n.max(1.0)).max(1e-4)
    };
    let (mut noon_scene, noon_bodies) = snow_scene(43_200.0);
    ground_plane(&mut noon_scene);
    let noon_view = horizon_view(noon_bodies.sun, 12.0);
    let noon = render(&gpu, &noon_scene, &noon_view);
    let noon_dry = render(&gpu, &without_precip(&noon_scene), &noon_view);

    let dusk_warm = warmth(&img, &dry);
    let noon_warm = warmth(&noon, &noon_dry);
    eprintln!("snow_dusk r/b {dusk_warm:.3} vs noon {noon_warm:.3}");
    assert!(
        dusk_warm > noon_warm + 0.05,
        "dusk snow is not warmer than noon snow ({dusk_warm:.3} vs {noon_warm:.3}) \
         — the flakes are being lit by a constant, not by the sky"
    );
}

/// The **off path**, measured rather than asserted: a scene whose precipitation
/// is disabled renders **byte-identically** to one that never had a
/// `PrecipParams` at all. That is what keeps all 33 pre-P17.4 goldens intact —
/// the node returns before touching the encoder, so the command stream is the
/// one it always was.
#[test]
fn precipitation_off_is_byte_identical() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);

    let mut disabled = scene.clone();
    disabled.atmosphere.precip.enabled = false;
    disabled.mark_dirty();

    // Three ways to say "no rain", all of which must produce the same pixels:
    // never configured, explicitly disabled, and zero intensity.
    let bare = render(&gpu, &without_precip(&scene), &view);
    assert_eq!(render(&gpu, &disabled, &view), bare, "disabled != absent");
    let mut zero = scene.clone();
    zero.atmosphere.precip.intensity = 0.0;
    zero.mark_dirty();
    assert_eq!(render(&gpu, &zero, &view), bare, "zero intensity != absent");

    // …and the enabled one really is different, or all three compare nothing.
    assert_ne!(render(&gpu, &scene, &view), bare);
}

/// The **depth** contract: geometry in front of the camera occludes the
/// precipitation behind it. Without a depth test the drops would hang in front
/// of the world, which is the most visible way a particle layer goes wrong.
#[test]
fn precipitation_is_occluded_by_geometry() {
    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);

    // A wall a couple of metres ahead, filling the middle of the frame. Every
    // drop beyond it must go.
    let ahead = DVec3::new(bodies.sun.x, 0.05, bodies.sun.z).normalize();
    scene.instances.push(MeshInstance::lit(
        ahead * 2.5,
        Quat::IDENTITY,
        Vec3::new(20.0, 20.0, 0.4),
        [0.5, 0.1, 0.1, 1.0],
        1,
    ));
    scene.mark_dirty();
    let walled = render(&gpu, &scene, &view);
    let walled_dry = render(&gpu, &without_precip(&scene), &view);
    // The control: the identical scene with the wall removed, so the SAME screen
    // region is measured against itself with and without an occluder. An absolute
    // threshold would only be measuring how much of the box lies in front of
    // 2.5 m; the ratio is the depth test.
    let mut open_scene = scene.clone();
    open_scene.instances.pop();
    open_scene.mark_dirty();
    let open = render(&gpu, &open_scene, &view);
    let open_dry = render(&gpu, &without_precip(&open_scene), &view);

    let centre_changed = |a: &[u8], b: &[u8]| {
        let (x0, x1) = (W * 3 / 8, W * 5 / 8);
        let (y0, y1) = (H * 3 / 8, H * 5 / 8);
        let mut n = 0u32;
        for y in y0..y1 {
            for x in x0..x1 {
                let p = px(a, x, y);
                let q = px(b, x, y);
                let d: i32 = (0..3).map(|i| (p[i] as i32 - q[i] as i32).abs()).sum();
                if d > 4 {
                    n += 1;
                }
            }
        }
        n as f32 / ((x1 - x0) * (y1 - y0)) as f32
    };
    let behind_wall = centre_changed(&walled, &walled_dry);
    let open_air = centre_changed(&open, &open_dry);
    eprintln!("precip over a 2.5 m wall: {behind_wall:.4} vs open air {open_air:.4}");
    // A wall 2.5 m into a 40 m box hides ~94 % of the depth the rain occupies, so
    // what survives is the sliver of drops genuinely in front of it. Without the
    // depth test this region would be as rained-on as the open air beside it —
    // mutation-verified by removing `depth_compare` from the pipeline, which
    // takes the ratio from ~0.2 to ~1.0.
    assert!(
        behind_wall < open_air * 0.35,
        "rain is drawing through a wall 2.5 m away ({behind_wall:.4} vs {open_air:.4} in open air)"
    );
    assert!(open_air > 0.2, "the open-air control is not raining");
}

/// The precipitation field is a pure function of the level's clock, exactly like
/// the cloud wind: two scenes at the same `time_s` are byte-identical, and one a
/// tenth of a second later is not.
///
/// Adapter-free where it can be (the offsets), on the GPU where it must be (the
/// frame), so a nondeterministic placement surfaces at whichever level it is
/// introduced.
#[test]
fn precipitation_follows_the_level_clock() {
    // The CPU half runs everywhere, including CI legs with no adapter.
    let at = |t: f64| {
        PrecipParams {
            enabled: true,
            intensity: 1.0,
            wind_x: 22.0,
            wind_z: 9.0,
            time_s: t,
            ..PrecipParams::default()
        }
        .offsets()
    };
    assert_eq!(
        at(1_234.5),
        at(1_234.5),
        "the offsets are not a pure function"
    );
    assert_ne!(at(1_234.5), at(1_234.6));

    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);
    let a = render(&gpu, &scene, &view);
    let b = render(&gpu, &scene, &view);
    assert_eq!(a, b, "the precipitation pass is not deterministic");

    let mut later = scene.clone();
    later.atmosphere.precip.time_s += 0.1;
    later.mark_dirty();
    assert_ne!(render(&gpu, &later, &view), a, "the rain never fell");
}

/// The **tier** clamp reaches precipitation: a lower atmosphere quality draws
/// fewer particles, and never more. Asserted on the count (which is what the tier
/// actually controls) and confirmed on the frame, so "the tier is wired" and "the
/// tier is honoured" are separate claims.
#[test]
fn precipitation_density_follows_the_render_tier() {
    let p = PrecipParams {
        enabled: true,
        intensity: 1.0,
        ..PrecipParams::default()
    };
    let n = |q| p.count(PrecipQuality::from_atmosphere(q));
    assert!(n(AtmosphereQuality::Medium) < n(AtmosphereQuality::High));
    assert!(n(AtmosphereQuality::Low) < n(AtmosphereQuality::Medium));

    let Some(gpu) = gpu_or_skip() else { return };
    let (mut scene, bodies) = storm_scene(43_200.0);
    ground_plane(&mut scene);
    let view = horizon_view(bodies.sun, 8.0);
    // The control must be rendered at the SAME tier: a lower atmosphere quality
    // also shrinks the sky LUTs, so comparing a Low frame against a High dry one
    // measures the whole sky rather than the rain. Not hypothetical — the first
    // draft of this test reported Low drawing *thirty times* the rain of High,
    // which was the LUT difference in its entirety.
    let with = |q, precip: bool| {
        let mut s = RenderSettings::default();
        s.atmosphere.quality = q;
        let sc = if precip {
            scene.clone()
        } else {
            without_precip(&scene)
        };
        render_with(&gpu, &sc, &view, s)
    };
    let high = changed_fraction(
        &with(AtmosphereQuality::High, true),
        &with(AtmosphereQuality::High, false),
        H,
    );
    let low = changed_fraction(
        &with(AtmosphereQuality::Low, true),
        &with(AtmosphereQuality::Low, false),
        H,
    );
    eprintln!("precip coverage High {high:.4} vs Low {low:.4}");
    assert!(
        low < high,
        "the Low tier drew as much rain as High ({low:.4} vs {high:.4})"
    );
}
