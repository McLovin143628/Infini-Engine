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

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::gizmo::{self, GizmoAxis, GizmoMode};
use inf_render::golden::{image_diff, within_tolerance};
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, MeshInstance, RenderScene, RenderView,
    HEADLESS_FORMAT,
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
