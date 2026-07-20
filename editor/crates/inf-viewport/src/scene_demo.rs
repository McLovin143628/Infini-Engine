//! Phase 2 demo scene: 10 000 cubes (the roadmap's vsync target) laid out as
//! a district of deterministic pseudo-random "buildings", plus a few landmark
//! cubes near the origin. Replaced by the real ECS world in Phase 3.

use glam::{DVec3, Quat, Vec3};
use inf_render::{MeshInstance, RenderScene};

/// Deterministic hash → [0, 1). Keeps the scene identical across runs
/// (goldens, perf comparisons) without pulling in a rand dependency.
fn hash01(mut x: u32) -> f32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    (x >> 8) as f32 / ((1u32 << 24) as f32)
}

/// Cheap HSV→RGB for hue-spread instance colors (s/v fixed).
fn hue_color(h: f32) -> [f32; 4] {
    let f = |n: f32| {
        let k = (n + h * 6.0) % 6.0;
        0.72 - 0.55 * k.min(4.0 - k).clamp(0.0, 1.0)
    };
    [f(5.0), f(3.0), f(1.0), 1.0]
}

pub fn build() -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: true,
        ..Default::default()
    };

    // Field of varied cubes, a boulevard kept clear on both axes. 104×104
    // minus the boulevard stays above the roadmap's 10k-at-vsync target.
    const N: u32 = 104;
    const SPACING: f64 = 3.0;
    let offset = (N - 1) as f64 * SPACING * 0.5;
    for gz in 0..N {
        for gx in 0..N {
            let i = gz * N + gx;
            let h = hash01(i.wrapping_mul(2_654_435_761));
            let height = 0.4 + h * h * 4.2;
            let footprint = 0.6 + hash01(i ^ 0x9e37_79b9) * 1.3;
            let x = gx as f64 * SPACING - offset;
            let z = gz as f64 * SPACING - offset;
            // Clear a cross-shaped boulevard through the middle.
            if x.abs() < 4.0 || z.abs() < 4.0 {
                continue;
            }
            scene.instances.push(MeshInstance {
                translation: DVec3::new(x, height as f64 * 0.5, z),
                rotation: Quat::from_rotation_y(hash01(i ^ 0xdead_beef) * 0.35 - 0.175),
                scale: Vec3::new(footprint, height, footprint),
                color: hue_color(0.52 + hash01(i ^ 0x00c0_ffee) * 0.22),
                id: i + 1,
            });
        }
    }

    // Landmarks at the origin so the first frame reads instantly.
    for (i, (pos, size, hue)) in [
        (DVec3::new(0.0, 0.5, 0.0), 1.0f32, 0.58),
        (DVec3::new(2.2, 0.35, -1.4), 0.7, 0.08),
        (DVec3::new(-1.8, 0.9, -2.2), 1.8, 0.33),
    ]
    .into_iter()
    .enumerate()
    {
        scene.instances.push(MeshInstance {
            translation: pos,
            rotation: Quat::IDENTITY,
            scale: Vec3::splat(size),
            color: hue_color(hue),
            id: 1_000_000 + i as u32,
        });
    }

    scene.mark_dirty();
    tracing::info!(
        "inf-viewport: demo scene with {} cubes",
        scene.instances.len()
    );
    scene
}
