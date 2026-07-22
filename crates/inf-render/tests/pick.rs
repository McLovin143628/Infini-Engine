//! Headless Picker smoke gate. `Picker::new` compiles the composed pick
//! shader on a real device (the P13 composition drift that shipped a
//! `no definition in scope for identifier: shadow` parse error was only
//! reachable through this constructor — no test built a `Picker`, so it
//! escaped CI and killed the live viewport thread). This proves the module
//! AND the view-only pipeline layout end-to-end, then picks a cube.
//! Complements the no-GPU naga gate in `passes::shader_compose_tests`.
//! Skips when no adapter is available.

use glam::{DVec3, Quat, Vec3};
use inf_math::FloatingOrigin;
use inf_render::{GpuContext, MeshInstance, Picker, RenderScene, RenderView};

const W: u32 = 160;
const H: u32 = 120;

#[test]
fn picker_builds_and_picks_center_cube() {
    let gpu = match GpuContext::headless() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP pick: no GPU adapter ({e})");
            return;
        }
    };

    // Constructing the picker is itself the regression gate: a composition
    // error in the pick shader panics right here.
    let mut picker = Picker::new(&gpu);

    let mut scene = RenderScene::default();
    scene.instances.push(MeshInstance::lit(
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(1.0),
        [1.0, 0.2, 0.2, 1.0],
        7,
    ));

    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 0.0, 5.0),
        forward: Vec3::new(0.0, 0.0, -1.0),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    };

    // Center pixel lands on the cube; the top-left corner is background.
    assert_eq!(picker.pick(&gpu, &scene, &view, W / 2, H / 2), Some(7));
    assert_eq!(picker.pick(&gpu, &scene, &view, 1, 1), None);
}
