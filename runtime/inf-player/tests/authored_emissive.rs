//! **The authored twin of `hdr_bloom.png`** (wave VIS1b, clause 2).
//!
//! `golden_hdr_bloom` builds its three glowing cubes by writing
//! `MeshInstance::emissive = [8.0, 1.0, 0.4]` **by hand**, because a golden may
//! reach past the UI. Until wave VIS1a's schema window that was the only way to
//! get there at all: `Material::emissive` is an 8-bit sRGB colour, so it cannot
//! exceed 1.0 in any channel, and the renderer's default bloom threshold is a
//! linear luminance of 1.0. An emissive material authored in the editor could
//! *touch* the threshold and never cross it.
//!
//! `Material::emissive_intensity` closed that, and this file is the arm that says
//! so — it drives the same 9.0-class emissive through the **real component path**
//! (an ECS `Material` on an entity, `inf_player::render::project_scene`, the
//! shipped projection), and then renders it and measures the bloom.
//!
//! It lives here rather than beside the golden because `inf-render` cannot see
//! `inf-ecs` — the ring rule — so the only place the component path and the
//! renderer meet is a host's tests.

use glam::DVec2;
use glam::{DVec3, Vec3};
use inf_ecs::components::{BlendMode, Material, MeshRef, Primitive, Transform};
use inf_ecs::math::Color;
use inf_ecs::{EcsWorld, Vec3d};
use inf_player::runtime_sim::RuntimeSim;
use inf_render::{
    BloomSettings, EngineRenderer, GpuContext, HeadlessTarget, RenderScene, RenderSettings,
    RenderView, HEADLESS_FORMAT,
};

const W: u32 = 320;
const H: u32 = 180;
const HZ: f64 = 60.0;

/// The intensity the golden reaches by hand. Nine, so the arm is measuring the
/// same magnitude `hdr_bloom` does rather than a magnitude chosen to pass.
const AUTHORED_INTENSITY: f32 = 9.0;

fn view() -> RenderView {
    let eye = DVec3::new(0.0, 1.5, 7.0);
    RenderView {
        origin: inf_math::FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (DVec3::new(0.0, 0.5, 0.0) - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    }
}

/// A world with three cubes whose **`Material`** carries the emissive — the
/// colour an editor colour picker can produce, and an intensity a slider can.
fn authored_world(intensity: f32) -> RuntimeSim {
    let mut world = EcsWorld::new();
    for (i, (x, colour)) in [
        (-2.6f64, [1.0f32, 0.125, 0.05]),
        (0.0, [0.071, 1.0, 0.143]),
        (2.6, [0.067, 0.133, 1.0]),
    ]
    .into_iter()
    .enumerate()
    {
        // FIXED guids: the projection iterates in guid order, so two worlds
        // built with fresh guids would project their lamps in two different
        // orders and the multiplier comparison below would pair the red lamp
        // with the blue one. (It did, on the first run.)
        let e = world.spawn_with_guid(
            uuid::Uuid::from_u128(0xE_1155_1E0_0000_0001 + i as u128),
            &format!("Lamp{i}"),
            None,
        );
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: Vec3d::new(x, 0.5, 0.0),
                rotation: Vec3d::new(0.0, 17.2, 0.0),
                scale: Vec3d::new(0.6, 0.6, 0.6),
            },
            MeshRef {
                primitive: Primitive::Cube,
                ..MeshRef::default()
            },
            Material {
                base_color: Color::new(0.02, 0.02, 0.02, 1.0),
                metallic: 0.0,
                roughness: 0.5,
                // An 8-bit sRGB colour: every channel is inside [0,1] and the
                // picker can produce it.
                emissive: Color::new(colour[0], colour[1], colour[2], 1.0),
                emissive_intensity: intensity,
                blend: BlendMode::Opaque,
                alpha_cutoff: 0.5,
                asset: None,
            },
        ));
    }
    world.mark_dirty();
    world.propagate();
    RuntimeSim::new(world, Vec::new(), DVec2::new(0.0, -9.81), HZ)
}

fn project(sim: &RuntimeSim) -> RenderScene {
    let mut scene = RenderScene {
        grid_enabled: false,
        ..Default::default()
    };
    inf_player::render::project_scene(
        &mut scene,
        sim,
        0.0,
        &inf_player::vmesh::VmeshRegistry::new(),
    );
    scene.grid_enabled = false;
    scene.mark_dirty();
    scene
}

fn render(gpu: &GpuContext, scene: &RenderScene, settings: RenderSettings) -> Vec<u8> {
    let target = HeadlessTarget::new(gpu, W, H);
    let mut renderer = EngineRenderer::new(gpu, HEADLESS_FORMAT);
    renderer.set_settings(settings);
    renderer.render(gpu, scene, &view(), &target.view, (W, H));
    target.read_rgba(gpu).expect("readback")
}

fn luma(img: &[u8]) -> f64 {
    img.chunks(4)
        .map(|p| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64)
        .sum()
}

/// **The projection carries the intensity, and it carries it past 1.0.**
///
/// No GPU needed: this is the half that says the component path reaches the
/// renderer's input at all. The 8-bit ceiling is the thing being falsified, so it
/// is named — every authored channel is inside `[0, 1]` and at least one
/// projected channel is past the default bloom threshold.
#[test]
fn an_authored_material_drives_emissive_past_the_eight_bit_ceiling() {
    let scene = project(&authored_world(AUTHORED_INTENSITY));
    assert_eq!(scene.instances.len(), 3, "the three lamps did not project");

    let mut over_threshold = 0;
    for inst in &scene.instances {
        assert!(
            inst.emissive.iter().all(|c| c.is_finite()),
            "a projected emissive is not finite: {:?}",
            inst.emissive
        );
        if inst.emissive.iter().any(|&c| c > 1.0) {
            over_threshold += 1;
        }
    }
    assert_eq!(
        over_threshold,
        3,
        "no authored lamp crossed the default bloom threshold: {:?}",
        scene
            .instances
            .iter()
            .map(|i| i.emissive)
            .collect::<Vec<_>>()
    );
    // The brightest channel of the first lamp is `1.0 * 9.0` — the same magnitude
    // `golden_hdr_bloom` writes by hand as `8.0`.
    let peak = scene.instances[0]
        .emissive
        .iter()
        .fold(0.0f32, |a, &b| a.max(b));
    assert!(
        (peak - AUTHORED_INTENSITY).abs() < 1e-4,
        "peak authored emissive is {peak}, expected {AUTHORED_INTENSITY}"
    );

    // The identity holds: intensity 1.0 is the pre-v26 behaviour, byte for byte.
    let plain = project(&authored_world(1.0));
    for (a, b) in plain.instances.iter().zip(scene.instances.iter()) {
        assert!(a.emissive[0] <= 1.0 && a.emissive[1] <= 1.0 && a.emissive[2] <= 1.0);
        for c in 0..3 {
            assert!(
                (a.emissive[c] * AUTHORED_INTENSITY - b.emissive[c]).abs() < 1e-4,
                "intensity is not a multiplier"
            );
        }
    }
}

/// **And it actually blooms** — the twin of `golden_hdr_bloom`'s own assertion,
/// on a scene built the way an author would build it.
#[test]
fn an_authored_emissive_material_blooms_from_the_component_path() {
    let Some(gpu) = (match GpuContext::headless() {
        Ok(g) => Some(g),
        Err(e) => {
            eprintln!("SKIP authored_emissive: no GPU adapter ({e})");
            None
        }
    }) else {
        return;
    };

    let bloom_on = RenderSettings {
        bloom: BloomSettings {
            enabled: true,
            threshold: 1.0,
            knee: 0.6,
            intensity: 0.5,
            karis: false,
        },
        ..RenderSettings::default()
    };

    let authored = project(&authored_world(AUTHORED_INTENSITY));
    let on = render(&gpu, &authored, bloom_on);
    let off = render(&gpu, &authored, RenderSettings::default());
    let gain = luma(&on) - luma(&off);

    // The control: the SAME materials at the pre-v26 intensity of 1.0, where the
    // 8-bit colour cannot cross the threshold. This is the arm's whole point —
    // not "bloom works" but "the intensity field is what lets an authored
    // material reach it".
    let ceilinged = project(&authored_world(1.0));
    let ceil_on = render(&gpu, &ceilinged, bloom_on);
    let ceil_off = render(&gpu, &ceilinged, RenderSettings::default());
    let ceil_gain = luma(&ceil_on) - luma(&ceil_off);

    eprintln!(
        "authored emissive: intensity {AUTHORED_INTENSITY} blooms +{gain:.0}, \
         intensity 1.0 blooms +{ceil_gain:.0}"
    );
    assert!(
        gain > 1000.0,
        "an authored 9.0-class emissive did not bloom: +{gain:.0}"
    );
    assert!(
        gain > ceil_gain * 4.0,
        "the intensity field bought nothing: +{gain:.0} against +{ceil_gain:.0}"
    );
}
