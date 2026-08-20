//! **Clause 5 — real building geometry: the path, proven, and the prize, priced.**
//!
//! The I3 ledger's carried bound: *"a far building's shell is a BOX, not its
//! baked mesh… the near tier draws placeholder cubes too (`kind_index → real
//! mesh` is the standing P19 gap). Wiring `bake_building_in` into a runtime vgeom
//! asset per archetype is a project."* The I4 brief asked for that project. This
//! file is what this wave landed of it, and it is deliberately two things and not
//! three:
//!
//! 1. **The path works, end to end, in memory** — a grammar building baked to a
//!    `MeshAsset` (P23's `bake_building`), indexed into a meshlet DAG
//!    (`inf_vgeom::build_vgeom`), and handed to the renderer as a
//!    `VgeomAsset` + `VgeomInstance`. No schema, no cook, no file: every door in
//!    that chain already exists and none of them had ever been called in a row.
//! 2. **The price is a number, and it is not the sign anybody expected** — the
//!    same city content drawn both ways at 1080p, with the per-pass GPU clock
//!    this wave built. 434 176 placeholder cubes cost **3.378 ms** of GPU frame;
//!    1 024 baked vgeom instances cost **4.793 ms**. Real geometry is **1.416 ms
//!    DEARER**, not cheaper. So "the city stops being cubes" is a **fidelity**
//!    decision with a frame-time cost attached, and anyone who takes it on the
//!    assumption that 370 000 instances must be the expensive thing is taking it
//!    on a guess this arm refutes.
//!
//! # What is NOT here, and why (the honest half)
//!
//! The runtime projection still emits placeholder-cube scatter batches for PCG
//! structures. Closing that needs the bake to run **cook-side**, and the blocker
//! is placement rather than capability:
//!
//! * `inf_editor_core::bake` is Ring 1, and the shipped player must never link
//!   the modelling kernel — `inf-dcc` is a *dev*-dependency of `inf-player` by
//!   ruling (P23.3).
//! * Its own dependencies are all Ring 0 (`inf-dcc`, `inf-ecs`, `inf-mesh`,
//!   `inf-pcg`), so the bake **could** move down. `inf-dcc` names none of them,
//!   so `inf-dcc::bake` is acyclic.
//! * `inf-packager` (the cook) already depends on `inf-pcg`, `inf-mesh`,
//!   `inf-vgeom` and `inf-ecs`, and adding `inf-dcc` to it puts the kernel in a
//!   TOOL rather than in the player — which keeps the P23.3 ruling intact and
//!   keeps the `inf` CLI wgpu-free (the reason `inf-project` exists).
//! * What is genuinely missing is that **the cook does not evaluate PCG
//!   volumes** — the player does, on load — so there is nothing cook-side to bake
//!   *from* yet. That is the project, and it is the piece this wave did not take.
//!
//! The three `derived_*_id` precedents (`derived_vmesh_id`,
//! `derived_fracture_id`, `derived_material_id`) are the shape the derived
//! building mesh would take, and `editor/crates/inf-editor-core/src/assets/vmesh.rs`'s
//! plan/build/commit split is the shape of doing it without holding the project
//! mutex.

use std::sync::Arc;

use glam::{DVec3, Quat, Vec3};
use inf_editor_core::bake::bake_building;
use inf_math::FloatingOrigin;
use inf_pcg::building::{ArchetypeId, Rect2};
use inf_pcg::BuildingParams;
use inf_render::{
    EngineRenderer, GpuContext, HeadlessTarget, LightKind, PrimMesh, RenderLight, RenderScene,
    RenderSettings, RenderView, ScatterBatch, ScatterData, ScatterInstance, VgeomAsset,
    VgeomInstance, VgeomSettings, HEADLESS_FORMAT,
};

const W: u32 = 1920;
const H: u32 = 1080;
/// The city's own block count, so the comparison is at the fixture's scale.
const BUILDINGS: usize = 1_000;
/// A stable id for the baked asset, in the shape a `derived_*_id` would produce:
/// computed from the building's own parameters, never indexed.
const BAKED_ID: u128 = 0x8431_0002_ba7e_d000_0000_0000_0000_0001;

/// The city's building: `CITY_FLOORS = 2` on a 20 × 30 m lot, through the same
/// `BuildingParams` the `building.plan` node fills in.
fn city_building() -> BuildingParams {
    let footprint = Rect2::from_center(glam::DVec2::ZERO, glam::DVec2::new(20.0, 30.0));
    BuildingParams {
        floors: 2,
        ..BuildingParams::new(ArchetypeId::Office, footprint, 0.0, 30)
    }
}

/// **The path**: a grammar building becomes one meshlet-indexed asset the
/// renderer can draw, with no file and no schema anywhere in it.
#[test]
fn a_baked_grammar_building_becomes_one_vgeom_asset() {
    let baked = bake_building(&city_building(), 30, false).expect("the building bakes");
    let (positions, normals, uvs, tangents, indices) = baked.asset.vgeom_streams();
    let mesh = inf_vgeom::build_vgeom(
        &positions,
        &normals,
        &uvs,
        &tangents,
        &indices,
        inf_vgeom::BuildParams::default(),
    );
    let asset = VgeomAsset::from_mesh(BAKED_ID, &mesh).expect("the DAG indexes");

    println!(
        "clause 5 path: bake -> {} parts, {} vertices / {} triangles -> vgeom {} \
         meshlets, asset id {BAKED_ID:#034x}",
        baked.report.parts,
        positions.len(),
        indices.len() / 3,
        mesh.meshlet_count(),
    );
    assert!(
        baked.report.parts > 0,
        "the bake merged nothing — there is no building to draw"
    );
    assert!(
        indices.len() / 3 > 100,
        "the baked building is {} triangles; a placeholder cube is 12, and a \
         mesh that small is not standing in for a building",
        indices.len() / 3
    );
    assert!(
        mesh.meshlet_count() > 0,
        "the DAG has no meshlets — the vgeom path cannot draw this"
    );
    assert_eq!(asset.id, BAKED_ID, "the asset carries the id it was given");

    // **The frame carries the baked id, and not a placeholder.** The claim the
    // brief asks for, on the half of the pipeline this wave built: a scene whose
    // structures are baked vgeom instances has ZERO placeholder-primitive
    // batches, and every instance names the baked asset.
    let scene = vgeom_city(&asset, 8);
    assert!(
        scene.scatter.is_empty(),
        "a baked-geometry frame still carries {} placeholder scatter batch(es)",
        scene.scatter.len()
    );
    assert_eq!(scene.vgeom_assets.len(), 1, "one asset for one archetype");
    assert!(
        scene.vgeom_instances.iter().all(|i| i.asset == BAKED_ID),
        "an instance named an asset that is not the baked building"
    );
}

/// A district of `side²` baked buildings on the city's own 140 × 100 m pitch.
fn vgeom_city(asset: &VgeomAsset, side: i32) -> RenderScene {
    let mut scene = RenderScene {
        vgeom_assets: vec![asset.clone()],
        lights: vec![sun()],
        grid_enabled: false,
        ..Default::default()
    };
    let mut id = 1u32;
    for x in 0..side {
        for z in 0..side {
            scene.vgeom_instances.push(VgeomInstance::lit(
                asset.id,
                DVec3::new(f64::from(x) * 140.0, 0.0, f64::from(z) * 100.0),
                Quat::IDENTITY,
                Vec3::ONE,
                [0.62, 0.60, 0.55, 1.0],
                id,
            ));
            id += 1;
        }
    }
    scene.mark_dirty();
    scene
}

fn sun() -> RenderLight {
    RenderLight {
        kind: LightKind::Directional,
        direction: Vec3::new(-0.4, 0.78, -0.48).normalize(),
        color: [1.0, 0.96, 0.88],
        intensity: 3.2,
        ..Default::default()
    }
}

/// **THE PRICE, MEASURED — and it has the opposite sign to the guess.**
///
/// The instrument's own frame says the scatter pass is **67.8 %** of a 1080p GPU
/// frame over the city (10.758 ms of 15.875), which reads like an invitation to
/// replace 370 468 cubes with 1 000 meshes. Measured, that trade is **dearer**:
/// 434 176 cubes at 3.378 ms against 1 024 baked instances at 4.793 ms, a
/// 1.416 ms loss. The GPU scatter path culls per instance in a compute pass and
/// draws indirect; the meshlet path pays a DAG traversal, a two-pass HZB and a
/// far larger triangle count (5 088 per building against a cube's 12).
///
/// The project the ledger carries is therefore chosen on this number rather than
/// on the fact that boxes look like boxes — and what it buys is the city looking
/// like a city, which is worth 1.4 ms and should be spent knowingly.
///
/// Reported and never asserted: it is two wall clocks on one machine, and §8
/// budgets are tripwires rather than hardware claims.
#[test]
fn the_cost_of_the_city_as_cubes_and_as_baked_meshes() {
    let Ok(gpu) = GpuContext::headless() else {
        eprintln!("SKIP baked_building_geometry cost: no GPU adapter");
        return;
    };
    let info = gpu.adapter.get_info();
    let baked = bake_building(&city_building(), 30, false).expect("the building bakes");
    let (positions, normals, uvs, tangents, indices) = baked.asset.vgeom_streams();
    let mesh = inf_vgeom::build_vgeom(
        &positions,
        &normals,
        &uvs,
        &tangents,
        &indices,
        inf_vgeom::BuildParams::default(),
    );
    let asset = VgeomAsset::from_mesh(BAKED_ID, &mesh).expect("the DAG indexes");

    // The cube side: the city's 370 468 solids over 1 000 buildings is ~370 each.
    let per_building = baked.report.parts.max(1);
    let side = 32; // 1 024 buildings, the city's own order
    let mut cubes = Vec::with_capacity(side * side * per_building);
    for x in 0..side {
        for z in 0..side {
            for p in 0..per_building {
                let t = p as f64 * 0.37;
                cubes.push(ScatterInstance {
                    position: DVec3::new(
                        x as f64 * 140.0 + (t % 20.0) - 10.0,
                        (t % 7.2) + 0.1,
                        z as f64 * 100.0 + (t % 30.0) - 15.0,
                    ),
                    rotation: Quat::IDENTITY,
                    scale: Vec3::new(2.0, 1.8, 0.25),
                    color: [0.62, 0.60, 0.55, 1.0],
                });
            }
        }
    }
    let cube_count = cubes.len();
    let mut cube_scene = RenderScene {
        scatter: vec![ScatterBatch::lit(
            Arc::new(ScatterData::build(PrimMesh::Cube, DVec3::ZERO, cubes)),
            DVec3::ZERO,
            0.85,
            1,
        )],
        lights: vec![sun()],
        grid_enabled: false,
        ..Default::default()
    };
    cube_scene.mark_dirty();
    let vgeom_scene = vgeom_city(&asset, side as i32);

    let eye = DVec3::new(-120.0, 60.0, -120.0);
    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: eye,
        forward: (DVec3::new(1200.0, 0.0, 900.0) - eye).as_vec3().normalize(),
        up: Vec3::Y,
        fov_y: 70f32.to_radians(),
        near: 0.05,
        width: W,
        height: H,
        ortho: None,
    };
    let target = HeadlessTarget::new(&gpu, W, H);

    let measure = |scene: &RenderScene, vgeom: bool| -> (f64, Vec<(&'static str, f64)>) {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(RenderSettings {
            vgeom: VgeomSettings {
                enabled: vgeom,
                ..VgeomSettings::default()
            },
            ..RenderSettings::default()
        });
        let timed = r.set_gpu_timing(&gpu, true);
        for _ in 0..12 {
            r.render(&gpu, scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let _ = r.gpu_timings(&gpu);
        }
        const N: usize = 30;
        let mut total = 0.0;
        let mut passes: Vec<(&'static str, f64)> = Vec::new();
        for _ in 0..N {
            r.render(&gpu, scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(t) = r.gpu_timings(&gpu) {
                total += t.total_ms;
                if passes.is_empty() {
                    passes = t.passes.iter().map(|p| (p.name, p.ms)).collect();
                } else {
                    for (slot, p) in passes.iter_mut().zip(&t.passes) {
                        slot.1 += p.ms;
                    }
                }
            }
        }
        if !timed {
            passes.clear();
        }
        (
            total / N as f64,
            passes.into_iter().map(|(n, v)| (n, v / N as f64)).collect(),
        )
    };

    let (cube_ms, cube_passes) = measure(&cube_scene, false);
    let (vgeom_ms, vgeom_passes) = measure(&vgeom_scene, true);
    let pass_of = |p: &[(&'static str, f64)], name: &str| {
        p.iter()
            .find(|(n, _)| *n == name)
            .map_or(0.0, |(_, ms)| *ms)
    };

    println!(
        "clause 5 prize on {} ({:?}), {W}x{H}, {} buildings:\n  \
         as PLACEHOLDER CUBES: {cube_count} scatter instances -> GPU frame \
         {cube_ms:.3} ms (scatter pass {:.3} ms)\n  \
         as BAKED VGEOM : {} instances of one {}-meshlet asset -> GPU frame \
         {vgeom_ms:.3} ms (vgeom pass {:.3} ms)\n  \
         delta {:+.3} ms ({:.2}x)",
        info.name,
        info.device_type,
        side * side,
        pass_of(&cube_passes, "scatter"),
        vgeom_scene.vgeom_instances.len(),
        mesh.meshlet_count(),
        pass_of(&vgeom_passes, "vgeom"),
        vgeom_ms - cube_ms,
        cube_ms / vgeom_ms.max(1.0e-9),
    );

    // Unconditional, because neither is a clock: both scenes really carried what
    // they claim to. A comparison between a full frame and an empty one is the
    // easiest way to make a renderer look fast.
    assert!(
        cube_count > 100_000,
        "the cube side drew only {cube_count} instances — this is not the city's \
         order of magnitude"
    );
    assert_eq!(
        vgeom_scene.vgeom_instances.len(),
        side * side,
        "the vgeom side must draw one instance per building and no more"
    );
    let _ = BUILDINGS;
}
