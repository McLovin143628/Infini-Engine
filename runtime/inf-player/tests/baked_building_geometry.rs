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
    RenderSettings, RenderView, ScatterAudit, ScatterBatch, ScatterData, ScatterInstance,
    VgeomAsset, VgeomAudit, VgeomInstance, VgeomSettings, HEADLESS_FORMAT,
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

    // **The asset really indexed the DAG.** `assert_eq!(asset.id, BAKED_ID)` was
    // the first version of this line, and the I4 audit measured it: `from_mesh`
    // assigns the id it is given straight into the field, so it asserted a struct
    // literal against itself. What is worth asserting is that the packed source
    // is the mesh — the bounding sphere is read out of the written `.inf_vmesh`
    // header, so a radius that survives a round trip through the image is a
    // statement about the pack rather than about an argument.
    let (_, radius) = asset.bounds();
    println!("clause 5 path: the packed asset's bounding radius is {radius:.3} m");
    assert!(
        radius.is_finite() && radius > 1.0,
        "the packed `.inf_vmesh` header reports a {radius} m bounding radius for a \
         20 x 30 m building — the image the renderer pages from is not this mesh"
    );

    // **What this scene is and is not.** `vgeom_city` builds a `RenderScene` by
    // hand with `vgeom_instances` and nothing else, so "zero placeholder scatter
    // batches" here is a statement about a `Vec` the fixture never pushed to —
    // the I4 audit's finding, and the reason the claim is now written as what it
    // is. The load-bearing half of clause 5 is the PROJECTION, and the projection
    // still emits placeholder cubes for every PCG structure (`push_pcg_scatter`,
    // private to both hosts): that is the gap the wave carries and the cook-side
    // evaluation is the blocker on. What this arm can honestly say is that the
    // *renderer's* side of the path takes the baked asset end to end.
    //
    // Three assertions were deleted here rather than kept green: `asset.id ==
    // BAKED_ID`, `vgeom_assets.len() == 1` and `every instance names BAKED_ID`
    // are all identities of the two lines of fixture that produced both sides.
    // A deleted tautology is better than a green one, because a green one reads
    // like evidence in a ledger.
    let scene = vgeom_city(&asset, 8);
    assert_eq!(
        scene.vgeom_instances.len(),
        64,
        "the fixture must place one instance per lot before anything reads them"
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
/// replace 370 468 cubes with 1 000 meshes. The GPU scatter path culls per
/// instance in a compute pass and draws indirect; the meshlet path pays a DAG
/// traversal, a two-pass HZB and a far larger triangle count (5 088 per building
/// against a cube's 12).
///
/// # THE COMPARISON HAS TO BE OF ONE THING (the I4 audit)
///
/// The first version of this test ran the cube side at the **shipped** scatter
/// bands — `cull_distance_m` 400 m, `mesh_distance_m` 120 m — against a vgeom
/// side with **no distance cull at all**, over a 4 480 × 3 200 m district. Nearly
/// every cube was thrown away in the cull compute while every building was
/// rasterized out to five kilometres, and the delta it printed (+1.416 ms
/// "dearer") was a delta between two culling policies. Its own anti-vacuity
/// clause counted `Vec::len()` on the CPU, which a cull cannot see.
///
/// So the cube side is measured twice — as it ships, and with its bands opened
/// past the district — and every row carries the **audits**: how many instances
/// the cull saw, how many it threw past the cull distance, how many it drew, and
/// how many meshlet pairs the vgeom side actually rasterized. The delta against
/// the comparable row is the one a fidelity decision may be taken on.
///
/// # …and a mean of thirty frames was not a measurement either
///
/// The first version took one round's mean. Run four times on byte-identical
/// code it answered **−2.03, −0.28, +2.00 and −0.86 ms** — the *sign* of the
/// conclusion changed twice. The round spread this file now prints says why: on
/// this card a 30-frame mean of the same scene lands anywhere between **0.54 and
/// 5.76 ms**. With `fps_instrument`'s MIN-of-rounds discipline the same three
/// numbers reproduce to three decimals across runs:
///
/// ```text
/// cubes at the shipped bands (nearly all distance-culled)  0.539 ms
/// cubes with the bands opened past the district            1.25  ms
/// 1 024 baked vgeom buildings (146 067 meshlet pairs)      1.573 ms
/// -> +1.034 ms against the shipped bands, +0.32 ms against the comparable ones
/// ```
///
/// Real geometry is therefore **dearer, and by about a third of a millisecond**,
/// not the 1.416 ms the wave's first write-up recorded — that figure was one
/// unstable sample of a comparison between two culling policies. The direction
/// survives; the magnitude does not.
///
/// Reported and never asserted: they are wall clocks on one machine, and §8
/// budgets are tripwires rather than hardware claims. What IS asserted is that
/// both sides drew — which is what the audits are for.
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

    // **A DEVICE THAT CANNOT TIME AN ENCODER SEGMENT MUST NOT BE ASKED TO.**
    //
    // `set_gpu_timing` returns `false` on a software rasterizer and most
    // paravirtual runners, and `gpu_timings` then returns `None` for ever — so
    // the accumulation below never runs, `total` stays `0.0`, and this test used
    // to print `0.000 ms` against `0.000 ms` with a `+0.000 ms` delta and pass.
    // `crates/inf-render/tests/gpu_timing.rs` takes exactly this guard, in the
    // crate this file depends on; the I4 audit found it missing here. A harness
    // that silently believed it had per-pass numbers would print zeros as
    // measurements.
    {
        let mut probe = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        if !probe.set_gpu_timing(&gpu, true) {
            eprintln!(
                "SKIP baked_building_geometry cost: {} ({:?}) cannot time an \
                 encoder segment, and a comparison of two zeros is not a price",
                info.name, info.device_type
            );
            return;
        }
    }

    #[allow(clippy::type_complexity)]
    let measure = |scene: &RenderScene,
                   settings: RenderSettings|
     -> (
        f64,
        (f64, f64),
        Vec<(&'static str, f64)>,
        ScatterAudit,
        VgeomAudit,
    ) {
        let mut r = EngineRenderer::new(&gpu, HEADLESS_FORMAT);
        r.set_settings(settings);
        r.set_scatter_audit(true);
        r.set_vgeom_audit(true);
        let timed = r.set_gpu_timing(&gpu, true);
        for _ in 0..12 {
            r.render(&gpu, scene, &view, &target.view, (W, H));
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let _ = r.gpu_timings(&gpu);
        }
        // **MIN OF ROUNDS**, `fps_instrument`'s discipline, because a mean of one
        // round of thirty frames does not survive this card's boost clocks: the
        // I4 audit ran the original single-mean version four times and the sign of
        // the answer changed twice (−2.03, −0.28, +2.00, −0.86 ms on byte-identical
        // code). A round's slow half is a statement about the machine.
        const N: usize = 30;
        const ROUNDS: usize = 5;
        let mut best = f64::INFINITY;
        let mut best_passes: Vec<(&'static str, f64)> = Vec::new();
        let mut spread = (f64::INFINITY, 0.0f64);
        for _ in 0..ROUNDS {
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
            let mean = total / N as f64;
            spread = (spread.0.min(mean), spread.1.max(mean));
            if mean < best {
                best = mean;
                best_passes = passes.into_iter().map(|(n, v)| (n, v / N as f64)).collect();
            }
        }
        assert!(timed, "the probe above said this device can time an encoder segment and this renderer says it cannot");
        (
            best,
            spread,
            best_passes,
            r.scatter_audit(&gpu),
            r.vgeom_audit(&gpu),
        )
    };

    let classic = RenderSettings {
        vgeom: VgeomSettings {
            enabled: false,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    };
    let meshlets = RenderSettings {
        vgeom: VgeomSettings {
            enabled: true,
            ..VgeomSettings::default()
        },
        ..RenderSettings::default()
    };
    // **THE CULL POLICIES ARE NOT THE SAME, AND THE FIRST COMPARISON DID NOT SAY
    // SO** (the I4 audit). `ScatterSettings::default().cull_distance_m` is 400 m
    // and `mesh_distance_m` is 120 m, while `VgeomSettings` has **no distance
    // cull at all** — only screen-space error, frustum, cone and HZB. The
    // district below is 4 480 x 3 200 m, so at the shipped defaults the cube side
    // throws nearly every instance away in the cull compute while the vgeom side
    // rasterizes all 1 024 buildings out to five kilometres. A delta between
    // those two is a delta between two policies, not between two geometries.
    //
    // So the cube side is measured **twice**: once as it ships, and once with its
    // bands opened past the district so both sides draw the same city. The audits
    // are printed beside every row, because "did this side draw anything" is the
    // question a wall clock cannot answer.
    let far = 12_000.0;
    let mut wide = classic;
    wide.scatter.cull_distance_m = far;
    wide.scatter.mesh_distance_m = far;

    let (cube_ms, cube_spread, cube_passes, cube_audit, _) = measure(&cube_scene, classic);
    let (wide_ms, wide_spread, wide_passes, wide_audit, _) = measure(&cube_scene, wide);
    let (vgeom_ms, vgeom_spread, vgeom_passes, _, vgeom_audit) = measure(&vgeom_scene, meshlets);
    let pass_of = |p: &[(&'static str, f64)], name: &str| {
        p.iter()
            .find(|(n, _)| *n == name)
            .map_or(0.0, |(_, ms)| *ms)
    };

    println!(
        "clause 5 price on {} ({:?}), {W}x{H}, {} buildings over a {:.0} x {:.0} m \
         district:\n  \
         CUBES as shipped (cull {} m / mesh {} m): {cube_count} instances -> GPU \
         frame {cube_ms:.3} ms (scatter pass {:.3} ms); cull saw {} candidates, \
         threw {} past the cull distance, drew {} as meshes and {} as impostors\n  \
         CUBES with the bands opened to {far:.0} m: GPU frame {wide_ms:.3} ms \
         (scatter pass {:.3} ms); cull saw {} candidates, threw {} past the cull \
         distance, drew {} as meshes and {} as impostors\n  \
         BAKED VGEOM: {} instances of one {}-meshlet asset -> GPU frame \
         {vgeom_ms:.3} ms (vgeom pass {:.3} ms); base cut {} pairs, {} occluded, \
         {} drawn early + {} late\n  \
         round spreads (MIN of 5 x 30 frames): cubes {:.3}..{:.3}, wide \
         {:.3}..{:.3}, vgeom {:.3}..{:.3} ms\n  \
         delta against the SHIPPED cube bands {:+.3} ms; against the COMPARABLE \
         ones {:+.3} ms",
        info.name,
        info.device_type,
        side * side,
        side as f64 * 140.0,
        side as f64 * 100.0,
        classic.scatter.cull_distance_m,
        classic.scatter.mesh_distance_m,
        pass_of(&cube_passes, "scatter"),
        cube_audit.candidates,
        cube_audit.distance_culled,
        cube_audit.mesh,
        cube_audit.impostor,
        pass_of(&wide_passes, "scatter"),
        wide_audit.candidates,
        wide_audit.distance_culled,
        wide_audit.mesh,
        wide_audit.impostor,
        vgeom_scene.vgeom_instances.len(),
        mesh.meshlet_count(),
        pass_of(&vgeom_passes, "vgeom"),
        vgeom_audit.base_cut,
        vgeom_audit.occluded,
        vgeom_audit.early_drawn,
        vgeom_audit.late_drawn,
        cube_spread.0,
        cube_spread.1,
        wide_spread.0,
        wide_spread.1,
        vgeom_spread.0,
        vgeom_spread.1,
        vgeom_ms - cube_ms,
        vgeom_ms - wide_ms,
    );

    // Unconditional, because none of these is a clock: **both sides really drew**.
    // The first version of this block asserted `Vec::len()` on the CPU and called
    // it "both scenes really carried what they claim to" — which is exactly the
    // "a comparison between a full frame and an empty one" hazard it named, since
    // a `Vec` of half a million instances that the cull throws away costs a frame
    // nothing. The audits are what answer the question.
    assert!(
        cube_count > 100_000,
        "the cube side offered only {cube_count} instances — this is not the \
         city's order of magnitude"
    );
    assert_eq!(
        cube_audit.candidates as usize, cube_count,
        "the scatter cull saw {} of the {cube_count} instances the scene carries \
         — the cube side is not the scene this test built",
        cube_audit.candidates
    );
    assert!(
        wide_audit.mesh > cube_audit.mesh * 10,
        "opening the cube bands to {far:.0} m took the drawn-as-mesh count from \
         {} to {} — the shipped bands were not the reason the cube side is cheap, \
         so the comparable row above is measuring something else",
        cube_audit.mesh,
        wide_audit.mesh
    );
    assert!(
        vgeom_audit.early_drawn + vgeom_audit.late_drawn > 0,
        "the vgeom side drew ZERO meshlet pairs ({} base cut, {} occluded) — its \
         milliseconds are the cost of an empty frame",
        vgeom_audit.base_cut,
        vgeom_audit.occluded
    );
    assert_eq!(
        vgeom_scene.vgeom_instances.len(),
        side * side,
        "the vgeom side must carry one instance per building and no more"
    );
    let _ = BUILDINGS;
}
