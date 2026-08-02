//! P13.1b player integration: the runtime resolves a mesh's cook-derived
//! `.inf_vmesh` from a pack and the GPU meshlet path **activates** (the cull
//! compute reports visible meshlets > 0), plus a drift guard on the derived-id
//! salt shared with the cook. GPU parts skip cleanly with no adapter.

use glam::{DVec3, Quat, Vec3};
use uuid::Uuid;

use inf_asset::{AssetId, AssetKind, PackReader, PackWriter};
use inf_math::FloatingOrigin;
use inf_player::vmesh::{derived_vmesh_id, pack_has_vmesh, VmeshRegistry};
use inf_render::{cull_visible_source, GpuContext, RenderView, VgeomInstance, VgeomSettings};

// A dense procedural mesh (subdivided displaced grid) → a meshlet DAG, exactly as
// the cook's `derive_vmesh` builds one from a mesh's streams. One shared generator
// with the renderer's vgeom suites (`inf_vgeom::test_support`), with bit-portable
// trig — this file used to carry its own copy, written with the `0.3 · 3.0` normal
// constants folded by hand, which is precisely how nine copies drift.
use inf_vgeom::test_support::dense_grid_mesh as dense_vmesh;

/// The player's derived-id salt must match the cook's, or the runtime would never
/// find a cooked vmesh. Drift guard against `inf_packager::derived_vmesh_id`.
#[test]
fn derived_vmesh_id_matches_cook() {
    for raw in [0x1u128, 0xDEAD_BEEF, 0xFEED_FACE_CAFE_0001] {
        let mesh_id = Uuid::from_u128(raw);
        let player = derived_vmesh_id(mesh_id);
        let cook = inf_packager::derived_vmesh_id(AssetId(mesh_id)).uuid();
        assert_eq!(player, cook, "player/cook derived-vmesh-id salt drifted");
    }
}

/// Cook (mirror) a dense-mesh vmesh into a pack, load it with the runtime
/// registry, pick it with vgeom **on**, and assert the GPU meshlet path activates
/// (visible meshlets > 0 via the cull readback). Skips with no GPU adapter.
#[test]
fn vgeom_path_activates_from_pack() {
    // 1. Build the derived vmesh and write it into a pack under the cook's
    //    derived id (what `cook` does for a dense `.inf_mesh`).
    let mesh = dense_vmesh(40);
    let mesh_id = Uuid::from_u128(0x00C0_FFEE);
    let vmesh_id = derived_vmesh_id(mesh_id);

    let mut writer = PackWriter::new();
    let bytes = inf_asset::encode(&mesh).expect("encode vmesh");
    writer
        .add_bytes(AssetId(vmesh_id), AssetKind::MeshletMesh, &bytes)
        .expect("add vmesh");

    let dir = tempfile::tempdir().unwrap();
    let pack_path = dir.path().join("content.inf_pack");
    writer.write_to_file(&pack_path).expect("write pack");

    // 2. Load the runtime registry from the pack + the pick rule.
    let reader = PackReader::open(&pack_path).expect("open pack");
    assert!(
        pack_has_vmesh(&reader, mesh_id),
        "pack should carry the vmesh"
    );
    let reg = VmeshRegistry::from_pack(std::sync::Arc::new(reader)).expect("registry");
    assert_eq!(reg.len(), 1);

    assert!(
        reg.pick(mesh_id, false).is_none(),
        "vgeom disabled ⇒ classic path (no vmesh)"
    );
    let (asset_id, picked) = reg.pick(mesh_id, true).expect("vgeom on ⇒ vmesh picked");
    assert_eq!(asset_id, vmesh_id.as_u128());

    // 3. GPU: the meshlet path activates (cull reports visible meshlets > 0).
    let gpu = match GpuContext::headless() {
        Ok(g) => g,
        Err(e) => {
            eprintln!("SKIP vgeom activation: no GPU adapter ({e})");
            return;
        }
    };
    let view = RenderView {
        origin: FloatingOrigin::new(DVec3::ZERO),
        eye_world: DVec3::new(0.0, 3.0, 4.5),
        forward: (DVec3::ZERO - DVec3::new(0.0, 3.0, 4.5))
            .as_vec3()
            .normalize(),
        up: Vec3::Y,
        fov_y: 60f32.to_radians(),
        near: 0.05,
        width: 320,
        height: 180,
        ortho: None,
    };
    let inst = VgeomInstance::lit(
        asset_id,
        DVec3::ZERO,
        Quat::IDENTITY,
        Vec3::splat(2.0),
        [0.7, 0.6, 0.4, 1.0],
        1,
    );
    let settings = VgeomSettings {
        enabled: true,
        ..VgeomSettings::default()
    };
    let visible =
        cull_visible_source(&gpu, &picked, std::slice::from_ref(&inst), &view, &settings).pairs;
    assert!(
        !visible.is_empty(),
        "vgeom path activated: expected visible meshlets > 0"
    );
    eprintln!(
        "vgeom activation: {} of {} meshlets visible",
        visible.len(),
        picked.meshlet_count()
    );
}
