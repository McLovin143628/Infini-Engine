//! **The viewport's virtual-texture invalidation, executed** (P26.5).
//!
//! The P26.4 audit's remainder, verbatim:
//!
//! > **The editor's VT invalidation is pinned by a SOURCE assertion.** The rule
//! > is asserted on bytes in Ring 1 (`EditorRenderAssets::index_generation` vs a
//! > re-imported `.inf_tex`), but the host's *use* of it needs a live
//! > `EngineHost` — a window and a device — so `projector_mirror` pins the call
//! > and nothing executes it. Same class as every other viewport-host claim.
//!
//! # Why the arm is here and not in the host
//!
//! `EngineHost::new` takes a `SurfaceTarget` and calls `create_surface` on it:
//! there is no headless constructor, and `host.rs` is `#[cfg(any(windows,
//! target_os = "macos"))]`, so a test written beside it would also be invisible
//! to the Linux CI leg. Rather than memo that as unreachable, P26.5 **moved the
//! rule**: the two terms the rebuild is gated on are now
//! [`inf_editor_core::render_assets::VtLevelKey`], produced by
//! `EditorRenderAssets::vt_level_key`, and the host's early-out is one `!=` over
//! that value. Ring 1 compiles and tests everywhere, so the decision does too.
//!
//! What is still a source pin is the *call* — `projector_mirror` and the host's
//! own doc — and that is now a pin on one line instead of on a three-term
//! condition. The residual is ledgered in the Phase 26 completion block.
//!
//! # What this asserts
//!
//! Not the counter. The whole sequence the host runs, through the **real**
//! registration door (`inf_render::build_vt_level`, on a real device), with the
//! world asserted at each end: after a re-import that changes a texture's
//! EXTENT, the rebuilt registry's own descriptor carries the new extent. A
//! registry that had kept the first read's bytes would carry the old one, and
//! nothing about the key or the counter would say so.

use std::collections::BTreeMap;

use inf_editor_core::assets::AssetProject;
use inf_editor_core::render_assets::EditorRenderAssets;
use inf_editor_core::scene::SceneDoc;
use inf_render::GpuContext;
use uuid::Uuid;

/// A real v2 tiled container of `n × n`, through the one writer.
fn tiled(n: u32) -> Vec<u8> {
    inf_material::build_tiled_texture(
        vec![180u8; (n * n * 4) as usize],
        n,
        n,
        inf_material::TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: inf_material::TextureCompression::None,
            hdr: false,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

fn gpu_or_skip(what: &str) -> Option<GpuContext> {
    match GpuContext::headless() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("SKIP: no GPU adapter available for {what} ({e})");
            None
        }
    }
}

/// Build the level the host would build from `key`'s bindings, and report the
/// mip-0 extent the registry ended up with — i.e. **which bytes it read**.
fn built_extent(
    gpu: &GpuContext,
    store: &mut EditorRenderAssets,
    bindings: impl IntoIterator<Item = Uuid>,
    albedo: Uuid,
) -> Option<u32> {
    let content = store.material_content(bindings);
    let materials: BTreeMap<u128, inf_render::VtMaterialMaps> = content.materials.clone();
    let (lib, _pools, report) = inf_render::build_vt_level(
        &gpu.device,
        &gpu.queue,
        &inf_render::RenderSettings::default(),
        inf_render::DEFAULT_VT_BUDGET_BYTES,
        &materials,
        |g| content.source(g),
    )?;
    assert_eq!(report.refused, 0, "a bound texture was refused");
    let handle = lib.handle(albedo.as_u128())?;
    let desc = lib.residency().desc(handle)?;
    Some(desc.mips[0].width)
}

#[test]
fn a_reimported_texture_reaches_the_viewports_rebuilt_level() {
    let Some(gpu) = gpu_or_skip("the editor VT invalidation") else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let root = dir.path().to_path_buf();
    let mut proj = AssetProject::open(&root).expect("open project");
    let content_dir = proj.content_dir("Materials").expect("content dir");

    // The RAW-IMAGE door: a v2 container is never written through
    // `inf_asset::encode`, so a fixture using the generic writer is a fixture the
    // reader refuses.
    let first = tiled(128);
    let tex_path = proj
        .unique_asset_path(&content_dir, "Albedo", "inf_tex")
        .expect("path");
    std::fs::write(&tex_path, &first).expect("write texture");
    let albedo = proj
        .register_written_asset(
            tex_path.clone(),
            inf_asset::AssetKind::Texture,
            inf_asset::ContentHash::of(&first),
            None,
            None,
            None,
        )
        .expect("register texture");
    let mat = proj
        .write_asset(
            &content_dir,
            "Wall",
            &inf_material::MaterialAsset {
                base_color_texture: Some(albedo),
                ..Default::default()
            },
            None,
            vec![albedo],
            None,
        )
        .expect("write material");

    // A document that binds it — one cube with `Material.asset`.
    let mut doc = SceneDoc::new();
    let cube = doc.edit_create(inf_editor_core::ipc::SpawnKind::Cube, "Wall", None);
    {
        let world = doc.world_mut();
        let e = world.entity_of(cube).expect("the cube exists");
        world
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::Material {
                asset: Some(mat.uuid()),
                ..Default::default()
            });
    }

    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(root.clone()));

    // ── the key, and what the host does with it ─────────────────────────────
    let key1 = store.vt_level_key(&doc);
    assert_eq!(
        key1.bindings.iter().copied().collect::<Vec<_>>(),
        vec![mat.uuid()],
        "the key did not pick up the document's binding"
    );
    assert!(!key1.is_empty());
    let extent1 = built_extent(&gpu, &mut store, key1.bindings.clone(), albedo.uuid())
        .expect("the level builds");
    assert_eq!(extent1, 128, "the registry read the wrong container");

    // The IDEMPOTENT case: nothing changed, so the key is equal and the host's
    // early-out fires. This is the half that makes the rebuild affordable at
    // all, and a key that changed every projection would rebuild an atlas per
    // gizmo drag.
    assert_eq!(
        store.vt_level_key(&doc),
        key1,
        "the key moved with nothing changed — the viewport would rebuild its \
         atlas on every document version"
    );

    // ── the re-import ───────────────────────────────────────────────────────
    let second = tiled(64);
    assert_ne!(first, second, "the fixture re-imported the same bytes");
    std::fs::write(&tex_path, &second).expect("re-import");
    store.refresh_index();

    let key2 = store.vt_level_key(&doc);
    assert_eq!(
        key2.bindings, key1.bindings,
        "the re-import changed the BINDING set, so this arm would pass on the \
         pre-P26.4 rule and prove nothing"
    );
    assert_ne!(
        key2, key1,
        "a re-imported .inf_tex left the key unchanged — the viewport's early-out \
         would fire and it would hold the bytes it read the first time for the \
         rest of the session"
    );

    // ── THE WORLD: the rebuilt registry read the NEW file ───────────────────
    let extent2 = built_extent(&gpu, &mut store, key2.bindings.clone(), albedo.uuid())
        .expect("the level rebuilds");
    assert_eq!(
        extent2, 64,
        "the rebuilt level's registry still describes a {extent1}-texel texture, \
         so it re-read nothing — asserted on the registry's own descriptor rather \
         than on the counter, because a monotone number that tracks nothing is \
         exactly as useless as no number"
    );

    // ── and unbinding is the third event ────────────────────────────────────
    {
        let world = doc.world_mut();
        let e = world.entity_of(cube).expect("the cube exists");
        world
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::Material::default());
    }
    let key3 = store.vt_level_key(&doc);
    assert!(
        key3.is_empty(),
        "unbinding the material left a binding in the key"
    );
    assert_ne!(key3, key2);
    // …and an empty key is the textureless path, which is what the host turns
    // into `set_vt_level(None)` and what all 50 goldens record.
    assert!(store.material_content(key3.bindings.clone()).is_empty());
}

/// **The key is a pure function of the document and the index generation.**
///
/// Two documents built in opposite orders produce the same key, because the
/// bindings are a `BTreeSet` — which is not a tidiness choice: the same set is
/// what `material_content` is handed, and `inf_render::registration_order`'s
/// purity in the registration SEQUENCE is the property both hosts' residency
/// rests on (the P26.3b audit's `HashMap`-walk finding, which produced two page
/// sets for one level on two machines).
#[test]
fn the_key_is_a_function_of_the_set_and_not_of_the_order() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mut store = EditorRenderAssets::new();
    store.set_content_root(Some(dir.path().to_path_buf()));

    let ids: Vec<Uuid> = (0..6).map(|i| Uuid::from_u128(0x2605_0000 + i)).collect();
    let build = |order: &[Uuid]| {
        let mut doc = SceneDoc::new();
        for (i, id) in order.iter().enumerate() {
            let e = doc.edit_create(
                inf_editor_core::ipc::SpawnKind::Cube,
                &format!("Wall{i}"),
                None,
            );
            let world = doc.world_mut();
            let entity = world.entity_of(e).expect("the cube exists");
            world
                .world_mut()
                .entity_mut(entity)
                .insert(inf_ecs::components::Material {
                    asset: Some(*id),
                    ..Default::default()
                });
        }
        doc
    };
    let forward = build(&ids);
    let mut backward: Vec<Uuid> = ids.clone();
    backward.reverse();
    let reversed = build(&backward);

    let a = store.vt_level_key(&forward);
    let b = store.vt_level_key(&reversed);
    assert_eq!(a, b, "the key depends on document order");
    assert_eq!(a.bindings.len(), ids.len());
    // ANTI-VACUITY: the two documents really were built in opposite orders, so
    // the equality is a statement about the set rather than about a constant.
    assert_ne!(ids, backward);
    // …and a NIL-bound material is a binding like any other — the cook names it
    // in `dangling_material_advisory` and every runtime path falls to the scalar
    // surface, so the key must not quietly drop it and make the two hosts
    // disagree about what was asked for.
    let nil = build(&[Uuid::nil()]);
    assert_eq!(store.vt_level_key(&nil).bindings.len(), 1);
}
