//! **The Phase 26 gate — the wire half (P26.3b).**
//!
//! P26.1 built the tiled container, P26.2 the pool and the residency, P26.3 the
//! WGSL sample — and the P26.3 ledger closed with the honest remainder that
//! spec clause 4 was **not** done: *".inf_mat texture references do not yet
//! resolve in the viewport or the player. Both projectors fill
//! `vt: Default::default()` … what is missing is the persisted binding, which
//! needs a scene schema bump (v21 → v22) and pack dependency edges so a cooked
//! pack carries the `.inf_tex` payloads a level names."*
//!
//! This file holds the arms for that wire. They are deliberately about **bytes
//! and worlds**, not pixels: what P26.3b built is the path from an authored
//! `.inf_mat` to a runtime record, along routes that must not disagree.
//!
//! * **(a) ONE DOOR.** The `.inf_matd` bytes a cooked pack carries are
//!   byte-identical to the ones the PIE payload carries, for the same project.
//!   The P22.2 law made executable: the cook and the payload builder both call
//!   `inf_material::derive_material_bytes` and are compared on their OUTPUT
//!   rather than trusted on their comment. (`fracture_equivalence.rs` is the
//!   precedent, one asset kind over.) **Two producers, not three** — the P26.3b
//!   audit's count: the editor viewport resolves no material at all yet and
//!   `render_assets` does not call the door, so the third path is P26.4's.
//! * **(b) The dependency edges are real.** A level binds a material, the
//!   material names a texture, and the cooked pack contains both — at exact
//!   counts, because `!is_empty()` would pass on a pack that lost the second of
//!   two.
//! * **(c) The payload carries the garment and the hairstyle.** The P24.4 debt.
//!   `phase24_gate::the_pie_payload_carries_no_garment_and_that_is_measured` was
//!   written to fail the day `ScenePayload` grew `cloths`; it fired, it is
//!   retired, and this is the positive arm that replaces its source read.
//! * **(d) PIE == shipping on a textured, clothed scene**, with the anti-vacuity
//!   control the P24.4 mutation matrix earned: the same level with nothing bound
//!   must fold a **different** trace, or the equality above is a statement about
//!   two empty worlds agreeing. A **real `--pie` subprocess** folds it too, since
//!   a boot path that drops an attachment does not crash, it agrees with itself
//!   (P21.4).
//! * **(e) The two hosts build the same material content** (P26.3b audit) — the
//!   maps a host binds and pages FROM, not just the bytes on the wire, and the
//!   registration order that the residency is a pure function of.
//! * **(f) Every silent material hazard raises its advisory** (P26.3b audit): a
//!   missing material, a **material instance**, a missing `.inf_tex` and a **v1**
//!   `.inf_tex` — four fixtures, plus the healthy control.
//!
//! The fixture carries an **unbound** material and texture on purpose, reached
//! through a mesh's own sidecar edges the way a glTF import writes them. Without
//! it (e) cannot tell a walk of the pack index from a walk of the level's
//! bindings, which is exactly the defect it found.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Duration;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash, PackReader};
use inf_ecs::components::{ClothSim, HairGuides, Material, SkeletalMesh};
use inf_editor_core::pie::PieSession;
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_material::{MatBlend, MaterialAsset, TextureCompression, TextureImportSettings};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;
use inf_runtime::pie::PlayerToEditor;
use uuid::Uuid;

// ── the fixture's stable ids ────────────────────────────────────────────────

const LEVEL: Uuid = Uuid::from_u128(0x2603_0000_0000_0001);
const MAT: Uuid = Uuid::from_u128(0x2603_0000_0000_0002);
const ALBEDO: Uuid = Uuid::from_u128(0x2603_0000_0000_0003);
const ORM: Uuid = Uuid::from_u128(0x2603_0000_0000_0004);
const CLOTH: Uuid = Uuid::from_u128(0x2603_0000_0000_0005);
const HAIR: Uuid = Uuid::from_u128(0x2603_0000_0000_0006);
/// A mesh whose SIDECAR depends on a material nothing in the level binds — the
/// ordinary glTF-import shape (P26.3b audit). `inf_editor_core::assets::import`
/// writes exactly these edges: mesh → its materials → their textures, and none
/// of them is a `Material.asset` binding.
const DECOR_MESH: Uuid = Uuid::from_u128(0x2603_0000_0000_0007);
const DECOR_MAT: Uuid = Uuid::from_u128(0x2603_0000_0000_0008);
const DECOR_TEX: Uuid = Uuid::from_u128(0x2603_0000_0000_0009);

/// The pack must hold exactly these, and the counts are exact rather than
/// non-zero for the P21.4 reason: a walk that lost the second texture of two
/// still satisfies `!is_empty()`.
///
/// **Two materials and three textures, and only ONE of each pair is bound.** The
/// unbound one arrives through the mesh's sidecar edges, which is how a real
/// project gets most of its materials — and it is what makes "the two hosts
/// build the same `MaterialContent`" a claim that can fail (P26.3b audit: the
/// pack side walked the pack INDEX and therefore saw both).
const EXPECTED_MATERIALS_IN_PACK: usize = 2;
const EXPECTED_DERIVED_MATERIALS_IN_PACK: usize = 2;
const EXPECTED_TEXTURES_IN_PACK: usize = 3;
/// …of which the LEVEL binds exactly one material and two textures.
const BOUND_MATERIALS: usize = 1;
const BOUND_TEXTURES: usize = 2;

// ── the fixture ─────────────────────────────────────────────────────────────

/// A `.inf_tex` **v2 tiled container** with a left-to-right red ramp, through
/// `inf_material::build_tiled_texture` — the one writer, so the bytes the pack
/// carries are the bytes a runtime pages.
fn texture_bytes(n: u32, tint: u8) -> Vec<u8> {
    let mut rgba = Vec::with_capacity((n * n * 4) as usize);
    for _y in 0..n {
        for x in 0..n {
            rgba.extend_from_slice(&[(x * 255 / (n - 1)) as u8, tint, 200, 255]);
        }
    }
    inf_material::build_tiled_texture(
        rgba,
        n,
        n,
        TextureImportSettings {
            srgb: true,
            generate_mips: true,
            compression: TextureCompression::None,
        },
    )
    .expect("the fixture tiles")
    .into_bytes()
}

/// The authored material: two texture slots bound, the third deliberately empty,
/// so `texture_dependencies()`'s slot order is exercised with a hole in it.
fn material() -> MaterialAsset {
    MaterialAsset {
        base_color: [0.9, 0.4, 0.2, 1.0],
        metallic: 0.25,
        roughness: 0.75,
        base_color_texture: Some(AssetId(ALBEDO)),
        normal_texture: None,
        metallic_roughness_texture: Some(AssetId(ORM)),
        blend: MatBlend::Masked,
        alpha_cutoff: 0.375,
        ..Default::default()
    }
}

/// The level: one materialed, **bound** cube plus one character wearing a
/// garment and a hairstyle. `bound` is the anti-vacuity switch — `false` leaves
/// every component authored and every reference `None`, which is the same level
/// with nothing to resolve.
fn doc_with_binding(bound: bool) -> SceneDoc {
    let mut doc = SceneDoc::new();
    let cube = doc.edit_create(inf_editor_core::ipc::SpawnKind::Cube, "Wall", None);
    let hero = doc.edit_create(inf_editor_core::ipc::SpawnKind::Empty, "Hero", None);
    {
        let world = doc.world_mut();
        let e = world.entity_of(cube).expect("the cube exists");
        // A real `.inf_mesh` reference, ALWAYS (not gated on `bound`): its
        // sidecar drags a material and a texture into the closure that nothing
        // binds, which is how a project gets most of its materials and what makes
        // "the two hosts build the same MaterialContent" falsifiable.
        world
            .world_mut()
            .entity_mut(e)
            .insert(inf_ecs::components::MeshRef {
                asset: Some(DECOR_MESH),
                ..Default::default()
            });
        world.world_mut().entity_mut(e).insert(Material {
            base_color: inf_ecs::math::Color::new(0.9, 0.4, 0.2, 1.0),
            metallic: 0.25,
            roughness: 0.75,
            // P26.3b: the persisted binding scene v22 added. `None` is the
            // scalars-only surface — which is exactly what the control builds.
            asset: bound.then_some(MAT),
            ..Default::default()
        });
        let h = world.entity_of(hero).expect("the hero exists");
        world
            .world_mut()
            .entity_mut(h)
            .insert(SkeletalMesh::default());
        world.world_mut().entity_mut(h).insert(ClothSim {
            asset: bound.then_some(CLOTH),
            enabled: true,
            quality: 1,
        });
        world.world_mut().entity_mut(h).insert(HairGuides {
            asset: bound.then_some(HAIR),
            enabled: true,
            quality: 1,
        });
    }
    doc
}

/// Write `payload` under `content` with a sidecar, so the asset database finds
/// it at the id the level names.
fn put(content: &Path, file: &str, guid: Uuid, bytes: &[u8], kind: AssetKind) {
    put_with_deps(content, file, guid, bytes, kind, &[]);
}

/// [`put`] with explicit sidecar dependency edges — what the glTF importer writes
/// (a mesh names its materials, a material names its textures) and what
/// `dependency_closure` follows through `AssetDb::references_of`.
fn put_with_deps(
    content: &Path,
    file: &str,
    guid: Uuid,
    bytes: &[u8],
    kind: AssetKind,
    deps: &[Uuid],
) {
    let path = content.join(file);
    std::fs::write(&path, bytes).expect("write asset");
    let mut side = AssetSidecar::new(AssetId(guid), kind, ContentHash::of(bytes));
    side.dependencies = deps.iter().copied().map(AssetId).collect();
    side.save(&path).expect("write sidecar");
}

/// A project on disk: the level plus every asset it names.
fn scaffold(tmp: &Path, bound: bool) -> (PathBuf, SceneDoc) {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 26 Wire", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let doc = doc_with_binding(bound);
    let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode level");
    put(&content, "Wire.inf_lvl", LEVEL, &level, AssetKind::Level);
    put(
        &content,
        "Wall.inf_mat",
        MAT,
        &inf_asset::encode(&material()).expect("encode material"),
        AssetKind::Material,
    );
    put(
        &content,
        "Albedo.inf_tex",
        ALBEDO,
        &texture_bytes(128, 40),
        AssetKind::Texture,
    );
    put(
        &content,
        "Orm.inf_tex",
        ORM,
        &texture_bytes(128, 90),
        AssetKind::Texture,
    );
    // The glTF-import shape: a mesh whose sidecar names a material, whose sidecar
    // names a texture — and no `Material.asset` binding anywhere near them.
    let (decor_mesh, _) =
        inf_dcc::to_mesh_asset(&inf_dcc::cube(0.5), &inf_dcc::ExportOptions::default());
    put_with_deps(
        &content,
        "Decor.inf_mesh",
        DECOR_MESH,
        &inf_asset::encode(&decor_mesh).expect("encode mesh"),
        AssetKind::Mesh,
        &[DECOR_MAT],
    );
    put_with_deps(
        &content,
        "Decor.inf_mat",
        DECOR_MAT,
        &inf_asset::encode(&MaterialAsset {
            base_color_texture: Some(AssetId(DECOR_TEX)),
            ..Default::default()
        })
        .expect("encode decor material"),
        AssetKind::Material,
        &[DECOR_TEX],
    );
    put(
        &content,
        "Decor.inf_tex",
        DECOR_TEX,
        &texture_bytes(128, 200),
        AssetKind::Texture,
    );
    put(
        &content,
        "Coat.inf_cloth",
        CLOTH,
        &inf_asset::encode(&garment()).expect("encode cloth"),
        AssetKind::Cloth,
    );
    put(
        &content,
        "Mane.inf_hair",
        HAIR,
        &inf_asset::encode(&hairstyle()).expect("encode hair"),
        AssetKind::Hair,
    );
    (proj, doc)
}

/// A garment through the Model Editor's own door (`inf_editor_core::groom`), so
/// the bytes on the wire are bytes an author could have made.
fn garment() -> inf_anim::ClothAsset {
    let mesh = inf_dcc::plane(1.0);
    let mut sel = inf_dcc::SelectionSet::new(0);
    for v in mesh.vert_ids().take(2) {
        sel.set_vert(v, true);
    }
    let (asset, report) = inf_editor_core::groom::garment_from_session(
        &mesh,
        &sel,
        *Uuid::from_u128(0x2603_0000_0000_00A1).as_bytes(),
        inf_editor_core::groom::GarmentSpec::default(),
        None,
    )
    .expect("the plane is a garment");
    assert!(report.pinned > 0, "the fixture must pin something");
    asset
}

/// The hairstyle twin.
fn hairstyle() -> inf_anim::HairAsset {
    let mesh = inf_dcc::plane(1.0);
    let mut sel = inf_dcc::SelectionSet::new(0);
    for f in mesh.face_ids() {
        sel.set_face(f, true);
    }
    let (asset, report) = inf_editor_core::groom::groom_from_session(
        &mesh,
        &sel,
        *Uuid::from_u128(0x2603_0000_0000_00A2).as_bytes(),
        inf_editor_core::groom::GroomSpec {
            length_m: 0.3,
            segments: 4,
            ..Default::default()
        },
        None,
    )
    .expect("the plane grows guides");
    assert!(report.strands > 0, "the fixture must grow strands");
    asset
}

fn player_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_inf-player"))
}

fn cook_opts() -> CookOptions {
    CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// The PIE payload for `doc`, with every one of the fixture's assets served from
/// the project on disk — the shape the Ring-2 command builds, minus Tauri.
fn payload_for(proj: &Path, doc: &SceneDoc) -> inf_runtime::pie::ScenePayload {
    // Indexed by walking the sidecars directly rather than through
    // `render_assets::content_paths_by_guid`, whose `INDEXED_EXTENSIONS` list is
    // the *render* store's four kinds and does not include `.inf_mat` /
    // `.inf_tex` / `.inf_cloth` / `.inf_hair`. Using it here would have made
    // every arm below fail for a reason that has nothing to do with the wire.
    let content = proj.join("Content");
    let mut by_guid: HashMap<Uuid, PathBuf> = HashMap::new();
    for e in std::fs::read_dir(&content).expect("content dir") {
        let p = e.expect("dir entry").path();
        if let Ok(side) = AssetSidecar::load(&p) {
            by_guid.insert(side.guid.uuid(), p);
        }
    }
    let read = move |g: Uuid| by_guid.get(&g).and_then(|p| std::fs::read(p).ok());
    inf_editor_core::pie::build_scene_payload(
        doc,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        read,
        0,
        false,
    )
    .expect("the payload builds")
}

// ── (a) ONE DOOR, three paths ───────────────────────────────────────────────

/// **The `.inf_matd` a pack carries and the one a payload carries are the same
/// BYTES** (P22.2's law, executable).
///
/// The cook and the PIE payload builder are separate code walking separate
/// representations — a `RuntimeLevel` decoded off disk and a live `SceneDoc` —
/// and P22.2's finding was that exactly this arrangement, with comments claiming
/// agreement "by construction", did not agree: one walked archetype order and
/// the other document order, and one skipped a refusal.
///
/// So they are compared on their output, not trusted on their comment. Both call
/// `inf_material::derive_material_bytes`; a second flattening on either side
/// fails here, and so does a divergent id derivation, because the KEY is
/// compared too.
#[test]
fn the_pack_and_the_payload_derive_the_same_material_bytes() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let derived_id = inf_asset::derived_material_id(AssetId(MAT));
    let reader = PackReader::open(&report.pack_path).expect("open pack");
    let from_pack = reader
        .read(derived_id)
        .expect("the pack carries the record");

    let payload = payload_for(&proj, &doc);
    assert_eq!(
        payload.materials.len(),
        1,
        "the payload carries no derived material — the comparison below would be \
         about nothing"
    );
    let (payload_key, from_payload) = payload.materials[0].clone();
    assert_eq!(
        payload_key,
        derived_id.uuid(),
        "the payload keys its material differently from the pack, so a runtime \
         would need two lookup rules"
    );
    assert_eq!(
        from_payload, from_pack,
        "the cook and the PIE payload builder flattened the same .inf_mat into \
         different bytes — there are two doors, not one"
    );

    // …and the record really says what the material said, so the equality above
    // is not two identical defaults agreeing.
    let rec: inf_asset::DerivedMaterial = inf_asset::decode(&from_pack).expect("decode record");
    assert_eq!(rec.albedo, Some(AssetId(ALBEDO)));
    assert_eq!(rec.normal, None, "the empty slot must stay empty");
    assert_eq!(rec.orm, Some(AssetId(ORM)));
    assert_eq!(rec.blend, inf_asset::DerivedBlend::Masked);
    assert_eq!(rec.alpha_cutoff, 0.375);
}

// ── (b) the dependency edges ────────────────────────────────────────────────

/// **A cooked pack carries the material a level binds and the textures that
/// material names**, at exact counts.
///
/// Two edges, and neither existed before this batch: `Material.asset` (scene
/// v22) and `.inf_mat` → its `.inf_tex` maps. Without the first the pack has no
/// material at all; without the second it has a record naming textures the
/// player asks for and cannot find — which renders as an untextured surface, is
/// indistinguishable from an authored flat colour, and is exactly what the
/// advisory doctrine exists for.
#[test]
fn the_cooked_pack_carries_the_material_and_its_textures() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, _doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");
    let reader = PackReader::open(&report.pack_path).expect("open pack");
    let kinds: Vec<AssetKind> = reader.index().map(|e| e.kind).collect();
    let count = |k: AssetKind| kinds.iter().filter(|x| **x == k).count();

    assert_eq!(
        count(AssetKind::Material),
        EXPECTED_MATERIALS_IN_PACK,
        "the level → .inf_mat edge did not close — kinds: {kinds:?}"
    );
    assert_eq!(
        count(AssetKind::Texture),
        EXPECTED_TEXTURES_IN_PACK,
        "the .inf_mat → .inf_tex edge did not close — kinds: {kinds:?}"
    );
    assert_eq!(
        count(AssetKind::DerivedMaterial),
        EXPECTED_DERIVED_MATERIALS_IN_PACK,
        "no .inf_matd was derived — kinds: {kinds:?}"
    );
    assert_eq!(report.materials_derived, EXPECTED_DERIVED_MATERIALS_IN_PACK);

    // The pack's texture entries really are the tiled containers, readable by the
    // door the player pages through — not merely bytes of the right length.
    for tex in [ALBEDO, ORM] {
        let bytes = reader.read(AssetId(tex)).expect("texture bytes");
        inf_vt::TiledTextureReader::new(bytes).expect("the pack's .inf_tex is a v2 container");
    }

    // And the CONTROL: an unbound level pulls none of it in. Without this the
    // three counts above would pass on a cook that packed the whole content root
    // regardless of what the level references.
    //
    // Sharper than a count since the fixture grew its glTF-shaped decor
    // (P26.3b audit): the unbound level still pulls the MESH's material through
    // the mesh's own sidecar — a real edge, and not this batch's — so what must
    // disappear is the material the LEVEL BOUND, and what must not is the one it
    // never bound. A count of zero would now be measuring the wrong thing.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, _) = scaffold(bare.path(), false);
    let bare_report =
        cook(&bare_proj, &bare.path().join("out"), &cook_opts()).expect("the bare project cooks");
    let bare_reader = PackReader::open(&bare_report.pack_path).expect("open bare pack");
    assert!(
        !bare_reader.contains(inf_asset::derived_material_id(AssetId(MAT))),
        "an unbound level still dragged its BOUND material into the pack — the \
         counts above are measuring the content root, not the closure"
    );
    assert!(
        !bare_reader.contains(AssetId(ALBEDO)) && !bare_reader.contains(AssetId(ORM)),
        "the bound material's textures reached a pack whose level binds no material"
    );
    assert!(
        bare_reader.contains(inf_asset::derived_material_id(AssetId(DECOR_MAT)))
            && bare_reader.contains(AssetId(DECOR_TEX)),
        "the MESH's material and its texture vanished from the unbound pack too, \
         so this control is measuring a cook that ships nothing rather than the \
         level→material edge"
    );
}

// ── (c) the payload carries the garment, the hairstyle and the surface ──────

/// **The P24.4 debt, discharged and measured** (P26.3b).
///
/// `phase24_gate::the_pie_payload_carries_no_garment_and_that_is_measured` read
/// `ScenePayload`'s own declaration and was built to fail the day it grew
/// `cloths`. It fired. This is what replaces it, and the difference is the
/// point: that arm could only say the field exists, and this one says the field
/// is FILLED — at an exact count, from a real project, through the real builder.
#[test]
fn the_payload_carries_the_garment_the_hairstyle_and_the_material() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let payload = payload_for(&proj, &doc);

    assert_eq!(
        payload.schema_version,
        inf_runtime::pie::SCENE_PAYLOAD_VERSION
    );
    assert_eq!(payload.cloths.len(), 1, "no .inf_cloth crossed the wire");
    assert_eq!(payload.hairs.len(), 1, "no .inf_hair crossed the wire");
    assert_eq!(payload.materials.len(), 1, "no material record crossed");
    assert_eq!(
        payload.textures.len(),
        2,
        "the material's two maps did not both cross"
    );
    assert_eq!(payload.cloths[0].0, CLOTH);
    assert_eq!(payload.hairs[0].0, HAIR);

    // The bytes are the authored ones, not a default: decode them back through
    // the same door the player takes.
    let coat: inf_anim::ClothAsset =
        inf_asset::decode(&payload.cloths[0].1).expect("the garment decodes");
    assert_eq!(coat, garment(), "the wire carried a different garment");

    // …and the world the payload builds really wears it. The P21.4 rule: assert
    // the WORLD before comparing two of them.
    let mut sim = inf_player::sim_from_payload(&payload)
        .expect("the payload builds a sim")
        .sim;
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    assert!(
        !inf_ecs::cloth::cloth_state_bytes(sim.world()).is_empty(),
        "the PIE world simulates NO garment — the payload's cloths reached no sim"
    );
    assert!(
        !inf_ecs::hair::hair_state_bytes(sim.world()).is_empty(),
        "the PIE world simulates NO hair"
    );

    // The CONTROL: the same level with nothing bound carries nothing, so the
    // counts above are a measurement of the bindings rather than of the resolver.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_payload = payload_for(&bare_proj, &bare_doc);
    assert!(bare_payload.cloths.is_empty());
    assert!(bare_payload.hairs.is_empty());
    assert!(bare_payload.materials.is_empty());
    assert!(bare_payload.textures.is_empty());
}

// ── (d) PIE == shipping ─────────────────────────────────────────────────────

/// The determinism trace of a world built from a cooked pack.
fn pack_trace(pack: &Path, frames: u64) -> u128 {
    let source = inf_player::level::PackLevelSource::open(pack).expect("open pack");
    let built = inf_player::build_world_from_pack(&source).expect("build world from pack");
    let mut sim = inf_player::sim_from_built(built);
    // ASSERT THE WORLD BEFORE COMPARING TWO OF THEM (P21.4, and the P24.4
    // mutation that proved it again): with no garment in the pack both sides
    // simulate nothing and agree perfectly.
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    assert!(
        !inf_ecs::cloth::cloth_state_bytes(sim.world()).is_empty(),
        "the cooked pack's character simulates NO garment — the comparison would \
         be two empty worlds agreeing"
    );
    inf_player::fold_trace_sim(sim, frames, None)
}

/// The same, from a streamed payload.
///
/// **The same one-step guard**, and the symmetry is load-bearing rather than
/// tidy: `pack_trace` steps once before folding in order to assert the world has
/// a coat in it, so a payload side that folded from step 0 would be comparing
/// frames 0..n against frames 1..n+1 and reporting a phase offset as a content
/// divergence. (Measured — that is exactly what the first draft did.)
fn payload_trace(payload: &inf_runtime::pie::ScenePayload, frames: u64) -> u128 {
    let mut sim = inf_player::sim_from_payload(payload)
        .expect("the payload builds a sim")
        .sim;
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    inf_player::fold_trace_sim(sim, frames, None)
}

/// **PIE == shipping on a textured, clothed scene.**
///
/// The same level, one world built from a cooked pack and one from the streamed
/// payload, folded over the same fixed steps. They must agree — and the control
/// below must NOT, or the agreement is a statement about two worlds in which
/// nothing was bound.
#[test]
fn pie_equals_shipping_on_a_textured_clothed_scene() {
    const FRAMES: u64 = 16;
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let shipped = pack_trace(&report.pack_path, FRAMES);
    let previewed = payload_trace(&payload_for(&proj, &doc), FRAMES);
    assert_eq!(
        previewed, shipped,
        "a PIE preview and the shipped build folded different worlds out of one \
         level — the payload and the pack are not carrying the same content"
    );

    // ANTI-VACUITY: unbind everything and the trace must MOVE. Without this the
    // equality above is satisfied by two worlds that simulate nothing, which is
    // exactly the failure the P24.4 matrix produced by severing one cook edge.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_payload = payload_for(&bare_proj, &bare_doc);
    assert_ne!(
        payload_trace(&bare_payload, FRAMES),
        previewed,
        "the same level with and without its garment folded the SAME trace — \
         nothing in the payload is being simulated"
    );
}

/// **The REAL `--pie` subprocess folds the clothed level** — the arm
/// `phase24_gate`'s retirement note said lived here (P26.3b audit).
///
/// It did not. The note claims the replacement for the retired trip-wire includes
/// "a real `--pie` subprocess folds them", and every arm above this one runs
/// in-process. That distinction is not pedantry in this repository: P21.4's
/// finding was a `--pie` binary that built its sim with a bare `RuntimeSim::new`
/// and therefore *agreed with itself* about a world missing an attachment, with
/// no gate running the binary. `sim_from_payload` is the one seam now, and this
/// is what keeps it that way for the four slots v8 added.
///
/// The anti-vacuity guard is first and the trace must MOVE, or a subprocess that
/// folded an empty world would match a reference that folded an empty world.
#[test]
fn the_real_pie_subprocess_folds_the_garment_and_the_binding() {
    const N: u32 = 8;
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let payload = payload_for(&proj, &doc);
    assert_eq!(payload.cloths.len(), 1, "nothing to fold");
    assert_eq!(payload.hairs.len(), 1, "nothing to fold");
    assert_eq!(payload.materials.len(), 1, "no binding on the wire");

    let mut session = PieSession::spawn_scene(&player_bin(), &payload).expect("scene session");
    session.step(N).expect("step N");
    let mut got = Vec::with_capacity(N as usize);
    for _ in 0..N {
        let ev = session
            .wait_for(Duration::from_secs(20), |e| {
                matches!(e, PlayerToEditor::Frame { .. })
            })
            .expect("a frame per step");
        if let PlayerToEditor::Frame { state_hash, .. } = ev {
            got.push(state_hash);
        }
    }
    let want = inf_player::scene_trace(&payload, N as u64).expect("in-process trace");
    assert_eq!(
        got, want,
        "the REAL --pie subprocess folded a different world from the in-process \
         one built out of the SAME payload — a boot path is dropping one of the \
         four slots v8 added"
    );
    assert!(
        got.windows(2).any(|w| w[0] != w[1]),
        "the trace never changed across {N} steps — the garment is not being \
         simulated, so the equality above compares two static worlds"
    );
    session
        .stop(Duration::from_secs(10))
        .expect("graceful stop");
}

// ── (e) the two hosts' material content ─────────────────────────────────────

/// **The pack path and the payload path hand a runtime the SAME material
/// content** (P26.3b audit).
///
/// Arm (a) proves the derived BYTES agree. It cannot see the thing a residency
/// trace is actually a function of: the *maps a host binds from*.
/// `PackLevelSource::material_content` walks a pack index and
/// `materials_from_payload` walks a wire vector — two collectors over two
/// containers, which is exactly the P22.2 arrangement that did not agree. Before
/// this arm neither function had a caller **or a test** anywhere in the tree,
/// under a batch whose headline is that the two wires carry one answer.
///
/// The first version of the pack side collected **every** `.inf_tex` and every
/// `.inf_matd` in the pack rather than what the level binds, so the two sides
/// diverged the moment a closure contained a material no entity bound — which is
/// the ordinary glTF-import case, since a `.inf_mesh` sidecar depends on its
/// materials and they depend on their textures. The sets are compared here, not
/// just their lengths.
#[test]
fn the_pack_and_the_payload_build_the_same_material_content() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let source = inf_player::level::PackLevelSource::open(&report.pack_path).expect("open pack");
    let from_pack = source.material_content();
    let from_payload = inf_player::materials_from_payload(&payload_for(&proj, &doc));

    // ANTI-VACUITY, both halves: a comparison of two empty maps passes.
    assert!(!from_pack.is_empty(), "the pack path resolved no material");
    assert!(
        !from_payload.is_empty(),
        "the payload path resolved no material"
    );
    // BOUND, not PACKED. The pack carries the mesh's material and its texture as
    // well, and a host that registered those would page a set the payload side
    // can never produce — which is exactly what the first cut did, because it
    // walked the pack INDEX.
    assert_eq!(from_pack.materials.len(), BOUND_MATERIALS);
    assert_eq!(from_pack.textures.len(), BOUND_TEXTURES);
    // A `const` block on purpose (clippy's `assertions_on_constants` is right
    // that this is constant-valued, and its remedy is the stronger one): the day
    // someone trims the fixture back to a bound-only closure, this arm stops
    // being able to tell a walk of the pack INDEX from a walk of the level's
    // bindings — and that must fail the BUILD rather than pass a test that has
    // quietly become a tautology.
    const {
        assert!(
            EXPECTED_DERIVED_MATERIALS_IN_PACK > BOUND_MATERIALS
                && EXPECTED_TEXTURES_IN_PACK > BOUND_TEXTURES,
            "the fixture's pack holds no UNBOUND material, so this arm cannot \
             tell a walk of the pack index from a walk of the level's bindings"
        );
    }
    assert!(
        !from_pack.materials.contains_key(&DECOR_MAT)
            && !from_pack.textures.contains_key(&DECOR_TEX),
        "the pack path resolved a material the level never bound — its \
         registration order, and therefore its residency, would be a function of \
         the pack rather than of the level"
    );

    // Keyed by the `.inf_mat` GUID on BOTH sides — the salt is inverted at the
    // boundary, so a projector never sees it. A side that forgot to un-salt keys
    // its map by `derived_material_id(MAT)` and fails here rather than as a
    // lookup miss in a frame.
    let pack_records: BTreeMap<_, _> = from_pack.materials.iter().collect();
    let payload_records: BTreeMap<_, _> = from_payload.materials.iter().collect();
    assert_eq!(
        pack_records, payload_records,
        "the pack and the payload resolved different material records for one level"
    );
    assert!(
        pack_records.contains_key(&MAT),
        "the records are not keyed by the .inf_mat GUID the scene names"
    );

    let pack_textures: BTreeMap<_, _> = from_pack.textures.iter().collect();
    let payload_textures: BTreeMap<_, _> = from_payload.textures.iter().collect();
    assert_eq!(
        pack_textures, payload_textures,
        "the pack and the payload carry different texture BYTES for one level"
    );

    // **The registration order is the residency trace.** `want_floor` is a pure
    // function of the registration SEQUENCE (the P26.3 handle law: the handles
    // may differ across hosts, the pages may not), so the two hosts must walk
    // these identically — and in the fixed slot order, with the empty normal slot
    // skipped rather than shifting the ORM into its place.
    assert_eq!(from_pack.registration_order(), vec![ALBEDO, ORM]);
    assert_eq!(
        from_payload.registration_order(),
        from_pack.registration_order()
    );
}

/// **THE REGISTRATION GAP, CLOSED** (P26.4, clause 0): a cooked pack's bound
/// material becomes a per-instance texture set, and the streamed payload's
/// becomes the same one **by GUID**.
///
/// Every layer under this shipped in P26.1–P26.3b and none of them had a
/// non-test caller: the container, the residency, the WGSL sample, the
/// registration door, the material rule and the want floor were all built and
/// exercised while `no projector called VtTextures::register` and both filled
/// `vt: Default::default()`. So a `.inf_mat`'s textures reached a runtime record
/// on every path and were sampled by nothing.
///
/// GPU-free on purpose: what this asserts is the *decision* — which textures are
/// registered, in what order, and which handles a surface's three slots name —
/// and that decision is `inf-render`'s registry, which needs no adapter. The
/// pixels are `inf-render`'s own `a_virtual_texture_reaches_the_lit_pixel`.
///
/// The cross-host comparison is **by GUID and never by handle**, which is the
/// P26.3 LAW: the editor walks document order and the player walks `Guid` order,
/// so one level mints different handles on the two sides and comparing the
/// integers would be comparing two correct answers and calling them wrong.
#[test]
fn a_bound_material_becomes_a_per_instance_texture_set_on_both_wires() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let (proj, doc) = scaffold(tmp.path(), true);
    let report = cook(&proj, &tmp.path().join("out"), &cook_opts()).expect("the project cooks");

    let build = |content: &inf_player::MaterialContent| {
        let mats = content.vt_materials();
        assert_eq!(mats.len(), BOUND_MATERIALS, "no binding to register");
        let (mut lib, _) = inf_render::VtTextures::new(inf_vt::VtPoolConfig {
            // The fixture's containers are written with `TextureCompression::None`,
            // so their stored pages ARE RGBA8 — the same decision
            // `build_vt_level` makes from the payload's own header, spelled here
            // because this arm builds the registry without a device.
            format: inf_vt::PageFormat::Rgba8,
            stored_tile_size: inf_vt::STORED_TILE_SIZE,
            budget_bytes: inf_vt::DEFAULT_VT_BUDGET_BYTES,
            max_texture_dim: 8192,
        });
        let n = lib.register_materials(&mats, |g| content.source(g));
        assert_eq!(n, BOUND_TEXTURES, "the door registered the wrong count");
        assert!(
            lib.refusals().is_empty(),
            "a bound texture was refused: {:?}",
            lib.refusals()
        );
        // The transaction that carries the root pages — until it runs, the warm
        // gate correctly refuses to name anything.
        assert!(
            lib.set_for_material(MAT.as_u128()).is_none(),
            "a cold registry named a texture"
        );
        let floor = lib.want_floor();
        let txn = lib.residency_mut().apply_wants(&floor);
        assert_eq!(txn.deferred, 0, "the floor did not fit: {}", txn.trace());
        lib
    };

    let source = inf_player::level::PackLevelSource::open(&report.pack_path).expect("open pack");
    let shipped = build(&source.material_content());
    let previewed = build(&inf_player::materials_from_payload(&payload_for(
        &proj, &doc,
    )));

    // The set a surface bound to MAT gets. Not `NONE` — which is what every
    // projector produced before this batch, on every path, for every level.
    let ship_set = shipped.set_for_material(MAT.as_u128());
    let prev_set = previewed.set_for_material(MAT.as_u128());
    assert!(
        !ship_set.is_none(),
        "the shipped path resolved no textures for a bound material"
    );
    assert_eq!(ship_set.normal, 0, "the empty normal slot must stay empty");
    assert_ne!(ship_set.albedo, 0);
    assert_ne!(ship_set.orm, 0);
    assert_ne!(
        ship_set.albedo, ship_set.orm,
        "one texture is serving two slots"
    );

    // BY GUID (the P26.3 LAW), slot by slot: each side's slot must resolve to the
    // texture the material actually names.
    for (slot, guid) in [(ship_set.albedo, ALBEDO), (ship_set.orm, ORM)] {
        assert_eq!(
            shipped.handle(guid.as_u128()).map(|h| h.0 + 1),
            Some(slot),
            "the shipped set's slot does not name {guid}"
        );
    }
    for (slot, guid) in [(prev_set.albedo, ALBEDO), (prev_set.orm, ORM)] {
        assert_eq!(
            previewed.handle(guid.as_u128()).map(|h| h.0 + 1),
            Some(slot),
            "the previewed set's slot does not name {guid}"
        );
    }

    // …and the residency the two hosts arrive at is the same WORLD: the same
    // tiles of the same textures, keyed by GUID. Handles may differ; pages may
    // not.
    let resident = |lib: &inf_render::VtTextures| {
        let mut out: Vec<(Uuid, inf_vt::TileCoord)> = Vec::new();
        for tex in [ALBEDO, ORM] {
            let h = lib.handle(tex.as_u128()).expect("registered");
            let desc = lib.residency().desc(h).expect("registered").clone();
            for mip in 0..desc.mip_count() {
                let m = desc.mips[mip as usize];
                for y in 0..m.tiles_y {
                    for x in 0..m.tiles_x {
                        let at = inf_vt::TileCoord::new(mip, x, y);
                        if lib.residency().is_resident(h, at) {
                            out.push((tex, at));
                        }
                    }
                }
            }
        }
        out
    };
    let ship_res = resident(&shipped);
    assert!(!ship_res.is_empty(), "nothing is resident");
    assert_eq!(
        ship_res,
        resident(&previewed),
        "a cooked pack and a PIE payload paged different tiles for one level"
    );

    // ANTI-VACUITY: the same level with nothing bound registers nothing, so every
    // equality above is a measurement of the binding rather than of two empty
    // registries agreeing.
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, bare_doc) = scaffold(bare.path(), false);
    let bare_content = inf_player::materials_from_payload(&payload_for(&bare_proj, &bare_doc));
    assert!(bare_content.vt_materials().is_empty());
}

/// **`registration_order` is a function of the SET, not of how the map was
/// built** (P26.3b audit) — the property that makes cross-host agreement
/// possible at all.
///
/// The P26.3 LAW says a handle is a per-registry index, so the editor (document
/// order) and the player (`Guid` order) mint different handles for one texture
/// and compare by GUID. It explicitly does **not** license the two hosts to page
/// different tiles, and `want_floor` is pure in the registration *sequence* — so
/// a `HashMap` walk would give one level two page sets on two machines, silently,
/// and only under a pool small enough to matter.
///
/// Built twice from opposite insertion orders. `std`'s `RandomState` is seeded
/// per map, so two separately-built `HashMap`s of the same keys genuinely iterate
/// differently; six materials make an accidental agreement negligible. The
/// anti-vacuity assertion is that the sorted answer is NOT the insertion order,
/// or this would be a statement about a constant.
#[test]
fn the_registration_order_is_sorted_not_inserted() {
    let mat = |n: u128| Uuid::from_u128(0x2603_0000_1000_0000_0000_0000_0000_0000 + n);
    let tex = |n: u128| Uuid::from_u128(0x2603_0000_2000_0000_0000_0000_0000_0000 + n);
    // Material `i` names texture `i` in its albedo slot, so the expected order is
    // the materials' GUID order projected onto textures.
    let build = |rev: bool| {
        let mut c = inf_player::MaterialContent::default();
        let mut ids: Vec<u128> = (0..6).collect();
        if rev {
            ids.reverse();
        }
        for i in ids {
            c.materials.insert(
                mat(i),
                inf_asset::DerivedMaterial {
                    albedo: Some(AssetId(tex(i))),
                    ..Default::default()
                },
            );
        }
        c
    };
    let forward = build(false);
    let reversed = build(true);
    let want: Vec<Uuid> = (0..6).map(tex).collect();

    assert_eq!(forward.registration_order(), want);
    assert_eq!(
        reversed.registration_order(),
        want,
        "the registration order depends on the order the map was BUILT — two \
         hosts would page different tiles for one level"
    );
    // Anti-vacuity: the sorted answer is not the reversed insertion order, so the
    // equality above is a statement about sorting rather than about a constant.
    let mut backwards = want.clone();
    backwards.reverse();
    assert_ne!(want, backwards);
}

// ── (f) the advisories fire ─────────────────────────────────────────────────

/// **Every silent material hazard raises a named advisory** (P26.3b audit).
///
/// P26.3b shipped two advisories and this audit added two more, and *none* of
/// them had a caller in any test: a `dangling_material_refs` that returned
/// `Vec::new()` unconditionally was invisible, which is the same shape as the
/// counter that never moves. Four levels, four fixtures, four messages — plus the
/// control, because an advisory list that is never empty is noise.
///
/// The two this audit added are the ones the P16 law names directly: a `.inf_mat`
/// whose `.inf_tex` is **missing**, and one whose `.inf_tex` is a **v1** payload
/// `inf_vt::TiledTextureReader` refuses. Both ship a pack that loads, renders and
/// is textureless.
#[test]
fn every_silent_material_hazard_raises_its_advisory() {
    // The healthy control FIRST: none of the four fires on the good fixture, so
    // the four below are measuring their triggers and not a cook that warns
    // about everything.
    let ok = tempfile::tempdir().expect("tempdir");
    let (ok_proj, _) = scaffold(ok.path(), true);
    let ok_report = cook(&ok_proj, &ok.path().join("out"), &cook_opts()).expect("cooks");
    for w in &ok_report.warnings {
        assert!(
            !w.contains("bound to") && !w.contains("references texture"),
            "the healthy fixture raised a material advisory: {w}"
        );
    }

    // 1. A binding naming an asset the project does not have.
    let case = |mutate: &dyn Fn(&Path, &mut SceneDoc)| -> Vec<String> {
        let tmp = tempfile::tempdir().expect("tempdir");
        let proj = tmp.path().join("proj");
        ProjectManifest::new("Advisory", "blank-3d")
            .save(&proj)
            .unwrap();
        let content = proj.join("Content");
        std::fs::create_dir_all(&content).unwrap();
        let mut doc = doc_with_binding(true);
        mutate(&content, &mut doc);
        let level = serialize::encode(&serialize::to_scene_file(&doc)).expect("encode");
        put(&content, "Wire.inf_lvl", LEVEL, &level, AssetKind::Level);
        put(
            &content,
            "Coat.inf_cloth",
            CLOTH,
            &inf_asset::encode(&garment()).unwrap(),
            AssetKind::Cloth,
        );
        put(
            &content,
            "Mane.inf_hair",
            HAIR,
            &inf_asset::encode(&hairstyle()).unwrap(),
            AssetKind::Hair,
        );
        cook(&proj, &tmp.path().join("out"), &cook_opts())
            .expect("an advisory is not a build failure")
            .warnings
    };

    // (a) the binding names nothing — no `.inf_mat` is written at all.
    let dangling = case(&|_content, _doc| {});
    assert!(
        dangling.iter().any(|w| w.contains("is bound to")
            && w.contains(&MAT.to_string())
            && w.contains("not in \nthe project".replace('\n', "").as_str())),
        "a binding naming a missing material cooked silently: {dangling:?}"
    );

    // (b) the binding names an asset of the WRONG KIND — the `.inf_mati` case,
    // reachable through the shipped Content Drawer before this audit.
    let wrong_kind = case(&|content, _doc| {
        put(
            content,
            "Wall.inf_mati",
            MAT,
            &inf_asset::encode(&inf_material::MaterialInstance::new(AssetId(ALBEDO))).unwrap(),
            AssetKind::MaterialInstance,
        );
    });
    assert!(
        wrong_kind
            .iter()
            .any(|w| w.contains("material_instance") && w.contains("TEXTURELESS")),
        "a binding naming a material INSTANCE cooked silently — the asset is in \
         the project, so nothing looked dangling: {wrong_kind:?}"
    );

    // (c) the material names a texture the project does not have.
    let missing_tex = case(&|content, _doc| {
        put(
            content,
            "Wall.inf_mat",
            MAT,
            &inf_asset::encode(&material()).unwrap(),
            AssetKind::Material,
        );
    });
    assert!(
        missing_tex
            .iter()
            .any(|w| w.contains("references texture") && w.contains("not in the")),
        "a material naming a missing .inf_tex cooked silently: {missing_tex:?}"
    );

    // (d) the material names a **v1** `.inf_tex`: present, valid, and unpageable.
    let v1 = case(&|content, _doc| {
        put(
            content,
            "Wall.inf_mat",
            MAT,
            &inf_asset::encode(&material()).unwrap(),
            AssetKind::Material,
        );
        for id in [ALBEDO, ORM] {
            let payload = inf_asset::encode(&inf_material::TextureAsset {
                schema_version: inf_material::TextureAsset::CURRENT_VERSION,
                width: 4,
                height: 4,
                format: inf_material::TextureFormat::Rgba8,
                srgb: true,
                mips: vec![inf_material::TextureMip {
                    width: 4,
                    height: 4,
                    data: vec![0u8; 4 * 4 * 4],
                }],
            })
            .unwrap();
            assert!(
                !inf_material::tiles::is_v2(&payload),
                "the v1 fixture is not v1, so this case measures nothing"
            );
            put(
                content,
                &format!("Tex{}.inf_tex", id.as_u128() & 0xF),
                id,
                &payload,
                AssetKind::Texture,
            );
        }
    });
    assert!(
        v1.iter()
            .any(|w| w.contains("v1 \n.inf_tex".replace('\n', "").as_str())
                || (w.contains("v1") && w.contains("tiled container"))),
        "a material naming a v1 .inf_tex cooked silently — the pack is bigger and \
         the surface is textureless: {v1:?}"
    );
}
