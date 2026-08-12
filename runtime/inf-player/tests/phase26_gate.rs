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
//! `.inf_mat` to a runtime record, along three routes that must not disagree.
//!
//! * **(a) ONE DOOR, three paths.** The `.inf_matd` bytes a cooked pack carries
//!   are byte-identical to the ones the PIE payload carries, for the same
//!   project. The P22.2 law made executable: the cook and the payload builder
//!   both call `inf_material::derive_material_bytes` and are compared on their
//!   OUTPUT rather than trusted on their comment. (`fracture_equivalence.rs` is
//!   the precedent, one asset kind over.)
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
//!   two empty worlds agreeing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash, PackReader};
use inf_ecs::components::{ClothSim, HairGuides, Material, SkeletalMesh};
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_material::{MatBlend, MaterialAsset, TextureCompression, TextureImportSettings};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;
use uuid::Uuid;

// ── the fixture's stable ids ────────────────────────────────────────────────

const LEVEL: Uuid = Uuid::from_u128(0x2603_0000_0000_0001);
const MAT: Uuid = Uuid::from_u128(0x2603_0000_0000_0002);
const ALBEDO: Uuid = Uuid::from_u128(0x2603_0000_0000_0003);
const ORM: Uuid = Uuid::from_u128(0x2603_0000_0000_0004);
const CLOTH: Uuid = Uuid::from_u128(0x2603_0000_0000_0005);
const HAIR: Uuid = Uuid::from_u128(0x2603_0000_0000_0006);

/// The pack must hold exactly these, and the counts are exact rather than
/// non-zero for the P21.4 reason: a walk that lost the second texture of two
/// still satisfies `!is_empty()`.
const EXPECTED_MATERIALS_IN_PACK: usize = 1;
const EXPECTED_DERIVED_MATERIALS_IN_PACK: usize = 1;
const EXPECTED_TEXTURES_IN_PACK: usize = 2;

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
    let path = content.join(file);
    std::fs::write(&path, bytes).expect("write asset");
    AssetSidecar::new(AssetId(guid), kind, ContentHash::of(bytes))
        .save(&path)
        .expect("write sidecar");
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
    let bare = tempfile::tempdir().expect("tempdir");
    let (bare_proj, _) = scaffold(bare.path(), false);
    let bare_report =
        cook(&bare_proj, &bare.path().join("out"), &cook_opts()).expect("the bare project cooks");
    let bare_reader = PackReader::open(&bare_report.pack_path).expect("open bare pack");
    let bare_kinds: Vec<AssetKind> = bare_reader.index().map(|e| e.kind).collect();
    assert_eq!(
        bare_kinds
            .iter()
            .filter(|k| **k == AssetKind::DerivedMaterial)
            .count(),
        0,
        "an unbound level still dragged a material into its pack — the counts \
         above are measuring the content root, not the closure: {bare_kinds:?}"
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
