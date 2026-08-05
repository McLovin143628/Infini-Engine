//! **P22.3 re-audit: the cook and the editor derive the SAME fracture bytes.**
//!
//! Three paths produce a `.inf_fracture` for one mesh — `inf_packager::cook`
//! (what a player ships), `inf_editor_core::pie` (what a PIE session previews),
//! and the editor's Simulate seeder (which calls the same `pie` helpers). Two of
//! them are separate code, and both carried comments claiming they agreed "by
//! construction". They did not: the re-audit found the editor resolving a shared
//! mesh's parameters in ECS **archetype** order while the cook used document
//! order, and — earlier — the editor skipping the cook's collidability refusal
//! entirely.
//!
//! Comments cannot hold that. This file compares the **bytes**, out of a real
//! cooked pack, against a real editor derivation, over one level document that
//! feeds both.
//!
//! The fixture is deliberately the case that broke: **two destructible actors
//! sharing one mesh and disagreeing about its chunk count**, with the second
//! carrying an `AlwaysLoaded` so the two sit in different ECS archetypes and its
//! components inserted first so its archetype is created first. Exactly one of
//! them decides the chunking, and both paths must pick the same one.

use std::collections::BTreeMap;
use std::path::Path;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{AlwaysLoaded, Destructible, MeshRef};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::{serialize, SceneDoc};
use inf_mesh::{MeshAsset, MeshVertex, SubMesh};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;
use uuid::Uuid;

const MESH: Uuid = Uuid::from_u128(0x2203_3001);
const LEVEL: Uuid = Uuid::from_u128(0x2203_3002);
/// The FIRST actor in document order — its chunk count must win on both paths.
const FIRST_CHUNKS: u32 = 8;
/// The second's, which must lose on both paths.
const SECOND_CHUNKS: u32 = 24;

/// A 2 m cube — big enough to fracture, simple enough to reason about.
fn cube_mesh() -> MeshAsset {
    let c = |x: f32, y: f32, z: f32| MeshVertex {
        position: [x, y, z],
        ..Default::default()
    };
    MeshAsset::new(
        vec![SubMesh {
            name: "wall".into(),
            vertices: vec![
                c(-1.0, -1.0, -1.0),
                c(1.0, -1.0, -1.0),
                c(1.0, 1.0, -1.0),
                c(-1.0, 1.0, -1.0),
                c(-1.0, -1.0, 1.0),
                c(1.0, -1.0, 1.0),
                c(1.0, 1.0, 1.0),
                c(-1.0, 1.0, 1.0),
            ],
            indices: vec![
                0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0,
                7, 4, 1, 5, 6, 1, 6, 2,
            ],
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec!["Concrete".into()],
    )
}

/// Two destructible actors sharing one mesh and disagreeing about it.
fn two_actor_doc() -> SceneDoc {
    let mut doc = SceneDoc::new();
    let first = doc.create(SpawnKind::Empty, "First", None);
    let second = doc.create(SpawnKind::Empty, "Second", None);
    {
        let world = doc.world_mut();
        // Second-actor-first, so its archetype is created first and an
        // archetype-order walk visits it before the document's first actor.
        // Without this the two orders coincide and the gate proves nothing.
        let b = world.entity_of(second).expect("second");
        world.world_mut().entity_mut(b).insert((
            Destructible {
                chunk_count: SECOND_CHUNKS,
                ..Destructible::default()
            },
            MeshRef {
                asset: Some(MESH),
                ..Default::default()
            },
            AlwaysLoaded,
        ));
        let a = world.entity_of(first).expect("first");
        world.world_mut().entity_mut(a).insert((
            Destructible {
                chunk_count: FIRST_CHUNKS,
                ..Destructible::default()
            },
            MeshRef {
                asset: Some(MESH),
                ..Default::default()
            },
        ));
        world.mark_dirty();
    }
    doc
}

/// The bytes the **editor** produces — PIE payloads and Simulate both go through
/// this exact pair of helpers.
fn editor_bytes(doc: &SceneDoc) -> BTreeMap<Uuid, Vec<u8>> {
    let mesh = cube_mesh();
    inf_editor_core::pie::destructible_mesh_params(doc)
        .into_iter()
        .filter_map(|(mesh_id, params)| {
            let asset = inf_editor_core::pie::derive_fracture(&mesh, mesh_id, params)?;
            let bytes = inf_asset::encode(&asset).ok()?;
            Some((
                inf_mesh::derived_fracture_id(AssetId(mesh_id)).uuid(),
                bytes,
            ))
        })
        .collect()
}

fn put(root: &Path, name: &str, id: AssetId, kind: AssetKind, bytes: &[u8]) {
    let path = root.join("Content").join(name);
    std::fs::write(&path, bytes).unwrap();
    AssetSidecar::new(id, kind, ContentHash::of(bytes))
        .save(&path)
        .unwrap();
}

/// The bytes the **cook** produces, out of a real cooked pack, from the SAME
/// document — serialized by the editor's own writer, so this is the level a Save
/// would have written rather than a hand-mirrored copy of it.
fn cook_bytes(doc: &SceneDoc, dir: &Path) -> BTreeMap<Uuid, Vec<u8>> {
    let (proj, out) = (dir.join("proj"), dir.join("out"));
    ProjectManifest::new("Fracture Equivalence", "blank-3d")
        .save(&proj)
        .unwrap();
    std::fs::create_dir_all(proj.join("Content")).unwrap();

    let level = serialize::encode(&serialize::to_scene_file(doc)).expect("encode level");
    put(
        &proj,
        "Level.inf_lvl",
        AssetId(LEVEL),
        AssetKind::Level,
        &level,
    );
    let mesh = inf_asset::encode(&cube_mesh()).unwrap();
    put(
        &proj,
        "Wall.inf_mesh",
        AssetId(MESH),
        AssetKind::Mesh,
        &mesh,
    );

    // vgeom off: its derivation dominates the runtime of a test that only cares
    // about fracture (the `cook_fracture` suite's own rationale).
    let opts = CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let report = cook(&proj, &out, &opts).expect("the level cooks");
    let reader = inf_asset::PackReader::open(&report.pack_path).expect("open pack");
    reader
        .index()
        .filter(|e| e.kind == AssetKind::Fracture)
        .map(|e| (e.guid.uuid(), reader.read(e.guid).expect("read fracture")))
        .collect()
}

/// **THE EQUIVALENCE GATE.** Byte-for-byte, for one level.
#[test]
fn the_cook_and_the_editor_derive_the_same_fracture_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let doc = two_actor_doc();
    let editor = editor_bytes(&doc);
    let cooked = cook_bytes(&doc, dir.path());

    // Anti-vacuity FIRST: two empty maps agree (the P21.4 lesson), and this
    // fixture exists to make that impossible to miss.
    assert_eq!(editor.len(), 1, "the editor derived no fracture");
    assert_eq!(cooked.len(), 1, "the cook shipped no fracture");

    let want = inf_mesh::derived_fracture_id(AssetId(MESH)).uuid();
    assert!(
        editor.contains_key(&want),
        "the editor used a different key"
    );
    assert!(cooked.contains_key(&want), "the cook used a different key");
    assert_eq!(
        editor[&want], cooked[&want],
        "the editor and the cook derived DIFFERENT fracture bytes for one mesh — a \
         PIE session would break a wall the shipped build breaks differently"
    );

    // …and the winner really is the FIRST actor in document order. Without this
    // the byte comparison could pass with both paths agreeing on the wrong one.
    let decoded =
        inf_asset::decode::<inf_mesh::FractureAsset>(&cooked[&want]).expect("the fracture decodes");
    assert_eq!(
        decoded.requested_chunks, FIRST_CHUNKS,
        "the shared mesh took the SECOND actor's chunk count — an archetype-order \
         walk answers {SECOND_CHUNKS} here"
    );
    // The adjacency the cook shipped is symmetric and faceless-free, which is what
    // the runtime's `estimated_bonds` invariant rests on.
    for (i, c) in decoded.chunks.iter().enumerate() {
        for &n in &c.neighbors {
            assert!(
                decoded.chunks[n as usize].neighbors.contains(&(i as u32)),
                "the cook shipped a one-sided adjacency edge {i}-{n}"
            );
            assert!(
                decoded.shared_face_area_m2(i as u32, n) > 0.0,
                "the cook shipped a FACELESS adjacency edge {i}-{n} — a phantom \
                 load path for the support solve"
            );
        }
    }
}

/// The fixture must actually reproduce the archetype/document split, or the gate
/// above is comparing two walks that were never going to differ.
#[test]
fn the_fixture_really_reproduces_the_order_split() {
    let doc = two_actor_doc();
    let archetype_first = doc
        .world()
        .world()
        .iter_entities()
        .find_map(|e| e.get::<Destructible>().map(|d| d.chunk_count))
        .expect("a destructible");
    assert_eq!(
        archetype_first, SECOND_CHUNKS,
        "an archetype walk visits the document's FIRST actor first, so the two \
         orders coincide and the equivalence gate proves nothing about ordering"
    );
    assert_eq!(
        inf_editor_core::pie::destructible_mesh_params(&doc)[&MESH].chunk_count,
        FIRST_CHUNKS
    );
}
