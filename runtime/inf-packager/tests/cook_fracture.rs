//! **The P22.2 cook path, end to end.**
//!
//! The derivation is gated on a fact that lives in a *different asset* from the
//! one it acts on — "some entity, in some level, marks this mesh destructible" —
//! which is exactly the shape of edge the P21.4 audit found cold: if it is wrong,
//! the level cooks, the pack verifies, the player boots, and the wall is simply
//! indestructible in the shipped build with nothing anywhere saying so.
//!
//! So these tests assert the **pack**, not the report: that the derived id is
//! present, that its bytes decode into a real chunk set with real volume, and —
//! the other half — that a project with no `Destructible` anywhere derives
//! nothing at all.

use std::path::Path;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{Destructible, MeshRef, Primitive, Transform};
use inf_mesh::{MeshAsset, MeshVertex, SubMesh};
use inf_packager::{cook, derived_fracture_id, CookOptions};
use inf_project::ProjectManifest;

const LEVEL_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000022020001"));
const MESH_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000022020002"));
const FLAT_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000022020003"));
/// A GUID no asset in the project has — the dangling case.
const MISSING_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-0000220200ff"));
/// A GUID that resolves to something that is not a mesh.
const NOT_A_MESH_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-0000220200fe"));
/// A second level, for the closure-exclusion arm.
const OTHER_LEVEL_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-0000220200fd"));

fn g(n: u128) -> uuid::Uuid {
    uuid::Uuid::from_u128(n)
}

/// A bare entity record with every slot `None`.
fn rec(guid: u128, name: &str) -> inf_scene::RuntimeEntity {
    inf_scene::RuntimeEntity {
        guid: g(guid),
        name: name.into(),
        parent: None,
        transform: Transform::IDENTITY,
        visible: true,
        mesh: None,
        material: None,
        light: None,
        camera: None,
        sprite: None,
        tilemap: None,
        nine_slice: None,
        text2d: None,
        light_2d: None,
        rigid_body_2d: None,
        collider_2d: None,
        character_controller_2d: None,
        rigid_body_3d: None,
        collider_3d: None,
        character_controller_3d: None,
        actor: None,
        terrain: None,
        pcg_volume: None,
        skeletal_mesh: None,
        anim_player: None,
        anim_state_machine: None,
        root_motion: None,
        attached_to: None,
        joint_2d: None,
        joint_3d: None,
        audio_source: None,
        audio_listener: None,
        decal: None,
        volume: None,
        spline: None,
        foliage: None,
        streaming_source: None,
        always_loaded: None,
        time_of_day: None,
        sky_atmosphere: None,
        water_body: None,
        buoyancy: None,
        voxel_volume: None,
        destructible: None,
        ik_target: None,
        cloth_sim: None,
        hair_guides: None,
    }
}

/// An entity drawing `asset`, optionally destructible.
fn prop(
    guid: u128,
    name: &str,
    asset: Option<AssetId>,
    d: Option<Destructible>,
) -> inf_scene::RuntimeEntity {
    inf_scene::RuntimeEntity {
        mesh: asset.map(|a| MeshRef {
            primitive: Primitive::Cube,
            asset: Some(a.uuid()),
        }),
        destructible: d,
        ..rec(guid, name)
    }
}

fn level(entities: Vec<inf_scene::RuntimeEntity>) -> inf_scene::RuntimeLevel {
    inf_scene::RuntimeLevel {
        title: "Rubble".into(),
        entities,
        settings: inf_scene::RuntimeSettings::default(),
    }
}

/// A solid 2 m cube mesh — a real breakable wall's worth of geometry.
fn cube_mesh(half: f32) -> MeshAsset {
    let c = |x: f32, y: f32, z: f32| MeshVertex {
        position: [x * half, y * half, z * half],
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

/// A long thin box rotated onto the diagonal: its hull fills ~0.2% of its AABB,
/// so most Voronoi sites land outside the solid and yield no chunk.
fn diagonal_plank() -> MeshAsset {
    let q = glam::DQuat::from_xyzw(1.0, 2.0, 3.0, 4.0).normalize();
    let (hx, hy) = (2.0f64, 0.02f64);
    let mut vertices = Vec::new();
    for sx in [-1.0, 1.0f64] {
        for sy in [-1.0, 1.0f64] {
            for sz in [-1.0, 1.0f64] {
                let r = q * glam::DVec3::new(sx * hx, sy * hy, sz * hy);
                vertices.push(MeshVertex {
                    position: [r.x as f32, r.y as f32, r.z as f32],
                    ..Default::default()
                });
            }
        }
    }
    // **A real closed box, not `(0..8)`** (Hardening Wave J). This fixture used
    // to hand over its eight corner indices in order, which is two triangles
    // and two leftover indices — not a whole number of triangles, and not the
    // box the doc above describes. It cooked anyway because nothing on this
    // path read the index buffer; `MeshAsset::validate` now refuses it at the
    // decode, which is how it was found. The corner index is
    // `4·(x>0) + 2·(y>0) + (z>0)`, in the generation order above.
    #[rustfmt::skip]
    let indices = vec![
        0, 1, 3,  0, 3, 2, // -x
        4, 6, 7,  4, 7, 5, // +x
        0, 4, 5,  0, 5, 1, // -y
        2, 3, 7,  2, 7, 6, // +y
        0, 2, 6,  0, 6, 4, // -z
        1, 5, 7,  1, 7, 3, // +z
    ];
    MeshAsset::new(
        vec![SubMesh {
            name: "plank".into(),
            vertices,
            indices,
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec!["Pine".into()],
    )
}

/// A flat quad: real triangles, no volume — the `Degenerate` refusal.
fn flat_mesh() -> MeshAsset {
    let v = |x: f32, z: f32| MeshVertex {
        position: [x, 0.0, z],
        ..Default::default()
    };
    MeshAsset::new(
        vec![SubMesh {
            name: "sheet".into(),
            vertices: vec![v(0.0, 0.0), v(2.0, 0.0), v(2.0, 2.0), v(0.0, 2.0)],
            indices: vec![0, 1, 2, 0, 2, 3],
            material_slot: None,
            skin: Vec::new(),
        }],
        vec![],
    )
}

fn put(root: &Path, name: &str, id: AssetId, kind: AssetKind, bytes: &[u8]) {
    let path = root.join("Content").join(name);
    std::fs::write(&path, bytes).unwrap();
    AssetSidecar::new(id, kind, ContentHash::of(bytes))
        .save(&path)
        .unwrap();
}

/// Write a project holding `level` plus the given meshes.
fn make_project(root: &Path, lvl: &inf_scene::RuntimeLevel, meshes: &[(AssetId, &str, MeshAsset)]) {
    ProjectManifest::new("Fracture Cook", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let bytes = lvl.encode().unwrap();
    put(root, "Rubble.inf_lvl", LEVEL_ID, AssetKind::Level, &bytes);
    for (id, name, mesh) in meshes {
        let payload = inf_asset::encode(mesh).unwrap();
        put(root, name, *id, AssetKind::Mesh, &payload);
    }
}

/// Cook with vgeom off: it is irrelevant here and its derivation dominates the
/// runtime of a test that only cares about fracture.
fn opts() -> CookOptions {
    CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// **THE CLOSURE EDGE.** A level with one `Destructible` derives an
/// `.inf_fracture` for its mesh, packs it under the derived id, and those bytes
/// decode into a chunk set that conserves the mesh's volume.
///
/// Asserting the PACK rather than `report.fractures_derived` is the point: a
/// counter can be incremented by a derivation whose bytes never reach the pack,
/// and the failure mode this guards is precisely "the shipped build has no
/// fracture data and nothing notices".
#[test]
fn a_destructible_level_derives_and_packs_its_fracture() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    let d = Destructible {
        fracture_seed: 3,
        chunk_count: 10,
        strength: 4.0e6,
        density_kg_m3: 2400.0,
        runtime_destruct: true,
    };
    make_project(
        &proj,
        &level(vec![prop(0xD001, "Wall", Some(MESH_ID), Some(d))]),
        &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
    );

    let report = cook(&proj, &out, &opts()).expect("the level cooks");
    assert_eq!(report.levels_rewritten, 1);
    assert_eq!(report.fractures_derived, 1);
    assert!(
        report.warnings.is_empty(),
        "a well-formed destructible must be silent: {:?}",
        report.warnings
    );
    assert_eq!(report.kinds.get("fracture"), Some(&1));

    // The pack really contains it, at the id a runtime will compute.
    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    let frac_id = derived_fracture_id(MESH_ID);
    assert_ne!(frac_id, MESH_ID);
    let entry = reader.entry(frac_id).expect(
        "the .inf_fracture must be IN the pack — without it a shipped wall \
         cannot break and nothing says so",
    );
    assert_eq!(entry.kind, AssetKind::Fracture);
    assert!(
        entry.compressed || entry.uncompressed_len < 64,
        "a fracture is read whole, so it compresses like any other payload"
    );

    // …and the bytes are a real chunk set, not an empty container.
    let asset: inf_mesh::FractureAsset = inf_asset::decode(&reader.read(frac_id).unwrap()).unwrap();
    assert_eq!(asset.source_mesh, *MESH_ID.uuid().as_bytes());
    assert_eq!(asset.seed, 3);
    assert!(
        asset.chunks.len() >= 2 && asset.chunks.len() <= 10,
        "got {} chunks",
        asset.chunks.len()
    );
    // A 2 m cube is 8 m3 and the chunks tile it.
    let total = asset.total_volume_m3();
    assert!((total - 8.0).abs() < 1e-5, "chunks summed to {total} m3");
    for c in &asset.chunks {
        assert!(c.volume_m3 > 0.0);
        assert!(c.hull_points.len() >= 4, "a chunk needs a collider hull");
        assert!(!c.indices.is_empty());
        // The mass the runtime will give it is real, from the honest density.
        assert!(d.chunk_mass_kg(c.volume_m3) > 0.0);
    }
    assert!(!asset.adjacency_pairs().is_empty(), "no structural graph");
    // The generated interior slot exists and is the mesh's slots + 1.
    assert_eq!(asset.slots.len(), 2);
    assert_eq!(asset.slots[0], "Concrete");
    assert_eq!(asset.interior_slot, 1);
}

/// **The other half.** Content with no `Destructible` anywhere derives nothing —
/// the cook does not fracture every mesh it sees, and a project that never asked
/// for destruction pays neither the cook time nor the ship bytes.
#[test]
fn content_with_no_destructible_derives_no_fracture() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    make_project(
        &proj,
        &level(vec![prop(0xD002, "Wall", Some(MESH_ID), None)]),
        &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
    );

    let report = cook(&proj, &out, &opts()).expect("the level cooks");
    assert_eq!(report.fractures_derived, 0);
    assert_eq!(report.kinds.get("fracture"), None);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);

    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    assert!(reader.entry(MESH_ID).is_some(), "the mesh still ships");
    assert!(
        reader.entry(derived_fracture_id(MESH_ID)).is_none(),
        "an undeclared mesh must not be fractured"
    );
}

/// The build-level off switch stops the derivation without touching what is
/// authored — and says nothing, because turning it off is a decision, not a
/// hazard.
#[test]
fn the_fracture_derivation_can_be_disabled_for_a_build() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    make_project(
        &proj,
        &level(vec![prop(
            0xD003,
            "Wall",
            Some(MESH_ID),
            Some(Destructible::default()),
        )]),
        &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
    );

    let mut o = opts();
    o.fracture.enabled = false;
    let report = cook(&proj, &out, &o).expect("the level cooks");
    assert_eq!(report.fractures_derived, 0);
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    assert!(reader.entry(derived_fracture_id(MESH_ID)).is_none());
}

/// **Advisory: a `Destructible` on a mesh that cannot break.** The level says the
/// sheet shatters; the cook can see it bounds no volume. Non-fatal — the level is
/// valid — so it is *said*, with the reason and the fix.
#[test]
fn a_destructible_flat_mesh_raises_an_advisory_and_ships_no_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    make_project(
        &proj,
        &level(vec![prop(
            0xD004,
            "Sheet",
            Some(FLAT_ID),
            Some(Destructible::default()),
        )]),
        &[(FLAT_ID, "Sheet.inf_mesh", flat_mesh())],
    );

    let report = cook(&proj, &out, &opts()).expect("the level still cooks");
    assert_eq!(report.fractures_derived, 0);
    let msg = report
        .warnings
        .iter()
        .find(|w| w.contains("was NOT") && w.contains("fractured"))
        .unwrap_or_else(|| panic!("no fracture advisory in {:?}", report.warnings));
    assert!(msg.contains(&FLAT_ID.to_string()), "{msg}");
    assert!(msg.contains("Sheet"), "{msg}");
    assert!(msg.contains("flat or collinear"), "{msg}");
    assert!(msg.contains("remove the Destructible"), "{msg}");

    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    assert!(reader.entry(derived_fracture_id(FLAT_ID)).is_none());
}

/// **Advisory: a `Destructible` with no mesh at all.** There is nothing to
/// derive a fracture *from*, and the component references no asset of its own, so
/// this is the only place the mistake is visible.
#[test]
fn a_destructible_without_a_mesh_raises_an_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    make_project(
        &proj,
        &level(vec![prop(
            0xD005,
            "Ghost",
            None,
            Some(Destructible::default()),
        )]),
        &[],
    );

    let report = cook(&proj, &out, &opts()).expect("the level still cooks");
    assert_eq!(report.fractures_derived, 0);
    let msg = report
        .warnings
        .iter()
        .find(|w| w.contains("references no mesh asset"))
        .unwrap_or_else(|| panic!("no advisory in {:?}", report.warnings));
    assert!(msg.contains("Ghost"), "{msg}");
    assert!(msg.contains(&g(0xD005).to_string()), "{msg}");
}

/// **Advisory: an out-of-range chunk count.** The cook clamps it — a build must
/// not fail on a number — and quotes both the asked and the clamped value, so the
/// level's claim and the shipped reality are both in the report.
#[test]
fn an_out_of_range_chunk_count_is_clamped_with_an_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    make_project(
        &proj,
        &level(vec![prop(
            0xD006,
            "Wall",
            Some(MESH_ID),
            Some(Destructible {
                chunk_count: 5000,
                ..Destructible::default()
            }),
        )]),
        &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
    );

    let report = cook(&proj, &out, &opts()).expect("the level cooks");
    assert_eq!(report.fractures_derived, 1, "it still fractures");
    let msg = report
        .warnings
        .iter()
        .find(|w| w.contains("fracture chunks"))
        .unwrap_or_else(|| panic!("no clamp advisory in {:?}", report.warnings));
    assert!(msg.contains("5000"), "{msg}");
    assert!(
        msg.contains(&inf_mesh::MAX_CHUNK_COUNT.to_string()),
        "the advisory must quote the clamped value: {msg}"
    );

    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    let asset: inf_mesh::FractureAsset =
        inf_asset::decode(&reader.read(derived_fracture_id(MESH_ID)).unwrap()).unwrap();
    assert!(asset.chunks.len() <= inf_mesh::MAX_CHUNK_COUNT as usize);
}

/// **Advisory: two `Destructible`s, one mesh, different fractures.** Only one
/// `.inf_fracture` can exist per mesh — its id is a function of the mesh's — so
/// the second entity silently gets the first's break. Named, with the fix.
#[test]
fn conflicting_destructibles_on_one_mesh_raise_an_advisory() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    make_project(
        &proj,
        &level(vec![
            prop(
                0xD007,
                "Wall A",
                Some(MESH_ID),
                Some(Destructible {
                    chunk_count: 8,
                    ..Destructible::default()
                }),
            ),
            prop(
                0xD008,
                "Wall B",
                Some(MESH_ID),
                Some(Destructible {
                    chunk_count: 30,
                    ..Destructible::default()
                }),
            ),
        ]),
        &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
    );

    let report = cook(&proj, &out, &opts()).expect("the level cooks");
    assert_eq!(report.fractures_derived, 1, "exactly one per mesh");
    let msg = report
        .warnings
        .iter()
        .find(|w| w.contains("two different fractures"))
        .unwrap_or_else(|| panic!("no conflict advisory in {:?}", report.warnings));
    assert!(msg.contains(&MESH_ID.to_string()), "{msg}");
    assert!(msg.contains('8') && msg.contains("30"), "{msg}");
    assert!(msg.contains("separate meshes"), "{msg}");

    // The first in cook (entity) order won, and the pack proves which.
    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    let asset: inf_mesh::FractureAsset =
        inf_asset::decode(&reader.read(derived_fracture_id(MESH_ID)).unwrap()).unwrap();
    assert!(asset.chunks.len() <= 8, "got {}", asset.chunks.len());
}

/// Two cooks of one project produce byte-identical packs — the fracture
/// derivation must not be the thing that breaks the P9.2 determinism guarantee.
#[test]
fn two_cooks_of_a_destructible_project_are_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_project(
        &proj,
        &level(vec![prop(
            0xD009,
            "Wall",
            Some(MESH_ID),
            Some(Destructible {
                fracture_seed: 42,
                chunk_count: 16,
                ..Destructible::default()
            }),
        )]),
        &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
    );

    let a = cook(&proj, &dir.path().join("out_a"), &opts()).unwrap();
    let b = cook(&proj, &dir.path().join("out_b"), &opts()).unwrap();
    assert_eq!(a.fractures_derived, 1);
    assert_eq!(
        std::fs::read(&a.pack_path).unwrap(),
        std::fs::read(&b.pack_path).unwrap(),
        "two cooks of one destructible project diverged"
    );
}

/// **THE SEVENTH CROSS-ASSET ADVISORY (P22.2 audit).** A `Destructible` whose
/// mesh never reaches the cook ships silently — and there are three ways for
/// that to happen, all of which cooked clean before this landed.
///
/// The component names no asset of its own, which is exactly why nothing else
/// notices: the level is valid, the pack verifies, the player boots, and a wall
/// the level calls destructible cannot break. Arm (c) is the positive control —
/// a reachable mesh must stay silent, or the advisory is noise.
#[test]
fn a_destructible_whose_mesh_never_reaches_the_cook_is_advised() {
    // (a) the GUID resolves to nothing.
    {
        let dir = tempfile::tempdir().unwrap();
        let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
        make_project(
            &proj,
            &level(vec![prop(
                0xE001,
                "Wall",
                Some(MISSING_ID),
                Some(Destructible::default()),
            )]),
            &[],
        );
        let report = cook(&proj, &out, &opts()).expect("the level still cooks");
        let msg = report
            .warnings
            .iter()
            .find(|w| w.contains("CANNOT BREAK"))
            .unwrap_or_else(|| panic!("no advisory in {:?}", report.warnings));
        assert!(msg.contains("does not have"), "{msg}");
        assert!(msg.contains(&MISSING_ID.to_string()), "{msg}");
        assert_eq!(report.fractures_derived, 0);
    }

    // (b) the GUID resolves to something that is not a mesh.
    {
        let dir = tempfile::tempdir().unwrap();
        let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
        make_project(
            &proj,
            &level(vec![prop(
                0xE002,
                "Wall",
                Some(NOT_A_MESH_ID),
                Some(Destructible::default()),
            )]),
            &[],
        );
        // A texture standing in for the mis-pointed reference.
        put(
            &proj,
            "Bark.inf_tex",
            NOT_A_MESH_ID,
            AssetKind::Texture,
            b"not really a texture, but the sidecar says it is",
        );
        let report = cook(&proj, &out, &opts()).expect("the level still cooks");
        let msg = report
            .warnings
            .iter()
            .find(|w| w.contains("CANNOT BREAK"))
            .unwrap_or_else(|| panic!("no advisory in {:?}", report.warnings));
        assert!(msg.contains("is not an .inf_mesh"), "{msg}");
    }

    // (c) the positive control: a mesh that IS reachable stays silent.
    {
        let dir = tempfile::tempdir().unwrap();
        let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
        make_project(
            &proj,
            &level(vec![prop(
                0xE003,
                "Wall",
                Some(MESH_ID),
                Some(Destructible::default()),
            )]),
            &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
        );
        // A cook rooted at the level would follow the MeshRef edge, so the
        // exclusion is forced by rooting at an unrelated asset.
        let lone = level(vec![prop(0xE004, "Bare", None, None)]);
        let bytes = lone.encode().unwrap();
        put(
            &proj,
            "Lone.inf_lvl",
            OTHER_LEVEL_ID,
            AssetKind::Level,
            &bytes,
        );
        let mut o = opts();
        o.roots = Some(vec![LEVEL_ID]);
        // Rooting at the level DOES pull the mesh, so this arm proves the
        // opposite: with the mesh present and reachable, the advisory is silent.
        let report = cook(&proj, &out, &o).expect("the level cooks");
        assert!(
            !report.warnings.iter().any(|w| w.contains("CANNOT BREAK")),
            "a reachable mesh must be silent: {:?}",
            report.warnings
        );
        assert_eq!(report.fractures_derived, 1);
    }
}

/// **Dropped sites are counted and, past a stated floor, advised.** A diagonal
/// plank's Voronoi cells mostly miss the solid, so an author who asked for 32
/// pieces ships far fewer — legal, invisible in every other number, and a break
/// that reads as a cut.
#[test]
fn a_large_chunk_shortfall_is_counted_and_advised() {
    let dir = tempfile::tempdir().unwrap();
    let (proj, out) = (dir.path().join("proj"), dir.path().join("out"));
    make_project(
        &proj,
        &level(vec![prop(
            0xE010,
            "Plank",
            Some(MESH_ID),
            Some(Destructible {
                chunk_count: 32,
                ..Destructible::default()
            }),
        )]),
        &[(MESH_ID, "Plank.inf_mesh", diagonal_plank())],
    );

    let report = cook(&proj, &out, &opts()).expect("the plank cooks");
    assert_eq!(report.fractures_derived, 1);
    assert!(
        report.fracture_chunks_dropped > 0,
        "the fixture must actually drop sites"
    );
    assert_eq!(
        report.fracture_chunks + report.fracture_chunks_dropped,
        32,
        "shipped + dropped must account for every requested site"
    );
    let msg = report
        .warnings
        .iter()
        .find(|w| w.contains("of the 32 its Destructible asked for"))
        .unwrap_or_else(|| panic!("no yield advisory in {:?}", report.warnings));
    assert!(msg.contains("coarser"), "{msg}");

    // …and the shipped asset records the shortfall, so it stays checkable from
    // the pack afterwards.
    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    let asset: inf_mesh::FractureAsset =
        inf_asset::decode(&reader.read(derived_fracture_id(MESH_ID)).unwrap()).unwrap();
    assert_eq!(asset.requested_chunks, 32);
    assert!(asset.chunks.len() < 32);
    assert_eq!(asset.chunks.len(), report.fracture_chunks);

    // A well-shaped mesh at the same count does NOT trip it — the floor is a
    // judgement, not a blanket.
    let dir2 = tempfile::tempdir().unwrap();
    let (proj2, out2) = (dir2.path().join("proj"), dir2.path().join("out"));
    make_project(
        &proj2,
        &level(vec![prop(
            0xE011,
            "Wall",
            Some(MESH_ID),
            Some(Destructible {
                chunk_count: 32,
                ..Destructible::default()
            }),
        )]),
        &[(MESH_ID, "Wall.inf_mesh", cube_mesh(1.0))],
    );
    let clean = cook(&proj2, &out2, &opts()).expect("the cube cooks");
    assert!(
        !clean.warnings.iter().any(|w| w.contains("asked for")),
        "a cube must not trip the yield floor: {:?}",
        clean.warnings
    );
}

/// The `Destructible` component's default chunk count must equal the mesh
/// crate's, which `inf-ecs` cannot say because it must not depend on `inf-mesh`.
/// This crate can see both, so the pin lives here.
#[test]
fn destructible_defaults_match_the_mesh_crate() {
    assert_eq!(
        inf_ecs::components::DEFAULT_DESTRUCTIBLE_CHUNKS,
        inf_mesh::DEFAULT_CHUNK_COUNT,
        "the ECS default and the fracture default drifted"
    );
    // The default must not itself need clamping — asserted through the clamp
    // rather than against the bounds, so it is a behavioural check rather than a
    // constant expression the compiler can fold away.
    assert_eq!(
        inf_mesh::clamp_chunk_count(inf_ecs::components::DEFAULT_DESTRUCTIBLE_CHUNKS),
        inf_ecs::components::DEFAULT_DESTRUCTIBLE_CHUNKS,
        "the default chunk count is outside the supported range"
    );
}
