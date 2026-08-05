//! **Destruction is runtime-only: a broken wall is never written into the
//! author's `.inf_lvl`** (P22.4).
//!
//! The twin of `simulate_carve_not_persisted`, and the *whole* of P22.4's
//! persistence deliverable made checkable. The roadmap's P22.4 line reads
//! "destruction state persisted in the save"; the ruling this batch took is that
//! it is **not**, because there is no save-game container in this engine to put it
//! in (`.inf_lvl` is the author's document, not a player's progress) and inventing
//! one inside a destruction batch would be building the wrong thing twice.
//!
//! So the claim is the negative one, and a negative claim about persistence is
//! only worth anything if something asserts it:
//!
//! > Simulate → blow the wall apart → Stop → Ctrl+S
//!
//! must produce a **byte-identical** level file.
//!
//! It is not obviously safe. `FractureState` is sim state the way the voxel map is
//! sim state, and the voxel map's twin of this test exists because the first cut of
//! that fold *did* write a player's craters into the authored asset. What makes
//! destruction safe is structural — chunks are not entities, so there is nothing in
//! the document for them to be written into — and structural safety is exactly the
//! kind that a later batch can remove without noticing. Hence this file.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_blueprint::{
    BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty,
};
use inf_ecs::components::{ActorClass, Destructible, MeshRef, Terrain, Transform};
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::serialize;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession, SIM_HZ};
use inf_mesh::{Aabb, ChunkSection, FractureAsset, FractureChunk, MeshVertex};
use inf_physics::d3::{resolve_fracture_states, FractureState, CRACK_OPENING_M};
use inf_terrain::TerrainData;

const TERRAIN_GUID: u128 = 0x2204_2001;
const WALL_GUID: u128 = 0x2204_2002;
const MESH_GUID: u128 = 0x2204_20FF;
const WALL_CLASS_GUID: u128 = 0x2204_20AC;
const LEVEL_GUID: Uuid = Uuid::from_u128(0x2204_20FE);
const TOWER_CHUNKS: u32 = 4;

// ── the fixture (the `simulate_destruct` tower, trimmed to what this needs) ───

fn box_chunk(centre: DVec3, neighbors: Vec<u32>) -> FractureChunk {
    let h = 0.5;
    let mut hull = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                hull.push([centre.x + sx * h, centre.y + sy * h, centre.z + sz * h]);
            }
        }
    }
    let vertices: Vec<MeshVertex> = hull
        .iter()
        .map(|p| MeshVertex {
            position: [p[0] as f32, p[1] as f32, p[2] as f32],
            ..Default::default()
        })
        .collect();
    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0, 1, 3, 0, 3, 2,  4, 7, 5, 4, 6, 7,
        0, 4, 5, 0, 5, 1,  2, 3, 7, 2, 7, 6,
        0, 2, 6, 0, 6, 4,  1, 5, 7, 1, 7, 3,
    ];
    let index_count = indices.len() as u32;
    FractureChunk {
        vertices,
        indices,
        sections: vec![ChunkSection {
            material_slot: 0,
            first_index: 0,
            index_count,
        }],
        hull_points: hull,
        volume_m3: 1.0,
        center_of_mass: centre.to_array(),
        neighbors,
    }
}

fn tower() -> Arc<FractureAsset> {
    let n = TOWER_CHUNKS;
    let chunks: Vec<FractureChunk> = (0..n)
        .map(|i| {
            let mut nb = Vec::new();
            if i > 0 {
                nb.push(i - 1);
            }
            if i + 1 < n {
                nb.push(i + 1);
            }
            box_chunk(DVec3::new(0.0, i as f64 + 0.5, 0.0), nb)
        })
        .collect();
    Arc::new(FractureAsset {
        schema_version: FractureAsset::CURRENT_VERSION,
        source_mesh: *Uuid::from_u128(MESH_GUID).as_bytes(),
        bounds: Aabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, n as f32, 0.5],
        },
        seed: 0,
        requested_chunks: n,
        slots: vec!["Concrete".into(), "Fracture Interior".into()],
        interior_slot: 1,
        chunks,
    })
}

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        if let Some(e) = $doc.entity_of($guid) {
            $doc.world_mut().world_mut().entity_mut(e).insert($comp);
            $doc.world_mut().mark_dirty();
        }
    }};
}

fn wall_doc() -> SceneDoc {
    let mut doc = SceneDoc::new();
    let terrain = Uuid::from_u128(TERRAIN_GUID);
    let wall = Uuid::from_u128(WALL_GUID);

    doc.create_with_guid(terrain, SpawnKind::Empty, "Terrain", None);
    insert!(doc, terrain, Transform::IDENTITY);
    let mut data = TerrainData::new(5, 4.0);
    for c in [(-1, -1), (-1, 0), (0, -1), (0, 0)] {
        data.author_tile(c, |_, _| 0.0);
    }
    insert!(
        doc,
        terrain,
        Terrain {
            meters_per_sample: 4.0,
            tile_resolution: 5,
            data,
            ..Terrain::default()
        }
    );

    doc.create_with_guid(wall, SpawnKind::Empty, "Wall", None);
    insert!(doc, wall, Transform::IDENTITY);
    insert!(
        doc,
        wall,
        Destructible {
            chunk_count: TOWER_CHUNKS,
            ..Destructible::default()
        }
    );
    insert!(
        doc,
        wall,
        MeshRef {
            asset: Some(Uuid::from_u128(MESH_GUID)),
            ..Default::default()
        }
    );
    insert!(doc, wall, ActorClass(Uuid::from_u128(WALL_CLASS_GUID)));
    doc.world_mut().propagate();
    doc.mark_saved();
    doc
}

/// A Blueprint that blows its own wall apart on every Tick.
fn wall_class() -> BlueprintClass {
    let entity = Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let energy = Destructible::default().strength * 1.0 * CRACK_OPENING_M * 3.5;
    let mut class = BlueprintClass::new("act:destructible-wall", "Wall");
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body: vec![Stmt::ExprStmt(Expr::Call {
                path: vec!["destruct".into(), "apply_damage".into()],
                args: vec![entity, Expr::Lit(Lit::Float(energy))],
            })],
        },
    }];
    class
}

// ── the gate ────────────────────────────────────────────────────────────────

/// **THE CLAIM.** Save, break the wall in Simulate, stop, save again: the two
/// files are byte-identical.
#[test]
fn a_simulate_destruction_leaves_the_level_file_byte_identical() {
    let dir = tempfile::tempdir().unwrap();
    let before = dir.path().join("Before.inf_lvl");
    let after = dir.path().join("After.inf_lvl");

    let mut doc = wall_doc();
    serialize::save(&doc, &before, Some(LEVEL_GUID)).expect("the pre-damage save");
    let pre = std::fs::read(&before).expect("read pre");

    // Play → the Blueprint breaks its own wall → Stop.
    let mut session = SimSession::enter(
        &mut doc,
        vec![(Uuid::from_u128(WALL_GUID), wall_class())],
        DVec2::ZERO,
        SIM_HZ,
    );
    let asset = tower();
    let states: BTreeMap<Uuid, FractureState> = resolve_fracture_states(doc.world(), |mesh| {
        (mesh == Uuid::from_u128(MESH_GUID)).then(|| asset.clone())
    });
    assert_eq!(
        states.len(),
        1,
        "the fixture must seed exactly one fracture"
    );
    session.set_fractures(states);
    for _ in 0..30 {
        session.tick(&mut doc, 1.0 / SIM_HZ, SimInput::default());
    }

    // **ANTI-VACUITY, before the claim rather than after it.** A session that
    // broke nothing would satisfy every equality below, which is precisely how
    // the P21.4 gate certified a no-op.
    let state = &session.fractures()[&Uuid::from_u128(WALL_GUID)];
    assert!(
        !state.is_intact(),
        "the wall never broke — this proves nothing"
    );
    let detached = state.chunks().iter().filter(|c| c.detached).count();
    assert!(
        detached >= 2,
        "only {detached} chunk(s) came off over 30 ticks"
    );
    let moved = state
        .chunks()
        .iter()
        .enumerate()
        .filter(|(i, c)| {
            c.detached && (c.translation - DVec3::new(0.0, *i as f64 + 0.5, 0.0)).length() > 0.1
        })
        .count();
    assert!(
        moved >= 1,
        "the rubble never moved, so nothing was simulated"
    );

    session.exit(&mut doc);

    // THE CLAIM.
    serialize::save(&doc, &after, Some(LEVEL_GUID)).expect("the post-damage save");
    assert_eq!(
        std::fs::read(&after).expect("read post"),
        pre,
        "a Simulate session's destruction reached the author's .inf_lvl — the \
         phase's ruling is that destruction is RUNTIME-ONLY, and a save that \
         carried rubble would mean an author who pressed Play could not press \
         Ctrl+S again"
    );
    // The sidecar too: it is the git-diffable half of the same document, and a
    // TOML that gained twelve derived rows would be just as wrong as a payload
    // that did.
    assert_eq!(
        std::fs::read(after.with_extension("inf_lvl.toml")).expect("post sidecar"),
        std::fs::read(before.with_extension("inf_lvl.toml")).expect("pre sidecar"),
        "the level's TOML sidecar changed across a Simulate destruction"
    );

    // …and the document itself is not merely *saving* the same, it is back to
    // being the same: the wall is one intact entity with its `Destructible`, and
    // no chunk ever became one.
    assert_eq!(
        doc.order().len(),
        2,
        "the document gained or lost entities across a Simulate session — chunks \
         are not entities, and nothing may have made them into some"
    );
    let wall = doc.entity_of(Uuid::from_u128(WALL_GUID)).expect("the wall");
    assert!(
        doc.world().world().get::<Destructible>(wall).is_some(),
        "the wall lost its Destructible"
    );
}

/// A guard on the guard: this level file **can** change, so "byte-identical"
/// above is a statement about destruction and not about a save path that stopped
/// working.
#[test]
fn an_authoring_edit_does_change_the_level_file() {
    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("A.inf_lvl");
    let b = dir.path().join("B.inf_lvl");

    let mut doc = wall_doc();
    serialize::save(&doc, &a, Some(LEVEL_GUID)).expect("save a");
    if let Some(e) = doc.entity_of(Uuid::from_u128(WALL_GUID)) {
        let mut t = doc
            .world_mut()
            .world_mut()
            .get_mut::<Transform>(e)
            .expect("the wall has a transform");
        t.translation = inf_ecs::Vec3d::new(7.0, 0.0, 0.0);
    }
    doc.world_mut().mark_dirty();
    doc.world_mut().propagate();
    serialize::save(&doc, &b, Some(LEVEL_GUID)).expect("save b");
    assert_ne!(
        std::fs::read(&a).unwrap(),
        std::fs::read(&b).unwrap(),
        "moving a wall did not change the level file — the save seam is broken, \
         and the test above passes for the wrong reason"
    );
}
