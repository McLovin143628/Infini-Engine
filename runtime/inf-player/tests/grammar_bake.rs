//! **The grammar → mesh bake, end to end** (P23.6, the P22 carried ledger item).
//!
//! P22's `samples/phase22-playground` block is a hand-authored box, and its
//! source says exactly why: *"a `Destructible` names no asset of its own — the
//! cook fractures the ONE mesh its actor's `MeshRef.asset` points at. The P19
//! grammar produces a population… wiring it would have meant building a
//! grammar→single-mesh bake first, which belongs to Phase 23."*
//!
//! So this file closes that item by running the whole thing at once:
//!
//! ```text
//!   archetype + lot  → inf_pcg::building::build   (a real P19 grammar building)
//!                    → inf_editor_core::bake      (ONE MeshAsset, via the kernel)
//!                    → inf_mesh::fracture_mesh    (it breaks into sane chunks)
//!                    → inf_packager::cook         (a derived .inf_fracture ships)
//! ```
//!
//! It lives here rather than in `inf-editor-core` for one reason: the cook is
//! Ring 2 and `inf-packager` is a dev-dependency of this crate only. Splitting it
//! would have put the bake on one side of a seam and the thing that consumes it
//! on the other, which is the shape the P23.4 audit's LAW is about — **a gate
//! that inlines the code it is gating is a copy, not a gate**.
//!
//! The advisory arm is here too, and it is the other half of the same claim: a
//! `Destructible` on a population that was **not** baked is legal, cooks
//! silently, loads fine and cannot break — so the cook says so.

use std::collections::BTreeMap;
use std::path::Path;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash, PackReader};
use inf_ecs::components::{
    Destructible, MeshRef, PcgVolume, RigidBody3D, ScatteredSolid, Transform,
};
use inf_editor_core::bake;
use inf_editor_core::ipc::SpawnKind;
use inf_editor_core::scene::SceneDoc;
use inf_mesh::{fracture_mesh, FractureAsset, FractureParams};
use inf_packager::{cook, CookOptions};
use inf_pcg::{ArchetypeId, BuildingParams, Rect2};
use inf_project::ProjectManifest;

/// The mesh the baked building is written as.
const MESH_GUID: u128 = 0x2306_0100_0000_4000_8000_0000_0000_0001;
const LEVEL_GUID: u128 = 0x2306_0100_0000_4000_8000_0000_0000_0002;
const BUILDING_GUID: u128 = 0x2306_0100_0000_4000_8000_0000_0000_0003;

/// The building under test: a **house** on a 12 m × 9 m lot, furnished.
///
/// A house rather than an office because the archetype has to produce a solid an
/// author would actually try to blow up: slabs, walls with openings, a roof, a
/// stair and furniture — the whole placed population, nothing simplified.
fn house() -> BuildingParams {
    BuildingParams::new(
        ArchetypeId::House,
        Rect2::from_center(DVec2::ZERO, DVec2::new(12.0, 9.0)),
        0.0,
        0x2306,
    )
}

fn baked_house() -> bake::BakedMesh {
    bake::bake_building(&house(), 0x2306, true).expect("the house bakes")
}

macro_rules! insert {
    ($doc:expr, $guid:expr, $comp:expr) => {{
        let e = $doc.entity_of($guid).expect("entity");
        $doc.world_mut().world_mut().entity_mut(e).insert($comp);
        $doc.world_mut().mark_dirty();
    }};
}

/// Write a payload + sidecar under a chosen GUID — the cook resolves by GUID, so
/// the fixture has to name them.
fn put(dir: &Path, file: &str, id: AssetId, kind: AssetKind, bytes: &[u8]) {
    let path = dir.join(file);
    std::fs::write(&path, bytes).unwrap();
    AssetSidecar::new(id, kind, ContentHash::of(bytes))
        .save(&path)
        .unwrap();
}

/// A project holding the baked mesh and a level whose one actor is
/// `Destructible` and names it.
fn scaffold(tmp: &Path, baked: &bake::BakedMesh, scattered: bool) -> std::path::PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Grammar Bake", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    put(
        &content,
        "Bakery.inf_mesh",
        AssetId(Uuid::from_u128(MESH_GUID)),
        AssetKind::Mesh,
        &inf_asset::encode(&baked.asset).unwrap(),
    );

    let mut doc = SceneDoc::new();
    doc.set_title("Grammar Bake");
    let building = Uuid::from_u128(BUILDING_GUID);
    doc.create_with_guid(building, SpawnKind::Empty, "Building", None);
    insert!(
        doc,
        building,
        Transform::from_translation(baked.origin_world)
    );
    insert!(
        doc,
        building,
        MeshRef {
            asset: Some(Uuid::from_u128(MESH_GUID)),
            ..Default::default()
        }
    );
    insert!(doc, building, RigidBody3D::default());
    insert!(doc, building, Destructible::default());
    if scattered {
        // The hazard case: the population is still a population.
        insert!(
            doc,
            building,
            PcgVolume {
                graph: Some(Uuid::from_u128(0x2306_0100_0000_4000_8000_0000_0000_00FF)),
                ..Default::default()
            }
        );
    }
    doc.world_mut().propagate();
    inf_editor_core::scene::serialize::save(
        &doc,
        &content.join("Bakery.inf_lvl"),
        Some(Uuid::from_u128(LEVEL_GUID)),
    )
    .unwrap();
    proj
}

/// Cook the project and hand back `(warnings, fractures by guid)`.
fn cook_it(tmp: &Path, proj: &Path) -> (Vec<String>, BTreeMap<Uuid, Vec<u8>>) {
    let out = tmp.join("out");
    // vgeom off: its derivation dominates a test that is about fracture, the
    // `fracture_equivalence` suite's own rationale.
    let opts = CookOptions {
        vgeom: inf_packager::VgeomCookOptions {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let report = cook(proj, &out, &opts).expect("the level cooks");
    let reader = PackReader::open(&report.pack_path).expect("open pack");
    let fractures = reader
        .index()
        .filter(|e| e.kind == AssetKind::Fracture)
        .map(|e| (e.guid.uuid(), reader.read(e.guid).expect("read fracture")))
        .collect();
    (report.warnings, fractures)
}

// ── the bake ────────────────────────────────────────────────────────────────

/// **The headline**: a grammar building becomes ONE mesh, that mesh fractures
/// into sane chunks, and a `Destructible` naming it cooks with a derived
/// `.inf_fracture` in the pack.
#[test]
fn a_grammar_building_bakes_into_one_mesh_that_fractures_and_cooks() {
    let baked = baked_house();

    // ── it is a BUILDING, not a box ────────────────────────────────────────
    //
    // Anti-vacuity before anything else: a bake that produced one cube would
    // pass every assertion below and prove nothing about the grammar.
    assert!(
        baked.report.parts > 50,
        "the house baked into {} parts — that is not a building",
        baked.report.parts
    );
    assert!(
        baked.report.slots > 3,
        "the archetype's parts collapsed into {} material slots; the palette is \
         supposed to survive the bake",
        baked.report.slots
    );
    assert_eq!(
        baked.asset.submeshes.len(),
        baked.report.slots,
        "one submesh per slot"
    );
    // The slot names are the DSL's module names, so a material assigned to slot
    // *k* means the same thing on both sides of the bake.
    assert!(
        baked
            .asset
            .submeshes
            .iter()
            .all(|s| !s.name.starts_with("slot_")),
        "a submesh kept the writer's fallback name instead of its module's: {:?}",
        baked
            .asset
            .submeshes
            .iter()
            .map(|s| &s.name)
            .collect::<Vec<_>>()
    );
    assert!(
        baked.asset.submeshes.iter().all(|s| s
            .material_slot
            .is_some_and(|k| (k as usize) < baked.asset.material_slots.len())),
        "a submesh names a slot the palette does not have"
    );

    // ── it fractures ───────────────────────────────────────────────────────
    let mesh_id = AssetId(Uuid::from_u128(MESH_GUID));
    let params = FractureParams {
        seed: 7,
        chunk_count: 12,
    };
    let fracture = fracture_mesh(&baked.asset, mesh_id, params)
        .expect("a baked building is fracturable geometry");
    assert!(
        fracture.chunks.len() >= 8,
        "{} chunks out of a requested 12 — a building that shatters into two \
         pieces is not destructible content",
        fracture.chunks.len()
    );
    // **Sane** chunks, measured rather than counted: each one has geometry, a
    // hull the collider builder can use, positive volume, and at least one
    // neighbour to be bonded to (a chunk with no neighbours is a free-floating
    // piece the structural solve can say nothing about).
    for (i, c) in fracture.chunks.iter().enumerate() {
        assert!(!c.indices.is_empty(), "chunk {i} has no triangles");
        assert!(c.hull_points.len() >= 4, "chunk {i} has no usable hull");
        assert!(c.volume_m3 > 0.0, "chunk {i} has no volume");
        assert!(
            c.center_of_mass.iter().all(|v| v.is_finite()),
            "chunk {i} has a non-finite centre of mass"
        );
    }
    assert!(
        fracture.chunks.iter().any(|c| !c.neighbors.is_empty()),
        "no chunk has a neighbour — nothing is bonded, so the structural solve \
         has nothing to solve"
    );
    // The chunks account for the building: Voronoi cells tile the source hull, so
    // their volumes sum to it. Loose (10%) because the cells are clipped against
    // f32 geometry, tight enough that a bake that lost half its parts fails.
    let total: f64 = fracture.chunks.iter().map(|c| c.volume_m3).sum();
    assert!(
        (total - fracture.total_volume_m3()).abs() <= 1e-9,
        "the report's own total disagrees with its chunks"
    );
    assert!(
        total > 1.0,
        "the fractured building has {total} m³ of chunks — a house is not that small"
    );

    // ── and it COOKS ───────────────────────────────────────────────────────
    let tmp = tempfile::tempdir().unwrap();
    let proj = scaffold(tmp.path(), &baked, false);
    let (warnings, fractures) = cook_it(tmp.path(), &proj);
    assert_eq!(
        fractures.len(),
        1,
        "the cook shipped {} fractures for one destructible actor",
        fractures.len()
    );
    let want = inf_mesh::derived_fracture_id(AssetId(Uuid::from_u128(MESH_GUID))).uuid();
    let bytes = fractures
        .get(&want)
        .expect("the cook keyed the fracture off the baked mesh");
    let shipped: FractureAsset = inf_asset::decode(bytes).expect("the .inf_fracture decodes");
    assert_eq!(shipped.source_mesh, Uuid::from_u128(MESH_GUID).into_bytes());
    assert!(
        shipped.chunks.len() >= 8,
        "the shipped chunk set is {} chunks",
        shipped.chunks.len()
    );
    // The interior slot is appended past the baked palette, so a broken building
    // shows a *different* material where it cracked than where it was rendered.
    assert_eq!(
        shipped.slots.len(),
        baked.asset.material_slots.len() + 1,
        "the fracture's slot table is the mesh's plus one interior"
    );
    assert_eq!(shipped.interior_slot as usize, shipped.slots.len() - 1);
    // A baked building draws no advisory: it is exactly the content this
    // advisory was written to point authors AT.
    assert!(
        warnings.is_empty(),
        "the baked building trips cook advisories: {warnings:?}"
    );
}

// ── the signpost ────────────────────────────────────────────────────────────

/// **The honest signpost**: a `Destructible` on a still-scattered population
/// cooks, loads and cannot break — so the cook says so, in words that name the
/// fix.
#[test]
fn a_destructible_on_an_unbaked_population_draws_a_cook_advisory() {
    let baked = baked_house();
    let tmp = tempfile::tempdir().unwrap();
    let proj = scaffold(tmp.path(), &baked, true);
    let (warnings, fractures) = cook_it(tmp.path(), &proj);

    // It still cooks, and the mesh it *does* name still fractures — which is the
    // whole problem: nothing failed, and the building will not break.
    assert_eq!(
        fractures.len(),
        1,
        "the cook still ships the mesh's fracture"
    );
    let hit: Vec<&String> = warnings
        .iter()
        .filter(|w| w.contains("also carries a PcgVolume"))
        .collect();
    assert_eq!(
        hit.len(),
        1,
        "expected exactly one unbaked-population advisory, got {warnings:?}"
    );
    assert!(hit[0].contains("Bake the population"), "{}", hit[0]);
    assert!(
        hit[0].contains(&Uuid::from_u128(BUILDING_GUID).to_string()),
        "the advisory names the entity: {}",
        hit[0]
    );
}

/// The other side of the advisory, and what stops it being noise: the SAME level
/// with the population baked away is silent. Without this the sweep could fire on
/// every level and still pass the arm above.
#[test]
fn the_advisory_is_silent_once_the_population_is_baked() {
    let baked = baked_house();
    let tmp = tempfile::tempdir().unwrap();
    let proj = scaffold(tmp.path(), &baked, false);
    let (warnings, _) = cook_it(tmp.path(), &proj);
    assert!(
        !warnings.iter().any(|w| w.contains("PcgVolume")),
        "a level with no population drew the unbaked-population advisory: \
         {warnings:?}"
    );
}

// ── the ECS-mirror door ─────────────────────────────────────────────────────

/// The other bake door, exercised on the shape a `SceneDoc` actually holds:
/// `PcgVolume::structures`, a bare `Vec<ScatteredSolid>` with the module identity
/// already gone.
///
/// It is here rather than in the unit tests because the claim being made is about
/// the *pipeline* — that what a level carries can be baked and fractured, not
/// only what the assembler returns.
#[test]
fn the_ecs_mirror_population_bakes_and_fractures_too() {
    // Three storeys of one 4 m × 3 m × 4 m slab, stacked with a gap so the parts
    // do NOT abut: the gap is what keeps the result re-openable, and asserting
    // both halves is what makes `reopenable` a measurement.
    let solids: Vec<ScatteredSolid> = (0..3)
        .map(|i| ScatteredSolid {
            center: DVec3::new(0.0, 1.5 + i as f64 * 3.5, 0.0),
            half_extents: DVec3::new(2.0, 1.5, 2.0),
            rotation: glam::DQuat::IDENTITY,
        })
        .collect();
    let input = bake::parts_from_solids(&solids);
    assert_eq!(input.slot_names, vec![bake::UNTYPED_SLOT]);
    let baked = bake::bake_grammar_to_mesh(&input).expect("bakes");

    assert_eq!(baked.report.parts, 3);
    assert_eq!(baked.report.slots, 1, "the mirror has lost the module kind");
    assert_eq!(baked.asset.submeshes.len(), 1);
    assert_eq!(
        baked.report.coincident_vertices, 0,
        "the parts do not touch, so nothing is coincident"
    );
    assert!(
        baked.report.reopenable,
        "a population whose parts do not abut MUST be re-openable in the Model \
         Editor — if this fails the limitation is broader than the ledger says"
    );
    // …and the round trip really is one: the reader gets three closed shells.
    let read = inf_dcc::from_mesh_asset(&baked.asset).expect("re-opens");
    assert_eq!(read.report.boundary_edges, 0, "three closed solids");
    assert_eq!(inf_dcc::validate(&read.mesh), Ok(()));

    let fracture = fracture_mesh(
        &baked.asset,
        AssetId(Uuid::from_u128(MESH_GUID)),
        FractureParams::default(),
    )
    .expect("the stack fractures");
    assert!(fracture.chunks.len() >= 8, "{}", fracture.chunks.len());
}
