//! **The P21.1 cook path, end to end.**
//!
//! Three things about `.inf_voxel` at cook time had no coverage at all when they
//! landed, and each fails in a different, silent way:
//!
//! 1. **The dependency edge.** `VoxelVolume.asset` has to pull its `.inf_voxel`
//!    into the pack. If it does not, the level ships with its caves *referenced
//!    and absent* — no error, no warning, just solid ground where a tunnel was.
//!    Nothing else in the repo would notice: the level cooks, the pack verifies,
//!    the player boots.
//! 2. **The structural check.** A `.inf_voxel` is streaming-class, so it cooks
//!    **uncompressed** and the runtime sub-slices chunks straight out of the
//!    mapping by trusting a header and directory it validated once. A truncated,
//!    misaligned or accidentally framed payload must therefore fail the BUILD;
//!    reaching a player, it is a cave made of another chunk's bytes.
//! 3. **The dangling-ref advisory.** A reference the database cannot resolve is
//!    not a build failure — the level still loads and plays — so it has to be
//!    *said*. The P21.1 refactor also changed the terrain half of that advisory
//!    (one set, now carrying a kind discriminant), and neither output was
//!    asserted anywhere, so both message shapes are pinned here.

use std::path::Path;

use inf_asset::{AssetId, AssetKind, AssetSidecar, ContentHash};
use inf_ecs::components::{Transform, VoxelVolume};
use inf_packager::{cook, CookOptions};
use inf_project::ProjectManifest;
use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};

const LEVEL_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000021010001"));
const VOXEL_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000021010002"));
/// A GUID no asset in the project has — the dangling case.
const MISSING_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-0000210100ff"));

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
    }
}

fn cave(guid: u128, name: &str, asset: AssetId) -> inf_scene::RuntimeEntity {
    inf_scene::RuntimeEntity {
        voxel_volume: Some(VoxelVolume {
            asset: Some(asset.uuid()),
            ..VoxelVolume::default()
        }),
        ..rec(guid, name)
    }
}

fn level(entities: Vec<inf_scene::RuntimeEntity>) -> inf_scene::RuntimeLevel {
    inf_scene::RuntimeLevel {
        title: "Caves".into(),
        entities,
        settings: inf_scene::RuntimeSettings::default(),
    }
}

/// A real, valid `.inf_voxel` payload: a solid block with a ball carved out of
/// it, so the bytes are genuine content rather than an empty container.
fn voxel_payload() -> Vec<u8> {
    let mut data = VoxelData::new(0.5);
    for key in inf_voxel::chunk_range(ChunkKey::new(0, 0, 0), ChunkKey::new(1, 1, 1)) {
        data.insert_chunk(key, VoxelChunk::solid(1));
    }
    let (report, _) = data.apply_op(&inf_voxel::VoxelOp::carve(inf_voxel::VoxelShape::Sphere {
        center: glam::DVec3::splat(8.0),
        radius_m: 3.0,
    }));
    assert!(
        report.total_carved() > 0,
        "the fixture must carve something"
    );
    inf_voxel::build_voxel_asset(&data).unwrap().into_bytes()
}

/// Write a project whose level references `asset`, optionally writing that
/// `.inf_voxel` (with `bytes`) into the content root.
fn make_project(root: &Path, level: &inf_scene::RuntimeLevel, voxel: Option<(AssetId, Vec<u8>)>) {
    ProjectManifest::new("Voxel Cook", "blank-3d")
        .save(root)
        .unwrap();
    let content = root.join("Content");
    std::fs::create_dir_all(&content).unwrap();

    let bytes = level.encode().unwrap();
    let path = content.join("Caves.inf_lvl");
    std::fs::write(&path, &bytes).unwrap();
    AssetSidecar::new(LEVEL_ID, AssetKind::Level, ContentHash::of(&bytes))
        .save(&path)
        .unwrap();

    if let Some((id, payload)) = voxel {
        let vpath = content.join("Cave.inf_voxel");
        std::fs::write(&vpath, &payload).unwrap();
        AssetSidecar::new(id, AssetKind::VoxelVolume, ContentHash::of(&payload))
            .save(&vpath)
            .unwrap();
    }
}

/// **(b) THE CLOSURE EDGE.** A level that references a real `.inf_voxel` cooks it
/// into the pack — uncompressed, byte-identical, and reachable by its own GUID.
///
/// The byte-identity claim is the load-bearing one: a `.inf_voxel` rides through
/// the cook *verbatim* precisely so a runtime can sub-slice a chunk out of the
/// mapping, and an entry that were re-encoded (or compressed) would break every
/// offset in its directory while still "containing the asset".
#[test]
fn a_referenced_voxel_asset_cooks_into_the_pack_verbatim() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let out = dir.path().join("out");
    let payload = voxel_payload();
    make_project(
        &proj,
        &level(vec![cave(0xC001, "Cave System", VOXEL_ID)]),
        Some((VOXEL_ID, payload.clone())),
    );

    let report = cook(&proj, &out, &CookOptions::default()).expect("the level cooks");
    assert_eq!(report.levels_rewritten, 1);
    assert!(
        report.warnings.is_empty(),
        "a resolvable reference must be silent: {:?}",
        report.warnings
    );

    let reader = inf_asset::PackReader::open(&report.pack_path).unwrap();
    let entry = reader.entry(VOXEL_ID).expect(
        "the .inf_voxel must be IN the pack — without this edge a shipped \
                 level draws solid ground where its caves were",
    );
    assert_eq!(entry.kind, AssetKind::VoxelVolume);
    assert!(
        !entry.compressed,
        "a streaming-class kind must cook uncompressed or its chunk offsets are \
         unreachable from the mapping"
    );
    assert_eq!(
        reader.read(VOXEL_ID).unwrap(),
        payload,
        "the payload must ride through the cook verbatim"
    );
    // …and it is still a valid, sub-sliceable container on the far side.
    let cooked = inf_voxel::VoxelAssetReader::new(reader.read(VOXEL_ID).unwrap()).unwrap();
    assert_eq!(cooked.chunk_count(), 8);
    assert!(cooked.chunk_bytes(ChunkKey::new(0, 0, 0)).is_some());
}

/// **(c) THE ADVISORY.** A level referencing a `.inf_voxel` the project does not
/// have still cooks — and says so, naming the level, the asset and the
/// consequence.
///
/// Non-fatal on purpose: the level loads and plays, it just has no caves. That is
/// exactly the class of hazard an advisory exists for, and exactly the class that
/// ships unnoticed without one.
#[test]
fn a_dangling_voxel_reference_is_reported_and_still_cooks() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let out = dir.path().join("out");
    make_project(
        &proj,
        &level(vec![cave(0xC002, "Lost Cave", MISSING_ID)]),
        None,
    );

    let report = cook(&proj, &out, &CookOptions::default())
        .expect("a dangling voxel ref is an advisory, not a cook failure");
    assert_eq!(report.levels_rewritten, 1, "the level still cooked");

    let hit = report
        .warnings
        .iter()
        .find(|w| w.contains(&MISSING_ID.to_string()))
        .unwrap_or_else(|| {
            panic!(
                "the dangling voxel ref must be reported: {:?}",
                report.warnings
            )
        });
    assert!(
        hit.contains(&LEVEL_ID.to_string()),
        "names the level: {hit}"
    );
    assert!(hit.contains("voxel"), "names the KIND: {hit}");
    assert!(
        hit.contains("chunks will not stream"),
        "names the consequence: {hit}"
    );
    // It must not be mistaken for the terrain advisory it shares a function with.
    assert!(!hit.contains("terrain"), "{hit}");
    assert!(!hit.contains("tiles"), "{hit}");
}

/// The **terrain** half of the same advisory still says what it always said —
/// the P21.1 refactor folded a kind discriminant into that set and neither
/// message shape was asserted anywhere.
#[test]
fn the_terrain_advisory_survived_the_shared_refactor() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let out = dir.path().join("out");
    let entity = inf_scene::RuntimeEntity {
        terrain: Some(inf_ecs::components::Terrain {
            asset: Some(MISSING_ID.uuid()),
            ..inf_ecs::components::Terrain::default()
        }),
        ..rec(0xC003, "Lost Ground")
    };
    make_project(&proj, &level(vec![entity]), None);

    let report = cook(&proj, &out, &CookOptions::default()).expect("still cooks");
    let hit = report
        .warnings
        .iter()
        .find(|w| w.contains(&MISSING_ID.to_string()))
        .unwrap_or_else(|| panic!("{:?}", report.warnings));
    assert!(hit.contains("terrain"), "{hit}");
    assert!(hit.contains("tiles will not stream"), "{hit}");
    assert!(!hit.contains("voxel"), "{hit}");
}

/// A level with **no volumes at all** gains no warnings — the off-path half of
/// the advisory claim. An advisory that fires on correct content stops being read.
#[test]
fn a_level_without_volumes_is_silent() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    make_project(&proj, &level(vec![rec(0xC004, "Nothing")]), None);
    let report = cook(&proj, &dir.path().join("out"), &CookOptions::default()).unwrap();
    assert!(report.warnings.is_empty(), "{:?}", report.warnings);
}

/// **(a) THE STRUCTURAL CHECK.** A corrupt `.inf_voxel` fails the BUILD.
///
/// It cooks uncompressed and is sub-sliced by offsets the runtime validates once,
/// so a payload whose header or directory does not parse must never reach a pack.
/// Checked for three distinct corruptions, because a single "garbage bytes" case
/// would pass on a check that only looked at the magic.
#[test]
fn a_corrupt_voxel_asset_fails_the_cook() {
    let good = voxel_payload();

    // (i) not a voxel asset at all; (ii) truncated mid-directory; (iii) a valid
    // header whose blob_base no longer follows the directory.
    let mut misdirected = good.clone();
    let blob_base_at = 56;
    misdirected[blob_base_at..blob_base_at + 8].copy_from_slice(&9_999_999u64.to_le_bytes());

    for (label, payload) in [
        (
            "bad magic",
            b"definitely not an .inf_voxel payload".to_vec(),
        ),
        ("truncated", good[..good.len() / 3].to_vec()),
        ("bad blob_base", misdirected),
    ] {
        // The fixture must genuinely be broken, or the cook assertion is vacuous.
        assert!(
            inf_voxel::VoxelAssetReader::new(payload.as_slice()).is_err(),
            "{label}: the fixture parses, so it proves nothing"
        );

        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().join("proj");
        make_project(
            &proj,
            &level(vec![cave(0xC005, "Broken Cave", VOXEL_ID)]),
            Some((VOXEL_ID, payload)),
        );
        let err =
            cook(&proj, &dir.path().join("out"), &CookOptions::default()).expect_err(&format!(
                "{label}: a corrupt .inf_voxel must fail the BUILD — the runtime \
                 pages chunks by trusting a directory it validates once"
            ));
        let msg = err.to_string();
        assert!(
            msg.contains("voxel volume") && msg.contains(&VOXEL_ID.to_string()),
            "{label}: the error must name the asset, got: {msg}"
        );
    }
}
