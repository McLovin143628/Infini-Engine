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
/// The `.inf_terrain` the P21.2 hole advisories read.
const TERRAIN_ID: AssetId = AssetId(uuid::uuid!("00000000-0000-0000-0000-000021020001"));

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

/// A terrain entity backed by `asset` — the P21.2 half. The inline `data` stays
/// empty: a streamed terrain's tiles live in the `.inf_terrain`, and (since the
/// scene wire is pinned at tile generation 3) its hole mask can live nowhere
/// else at all.
fn ground(guid: u128, name: &str, asset: AssetId) -> inf_scene::RuntimeEntity {
    inf_scene::RuntimeEntity {
        terrain: Some(inf_ecs::components::Terrain {
            asset: Some(asset.uuid()),
            ..inf_ecs::components::Terrain::default()
        }),
        ..rec(guid, name)
    }
}

/// A real `.inf_terrain` payload: one flat tile, with `holes` carved into it at
/// the given samples.
fn terrain_payload(holes: &[(u32, u32)]) -> Vec<u8> {
    let res = 8;
    let mut t = inf_terrain::TerrainData::new(res, 1.0);
    t.author_tile((0, 0), |_, _| 0.0);
    {
        let tile = t.get_tile_mut((0, 0)).unwrap();
        for &(i, j) in holes {
            tile.set_hole(res, i, j, true);
        }
        assert_eq!(tile.has_holes(), !holes.is_empty());
    }
    let opts = inf_terrain::PyramidOptions::default();
    let pyramid = inf_terrain::build_pyramid(&t, opts);
    inf_terrain::build_terrain_asset(&t, &pyramid, opts)
        .unwrap()
        .into_bytes()
}

/// Write `bytes` into the content root as `name`, with a sidecar of `kind`.
fn put(root: &Path, name: &str, id: AssetId, kind: AssetKind, bytes: &[u8]) {
    let path = root.join("Content").join(name);
    std::fs::write(&path, bytes).unwrap();
    AssetSidecar::new(id, kind, ContentHash::of(bytes))
        .save(&path)
        .unwrap();
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

// ── P21.2 advisories ─────────────────────────────────────────────────

/// **(a) The scale mismatch is SAID, with both numbers.** The `.inf_voxel` header
/// and the `VoxelVolume` component each carry a voxel scale, the asset's is the
/// one that wins at runtime, and a component that disagrees is authored intent
/// being silently discarded.
///
/// Both measured values must appear in the message, because "these disagree" is
/// not actionable and "set it to 0.5" is.
#[test]
fn a_voxel_scale_mismatch_is_reported_with_both_numbers() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let out = dir.path().join("out");

    // The payload is built at 0.5 m/voxel (see `voxel_payload`); the component
    // claims 0.25.
    let mut entity = cave(0x21_0201, "Cave", VOXEL_ID);
    entity.voxel_volume.as_mut().unwrap().voxel_size_m = 0.25;
    make_project(
        &proj,
        &level(vec![entity]),
        Some((VOXEL_ID, voxel_payload())),
    );

    let report = cook(&proj, &out, &CookOptions::default()).unwrap();
    let hit = report
        .warnings
        .iter()
        .find(|w| w.contains("per voxel"))
        .unwrap_or_else(|| panic!("no scale advisory in {:?}", report.warnings));
    assert!(hit.contains("0.5"), "{hit}");
    assert!(hit.contains("0.25"), "{hit}");
    assert!(hit.contains(&VOXEL_ID.to_string()), "{hit}");
    // The fix, named: which value to change, and to what.
    assert!(hit.contains("VoxelVolume.voxel_size_m"), "{hit}");
}

/// **The matching silence.** A component that agrees with its asset says nothing.
///
/// Written as its own test rather than an assertion inside the one above,
/// because "the advisory fires" and "the advisory does not over-fire" fail
/// independently and an advisory that fires on correct content is the one that
/// stops being read (cook.rs: *noise is how advisories stop being read*).
#[test]
fn a_matching_voxel_scale_is_silent() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let out = dir.path().join("out");

    let mut entity = cave(0x21_0202, "Cave", VOXEL_ID);
    entity.voxel_volume.as_mut().unwrap().voxel_size_m = 0.5;
    make_project(
        &proj,
        &level(vec![entity]),
        Some((VOXEL_ID, voxel_payload())),
    );

    let report = cook(&proj, &out, &CookOptions::default()).unwrap();
    assert!(
        !report.warnings.iter().any(|w| w.contains("per voxel")),
        "a matching scale must be silent: {:?}",
        report.warnings
    );
}

/// **(b) A see-through pit is SAID, with the measured sample count.** Holed
/// heightfield samples with no voxel volume over them are ground you fall
/// through and sky you see from below — a hazard with no other alarm, because
/// the level loads, the terrain streams and the cook succeeds.
#[test]
fn an_uncovered_hole_is_reported_as_a_see_through_pit() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    let out = dir.path().join("out");

    make_project(
        &proj,
        &level(vec![ground(0x21_0203, "Ground", TERRAIN_ID)]),
        None,
    );
    put(
        &proj,
        "Ground.inf_terrain",
        TERRAIN_ID,
        AssetKind::Terrain,
        &terrain_payload(&[(1, 1), (1, 2), (2, 1), (2, 2)]),
    );

    let report = cook(&proj, &out, &CookOptions::default()).unwrap();
    let hit = report
        .warnings
        .iter()
        .find(|w| w.contains("see-through"))
        .unwrap_or_else(|| panic!("no pit advisory in {:?}", report.warnings));
    // The measured number, not a vague "some".
    assert!(hit.contains("4 holed sample"), "{hit}");
    assert!(hit.contains("(0,0)"), "the tile is named: {hit}");
    assert!(hit.contains(&TERRAIN_ID.to_string()), "{hit}");
    // The fix, in both forms it takes.
    assert!(hit.contains("extend a volume"), "{hit}");
    assert!(hit.contains("fill the holes"), "{hit}");
}

/// **The matching silence, twice over.** A terrain with no holes says nothing,
/// and — the load-bearing half — neither does a hole a voxel volume covers,
/// which is the *correct* authoring of a cave mouth and by far the common case.
#[test]
fn a_covered_hole_and_an_uncarved_terrain_are_both_silent() {
    for holes in [&[][..], &[(1u32, 1u32)][..]] {
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().join("proj");
        let out = dir.path().join("out");

        // The voxel fixture spans chunks (0,0,0)..(1,1,1) at 0.5 m/voxel from the
        // origin, i.e. world XZ [0, 16) — which contains sample (1, 1) of a 1 m
        // tile at the origin.
        let mut entity = cave(0x21_0204, "Cave", VOXEL_ID);
        entity.voxel_volume.as_mut().unwrap().voxel_size_m = 0.5;
        make_project(
            &proj,
            &level(vec![ground(0x21_0205, "Ground", TERRAIN_ID), entity]),
            Some((VOXEL_ID, voxel_payload())),
        );
        put(
            &proj,
            "Ground.inf_terrain",
            TERRAIN_ID,
            AssetKind::Terrain,
            &terrain_payload(holes),
        );

        let report = cook(&proj, &out, &CookOptions::default()).unwrap();
        assert!(
            !report.warnings.iter().any(|w| w.contains("see-through")),
            "holes {holes:?} must be silent: {:?}",
            report.warnings
        );
    }
}
