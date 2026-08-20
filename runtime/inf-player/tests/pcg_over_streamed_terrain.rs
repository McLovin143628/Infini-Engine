//! **PCG evaluates over a streamed terrain, and this arm asserts the world it
//! produces** (island phase, IB-1).
//!
//! # What this file used to be
//!
//! A pinned defect. The AAA-readiness certification measured it and this file
//! asserted it:
//!
//! | terrain | instances | y range |
//! |---|---|---|
//! | authored (inline tiles) | 220 | 50.068 … 59.813 — follows the hill |
//! | asset-backed (streamed) | **929** | **0.000 … 0.000 — all 929 at sea level** |
//!
//! Three facts composed into it. An asset-backed terrain ships **no inline
//! tiles** — its heights live in the `.inf_terrain`. `evaluate_pcg_volumes`
//! picked its height source with `if t.data.is_empty() { return None }`. And the
//! `None` arm was `FnHeight::new(|_, _| Some(0.0))`. The ordering made the first
//! unavoidable: PCG evaluates inside the world builder, `attach_terrain_streaming`
//! runs after `RuntimeSim::new`.
//!
//! The instance **count** differed too, which is the half that made it more than
//! a vertical offset: slope and height masks evaluated against a plane, so the
//! rule set produced a different population. Not a world that needed nudging
//! downward — a different world.
//!
//! # The fix, and why this file now asserts a positive
//!
//! `inf_player::level::page_terrains_for_pcg` pages the level-0 tiles covering
//! every `PcgVolume`'s region (and every `Spline`'s bounds) into the terrain's own
//! working set, from the same source the streamer will use, **before** the
//! evaluator runs. `t.data.is_empty()` was never a bug — it was a true statement
//! that the terrain had no heights yet, and the honest fix is to make it false.
//!
//! So the evaluator, the provider closure, the seed folding and both mirror gates
//! are byte-unchanged, and the two paths converge:
//!
//! **THE CLAIM THIS ARM MAKES IS EQUALITY.** The same volume over the same
//! heights placed as an *authored* terrain and as a *streamed* one must produce
//! the same instances — same count, same positions, bit for bit. A tolerance
//! would let a real divergence through; there is no floating-point reason for
//! one, because both cases read the same `f32` samples through the same bilinear
//! filter.
//!
//! # Assert the WORLD, not the report
//!
//! `page_terrains_for_pcg` returns a tile count and this file checks it, but that
//! is the anti-vacuity guard, not the claim. The claim is where the instances
//! *are*: they follow the hill, they are not at sea level, and they are the same
//! instances the authored terrain produced.

use std::collections::HashMap;

use glam::DVec2;
use inf_ecs::components::{Guid, PcgVolume, Terrain, Transform};
use inf_ecs::math::Vec2d;
use inf_ecs::world::EcsWorld;
use inf_pcg::asset::PcgAssetPayload;
use uuid::Uuid;

const TERRAIN_GUID: Uuid = Uuid::from_u128(0x0000_1111);
const VOLUME_GUID: Uuid = Uuid::from_u128(0x0000_2222);
const GRAPH_GUID: Uuid = Uuid::from_u128(0x0000_3333);
const ASSET_GUID: Uuid = Uuid::from_u128(0x0000_4444);

/// The tile grid the fixture authors on. Small enough that the whole thing is a
/// handful of tiles, large enough that the volume's region spans more than one —
/// a single-tile fixture would not exercise the want set at all.
const RESOLUTION: u32 = 33;
const METERS_PER_SAMPLE: f64 = 2.0;
/// World XZ span authored, metres (four level-0 tiles on a side).
const SPAN: f64 = (RESOLUTION as f64 - 1.0) * METERS_PER_SAMPLE * 4.0;

/// The fixture's hill. **Portable trig** (`psin64`/`pcos64`) because this shape
/// is written into a committed-shaped `.inf_terrain` inside the test and then
/// compared for equality against a second evaluation — `f32`/`f64` `std` trig is
/// not bit-portable across platforms and the P14 law covers exactly this.
fn hill(x: f64, z: f64) -> f64 {
    50.0 + 10.0 * inf_math::psin64(x * 0.05) * inf_math::pcos64(z * 0.05)
}

/// The authored heights, as a `TerrainData` on the fixture's grid.
fn authored_data() -> inf_terrain::TerrainData {
    let mut data = inf_terrain::TerrainData::new(RESOLUTION, METERS_PER_SAMPLE);
    data.write_region(DVec2::ZERO, DVec2::splat(SPAN), hill);
    assert!(!data.is_empty(), "the fixture authored no tiles");
    data
}

/// A world holding one terrain and one scatter volume over it.
///
/// `asset` chooses the two cases: `None` is an **authored** terrain carrying its
/// heights inline (what every committed sample does), `Some` is an
/// **asset-backed** one that streams them (what a 50 km² island must do — the
/// heights are hundreds of megabytes and a `.inf_lvl` is the author's document).
fn build(asset: Option<Uuid>) -> (EcsWorld, HashMap<Uuid, PcgAssetPayload>) {
    let mut world = EcsWorld::default();

    let t = world.spawn_with_guid(TERRAIN_GUID, "Terrain", None);
    let mut terrain = Terrain::configured(RESOLUTION, METERS_PER_SAMPLE);
    match asset {
        None => terrain.data = authored_data(),
        Some(a) => {
            terrain.asset = Some(a);
            assert!(
                terrain.data.is_empty(),
                "an asset-backed terrain ships no inline tiles — if this ever \
                 stops being true, the premise of this whole file has changed"
            );
        }
    }
    world.world_mut().entity_mut(t).insert(terrain);
    world.world_mut().entity_mut(t).insert(Transform::default());

    let v = world.spawn_with_guid(VOLUME_GUID, "Scatter", None);
    world.world_mut().entity_mut(v).insert(PcgVolume {
        graph: Some(GRAPH_GUID),
        extent: Vec2d::new(30.0, 30.0),
        ..PcgVolume::default()
    });
    world.world_mut().entity_mut(v).insert(Transform::default());
    world.propagate();

    let mut pcgs = HashMap::new();
    pcgs.insert(
        GRAPH_GUID,
        PcgAssetPayload {
            schema_version: PcgAssetPayload::CURRENT_VERSION,
            graph_json: None,
            document: inf_editor_core::samples::terrain_demo_pcg_document(),
        },
    );
    (world, pcgs)
}

/// Write the fixture's heights as a real `.inf_terrain`, pyramid and all — the
/// same writer the terrain-import wizard uses, so this exercises the shipping
/// container rather than a stand-in.
fn write_asset(dir: &std::path::Path) -> std::path::PathBuf {
    let data = authored_data();
    let opts = inf_terrain::PyramidOptions::default();
    let pyramid = inf_terrain::build_pyramid(&data, opts);
    let asset = inf_terrain::build_terrain_asset(&data, &pyramid, opts).expect("build asset");
    let path = dir.join("Fixture.inf_terrain");
    inf_terrain::write_terrain_asset(&path, &asset).expect("write asset");
    path
}

/// Every scattered instance's world position, in the order the evaluator wrote
/// them (which is a pure function of the content — `parallel_map` is an in-order
/// pure map, pinned by `portable_placement`).
fn instances(world: &EcsWorld) -> Vec<(f64, f64, f64)> {
    let w = world.world();
    w.iter_entities()
        .filter_map(|e| e.get::<Guid>().map(|g| (g.0, e.id())))
        .filter(|(g, _)| *g == VOLUME_GUID)
        .filter_map(|(_, e)| w.get::<PcgVolume>(e))
        .flat_map(|v| {
            v.evaluated
                .iter()
                .map(|i| (i.position.x, i.position.y, i.position.z))
        })
        .collect()
}

#[test]
fn a_streamed_terrain_scatters_the_same_world_an_authored_one_does() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_asset(tmp.path());

    // ── The control: an authored terrain, which has always worked. ───────────
    let (mut world, pcgs) = build(None);
    inf_player::level::evaluate_pcg_volumes(&mut world, &pcgs);
    let authored = instances(&world);
    assert!(
        !authored.is_empty(),
        "the control scattered nothing — the fixture is not posing the problem"
    );
    let (lo, hi) = (
        authored.iter().map(|i| i.1).fold(f64::INFINITY, f64::min),
        authored
            .iter()
            .map(|i| i.1)
            .fold(f64::NEG_INFINITY, f64::max),
    );
    assert!(
        lo > 40.0 && hi > lo + 1.0,
        "over an authored hill the scatter must FOLLOW it — got y in {lo}..{hi}"
    );

    // ── The subject: the same heights, streamed. ─────────────────────────────
    let (mut world, pcgs) = build(Some(ASSET_GUID));
    let paged = inf_player::level::page_terrains_for_pcg(&mut world, |g| {
        (g == ASSET_GUID)
            .then(|| inf_player::level::terrain_source_from_file(&path).expect("open source"))
    });
    assert!(
        paged > 0,
        "nothing was paged, so whatever the assertions below prove, they do not \
         prove that PCG can read a streamed terrain"
    );
    inf_player::level::evaluate_pcg_volumes(&mut world, &pcgs);
    let streamed = instances(&world);

    // THE CLAIM: the same world, instance for instance.
    assert_eq!(
        streamed.len(),
        authored.len(),
        "the streamed terrain produced a different POPULATION ({} vs {}), which \
         is the half of IB-1 that made it more than a vertical offset — the \
         slope and height masks are reading different ground",
        streamed.len(),
        authored.len()
    );
    assert_eq!(
        streamed, authored,
        "the streamed and authored terrains placed different instances. Both \
         read the same f32 samples through the same bilinear filter, so there is \
         no floating-point reason for a difference and no tolerance is owed."
    );

    // …and the sea-level answer is gone, stated as its own assertion so a future
    // fixture that accidentally authors a flat terrain cannot satisfy the
    // equality above while re-introducing the defect.
    let at_sea_level = streamed.iter().filter(|i| i.1.abs() < 1e-9).count();
    assert_eq!(
        at_sea_level,
        0,
        "{at_sea_level} of {} streamed instances are at exactly y = 0 — the \
         certification's measurement was 929 of 929",
        streamed.len()
    );
    eprintln!(
        "IB-1: authored {} instances (y {lo:.3}..{hi:.3}); streamed {} instances, \
         {paged} tile(s) paged, identical",
        authored.len(),
        streamed.len()
    );
}

/// **A terrain whose asset cannot be resolved still falls back**, rather than
/// failing the load.
///
/// The pre-existing contract: a dangling `Terrain.asset` leaves the terrain on
/// its inline data and the run continues (`TerrainStreaming::attach` warns and
/// carries on). The paging pre-pass must not turn that into a hard error, and it
/// must not page *something else* in its place.
#[test]
fn an_unresolvable_terrain_asset_pages_nothing_and_does_not_fail() {
    let (mut world, pcgs) = build(Some(ASSET_GUID));
    let paged = inf_player::level::page_terrains_for_pcg(&mut world, |_| None);
    assert_eq!(paged, 0, "an unresolvable asset must page nothing");
    inf_player::level::evaluate_pcg_volumes(&mut world, &pcgs);
    // The documented fallback: a flat plane. Asserted rather than assumed,
    // because "it did not crash" is not a behaviour.
    let ys = instances(&world);
    assert!(!ys.is_empty(), "the fallback scattered nothing at all");
    assert!(
        ys.iter().all(|i| i.1.abs() < 1e-9),
        "the fallback is a flat y = 0 plane and this is the ONE place that is \
         still true"
    );
}

/// **Paging an already-authored terrain is a no-op**, so the fix costs the
/// committed content nothing.
///
/// Every sample level in this repository carries inline tiles. If the pre-pass
/// touched them, every determinism trace and every golden in the tree would move,
/// and the equality arm above would be measuring the pre-pass rather than the
/// evaluator.
#[test]
fn an_authored_terrain_is_untouched_by_the_pre_pass() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_asset(tmp.path());
    let (mut world, pcgs) = build(None);

    let before = {
        let w = world.world();
        let e = world.entity_of(TERRAIN_GUID).unwrap();
        w.get::<Terrain>(e).unwrap().data.clone()
    };
    // A resolver that WOULD serve tiles — the point is that the terrain names no
    // asset, so nothing asks it.
    let paged = inf_player::level::page_terrains_for_pcg(&mut world, |_| {
        Some(inf_player::level::terrain_source_from_file(&path).expect("open source"))
    });
    assert_eq!(paged, 0, "an authored terrain names no asset to page from");

    let after = {
        let w = world.world();
        let e = world.entity_of(TERRAIN_GUID).unwrap();
        w.get::<Terrain>(e).unwrap().data.clone()
    };
    assert_eq!(
        before.tiles().count(),
        after.tiles().count(),
        "the pre-pass changed an authored terrain's residency"
    );
    inf_player::level::evaluate_pcg_volumes(&mut world, &pcgs);
    assert!(!instances(&world).is_empty());
}

/// **The region set is the volumes' and the splines', not a bounding box** —
/// the property that keeps this affordable at 50 km².
///
/// A union bounding box over two volumes a kilometre apart would page the
/// kilometre between them, which at island scale is the whole terrain and the
/// whole point of streaming. Measured as a *count* rather than described.
///
/// # The arm has to FALSIFY, and its first version did not
///
/// The I1 audit mutated `pcg_regions_of` to a union bounding box **in both
/// mirrors** and this test stayed green. Its fixture put both volumes on the same
/// `z` row, so their union was a thin strip — four tiles of sixteen — and the
/// bound was `paged < 16`. So the two volumes are now on a **diagonal**, where a
/// union really does span the whole authored world, the union's own cost is
/// *measured here* rather than asserted about, and the per-volume answer is
/// pinned to an exact count. A gate against a cheaper alternative has to price
/// the alternative.
#[test]
fn the_paged_footprint_follows_the_volumes_rather_than_their_bounding_box() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_asset(tmp.path());

    // One volume at the origin, one at the far corner. Their bounding box spans
    // the whole authored world; their own regions are two small squares.
    let corners = [(0.0f64, 0.0f64), (SPAN - 8.0, SPAN - 8.0)];
    let mut world = EcsWorld::default();
    let t = world.spawn_with_guid(TERRAIN_GUID, "Terrain", None);
    let mut terrain = Terrain::configured(RESOLUTION, METERS_PER_SAMPLE);
    terrain.asset = Some(ASSET_GUID);
    world.world_mut().entity_mut(t).insert(terrain);
    world.world_mut().entity_mut(t).insert(Transform::default());
    for (i, (x, z)) in corners.iter().enumerate() {
        let g = Uuid::from_u128(0x0000_5000 + i as u128);
        let v = world.spawn_with_guid(g, "Scatter", None);
        world.world_mut().entity_mut(v).insert(PcgVolume {
            graph: Some(GRAPH_GUID),
            extent: Vec2d::new(4.0, 4.0),
            ..PcgVolume::default()
        });
        world
            .world_mut()
            .entity_mut(v)
            .insert(Transform::from_translation(glam::DVec3::new(*x, 0.0, *z)));
    }
    world.propagate();

    let paged = inf_player::level::page_terrains_for_pcg(&mut world, |g| {
        (g == ASSET_GUID)
            .then(|| inf_player::level::terrain_source_from_file(&path).expect("open source"))
    });

    // THE ALTERNATIVE, PRICED: the same store, the same Ring-0 rule, one call
    // over the union of the two regions — which is what a "just take the bbox"
    // implementation would do. Measured on a fresh working set so it cannot be
    // confused with the count above.
    let store = inf_player::level::terrain_source_from_file(&path).expect("open source");
    let mut union_data = inf_terrain::TerrainData::new(RESOLUTION, METERS_PER_SAMPLE);
    let union = inf_terrain::residency::page_region(
        &mut union_data,
        store.store.as_ref(),
        DVec2::new(corners[0].0 - 4.0, corners[0].1 - 4.0),
        DVec2::new(corners[1].0 + 4.0, corners[1].1 + 4.0),
    )
    .loaded
    .len();

    // Four tiles on a side ⇒ 16 level-0 tiles authored, and two 8 m squares need
    // exactly one each.
    let total_level0 = 16;
    assert_eq!(
        paged, 2,
        "paged {paged} of {total_level0} level-0 tiles for two 8 m squares at \
         opposite corners — one per volume is the whole claim"
    );
    assert_eq!(
        union, total_level0,
        "the union bounding box cost {union} of {total_level0} tiles, so this \
         fixture does not distinguish the two rules and the arm above proves \
         nothing about them"
    );
    eprintln!(
        "IB-1 footprint: {paged} of {total_level0} level-0 tiles for two distant \
         volumes; their union bounding box costs {union}"
    );
}

// ── the production doors (I1 audit, A1) ─────────────────────────────────────
//
// Everything above calls `page_terrains_for_pcg` DIRECTLY, which is how the
// original gate was written and is why the I1 audit could delete the pre-pass
// from `InfSceneWorldBuilder::build` — the one place the shipped player runs it —
// and watch all 64 `inf-player` test binaries stay green. A gate must aim at the
// thing it names: the claim is not "this function works", it is "a level that
// loads scatters on its ground".

/// A level document holding one asset-backed terrain and one scatter volume over
/// it, as the `.inf_lvl` bytes a payload or a pack would carry.
fn level_bytes() -> Vec<u8> {
    use inf_editor_core::ipc::SpawnKind;
    use inf_editor_core::scene::{serialize, SceneDoc};

    let mut doc = SceneDoc::new();
    let t = doc.create_with_guid(TERRAIN_GUID, SpawnKind::Empty, "Terrain", None);
    let mut terrain = Terrain::configured(RESOLUTION, METERS_PER_SAMPLE);
    terrain.asset = Some(ASSET_GUID);
    assert!(
        terrain.data.is_empty(),
        "an asset-backed terrain ships no tiles"
    );
    doc.world_mut().world_mut().entity_mut(t).insert(terrain);
    doc.world_mut()
        .world_mut()
        .entity_mut(t)
        .insert(Transform::default());

    let v = doc.create_with_guid(VOLUME_GUID, SpawnKind::Empty, "Scatter", None);
    doc.world_mut().world_mut().entity_mut(v).insert(PcgVolume {
        graph: Some(GRAPH_GUID),
        extent: Vec2d::new(30.0, 30.0),
        ..PcgVolume::default()
    });
    doc.world_mut()
        .world_mut()
        .entity_mut(v)
        .insert(Transform::default());
    doc.world_mut().propagate();
    serialize::encode(&serialize::to_scene_file(&doc)).expect("the fixture level encodes")
}

/// The fixture's `.inf_pcg`, as the bytes a payload carries.
fn pcg_bytes() -> Vec<u8> {
    PcgAssetPayload {
        schema_version: PcgAssetPayload::CURRENT_VERSION,
        graph_json: None,
        document: inf_editor_core::samples::terrain_demo_pcg_document(),
    }
    .encode()
    .expect("the fixture graph encodes")
}

/// Every scattered instance in a built world, in evaluation order.
fn built_instances(built: &inf_player::level::BuiltWorld) -> Vec<(f64, f64, f64)> {
    instances(&built.world)
}

/// **THE PRODUCTION DOOR.** A `ScenePayload` that carries its `.inf_terrain`
/// scatters on the hill; the same payload without it scatters at sea level.
///
/// This is the arm the I1 audit found missing. `build_world_from_payload` is the
/// PIE path and it wires the resolver that `InfSceneWorldBuilder::build` feeds to
/// the pre-pass, so **both** the wiring and the pre-pass are load-bearing here:
/// stub either and the `y = 0` control below becomes the subject.
///
/// The negative half is the anti-vacuity guard, and it is the certification's own
/// measurement in miniature — 929 of 929 at exactly sea level, because a payload
/// with no terrain bytes has nothing to page and takes the documented flat
/// fallback.
#[test]
fn a_payload_that_carries_its_terrain_scatters_on_the_hill() {
    let tmp = tempfile::tempdir().unwrap();
    let path = write_asset(tmp.path());
    let terrain_bytes = std::fs::read(&path).expect("read the .inf_terrain");

    let base = || {
        inf_runtime::pie::ScenePayload::new("IB-1", level_bytes(), Vec::new(), 60, false)
            .with_pcgs(vec![(GRAPH_GUID, pcg_bytes())])
    };

    // WITH the terrain bytes: the shipped world builder pages them and the
    // scatter follows the hill.
    let built = inf_player::build_world_from_payload(
        &base().with_terrains(vec![(ASSET_GUID, terrain_bytes)]),
    )
    .expect("the payload builds a world");
    let got = built_instances(&built);
    assert!(
        !got.is_empty(),
        "the payload door scattered nothing — the fixture is not posing the problem"
    );
    let (lo, hi) = (
        got.iter().map(|i| i.1).fold(f64::INFINITY, f64::min),
        got.iter().map(|i| i.1).fold(f64::NEG_INFINITY, f64::max),
    );
    assert!(
        lo > 40.0 && hi > lo + 1.0,
        "the payload door scattered at y in {lo}..{hi}; the hill is 40..60 m. \
         Either the world builder stopped running the paging pre-pass or \
         `build_world_from_payload` stopped wiring its terrain resolver — the \
         two halves of IB-1's fix, and this is the only arm that runs them."
    );

    // …and it is the SAME world the authored terrain produces, which is the
    // equality claim, re-made through the door rather than through the function.
    let (mut authored_world, pcgs) = build(None);
    inf_player::level::evaluate_pcg_volumes(&mut authored_world, &pcgs);
    assert_eq!(
        got,
        instances(&authored_world),
        "the payload door and an authored terrain placed different instances"
    );

    // THE CONTROL: no terrain bytes, so nothing can be paged.
    let flat = inf_player::build_world_from_payload(&base()).expect("the payload builds a world");
    let flat = built_instances(&flat);
    assert!(!flat.is_empty(), "the control scattered nothing");
    assert!(
        flat.iter().all(|i| i.1.abs() < 1e-9),
        "the control is the documented flat fallback and did not take it — so \
         the arm above is not measuring the paging"
    );
    eprintln!(
        "IB-1 payload door: {} instance(s) at y {lo:.3}..{hi:.3}; with no terrain \
         bytes, {} at exactly sea level",
        got.len(),
        flat.len()
    );
}

/// **All three boot paths wire the resolver** — one door, three paths.
///
/// `build_world` (loose `.inf_lvl`), `build_world_from_pack` (the shipped pack)
/// and `build_world_from_payload` (PIE) each construct their own
/// `InfSceneWorldBuilder`, and only the last is reachable from a test in this
/// crate. A path that forgot the resolver would scatter at sea level while its
/// two siblings did not — the P21 "a boot path that forgets an attachment does
/// not crash, it agrees with itself" law, one boot path down.
#[test]
fn every_boot_path_wires_the_pcg_terrain_resolver() {
    let src = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs"),
    )
    .expect("read the player's boot paths");
    let n = src.matches(".with_terrain_resolver(").count();
    assert_eq!(
        n, 3,
        "the player wires the PCG terrain resolver on {n} boot path(s), not 3 \
         (loose dir, cooked pack, PIE payload). A path without it scatters every \
         instance over a streamed terrain at exactly y = 0."
    );
}
