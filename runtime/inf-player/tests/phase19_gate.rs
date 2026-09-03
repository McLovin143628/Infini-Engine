//! **THE PHASE 19 GATE** — biomes, PCG grammar and *enterable* structures.
//!
//! The fixture is the committed `samples/phase19-town`: a partitioned level over
//! a biome-painted terrain, a spline road with a grammar fence running its whole
//! length, and **one building lot per archetype**, each a `PcgVolume`
//! carrying its own `.inf_pcg`.
//!
//! Six arms, in the order the phase's claim needs them:
//!
//! * **(a) determinism** — two fresh loads of one pack agree on the whole trace:
//!   the instance population, the *solid* population, and the partition's cell
//!   directory. Plus pool-size invariance through the shipped content's own
//!   passes.
//! * **(b) cooked == uncooked** — the pack and the dev directory build the same
//!   town, bit for bit.
//! * **(c) PIE == shipping** — the editor's PIE payload builds the same town,
//!   bit for bit, on the P19.4 `bits()` standard, *and* on the solids.
//! * **(d) ENTERABILITY** — the headline. For one building per archetype: the
//!   room graph is connected on every floor; no collider intrudes into any door
//!   opening; and every floor is reachable **from outside** by a graph walk
//!   through the entrance door, the interior doors and the stair cores.
//! * **(e) partition** — the lots bin into the cells their transforms say they
//!   do, and the building's own content stays inside its lot.
//! * **(f) budget** — the composed scene builds inside the load budget.
//!
//! # Why the enterability arm asserts on the PLAN
//!
//! A door is a *hole*, and a hole is the absence of something. There is no
//! instance to look at and no collider to name — the assertion has to be "no
//! solid overlaps this rectangle", which needs the rectangle, which is a
//! property of the plan. So the gate re-derives the plans through
//! `inf_pcg::plans_of`, the same resolution `evaluate_buildings_in` performs, and
//! checks them against the colliders the *shipped pack* actually produced.
//! Neither half is re-invented here.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::DVec2;
use uuid::Uuid;

use inf_ecs::components::{PcgVolume, Transform};
use inf_ecs::{Guid, ScatteredInstance, ScatteredSolid};
use inf_editor_core::samples::{
    phase19_biome_at, phase19_biome_set, phase19_height, phase19_lot_pcg_guid,
    phase19_lot_position, phase19_town_dir, PHASE19_BIOME_SET_GUID, PHASE19_CELL_SIZE_M,
    PHASE19_LOT_EXTENT, PHASE19_LOT_FLOORS, PHASE19_ROAD_PCG_GUID,
};
use inf_packager::{cook, CookOptions, DEFAULT_PACK_NAME};
use inf_player::cell_stream::derived_partition_id;
use inf_player::level::{
    self, BuiltWorld, DevDirLevelSource, InfSceneWorldBuilder, PackLevelSource,
};
use inf_project::ProjectManifest;
use inf_scene::partition::{cell_of, CellCoord, PartitionAssetReader};

// Budgets are imported, never redeclared: a phase does not get its own budget for
// being new, and a private copy is somewhere for one to be quietly raised. Each arm
// takes the budget of its own *class* — the recurring-work arm the per-frame
// tripwire, the one-shot-load arm the load-class ceiling. They are not
// interchangeable, and neither may be restated as a literal here.
use inf_core::FRAME_BUDGET_MS;
use inf_player::budget::LOAD_BUDGET_MS;

/// Every committed file of the sample, so the fixture copy cannot silently miss
/// one as the sample grows.
fn sample_files() -> Vec<String> {
    let mut out = vec![
        "Phase19Town.inf_lvl".to_string(),
        "Phase19Town.inf_lvl.toml".to_string(),
        "Town.inf_biomes".to_string(),
        "Town.inf_biomes.toml".to_string(),
        "Roadside.inf_pcg".to_string(),
        "Roadside.inf_pcg.toml".to_string(),
    ];
    for id in inf_pcg::ArchetypeId::ALL {
        out.push(format!("Lot{}.inf_pcg", id.name()));
        out.push(format!("Lot{}.inf_pcg.toml", id.name()));
    }
    out
}

/// Scaffold a project holding the sample and cook it; returns `(content, pack)`.
fn cook_town(tmp: &Path) -> (PathBuf, PathBuf) {
    let proj = tmp.join("proj");
    ProjectManifest::new("Phase 19 Town", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = phase19_town_dir();
    for f in sample_files() {
        std::fs::copy(src.join(&f), content.join(&f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("cook succeeds");
    (content, out)
}

fn pack_built(pack: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack).expect("pack opens");
    inf_player::build_world_from_pack(&source).expect("pack world builds")
}

fn dir_built(content: &Path) -> BuiltWorld {
    let source = DevDirLevelSource::new(content.join("Phase19Town.inf_lvl"));
    let builder = InfSceneWorldBuilder::with_defaults(Vec::new())
        .with_pcgs(level::load_pcg_payloads_by_guid_from_dir(content));
    level::load(&source, &builder).expect("dev-dir world builds")
}

/// The editor's PIE payload for the committed sample: the document plus every
/// `.inf_pcg` its `PcgVolume.graph` refs resolve to.
fn pie_built() -> BuiltWorld {
    let dir = phase19_town_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("Phase19Town.inf_lvl"))
        .expect("the committed phase19 document loads");
    let mut pcgs: BTreeMap<Uuid, Vec<u8>> = BTreeMap::new();
    pcgs.insert(
        PHASE19_ROAD_PCG_GUID,
        std::fs::read(dir.join("Roadside.inf_pcg")).unwrap(),
    );
    for (i, id) in inf_pcg::ArchetypeId::ALL.into_iter().enumerate() {
        pcgs.insert(
            phase19_lot_pcg_guid(i),
            std::fs::read(dir.join(format!("Lot{}.inf_pcg", id.name()))).unwrap(),
        );
    }
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_| None,
        |guid| pcgs.get(&guid).cloned(),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        // P22.3: no destructible meshes in this fixture.
        |_| None,
        // P26.3b: the cloth / hair / material / texture byte resolver.
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    assert_eq!(
        payload.pcgs.len(),
        1 + inf_pcg::ArchetypeId::ALL.len(),
        "the road graph and every lot graph must ride the payload"
    );
    inf_player::build_world_from_payload(&payload).expect("PIE world builds")
}

// ── probes ──────────────────────────────────────────────────────────────────

/// Every volume's evaluated instances, in **Guid-sorted volume order** so the
/// probe is a function of the content rather than of ECS iteration order.
fn population(built: &BuiltWorld) -> Vec<ScatteredInstance> {
    volumes(built)
        .into_iter()
        .flat_map(|(_, v, _)| v.evaluated)
        .collect()
}

/// Every volume's evaluated **solids**, in the same order.
fn solids(built: &BuiltWorld) -> Vec<ScatteredSolid> {
    volumes(built)
        .into_iter()
        .flat_map(|(_, v, _)| v.structures)
        .collect()
}

/// `(guid, volume, world position)` for every `PcgVolume`, Guid-sorted.
fn volumes(built: &BuiltWorld) -> Vec<(Uuid, PcgVolume, DVec2)> {
    let w = built.world.world();
    let mut out: Vec<(Uuid, PcgVolume, DVec2)> = w
        .iter_entities()
        .filter_map(|e| {
            let g = e.get::<Guid>()?.0;
            let v = e.get::<PcgVolume>()?.clone();
            let t = e.get::<Transform>().copied().unwrap_or(Transform::IDENTITY);
            let p = t.translation.to_dvec3();
            Some((g, v, DVec2::new(p.x, p.z)))
        })
        .collect();
    out.sort_by_key(|(g, _, _)| *g);
    out
}

/// Every instance's placement as raw bits — the P19.4 standard, unchanged: a
/// low-bit divergence between two hosts is "the shipped wall is three nanometres
/// from the preview's", which is the shape of bug a player finds.
fn bits(v: &[ScatteredInstance]) -> Vec<[u64; 8]> {
    v.iter()
        .map(|i| {
            let r = i.rotation.to_array();
            [
                i.position.x.to_bits(),
                i.position.y.to_bits(),
                i.position.z.to_bits(),
                r[0].to_bits(),
                r[1].to_bits(),
                r[2].to_bits(),
                r[3].to_bits(),
                i.scale.to_bits(),
            ]
        })
        .collect()
}

/// The same standard for the **solid** half. Without it, two hosts could agree
/// about every visible instance and disagree about where the walls actually
/// stop — which is precisely the half that decides whether a doorway is a
/// doorway.
fn solid_bits(v: &[ScatteredSolid]) -> Vec<[u64; 10]> {
    v.iter()
        .map(|s| {
            let r = s.rotation.to_array();
            [
                s.center.x.to_bits(),
                s.center.y.to_bits(),
                s.center.z.to_bits(),
                s.half_extents.x.to_bits(),
                s.half_extents.y.to_bits(),
                s.half_extents.z.to_bits(),
                r[0].to_bits(),
                r[1].to_bits(),
                r[2].to_bits(),
                r[3].to_bits(),
            ]
        })
        .collect()
}

/// The plans the shipped content builds, one per lot, in
/// [`inf_pcg::ArchetypeId::ALL`] order.
///
/// Re-derived through `inf_pcg::plans_of` — the **same** lot and datum
/// resolution `evaluate_buildings_in` performs, pinned equal to assembly by
/// `building::pass::tests::plans_match_what_evaluation_builds`. The gate does
/// not re-implement the layout; it asks the runtime what it built.
fn shipped_plans(content: &Path) -> Vec<(inf_pcg::ArchetypeId, inf_pcg::BuildingPlan)> {
    let mut out = Vec::new();
    for (i, id) in inf_pcg::ArchetypeId::ALL.into_iter().enumerate() {
        let bytes = std::fs::read(content.join(format!("Lot{}.inf_pcg", id.name()))).unwrap();
        let payload = inf_pcg::PcgAssetPayload::decode(&bytes).expect("payload decodes");
        let graph = payload.graph().expect("the graph is the source of truth");
        let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
        assert!(lowered.ok, "{}: {:?}", id.name(), lowered.issues);
        assert_eq!(lowered.buildings.len(), 1, "{} has one plan", id.name());
        let p = phase19_lot_position(i);
        let height = inf_pcg::FnHeight::new(|x, z| Some(phase19_height(x, z)));
        let cx = inf_pcg::GrammarContext {
            entity: None,
            center: p,
            extent: DVec2::new(PHASE19_LOT_EXTENT.0, PHASE19_LOT_EXTENT.1),
            seed_offset: 100 + i as u64,
        };
        let plans = inf_pcg::plans_of(&lowered.buildings, &inf_pcg::NoSplines, &height, &cx);
        assert_eq!(plans.len(), 1, "{} resolves one lot", id.name());
        out.push((id, plans[0].clone()));
    }
    out
}

// ── (a) determinism ─────────────────────────────────────────────────────────

/// **Two fresh loads of one pack agree on the whole trace.** Not only the
/// instances: the *solids* too (a wall that renders in the same place and
/// collides in a different one is the specific failure this batch could
/// introduce), and the partition's cell directory beside them.
#[test]
fn the_whole_trace_is_identical_across_two_loads() {
    let dir = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_town(dir.path());

    let a = pack_built(&pack);
    let b = pack_built(&pack);
    let (pa, pb) = (population(&a), population(&b));
    assert!(
        pa.len() > 2_000,
        "the town must be substantial for this to mean anything, got {}",
        pa.len()
    );
    assert_eq!(bits(&pa), bits(&pb), "the population moved between loads");

    let (sa, sb) = (solids(&a), solids(&b));
    assert!(sa.len() > 1_000, "only {} solids", sa.len());
    assert_eq!(
        solid_bits(&sa),
        solid_bits(&sb),
        "the SOLIDS moved between loads"
    );

    // The partition's own directory: same cells, same entity counts.
    let cells_a = cell_directory(&pack);
    let cells_b = cell_directory(&pack);
    assert_eq!(cells_a, cells_b, "cell residency moved");
    assert!(cells_a.len() > 2, "the world must span several cells");
}

/// The cooked pack's cell directory, read out of the `.inf_part` the cook
/// derived: `coord → entity count`, with the persistent cell under `None`.
fn cell_directory(pack: &Path) -> BTreeMap<Option<CellCoord>, u32> {
    let bytes = partition_bytes(pack);
    let asset = PartitionAssetReader::new(bytes.as_slice()).expect("a real .inf_part");
    asset
        .directory()
        .iter()
        .map(|e| (e.key.coord(), e.entity_count))
        .collect()
}

/// The `.inf_part` payload for the cooked level, resolved the way the runtime
/// does: derive the partition's GUID from the level's, with no side index.
fn partition_bytes(pack: &Path) -> Vec<u8> {
    let reader = inf_asset::PackReader::open(&pack.join(DEFAULT_PACK_NAME)).unwrap();
    let level_id = reader
        .index()
        .find(|e| e.kind == inf_asset::AssetKind::Level)
        .expect("the pack has a level")
        .guid;
    let part_id = inf_asset::AssetId(derived_partition_id(level_id.uuid()));
    assert!(
        reader.contains(part_id),
        "a partitioned level must cook a .inf_part"
    );
    reader.read(part_id).unwrap()
}

/// **Pool-size invariance through the shipped content's own building passes.**
/// The passes are read back out of the cooked pack and driven through the real
/// `evaluate_buildings_in` seam at 1/2/4/8 workers, so the lot resolution, the
/// plan, the wall expansion, the furniture walk and the concatenation order all
/// participate on the very bytes a player would load.
#[test]
fn the_shipped_building_passes_are_invariant_under_pool_size() {
    use inf_core::JobPool;

    let dir = tempfile::tempdir().unwrap();
    let (content, _pack) = cook_town(dir.path());
    let bytes = std::fs::read(content.join("LotHotel.inf_pcg")).unwrap();
    let payload = inf_pcg::PcgAssetPayload::decode(&bytes).unwrap();
    let graph = payload.graph().expect("graph rides the payload");
    let lowered = inf_pcg::lower_graph(&graph, &inf_pcg::pcg_registry());
    assert!(lowered.has_buildings());

    let height = inf_pcg::FnHeight::new(|x, z| Some(phase19_height(x, z)));
    let idx = inf_pcg::ArchetypeId::ALL
        .iter()
        .position(|a| *a == inf_pcg::ArchetypeId::Hotel)
        .unwrap();
    let cx = inf_pcg::GrammarContext {
        entity: None,
        center: phase19_lot_position(idx),
        extent: DVec2::new(PHASE19_LOT_EXTENT.0, PHASE19_LOT_EXTENT.1),
        seed_offset: 100 + idx as u64,
    };
    let want = inf_pcg::evaluate_buildings_in(
        &JobPool::new(1),
        &lowered.buildings,
        &inf_pcg::NoSplines,
        &height,
        &cx,
    );
    assert!(
        want.instances.len() > 200,
        "only {} instances",
        want.instances.len()
    );
    for n in [2usize, 4, 8] {
        let got = inf_pcg::evaluate_buildings_in(
            &JobPool::new(n),
            &lowered.buildings,
            &inf_pcg::NoSplines,
            &height,
            &cx,
        );
        assert_eq!(want, got, "the building differs on an {n}-worker pool");
    }
}

// ── (b) cooked == uncooked ──────────────────────────────────────────────────

/// **The pack and the dev directory build the same town.** Both halves: the
/// visible instances and the solids, each bit for bit.
#[test]
fn cooked_equals_uncooked() {
    let dir = tempfile::tempdir().unwrap();
    let (content, pack) = cook_town(dir.path());

    let ship = pack_built(&pack);
    let dev = dir_built(&content);
    let (a, b) = (population(&ship), population(&dev));
    assert!(!a.is_empty());
    assert_eq!(
        a.len(),
        b.len(),
        "cooked and uncooked place different counts"
    );
    assert_eq!(bits(&a), bits(&b), "cooked != uncooked on placement bits");
    assert_eq!(
        solid_bits(&solids(&ship)),
        solid_bits(&solids(&dev)),
        "cooked != uncooked on the SOLIDS"
    );
}

/// The cook follows the whole chain with only the level as an explicit root: the
/// biome set, the roadside graph and every lot graph land in the pack.
#[test]
fn the_cook_follows_the_level_to_every_graph_and_the_biome_set() {
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join("proj");
    ProjectManifest::new("Phase 19 Town", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = phase19_town_dir();
    for f in sample_files() {
        std::fs::copy(src.join(&f), content.join(&f)).unwrap();
    }

    let mut db = inf_asset::AssetDb::new(content.clone());
    db.scan().unwrap();
    let level_id = db
        .iter()
        .find(|e| e.kind() == inf_asset::AssetKind::Level)
        .expect("level present")
        .id();
    let out = dir.path().join("roots-out");
    let report = cook(
        &proj,
        &out,
        &CookOptions {
            roots: Some(vec![level_id]),
            ..Default::default()
        },
    )
    .expect("cook succeeds");
    assert_eq!(
        report.kinds.get("pcg").copied(),
        Some(1 + inf_pcg::ArchetypeId::ALL.len()),
        "the road graph plus every lot graph must be reached from the level alone"
    );
    assert_eq!(
        report
            .kinds
            .get("biomeSet")
            .copied()
            .or(report.kinds.get("biome_set").copied()),
        Some(1),
        "the biome set must ride along; kinds were {:?}",
        report.kinds
    );
}

// ── (c) PIE == shipping ─────────────────────────────────────────────────────

/// **PIE == shipping, bit for bit, on both halves.**
#[test]
fn pie_matches_shipping_for_the_whole_town() {
    let dir = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_town(dir.path());
    let ship = pack_built(&pack);
    let pie = pie_built();

    let (a, b) = (population(&pie), population(&ship));
    assert!(
        b.len() > 2_000,
        "the parity arm must compare a real town, got {}",
        b.len()
    );
    assert_eq!(a.len(), b.len(), "PIE and shipping place different counts");
    assert_eq!(bits(&a), bits(&b), "PIE != shipping on placement bits");
    assert_eq!(
        solid_bits(&solids(&pie)),
        solid_bits(&solids(&ship)),
        "PIE != shipping on the SOLIDS — the halves that decide enterability"
    );
}

// ── (d) ENTERABILITY — the headline ─────────────────────────────────────────

/// **THE HEADLINE ASSERTION.** For **every** archetype, on the shipped content:
///
/// 1. every floor's room graph is connected through door openings;
/// 2. every door opening's rect contains **no** collider — walls, slabs,
///    lintels, stair treads and furniture alike;
/// 3. every floor is reachable **from outside**, by a graph walk from the
///    entrance door through the interior doors and up the stair cores.
///
/// The colliders are the ones the *pack* produced, not a re-derivation: the arm
/// asks the shipped world what it built and then asks the plan where the holes
/// are supposed to be.
#[test]
fn every_archetype_ships_an_enterable_building() {
    let dir = tempfile::tempdir().unwrap();
    let (content, pack) = cook_town(dir.path());
    let built = pack_built(&pack);
    let all_solids = solids(&built);
    assert!(!all_solids.is_empty(), "the pack shipped no solids at all");

    // The colliders, in `inf_pcg`'s own vocabulary, so `opening_is_clear` — the
    // one implementation of "is this hole a hole" — is what decides.
    let solids_pcg: Vec<inf_pcg::PcgCollider> = all_solids
        .iter()
        .map(|s| inf_pcg::PcgCollider {
            center: s.center,
            half_extents: s.half_extents,
            rotation: s.rotation,
        })
        .collect();

    for (id, plan) in shipped_plans(&content) {
        let what = id.name();
        assert_eq!(plan.floors, PHASE19_LOT_FLOORS, "{what}: storey count");
        assert!(!plan.rooms.is_empty(), "{what}: no rooms");

        // 1. Connectivity, floor by floor.
        for f in 0..plan.floors {
            assert!(
                plan.rooms_connected(f),
                "{what}: floor {f}'s room graph is not connected"
            );
        }

        // 3. Reachable from OUTSIDE — the walk that makes "enterable" mean
        //    something. Asserted before the clearance loop so a sealed building
        //    fails with the right message.
        assert!(
            plan.entrance.is_some(),
            "{what}: no entrance — the building is sealed"
        );
        assert!(
            plan.floors_reachable(),
            "{what}: a floor cannot be reached from outside"
        );
        assert!(
            plan.fully_reachable(),
            "{what}: a ROOM cannot be reached from outside"
        );
        assert_eq!(
            plan.stairs.len(),
            (PHASE19_LOT_FLOORS - 1) as usize,
            "{what}: wrong flight count"
        );

        // 2. No collider in any door's void. The margin is 2 cm — a jamb that
        //    stops exactly at the opening is not a blockage, and the exact-fill
        //    layout puts wall runs exactly there.
        let doors: Vec<&inf_pcg::Opening> = plan
            .openings
            .iter()
            .filter(|o| o.kind == inf_pcg::OpeningKind::Door)
            .collect();
        assert!(
            doors.len() >= plan.floors as usize,
            "{what}: only {} doors for {} floors",
            doors.len(),
            plan.floors
        );
        for (i, d) in doors.iter().enumerate() {
            assert!(
                plan.opening_is_clear(d, &solids_pcg, 0.02),
                "{what}: door {i} on wall {} is blocked by a collider",
                d.wall
            );
        }

        // **The control.** Every assertion above is of the form "no solid is
        // here", and such an assertion passes trivially if the predicate cannot
        // say *no*. (It could not, once: a degenerate void inverted under the
        // margin and `Rect2::intersection` never fired.) So: drop a slab through
        // the whole building and require every door to report blocked. This leg
        // is what keeps the enterability arm from silently disarming.
        let f = plan.footprint;
        let top = plan.floor_y(plan.floors);
        let block = [inf_pcg::PcgCollider {
            center: glam::DVec3::new(f.center().x, (plan.base_y + top) * 0.5, f.center().y),
            half_extents: glam::DVec3::new(f.size_x(), (top - plan.base_y) * 0.5 + 2.0, f.size_z()),
            rotation: glam::DQuat::IDENTITY,
        }];
        for (i, d) in doors.iter().enumerate() {
            assert!(
                !plan.opening_is_clear(d, &block, 0.02),
                "{what}: door {i} reads CLEAR through a solid building — \
                 the enterability predicate is vacuous"
            );
        }
        // Windows too: their band must be clear even though the wall below the
        // sill is solid.
        for w in plan
            .openings
            .iter()
            .filter(|o| o.kind == inf_pcg::OpeningKind::Window)
        {
            assert!(
                plan.opening_is_clear(w, &solids_pcg, 0.02),
                "{what}: a window band is blocked"
            );
        }
    }
}

/// The stair walk, stated as its own arm because it is the part a "rooms are
/// connected" assertion would miss: **a floor is only reachable if a flight
/// actually lands on it**. Removing a flight must break this.
#[test]
fn every_floor_is_reached_through_a_real_stair_core() {
    let dir = tempfile::tempdir().unwrap();
    let (content, _pack) = cook_town(dir.path());
    for (id, plan) in shipped_plans(&content) {
        let what = id.name();
        let core = plan.core.unwrap_or_else(|| panic!("{what}: no stair core"));
        // The core is the SAME rectangle on every storey — the alignment
        // guarantee, restated on the shipped plan.
        for f in 0..plan.floors {
            let r = plan.rooms[plan
                .stair_room(f)
                .unwrap_or_else(|| panic!("{what}: floor {f} has no stair room"))]
            .rect;
            assert_eq!(r, core, "{what}: floor {f}'s stair core moved");
        }
        // Deleting the flights makes the upper floors unreachable — so the walk
        // is really using them, and the assertion is not passing for some other
        // reason.
        let mut severed = plan.clone();
        severed.stairs.clear();
        assert!(
            !severed.floors_reachable(),
            "{what}: floors are reachable with NO stairs — the walk is vacuous"
        );
        assert!(plan.floors_reachable(), "{what}: floors unreachable");
    }
}

/// The fence really is solid, and a P19.4 grammar that declares no `collider`
/// still produces none — the opt-in stated as a shipped-content property.
#[test]
fn the_roadside_fence_is_a_real_barrier() {
    let dir = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_town(dir.path());
    let built = pack_built(&pack);
    let road = volumes(&built)
        .into_iter()
        .find(|(g, _, _)| *g == inf_editor_core::samples::PHASE19_ROAD_GUID)
        .expect("the road volume is in the world");
    assert!(
        road.1.structures.len() > 50,
        "the fence placed only {} solids",
        road.1.structures.len()
    );
    // Every fence solid stands on the road line, within the volume's own band.
    for s in &road.1.structures {
        let dz = (s.center.z - road.2.y).abs();
        assert!(
            dz < 4.0,
            "a fence solid {dz} m off the road line at {:?}",
            s.center
        );
    }
    // The verge scatter beside it declares no collider, so the fence's solids
    // are strictly fewer than its instances.
    assert!(
        road.1.structures.len() < road.1.evaluated.len(),
        "every instance became a solid — the opt-in is not opting"
    );
}

// ── (e) partition ───────────────────────────────────────────────────────────

/// **Entities bin into the cells their transforms say they do, and a building's
/// content stays inside its own lot.**
///
/// Two halves, because the engine has an honest seam between them:
///
/// * the **street lamps** are ordinary placed entities and really do stream —
///   each is in the grid cell `cell_of` puts it in, and nowhere else;
/// * the **lots** are `AlwaysLoaded` and therefore persistent, deliberately: PCG
///   evaluation is a load-time pass, so a volume in a streamed cell would spawn
///   with an empty building on it (the standing P10.6 remainder, restated in the
///   sample's own docs). What is asserted about them instead is the property
///   that survives that gap — every instance a lot places lies inside the lot's
///   own footprint, so the day evaluation follows streaming the content is
///   already in the right cell.
#[test]
fn the_lots_respect_the_partition_cells() {
    use inf_editor_core::samples::{phase19_lamp_guid, phase19_lamp_position, PHASE19_LAMPS};
    use inf_scene::partition::CellKey;

    let dir = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_town(dir.path());
    let bytes = partition_bytes(&pack);
    let view = PartitionAssetReader::new(bytes.as_slice()).expect("a real .inf_part");
    assert_eq!(view.cell_size_m(), PHASE19_CELL_SIZE_M);

    // ── the streamed half: every lamp is in exactly its own cell ──
    let mut want: BTreeMap<CellCoord, Vec<Uuid>> = BTreeMap::new();
    for i in 0..PHASE19_LAMPS {
        let p = phase19_lamp_position(i);
        want.entry(cell_of(p.x, p.z, PHASE19_CELL_SIZE_M))
            .or_default()
            .push(phase19_lamp_guid(i));
    }
    assert!(
        want.len() >= 3,
        "the streamed content must span several cells for this to mean anything ({want:?})"
    );
    for (coord, guids) in &want {
        let entities = view
            .cell(CellKey::grid(*coord))
            .expect("cell decodes")
            .unwrap_or_else(|| panic!("cell {coord:?} is not in the directory"));
        let got: std::collections::BTreeSet<Uuid> = entities.iter().map(|e| e.guid).collect();
        for g in guids {
            assert!(got.contains(g), "lamp {g} is not in cell {coord:?}");
        }
    }
    // …and nothing else streams: a lot that lost its marker would show up here.
    let streamed: std::collections::BTreeSet<Uuid> = view
        .grid_coords()
        .flat_map(|c| {
            view.cell(CellKey::grid(c))
                .expect("cell decodes")
                .unwrap_or_default()
        })
        .map(|e| e.guid)
        .collect();
    assert_eq!(
        streamed.len(),
        PHASE19_LAMPS,
        "exactly the lamps stream; a PcgVolume in a grid cell would never evaluate"
    );
    for i in 0..inf_pcg::ArchetypeId::ALL.len() {
        assert!(
            !streamed.contains(&inf_editor_core::samples::phase19_lot_guid(i)),
            "lot {i} is streamed — it would spawn with an empty building on it"
        );
    }

    // ── the persistent half: the buildings stay inside their lots ──
    // Every instance a lot placed is inside its footprint plus a metre of slack.
    let built = pack_built(&pack);
    for (i, id) in inf_pcg::ArchetypeId::ALL.into_iter().enumerate() {
        let guid = inf_editor_core::samples::phase19_lot_guid(i);
        let (_, vol, at) = volumes(&built)
            .into_iter()
            .find(|(g, _, _)| *g == guid)
            .unwrap_or_else(|| panic!("{} lot missing", id.name()));
        assert!(!vol.evaluated.is_empty(), "{} placed nothing", id.name());
        let slack = 1.0;
        for inst in &vol.evaluated {
            let d = DVec2::new(inst.position.x - at.x, inst.position.z - at.y).abs();
            assert!(
                d.x <= PHASE19_LOT_EXTENT.0 + slack && d.y <= PHASE19_LOT_EXTENT.1 + slack,
                "{}: an instance {d:?} from the lot centre escaped its footprint",
                id.name()
            );
        }
        // Every instance therefore bins into the same cell the volume does.
        let cell = cell_of(at.x, at.y, PHASE19_CELL_SIZE_M);
        for inst in &vol.evaluated {
            let c = cell_of(inst.position.x, inst.position.z, PHASE19_CELL_SIZE_M);
            assert!(
                (c.0 - cell.0).abs() <= 1 && (c.1 - cell.1).abs() <= 1,
                "{}: an instance binned to {c:?}, two cells from its volume's {cell:?}",
                id.name()
            );
        }
    }
}

/// The biomes really are painted and really do round-trip through the cook — the
/// P19.2 half of the phase, gated on the composed scene rather than only in its
/// own batch's tests.
#[test]
fn the_painted_biomes_survive_the_cook() {
    use inf_ecs::components::Terrain;

    let dir = tempfile::tempdir().unwrap();
    let (content, pack) = cook_town(dir.path());
    let built = pack_built(&pack);
    let w = built.world.world();
    let terrain = w
        .iter_entities()
        .find_map(|e| e.get::<Terrain>())
        .expect("the town has a terrain");
    assert_eq!(
        terrain.biome_set,
        Some(PHASE19_BIOME_SET_GUID),
        "the terrain lost its biome vocabulary"
    );
    assert!(
        !terrain.data.biomes_are_default(),
        "nothing is painted — the biome layer costs nothing and proves nothing"
    );
    // Both painted ids are present, and each reads back where it was painted.
    for (x, z) in [(120.0, 256.0), (400.0, 40.0), (60.0, 480.0), (300.0, 260.0)] {
        assert_eq!(
            terrain.data.biome_at(DVec2::new(x, z)),
            Some(phase19_biome_at(x, z)),
            "biome at ({x}, {z}) is not what the generator painted"
        );
    }
    // The set itself survives the cook, and its P19.2 `structure_hint` names a
    // real archetype — the field P19.2 declared "because it is what P19.5 will
    // ask a biome for", answered.
    let set: inf_terrain::BiomeSet =
        inf_asset::decode(&std::fs::read(content.join("Town.inf_biomes")).unwrap()).unwrap();
    assert_eq!(set, phase19_biome_set());
    for b in &set.biomes {
        if let Some(hint) = &b.structure_hint {
            assert!(
                inf_pcg::ArchetypeId::parse(hint).is_some(),
                "biome `{}` hints at `{hint}`, which names no archetype",
                b.name
            );
        }
    }
}

/// The town, cooked and booted, with a step profile armed.
fn timed_town(dir: &Path) -> (usize, inf_player::runtime_sim::RuntimeSim) {
    let (_content, pack) = cook_town(dir);
    let built = pack_built(&pack);
    let colliders = solids(&built).len();
    assert!(colliders > 1_000, "only {colliders} solids to time");
    let mut sim = inf_player::sim_from_built(built);
    sim.set_step_profiling(true);
    (colliders, sim)
}

/// Mean phase profile and wall-clock ms/step over `steps` settled steps.
fn time_steps(
    sim: &mut inf_player::runtime_sim::RuntimeSim,
    steps: u32,
) -> (f64, inf_player::step_profile::StepProfile) {
    let mut mean = inf_player::step_profile::StepProfile::default();
    let start = std::time::Instant::now();
    for _ in 0..steps {
        sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        mean.accumulate(&sim.step_profile());
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(steps);
    mean.scale(1.0 / f64::from(steps));
    (ms, mean)
}

/// Print a mean profile's dearest rows.
fn print_phases(mean: &inf_player::step_profile::StepProfile) {
    for (name, ms) in mean.dearest_first() {
        if ms > 0.05 {
            eprintln!("  {name:>18}  {ms:.3} ms");
        }
    }
}

/// **The per-fixed-step cost of the town's colliders.**
///
/// The budget arm below times a *load*; this one times the loop. `sync_from_world`
/// runs every fixed step at 60 Hz over the whole world, and the town's ~13 000
/// derived solids would otherwise be re-described and re-sorted 60 times a second
/// to discover that a wall has not moved. The change stamp on
/// `PcgVolume::structures_gen` is what makes that a no-op; this arm is where the
/// claim is measured rather than asserted.
///
/// The absolute figure is printed, not asserted (a number from one machine is not
/// a contract). What is asserted is the shape: a step over a fully collidered town
/// stays a small fraction of a frame, against the imported [`FRAME_BUDGET_MS`] —
/// the right *class* of budget for recurring work, and the only per-frame number
/// this repo has.
///
/// **Measured on this machine, 12 908 colliders:** 11.62 ms/step with the stamp
/// disabled, 4.94 ms/step with it — 6.7 ms of a 16.7 ms 60 Hz frame, reclaimed by
/// not re-describing walls that cannot move. That engineering claim is what the
/// two numbers above are for.
///
/// **Why the assertion is not against 16.6.** It used to be, as a private literal,
/// which made it a 60 fps *hardware* claim on machines that cannot make one:
/// `FRAME_BUDGET_MS` is 33 ms rather than 16.7 precisely because these gates run on
/// shared CI runners (~4× slower than dev hardware, noisy), and its own doc says a
/// budget nobody can meet is a budget everybody disables. The margin was already
/// thin — this arm measured **7.577 ms/step** merely because another cargo job was
/// running alongside it, and a 4× runner crosses 16.6 with nothing regressed. The
/// same category error took the load arm below red at 34.77 ms.
///
/// **And a third category error, found by wave EMS1 and fixed by splitting the
/// arm in two.** This measured 302 ms/step once the institutions stood in the
/// town — and 96% of it was `character move` for thirty-two crowd agents, a
/// phase that did not exist when the arm was minted (NPC1a is two waves later)
/// and that silently took it over. An arm named for colliders whose number is
/// the character controller is `is_venue`-as-a-proxy in a budget. The collider
/// claim keeps this arm, with the crowd banded out; the crowd's cost gets
/// [`a_full_crowd_agent_costs_more_than_the_whole_collider_band`], which
/// carries the finding as its own assertion.
///
#[test]
fn stepping_the_town_stays_cheap_with_its_collider_band() {
    let dir = tempfile::tempdir().unwrap();
    let (colliders, mut sim) = timed_town(dir.path());
    // **THE CROWD IS BANDED OUT, AND THAT IS WHAT MAKES THIS ARM ABOUT
    // COLLIDERS** (wave EMS1). See `a_full_crowd_agent_costs_more_than_the_whole
    // _collider_band` below for the measurement that forced the split: the
    // arm's name says colliders and its number was 96% the crowd's character
    // controller, which is `is_venue`-as-a-proxy one file over. Zero radii put
    // every agent `Dormant` — no entity, no controller — so what is timed here
    // is the static band and the streaming that carries it.
    sim.set_crowd_radii((0.0, 0.0, 0.0));
    // Warm: the FIRST sync is the one that builds every collider.
    let warm = std::time::Instant::now();
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
    let first_ms = warm.elapsed().as_secs_f64() * 1000.0;

    let (per_step_ms, mean) = time_steps(&mut sim, 60);
    let tiers = sim.crowd_stats().per_tier;
    eprintln!(
        "phase19 step cost: {colliders} colliders — first step {first_ms:.2} ms, \
         steady {per_step_ms:.3} ms/step (a 60 Hz frame is 16.7 ms; tripwire \
         {FRAME_BUDGET_MS} ms — read this number, it is where drift shows); \
         crowd tiers {tiers:?}"
    );
    print_phases(&mean);
    // Armed: a band with nothing in it is not a band.
    assert_eq!(
        tiers[0], 0,
        "an agent is still Full, so this is not a control"
    );
    assert!(
        tiers.iter().sum::<usize>() > 0,
        "the town holds no population at all, so 'the crowd is banded out' is a \
         statement about nothing"
    );
    assert!(
        per_step_ms < FRAME_BUDGET_MS,
        "a steady step costs {per_step_ms:.3} ms with {colliders} static colliders — \
         over the {FRAME_BUDGET_MS} ms frame budget (§8: investigate the regression, \
         never raise it)"
    );
}

/// **ONE FULL CROWD AGENT COSTS MORE THAN THE WHOLE COLLIDER BAND** (wave EMS1)
/// — the measurement the arm above was hiding, as its own claim.
///
/// # What was found, and how
///
/// EMS1 stood four institutions in this town. Its step went from a few
/// milliseconds to **302 ms**, and the obvious reading — "22 853 colliders is
/// too many" — is wrong. The four-point sweep below is the evidence:
///
/// ```text
///   radii (0, 0, 0)        1.24 ms/step   tiers [0, 0, 0, 252]
///   radii (8, 16, 32)      1.27 ms/step   tiers [0, 0, 32, 220]
///   radii (20, 40, 80)     1.31 ms/step   tiers [0, 32, 50, 170]
///   radii (32, 96, 512)  302.28 ms/step   tiers [32, 50, 170, 0]
///                                         character move 300.21 ms
/// ```
///
/// **The fourth row is `DEFAULT_CROWD_RADII`, which is the band this arm
/// actually sweeps** (EMS1 audit — the first write-up of this table quoted a
/// hand-picked `(40, 80, 160)` from an earlier run, so the doc named a
/// measurement the code does not make; re-measured at
/// `(32, 96, 512)`: 256.9 ms/step of which 254.9 is `character move`, 7.9 ms an
/// agent, on a quieter machine than the run above. The *shape* is what is
/// asserted below, because the milliseconds are a fact about a laptop.)
///
/// The static band is free. Fifty `Near` agents and a hundred and seventy `Far`
/// ones are free. **Thirty-two `Full` ones are 250–300 ms**, essentially all of
/// it `character move` — about **8–9.3 ms per standing agent per step**, against
/// a town whose whole physics sync and solve together are 1.5 ms.
///
/// # Why this is asserted the way round it is
///
/// It is a carried defect and not a budget, so the arm asserts the DIAGNOSIS:
/// the control is inside the frame budget, the default band is not, and the
/// difference is `character move`. The day somebody gives the character
/// controller a broadphase, **this arm goes red** and the ledger gets rewritten
/// — which is the P22 pattern (assert the outcome, so a fix cannot land
/// silently) and is the only honest thing to do with a number nobody may raise
/// a budget to.
#[test]
fn a_full_crowd_agent_costs_more_than_the_whole_collider_band() {
    let dir = tempfile::tempdir().unwrap();
    let (colliders, mut sim) = timed_town(dir.path());
    sim.step_once(inf_player::runtime_sim::RuntimeInput::default());

    /// One row of the sweep: the band, the wall-clock ms/step it produced, the
    /// tier census, and what `character move` cost inside it.
    type SweepRow = ((f64, f64, f64), f64, [usize; 4], f64);
    let mut rows: Vec<SweepRow> = Vec::new();
    for r in [
        (0.0f64, 0.0f64, 0.0f64),
        (8.0, 16.0, 32.0),
        (20.0, 40.0, 80.0),
        inf_ecs::crowd::DEFAULT_CROWD_RADII,
    ] {
        sim.set_crowd_radii(r);
        // Settle the tiering before timing it: a retier step is a spawn, and a
        // spawn is not what this table is about.
        for _ in 0..4 {
            sim.step_once(inf_player::runtime_sim::RuntimeInput::default());
        }
        let (ms, mean) = time_steps(&mut sim, 20);
        let tiers = sim.crowd_stats().per_tier;
        let idx = inf_player::step_profile::STEP_PHASE_NAMES
            .iter()
            .position(|n| *n == "character move")
            .expect("the `character move` phase exists");
        eprintln!(
            "  radii {r:?}: {ms:.3} ms/step, tiers {tiers:?}, character move \
             {:.3} ms",
            mean.ms[idx]
        );
        rows.push((r, ms, tiers, mean.ms[idx]));
    }
    let (_, control_ms, control_tiers, _) = rows[0];
    let (_, full_ms, full_tiers, full_move) = rows[rows.len() - 1];
    let agents = full_tiers[0];
    eprintln!(
        "EMS1 FINDING: {colliders} static colliders step in {control_ms:.3} ms \
         with the crowd banded out; {agents} Full agent(s) add \
         {:.3} ms — {:.2} ms per agent per step, {:.0}% of it `character move`",
        full_ms - control_ms,
        (full_ms - control_ms) / agents.max(1) as f64,
        100.0 * full_move / (full_ms - control_ms).max(1e-9)
    );

    // Armed both ways: the control must really hold no Full agent, and the
    // subject must really hold some.
    assert_eq!(control_tiers[0], 0, "the control has Full agents in it");
    assert!(
        agents > 0,
        "the default band materializes no Full agent at all, so this arm is two \
         controls"
    );
    assert!(
        control_ms < FRAME_BUDGET_MS,
        "the control costs {control_ms:.3} ms — the collider band itself is over \
         budget, which is a different regression from the one this arm names"
    );
    // **THE CARRIED DEFECT, ASSERTED SO A FIX CANNOT LAND SILENTLY.** If this
    // line fails, the character controller got cheaper: delete the arm, restore
    // the crowd to `stepping_the_town_stays_cheap_with_its_collider_band`, and
    // rewrite the ledger entry that quotes these numbers.
    assert!(
        full_ms > FRAME_BUDGET_MS,
        "{agents} Full agents now step in {full_ms:.3} ms, inside the \
         {FRAME_BUDGET_MS} ms frame budget — the character-controller cost this \
         arm was minted to carry is FIXED. That is good news and it makes this \
         arm a lie: fold the crowd back into the collider arm and rewrite the \
         EMS1 ledger."
    );
    assert!(
        full_move > 0.8 * (full_ms - control_ms),
        "`character move` is {full_move:.3} ms of a {:.3} ms difference — the \
         crowd's cost has moved to another phase and the diagnosis in this \
         arm's doc is stale",
        full_ms - control_ms
    );
}

// ── (f) budget ──────────────────────────────────────────────────────────────

/// **The composed scene builds inside the LOAD budget.**
///
/// Deliberately no new ratchet constant: the arm asserts against the shared
/// load-class ceiling [`LOAD_BUDGET_MS`], and the absolute milliseconds are
/// printed rather than asserted (a number from one machine is not a contract).
/// What is asserted is that building an entire ten-archetype furnished town —
/// every plan, every wall expansion, every furniture walk, every collider — stays
/// **bounded**: linear in the content it was handed, on any machine.
///
/// **Why not `FRAME_BUDGET_MS`.** This arm used to hold a one-shot build against
/// the 33 ms *per-frame* tripwire, which is a category error: a frame recurs sixty
/// times a second, a load happens once, and asserting a town builds in the time
/// one frame gets is a hardware claim rather than a growth check. It failed as
/// such — ~8 ms on a developer machine, **34.77 ms** on a shared `windows-latest`
/// runner, red, with nothing regressed but the runner. Loads get startup-class
/// ceilings (the P15.1 precedent); [`LOAD_BUDGET_MS`] carries the doctrine.
#[test]
fn the_composed_town_builds_inside_the_load_budget() {
    let dir = tempfile::tempdir().unwrap();
    let (_content, pack) = cook_town(dir.path());
    // Warm the mmap and the asset scan, so the measurement is the PCG evaluation
    // rather than first-touch page faults.
    let _ = pack_built(&pack);

    let start = std::time::Instant::now();
    let built = pack_built(&pack);
    let ms = start.elapsed().as_secs_f64() * 1000.0;
    let instances = population(&built).len();
    let colliders = solids(&built).len();
    eprintln!(
        "phase19 gate (f): {instances} instances + {colliders} solids built in {ms:.2} ms \
         (load tripwire {LOAD_BUDGET_MS} ms — read this number, it is where drift shows)"
    );
    assert!(
        instances > 2_000 && colliders > 1_000,
        "the town is too small to time"
    );
    assert!(
        ms < LOAD_BUDGET_MS,
        "building the town took {ms:.2} ms, over the {LOAD_BUDGET_MS} ms load budget \
         (the §8 budget only ratchets DOWN — investigate the regression, do not raise it)"
    );
}
