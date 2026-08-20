//! **The island's city gate** (wave I3): a thousand buildings, in a host.
//!
//! The AAA-readiness certification's IB-2 is one arithmetic, and it is about
//! scale rather than about correctness:
//!
//! > P19 town: 15 097 instances + 12 850 solids, steady **4.663 ms/step** with
//! > 12 850 colliders. Per collider: **0.363 µs/step**. The town is **seven**
//! > buildings → 60 fps ceiling ≈ **25 buildings**.
//!
//! Against `STREAMED_STEP_BUDGET_MS` — the 4.0 ms the phase-16 gate holds a
//! streamed world to — that is ~11 000 colliders, about seven buildings, for a
//! phase whose content is two cities and five towns.
//!
//! `crates/inf-physics/tests/city_collider_band.rs` measures the band's
//! *mechanism* over a programmatically-built city. **This file measures the
//! shipped one**: the committed `samples/phase30-city`, cooked, booted through
//! the real pack path, and driven. The distinction is the P28.5/P21.4 law — a
//! rule proven by calling it directly leaves the WIRING unarmed, and every one of
//! this wave's claims lives on a wire between a PCG graph, a derived cache, a
//! physics bridge and two hosts.
//!
//! # The arms
//!
//! * (a) the fixture really is a city — a thousand buildings, on lots, from the
//!   cooked pack;
//! * (b) the banded step holds the streamed budget, with the unbanded
//!   alternative priced in the same run;
//! * (c) a near building is whole and a far one is exactly one shell box;
//! * (d) **PIE == shipping on a scripted drive-through**, byte for byte, because
//!   the band is a function of sim state and a band that read a camera would
//!   make two hosts simulate different worlds;
//! * (e) the subdivision's world proof: the lots are oriented, disjoint, and
//!   identical across two independent builds.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{PcgVolume, StreamingSource, Transform};
use inf_ecs::{Guid, ScatteredSolid, StructureGroup, Vec3d};
use inf_editor_core::samples::{
    city_block_centre, city_block_guid, city_dir, city_drive_point, CITY_BLOCKS, CITY_BLOCK_M,
    CITY_DRIVER_GUID, CITY_PCG_GUID, CITY_STEPS,
};
use inf_math::Tier;
use inf_packager::{cook, CookOptions};
use inf_physics::d3::{pcg_shell_guid, pcg_structure_guid};
use inf_physics::PhysicsBridge3D;
use inf_player::level::{self, BuiltWorld, PackLevelSource};
use inf_project::ProjectManifest;

// The budget is IMPORTED, never restated: a phase does not get its own budget
// for being new, and a private copy is somewhere for one to be quietly raised.
use inf_player::budget::STREAMED_STEP_BUDGET_MS;

/// The certification's own measured cost of a static collider, in microseconds
/// per fixed step. Restated here because it is a *number from a document*, not a
/// constant this tree owns — and the arm that uses it says where it came from.
const CERT_US_PER_COLLIDER: f64 = 0.363;

/// Every committed file of the sample, so the fixture copy cannot silently miss
/// one as the sample grows.
fn sample_files() -> [&'static str; 7] {
    [
        "City.inf_lvl",
        "City.inf_lvl.toml",
        "CityBlock.inf_pcg",
        "CityBlock.inf_pcg.toml",
        "CityRoads.inf_mesh",
        "CityRoads.inf_mesh.toml",
        "README.md",
    ]
}

/// The manifest equals the directory listing — P21.4's finding, which was that a
/// hand-written fixture list was eight of nine from the day it was written.
#[test]
fn the_fixture_copies_every_committed_sample_file() {
    let dir = city_dir();
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .expect("the city sample directory exists")
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    on_disk.sort();
    let mut named: Vec<String> = sample_files().iter().map(|s| (*s).to_string()).collect();
    named.sort();
    assert_eq!(
        on_disk, named,
        "the gate's file manifest and the committed sample have diverged"
    );
}

/// Scaffold a project holding the sample and cook it; returns the pack dir.
fn cook_city(tmp: &Path) -> PathBuf {
    let proj = tmp.join("proj");
    ProjectManifest::new("Island City", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    let src = city_dir();
    for f in sample_files() {
        std::fs::copy(src.join(f), content.join(f)).unwrap_or_else(|e| panic!("copy {f}: {e}"));
    }
    let out = tmp.join("out");
    cook(&proj, &out, &CookOptions::default()).expect("the city cooks");
    out
}

/// The SHIPPING world: the cooked pack, through the real pack boot path.
fn pack_built(pack: &Path) -> BuiltWorld {
    let source = PackLevelSource::open(pack).expect("pack opens");
    inf_player::build_world_from_pack(&source).expect("pack world builds")
}

/// The editor's PIE payload for the committed sample, built through the ONE PIE
/// seam rather than by hand.
fn pie_built() -> BuiltWorld {
    let dir = city_dir();
    let doc = inf_editor_core::scene::serialize::load(&dir.join("City.inf_lvl"))
        .expect("the committed city document loads");
    let mut pcgs: BTreeMap<Uuid, Vec<u8>> = BTreeMap::new();
    pcgs.insert(
        CITY_PCG_GUID,
        std::fs::read(dir.join("CityBlock.inf_pcg")).unwrap(),
    );
    let payload = inf_editor_core::pie::build_scene_payload(
        &doc,
        |_| None,
        |guid| pcgs.get(&guid).cloned(),
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        |_| None,
        60,
        false,
    )
    .expect("payload builds");
    assert_eq!(
        payload.pcgs.len(),
        1,
        "the city's one block graph must ride the payload"
    );
    inf_player::build_world_from_payload(&payload).expect("PIE world builds")
}

// ── probes ──────────────────────────────────────────────────────────────────

/// `(guid, volume)` for every `PcgVolume`, Guid-sorted so every probe is a
/// function of the content rather than of ECS iteration order.
fn volumes(built: &BuiltWorld) -> Vec<(Uuid, PcgVolume)> {
    let w = built.world.world();
    let mut out: Vec<(Uuid, PcgVolume)> = w
        .iter_entities()
        .filter_map(|e| Some((e.get::<Guid>()?.0, e.get::<PcgVolume>()?.clone())))
        .collect();
    out.sort_by_key(|(g, _)| *g);
    out
}

fn solids(built: &BuiltWorld) -> Vec<ScatteredSolid> {
    volumes(built)
        .into_iter()
        .flat_map(|(_, v)| v.structures)
        .collect()
}

fn groups(built: &BuiltWorld) -> Vec<StructureGroup> {
    volumes(built)
        .into_iter()
        .flat_map(|(_, v)| v.structure_groups)
        .collect()
}

/// A shell's placement as raw bits. Two hosts agreeing about every *count* and
/// disagreeing about where a wall stops is exactly the divergence a byte
/// comparison exists for.
fn shell_bits(g: &[StructureGroup]) -> Vec<[u64; 12]> {
    g.iter()
        .map(|g| {
            let s = g.shell;
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
                u64::from(g.len),
                u64::from(g.inst_len),
            ]
        })
        .collect()
}

/// Put the driver at `step`'s point. Scripted, so the trace is a function of the
/// level alone — the phase-16 gate's discipline, applied to a drive-through.
fn drive_to(built: &mut BuiltWorld, step: u64) {
    let p = city_drive_point(step);
    let e = built
        .world
        .entity_of(CITY_DRIVER_GUID)
        .expect("the driver is in the world");
    if let Some(mut t) = built.world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::new(p.x, p.y, p.z);
    }
    built.world.mark_dirty();
    built.world.propagate();
}

/// One step of the drive-through, as the numbers a divergence would show in:
/// the active collider count, and a fold of every attached body's identity.
fn trace(built: &mut BuiltWorld, steps: usize) -> Vec<(usize, u64)> {
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let mut out = Vec::with_capacity(steps);
    for step in 0..steps as u64 {
        drive_to(built, step);
        bridge.sync_from_world(&built.world);
        // FNV-1a over the sorted body guids — the set, not a count. Two bands
        // holding the same NUMBER of different colliders is exactly the
        // divergence a count cannot see.
        let mut ids: Vec<Uuid> = bridge
            .tracked_bodies()
            .into_iter()
            .map(|(g, _, _)| g)
            .collect();
        ids.sort_unstable();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for g in &ids {
            for b in g.as_bytes() {
                h ^= u64::from(*b);
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        out.push((ids.len(), h));
    }
    out
}

// ── (a) the fixture is a city ───────────────────────────────────────────────

/// **A thousand buildings, on subdivided lots, out of the cooked pack.**
#[test]
fn the_shipped_city_is_a_thousand_buildings_on_lots() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let built = pack_built(&pack);

    let vols = volumes(&built);
    assert_eq!(
        vols.len(),
        (CITY_BLOCKS * CITY_BLOCKS) as usize,
        "the city lost blocks in the cook"
    );
    let gs = groups(&built);
    let ss = solids(&built);
    println!(
        "I3 city (shipped): {} blocks, {} buildings, {} solids, {} instances",
        vols.len(),
        gs.len(),
        ss.len(),
        vols.iter().map(|(_, v)| v.evaluated.len()).sum::<usize>()
    );
    assert!(
        gs.len() >= 1_000,
        "the shipped city is {} buildings; the brief asked for a thousand",
        gs.len()
    );
    // Each block really was SUBDIVIDED: ten buildings, not one.
    for (guid, v) in &vols {
        assert_eq!(
            v.structure_groups.len(),
            10,
            "block {guid} grew {} buildings — the `building.lots` node did not \
             reach the shipped pass",
            v.structure_groups.len()
        );
        // …and the groups partition the block's own lists exactly.
        let mut cursor = 0usize;
        for g in &v.structure_groups {
            assert_eq!(
                g.start as usize, cursor,
                "block {guid}: a gap in the ranges"
            );
            cursor += g.len as usize;
        }
        assert_eq!(cursor, v.structures.len());
    }
}

// ── (b) the budget ──────────────────────────────────────────────────────────

/// **THE HEADLINE.** The banded step holds `STREAMED_STEP_BUDGET_MS`, and the
/// unbanded alternative is priced in the same run.
///
/// The collider count is a function of the world alone, so it is asserted on
/// every machine. The wall clock is printed and never asserted — the standing
/// rule, and the reason `dig_stall_bench` is `#[ignore]`d.
#[test]
fn the_banded_city_holds_the_streamed_step_budget() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let mut built = pack_built(&pack);
    drive_to(&mut built, (CITY_STEPS / 2) as u64);

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&built.world);
    let banded = bridge.body_count();
    let (near_m, far_m) = bridge.collider_band_radii();

    // THE ALTERNATIVE, PRICED, on the same city: with the streaming source gone
    // there is no band and every solid is described. (Removed from *this* world
    // rather than from a copy — `EcsWorld` has no clone and a second cook is a
    // second thousand buildings.)
    let e = built.world.entity_of(CITY_DRIVER_GUID).expect("the driver");
    built
        .world
        .world_mut()
        .entity_mut(e)
        .remove::<StreamingSource>();
    built.world.mark_dirty();
    let mut plain = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    plain.sync_from_world(&built.world);
    let unbanded = plain.body_count();

    let cert_ms = |n: usize| n as f64 * CERT_US_PER_COLLIDER / 1000.0;
    println!(
        "IB-2a (shipped city): {unbanded} solids -> {banded} banded colliders at \
         near {near_m} m / far {far_m} m ({:.2} %); at the certification's \
         {CERT_US_PER_COLLIDER} us/collider that is {:.3} ms/step against \
         {:.3} ms, budget {STREAMED_STEP_BUDGET_MS} ms",
        100.0 * banded as f64 / unbanded as f64,
        cert_ms(banded),
        cert_ms(unbanded),
    );
    assert!(
        cert_ms(banded) < STREAMED_STEP_BUDGET_MS,
        "the banded city is {banded} colliders = {:.3} ms/step at the \
         certification's own rate, past the {STREAMED_STEP_BUDGET_MS} ms streamed \
         budget",
        cert_ms(banded)
    );
    // ANTI-VACUITY: the alternative must actually be past it, or this arm is a
    // statement about a small scene.
    assert!(
        cert_ms(unbanded) > STREAMED_STEP_BUDGET_MS * 10.0,
        "the unbanded city is {:.3} ms/step, only {:.1}x the budget — this \
         fixture is not a city",
        cert_ms(unbanded),
        cert_ms(unbanded) / STREAMED_STEP_BUDGET_MS
    );

    // The clock, printed, never asserted.
    let mut bench = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    let e = built.world.entity_of(CITY_DRIVER_GUID).unwrap();
    built
        .world
        .world_mut()
        .entity_mut(e)
        .insert(StreamingSource { radius_m: 256.0 });
    built.world.mark_dirty();
    bench.sync_from_world(&built.world);
    bench.sync_from_world(&built.world);
    const ITERS: u32 = 30;
    let t0 = std::time::Instant::now();
    for _ in 0..ITERS {
        bench.sync_from_world(&built.world);
        bench.step(1.0 / 60.0);
    }
    let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);
    println!("IB-2a (shipped city) clock: {ms:.4} ms per fixed step on this machine");
    assert!(ms > 0.0, "the fixed step took no measurable time");
}

// ── (c) the tiers ───────────────────────────────────────────────────────────

/// A near building keeps every part and grows no shell; a far one is exactly one
/// box. **The reduction is a SHELL, not a deletion** — that is the whole claim
/// IB-2b makes, and a count alone cannot tell the two apart.
#[test]
fn a_near_building_is_whole_and_a_far_one_is_one_box() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let mut built = pack_built(&pack);
    drive_to(&mut built, (CITY_STEPS / 2) as u64);

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&built.world);
    let (near_m, far_m) = bridge.collider_band_radii();
    let band = inf_ecs::SimBand::from_world(&built.world, near_m, far_m);

    let (mut whole, mut shelled, mut gone) = (0usize, 0usize, 0usize);
    for (guid, v) in volumes(&built) {
        for (gi, g) in v.structure_groups.iter().enumerate() {
            let tier = band.tier(g.shell.center, g.shell.half_extents, g.shell.rotation);
            let parts = g
                .range()
                .filter(|i| bridge.collider_of(pcg_structure_guid(guid, *i)).is_some())
                .count();
            let shell = bridge.collider_of(pcg_shell_guid(guid, gi)).is_some();
            match tier {
                Tier::Near => {
                    assert_eq!(parts, g.len as usize, "a NEAR building lost parts");
                    assert!(!shell, "a NEAR building grew a shell — it is sealed");
                    whole += 1;
                }
                Tier::Far => {
                    assert_eq!(parts, 0, "a FAR building kept {parts} parts");
                    assert!(shell, "a FAR building has no shell — it is walk-through");
                    shelled += 1;
                }
                Tier::Out => {
                    assert_eq!(parts, 0);
                    assert!(!shell);
                    gone += 1;
                }
            }
        }
    }
    println!(
        "IB-2b (shipped city) at near {near_m} m / far {far_m} m: {whole} whole, \
         {shelled} shells, {gone} out"
    );
    assert!(
        whole > 0,
        "no building is near the driver — nothing enterable"
    );
    assert!(
        shelled > 0,
        "no building is shelled — the far tier is unexercised"
    );
}

/// **The DRAW side, through the shipped projection** (IB-2b).
///
/// `both_projectors_band_a_structure_lod_the_same_way` compares the two hosts'
/// *source text*; nothing in it asserts that a `RenderScene` ever receives a
/// shell batch. That is the P21.4 law — a rule proven by reading the code that
/// implements it leaves the wiring unarmed — so this arm drives the real
/// `project_scene` over the real city and reads the batches out.
///
/// The property is that the bands are **complementary**: every parts batch is
/// bounded above by the LOD distance, every shell batch is bounded below by it,
/// and no batch spans the boundary. An overlap draws a solid box inside a
/// building; a gap deletes it from the skyline.
#[test]
fn the_shipped_projection_emits_complementary_parts_and_shell_batches() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let built = pack_built(&pack);
    let mut sim = inf_player::sim_from_built(built);
    let mut scene = inf_render::RenderScene::default();
    inf_player::render::project_scene(
        &mut scene,
        &sim,
        0.0,
        &inf_player::vmesh::VmeshRegistry::new(),
    );

    let lod = inf_render::STRUCTURE_LOD_M;
    let (mut parts, mut shells, mut loose) = (0usize, 0usize, 0usize);
    let (mut part_inst, mut shell_inst) = (0usize, 0usize);
    for b in &scene.scatter {
        let n = b.data.instances.len();
        assert!(
            b.near_distance == 0.0 || b.near_distance == lod,
            "a batch bands from {} m, which is neither 0 nor the LOD distance",
            b.near_distance
        );
        if b.near_distance == lod {
            // The shell band: bounded below by the LOD distance, and above only
            // by the volume's own authored draw distance.
            assert!(
                b.draw_distance > lod,
                "a shell batch spans [{lod}, {}) — an empty interval draws nothing",
                b.draw_distance
            );
            shells += 1;
            shell_inst += n;
        } else if b.draw_distance == lod {
            parts += 1;
            part_inst += n;
        } else {
            loose += n;
        }
    }
    println!(
        "IB-2b (shipped projection): {parts} parts batches ({part_inst} instances) \
         bounded above at {lod} m, {shells} shell batches ({shell_inst} instances) \
         bounded below at {lod} m, {loose} ungrouped instances"
    );
    assert_eq!(
        parts,
        (CITY_BLOCKS * CITY_BLOCKS) as usize,
        "one parts batch per block"
    );
    assert_eq!(shells, parts, "every parts batch needs its complement");
    // **The reduction, measured**: one instance a building against ~370 a
    // building. Without it the far field is the whole city's geometry.
    assert_eq!(
        shell_inst, 1_000,
        "one shell instance per building; got {shell_inst}"
    );
    assert!(
        part_inst > shell_inst * 100,
        "the far tier draws {shell_inst} instances against the near tier's \
         {part_inst} — only {:.0}x, which is not an LOD",
        part_inst as f64 / shell_inst.max(1) as f64
    );
    // A city with no buildings at all would satisfy "every batch is banded".
    assert!(
        loose == 0,
        "{loose} ungrouped instances in a city of buildings"
    );
    let _ = &mut sim;
}

// ── (d) PIE == shipping, driving ────────────────────────────────────────────

/// **PIE == shipping on a scripted drive-through.**
///
/// The band is a function of sim state — its anchors are `StreamingSource`
/// entities and nothing else — so the two hosts must agree on the active
/// collider SET at every step of a 240-step drive across the city. A band that
/// read a camera, a frame counter or a residency would pass a static comparison
/// and fail here at the first step the two hosts' cameras differed, which in the
/// editor is immediately.
///
/// Compared per step rather than as one `assert_eq!` on two vectors: the failure
/// this catches is a *divergence point*, and which step is most of the diagnosis.
#[test]
fn pie_equals_shipping_on_a_drive_through_the_city() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let mut ship = pack_built(&pack);
    let mut pie = pie_built();

    // The two hosts must be describing the same city before anything is
    // compared: the shells are what the band tiers, so a payload that lost the
    // grouping would make both sides agree on an ungrouped city and prove
    // nothing.
    let (gs, gp) = (groups(&ship), groups(&pie));
    assert!(
        gs.len() >= 1_000,
        "the shipped city is {} buildings",
        gs.len()
    );
    assert_eq!(
        shell_bits(&gs),
        shell_bits(&gp),
        "PIE and shipping derived different building shells"
    );

    let a = trace(&mut ship, CITY_STEPS);
    let b = trace(&mut pie, CITY_STEPS);
    for (i, (s, p)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            s, p,
            "step {i}: PIE banded {} colliders (fold {:#018x}) and shipping banded \
             {} (fold {:#018x}) — the active set is not a function of sim state",
            p.0, p.1, s.0, s.1
        );
    }

    // ANTI-VACUITY, and it is the clause that makes the arm mean something: the
    // drive must MOVE the band. A trace whose every step held the same set would
    // satisfy every line above and would be satisfied by a band that never
    // updated at all.
    let distinct: std::collections::BTreeSet<u64> = a.iter().map(|(_, h)| *h).collect();
    let (lo, hi) = (
        a.iter().map(|(n, _)| *n).min().unwrap(),
        a.iter().map(|(n, _)| *n).max().unwrap(),
    );
    println!(
        "IB-2a drive-through: {CITY_STEPS} steps, {} distinct active sets, \
         {lo}..={hi} colliders",
        distinct.len()
    );
    assert!(
        distinct.len() >= 7,
        "the {CITY_STEPS}-step drive produced {} distinct active sets — the band \
         is not tracking the driver",
        distinct.len()
    );
    assert!(lo > 0, "the band emptied entirely during the drive");
    // …and it SHRANK as well as grew. A run that only ever adds colliders is an
    // approach, and an approach cannot tell a band that releases from one that
    // accumulates — which is the leak this whole item would otherwise hide.
    let mut shrank = false;
    for w in a.windows(2) {
        shrank |= w[1].0 < w[0].0;
    }
    assert!(
        shrank,
        "the active set never fell across {CITY_STEPS} steps: the band grows and \
         never releases, which is a leak with the same symptoms as working"
    );
}

// ── (e) the subdivision's world proof ───────────────────────────────────────

/// **The lots are real lots**: oriented to their block, pairwise disjoint, and
/// identical across two independent builds of the same content.
///
/// Reads the SHIPPED shells rather than re-running the subdivider, because "the
/// rule produces disjoint rectangles" is a statement `subdivide_block`'s own arm
/// already makes — what this one adds is that the rectangles survive a cook, a
/// pack, a load and an evaluation, which is four places they could stop being
/// disjoint.
#[test]
fn the_shipped_lots_are_disjoint_and_deterministic() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let a = pack_built(&pack);
    let b = pack_built(&pack);
    assert_eq!(
        shell_bits(&groups(&a)),
        shell_bits(&groups(&b)),
        "two loads of one pack built different buildings"
    );

    // Per block, the shells must not overlap on the ground. Measured as a real
    // area rather than as a predicate, so the failure says how badly.
    let mut worst = 0.0f64;
    let mut checked = 0usize;
    for (i, (guid, v)) in volumes(&a).into_iter().enumerate() {
        assert_eq!(guid, city_block_guid(i as u32), "block order is not stable");
        let rects: Vec<(DVec2, DVec2)> = v
            .structure_groups
            .iter()
            .map(|g| {
                let c = DVec2::new(g.shell.center.x, g.shell.center.z);
                let h = DVec2::new(g.shell.half_extents.x, g.shell.half_extents.z);
                (c - h, c + h)
            })
            .collect();
        for x in 0..rects.len() {
            for y in (x + 1)..rects.len() {
                let ox = (rects[x].1.x.min(rects[y].1.x) - rects[x].0.x.max(rects[y].0.x)).max(0.0);
                let oz = (rects[x].1.y.min(rects[y].1.y) - rects[x].0.y.max(rects[y].0.y)).max(0.0);
                worst = worst.max(ox * oz);
                checked += 1;
            }
        }
        // …and every lot is inside its own block, which is where the setback and
        // the containment test are proven end to end.
        let c = city_block_centre(i as u32);
        for (lo, hi) in &rects {
            assert!(
                lo.x >= c.x - CITY_BLOCK_M.0 * 0.5 - 1e-6
                    && hi.x <= c.x + CITY_BLOCK_M.0 * 0.5 + 1e-6
                    && lo.y >= c.y - CITY_BLOCK_M.1 * 0.5 - 1e-6
                    && hi.y <= c.y + CITY_BLOCK_M.1 * 0.5 + 1e-6,
                "block {i}: a lot escaped its block into the street"
            );
        }
    }
    println!(
        "IB-2c (shipped city): {checked} lot pairs checked, worst overlap \
         {worst:.3e} m2"
    );
    assert!(checked > 4_000, "only {checked} pairs — that is not a city");
    assert!(
        worst < 1e-6,
        "two shipped lots overlap by {worst} m2 — a building is standing in \
         another's footprint"
    );
}

/// The cook is silent: a fixture that ships with an advisory is a fixture whose
/// numbers are about a scene the engine is complaining about.
#[test]
fn the_city_cooks_without_an_advisory() {
    let tmp = tempfile::tempdir().unwrap();
    let proj = tmp.path().join("proj");
    ProjectManifest::new("Island City", "blank-3d")
        .save(&proj)
        .unwrap();
    let content = proj.join("Content");
    std::fs::create_dir_all(&content).unwrap();
    for f in sample_files() {
        std::fs::copy(city_dir().join(f), content.join(f)).unwrap();
    }
    let report = cook(&proj, &tmp.path().join("out"), &CookOptions::default()).expect("cooks");
    assert!(
        report.warnings.is_empty(),
        "the city cooked with advisories: {:?}",
        report.warnings
    );
    let _ = level::PACK_FILE;
}
