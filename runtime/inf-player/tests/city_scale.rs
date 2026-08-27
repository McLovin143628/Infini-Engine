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

/// `(guid, world translation, volume)` — the batch anchor a projection uses is
/// the volume entity's own translation, so this is what pairs a `ScatterBatch`
/// back to the volume it came from.
fn placed_volumes(built: &BuiltWorld) -> Vec<(Uuid, DVec3, PcgVolume)> {
    let w = built.world.world();
    let mut out: Vec<(Uuid, DVec3, PcgVolume)> = w
        .iter_entities()
        .filter_map(|e| {
            let t = e.get::<Transform>()?;
            Some((
                e.get::<Guid>()?.0,
                DVec3::new(t.translation.x, t.translation.y, t.translation.z),
                e.get::<PcgVolume>()?.clone(),
            ))
        })
        .collect();
    out.sort_by_key(|(g, _, _)| *g);
    out
}

/// The widest shell's own half-diagonal — the distance by which a part may be
/// nearer the eye than its group's shell centre is, and therefore the width of
/// the overlap the parts band must carry (I3 audit).
fn reach_of_volume(v: &PcgVolume) -> f64 {
    v.structure_groups
        .iter()
        .map(|g| g.shell.half_extents.length())
        .fold(0.0_f64, f64::max)
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

/// Plant a `Camera` at `at` — the thing the band is forbidden to read (I3
/// audit).
///
/// `SimBand::from_world` takes an `&EcsWorld` and walks it for
/// `StreamingSource`; nothing in the type stops a later edit from reaching for a
/// `Camera` in the same walk, and a camera is exactly what differs between the
/// editor and a shipped build. So the drive-through gives the two hosts cameras
/// **a city apart** and requires the traces to stay identical: a band that read
/// one would diverge at step 0, and without this the fixture has no camera at
/// all, which is a hazard closed by the absence of a fixture rather than by an
/// arm.
fn plant_camera(built: &mut BuiltWorld, guid: Uuid, at: DVec3) {
    let e = built.world.spawn_with_guid(guid, "Camera", None);
    built
        .world
        .world_mut()
        .entity_mut(e)
        .insert(Transform {
            translation: Vec3d::new(at.x, at.y, at.z),
            ..Transform::IDENTITY
        })
        .insert(inf_ecs::components::Camera::default());
    built.world.mark_dirty();
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
/// The property is that the bands are **complementary**: every shell batch is
/// bounded below by the LOD distance, every parts batch is bounded above by it
/// **plus its widest shell's own half-diagonal**, and no batch bands from
/// anywhere else. The `reach` is the I3 audit's correction and it is what makes
/// a gap impossible — see
/// `no_eye_position_leaves_a_building_partly_drawn_with_no_shell`, which
/// measures the failure it prevents.
#[test]
fn the_shipped_projection_emits_complementary_parts_and_shell_batches() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let built = pack_built(&pack);
    // Read before the sim consumes the world: the expected parts cut is a
    // function of each volume's own shells.
    let reach_of: BTreeMap<[u64; 3], f64> = placed_volumes(&built)
        .into_iter()
        .map(|(_, t, v)| {
            (
                [t.x.to_bits(), t.y.to_bits(), t.z.to_bits()],
                reach_of_volume(&v),
            )
        })
        .collect();
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
    let mut worst_reach = 0.0f64;
    // **The BLOCKS a parts band was emitted for**, which is the claim; the batch
    // COUNT stopped being it in island wave I8b. `push_scatter` buckets on
    // (mesh, glow) now, so a block whose modules include a glazed one emits two
    // parts batches over the same band from the same anchor - one lit, one not.
    // Counting batches would make this arm a statement about the palette.
    let mut part_blocks: std::collections::BTreeSet<[u64; 3]> = std::collections::BTreeSet::new();
    for b in &scene.scatter {
        let n = b.data.instances.len();
        assert!(
            b.near_distance == 0.0 || b.near_distance == lod,
            "a batch bands from {} m, which is neither 0 nor the LOD distance",
            b.near_distance
        );
        let key = [
            b.anchor.x.to_bits(),
            b.anchor.y.to_bits(),
            b.anchor.z.to_bits(),
        ];
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
        } else if b.draw_distance > lod {
            // The parts band: bounded above by the LOD distance **plus this
            // volume's own reach**, which is what makes the pair gap-free.
            let reach = *reach_of
                .get(&key)
                .expect("a parts batch anchored where no volume is");
            assert!(reach > 0.0, "a building with a zero-size shell");
            assert_eq!(
                b.draw_distance,
                lod + reach,
                "a parts batch is bounded above at {} m, which is neither the LOD \
                 distance nor the LOD distance plus its own {reach} m reach",
                b.draw_distance
            );
            worst_reach = worst_reach.max(reach);
            parts += 1;
            part_blocks.insert(key);
            part_inst += n;
        } else {
            loose += n;
        }
    }
    println!(
        "IB-2b (shipped projection): {parts} parts batches ({part_inst} instances) \
         bounded above at {lod} m + a reach of at most {worst_reach:.3} m, {shells} \
         shell batches ({shell_inst} instances) bounded below at {lod} m, {loose} \
         ungrouped instances; the parts batches cover {} blocks",
        part_blocks.len()
    );
    assert_eq!(
        part_blocks.len(),
        (CITY_BLOCKS * CITY_BLOCKS) as usize,
        "every block must contribute a parts band"
    );
    assert!(
        parts >= part_blocks.len(),
        "{parts} parts batches over {} blocks",
        part_blocks.len()
    );
    assert_eq!(
        shells,
        part_blocks.len(),
        "every block's parts band needs its shell complement"
    );
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

/// **No eye position leaves a building partly drawn with nothing standing in for
/// it** (I3 audit).
///
/// The two bands are complementary in the **group's** distance; the cull is per
/// **instance**. A part sits up to its shell's own half-diagonal nearer the eye
/// than the shell's centre does, so with both cuts at `STRUCTURE_LOD_M` a
/// building whose shell centre is just *inside* the line loses the parts that
/// are just *outside* it — and grows no shell to stand in for them. That is a
/// hole through the back of a building, and on a city it is not a corner case:
/// it is every building in a `2 × reach`-wide annulus, at all times.
///
/// The property asserted is the disjunction, because it is what the eye needs:
/// **either the shell is drawn or every part is**. The naive equal cuts are
/// priced in the same run — a "zero" over a city that never straddles the line
/// would be a statement about the fixture rather than about the rule.
///
/// The bands are **read off the real `project_scene`**, never restated here: an
/// arm that recomputed the cuts would agree with a projection that had stopped
/// emitting shells at all.
#[test]
fn no_eye_position_leaves_a_building_partly_drawn_with_no_shell() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let built = pack_built(&pack);
    let lod = inf_render::STRUCTURE_LOD_M;

    // Per building: its shell's centre, its parts' own world positions, and the
    // anchor of the volume whose batches band it.
    let mut buildings: Vec<([u64; 3], DVec3, Vec<DVec3>)> = Vec::new();
    for (_, t, v) in placed_volumes(&built) {
        let key = [t.x.to_bits(), t.y.to_bits(), t.z.to_bits()];
        for g in &v.structure_groups {
            buildings.push((
                key,
                g.shell.center,
                v.evaluated[g.instance_range()]
                    .iter()
                    .map(|i| i.position)
                    .collect(),
            ));
        }
    }
    assert!(buildings.len() >= 1_000, "not a city");

    // The cuts, as the shipped projection states them: for each volume, the
    // widest band a parts batch ends at and the narrowest a shell batch begins
    // at. A volume with no shell batch keeps `INFINITY` — the honest reading of
    // "its shell is never drawn", and what makes this arm fail if the far tier
    // stops being emitted.
    let sim = inf_player::sim_from_built(built);
    let mut scene = inf_render::RenderScene::default();
    inf_player::render::project_scene(
        &mut scene,
        &sim,
        0.0,
        &inf_player::vmesh::VmeshRegistry::new(),
    );
    let mut cuts: BTreeMap<[u64; 3], (f64, f64)> = BTreeMap::new();
    for b in &scene.scatter {
        let key = [
            b.anchor.x.to_bits(),
            b.anchor.y.to_bits(),
            b.anchor.z.to_bits(),
        ];
        let e = cuts.entry(key).or_insert((0.0, f64::INFINITY));
        if b.near_distance > 0.0 {
            e.1 = e.1.min(b.near_distance);
        } else {
            e.0 = e.0.max(b.draw_distance);
        }
    }

    // A building is whole when the shell is drawn OR every part is.
    let gapped = |shipped: bool| {
        let mut worst = (0usize, 0u64);
        let mut total = 0usize;
        for step in (0..CITY_STEPS as u64).step_by(24) {
            let eye = city_drive_point(step);
            let mut n = 0usize;
            for (key, centre, parts) in &buildings {
                let (parts_cut, shell_cut) = cuts[key];
                // The alternative: both bands cut at the LOD distance, which is
                // what shipped before the reach was carried.
                let (parts_cut, shell_cut) = if shipped {
                    (parts_cut, shell_cut)
                } else {
                    (lod, lod)
                };
                let shell = (*centre - eye).length() >= shell_cut;
                let whole = parts.iter().all(|p| (*p - eye).length() < parts_cut);
                if !shell && !whole {
                    n += 1;
                }
            }
            total += n;
            if n > worst.0 {
                worst = (n, step);
            }
        }
        (total, worst)
    };

    let (shipped, _) = gapped(true);
    let (naive, naive_worst) = gapped(false);
    println!(
        "IB-2b gap sweep ({} buildings, {} eyes on the drive line): shipped \
         (parts band + reach) {shipped} part-drawn buildings with no shell; the \
         equal-cut alternative {naive}, worst {} at step {}",
        buildings.len(),
        CITY_STEPS / 24,
        naive_worst.0,
        naive_worst.1
    );
    assert_eq!(
        shipped, 0,
        "{shipped} buildings are drawn in pieces with no shell behind them — \
         the parts band no longer carries its reach"
    );
    // ANTI-VACUITY: the alternative this arm exists to refuse must really fail
    // on this fixture, or the fixture never straddles the LOD line.
    assert!(
        naive > 0,
        "the equal-cut alternative left nothing gapped — no building on this \
         city straddles the {lod} m line, so this arm measures nothing"
    );
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
    // A camera in each host, a city apart, and neither may reach the band.
    let cam = Uuid::from_u128(0x8430_0CA1);
    plant_camera(&mut ship, cam, DVec3::new(600.0, 40.0, 400.0));
    plant_camera(&mut pie, cam, DVec3::new(-600.0, 2.0, -400.0));

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
    // …and the same shells fold to a number, so "deterministic across two
    // PROCESSES" is a diff of two lines of output rather than a claim nobody can
    // check (I3 audit). Two loads in one process cannot see a difference that
    // needs a fresh address space to appear.
    let mut fold: u64 = 0xcbf2_9ce4_8422_2325;
    for row in shell_bits(&groups(&a)) {
        for w in row {
            for byte in w.to_le_bytes() {
                fold ^= u64::from(byte);
                fold = fold.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    println!("IB-2c (shipped city): shell digest {fold:#018x}");

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

/// **WHAT A CITY'S DOORS COST** (island wave I6) — measured on the shipped
/// fixture, in the two units that decide whether the design is affordable: how
/// many doorways a thousand buildings plan, and how many of them the band makes
/// SOLID.
///
/// The second number is the one that matters. IB-2a's whole finding is that the
/// cells decide what EXISTS and the band decides what is SOLID; a door system
/// that ignored the band would put twenty thousand kinematic bodies in a rapier
/// world sixty times a second. This prints both and asserts the ratio, so the
/// day the band stops applying to doors it fails here with a number rather than
/// as a frame-time mystery.
#[test]
fn the_citys_doorways_are_banded_like_its_walls() {
    let built = pie_built();
    let all: usize = volumes(&built).iter().map(|(_, v)| v.doorways.len()).sum();
    let blocks = volumes(&built).len();
    println!(
        "the shipped city: {blocks} blocks, {} solids, {all} doorways ({:.1} per block)",
        solids(&built).len(),
        all as f64 / blocks.max(1) as f64
    );
    assert!(
        all > 0,
        "the city plans no doors at all - the grammar's doorway emission is not reaching the population"
    );
    // The band: a bridge synced against the city's own `StreamingSource` (the
    // Driver), which is what the fixed step really does.
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&built.world);
    let band = bridge.sim_band(&built.world);
    let near = inf_physics::d3::door::placements_near(&built.world, &band).len();
    println!(
        "…of which {near} are inside the collider band - {:.2} % of {all}",
        100.0 * near as f64 / all.max(1) as f64
    );
    assert!(near > 0, "the band admitted no door at all");
    assert!(
        near < all,
        "every doorway in a 1.26 km city is solid - the band is not applying to doors"
    );
    // …and the same discipline the walls are held to: the admitted set is a
    // small fraction, not "most of them".
    assert!(
        (near as f64) < 0.15 * all as f64,
        "the band admitted {near} of {all} doorways, which is not a band"
    );
    // The leaves the bridge actually built match what the band admitted, so the
    // number above is about the world rather than about a list.
    let mut leaves = 0;
    for (g, v) in volumes(&built) {
        for i in 0..v.doorways.len() {
            let leaf =
                inf_physics::d3::door_leaf_guid(inf_physics::d3::door::pcg_doorway_guid(g, i));
            if bridge.body_of(leaf).is_some() {
                leaves += 1;
            }
        }
    }
    println!("the physics world holds {leaves} door leaves");
    assert_eq!(
        leaves, near,
        "the band's list and the bridge's bodies disagree"
    );
}

/// **The two hosts plan the same doors**, byte for byte — the cooked pack and
/// the PIE payload, through the two hand-written `population_of` mirrors.
///
/// `start` and `inst_start` were the mirror's own named hazard; a doorway adds
/// eight more fields to keep in step, and two of them (`closed_yaw_deg` and
/// `inside_yaw_deg`) are both angles, so **swapping them compiles** and would
/// hang every door in the city sideways.
#[test]
fn the_cooked_and_previewed_cities_plan_the_same_doors() {
    let tmp = tempfile::tempdir().unwrap();
    let pack = cook_city(tmp.path());
    let shipped = pack_built(&pack);
    let previewed = pie_built();
    let a: Vec<[u64; 10]> = doorway_bits(&shipped);
    let b: Vec<[u64; 10]> = doorway_bits(&previewed);
    println!(
        "the cooked city plans {} doorways and the previewed one {}",
        a.len(),
        b.len()
    );
    assert!(!a.is_empty(), "the cooked city plans no doors");
    assert_eq!(
        a.len(),
        b.len(),
        "the two hosts plan different numbers of doors"
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "doorway {i} differs between the cook and the preview");
    }
}

/// Every doorway's placement as raw bits, in volume order — the same discipline
/// `shell_bits` uses, and for the same reason: two hosts agreeing about a count
/// and disagreeing about an angle is exactly what a byte comparison is for.
///
/// **Every field of `DoorwaySlot`, not the ones that are angles** (the I6
/// audit). The first draft folded the eight scalars and left `exterior` and
/// `floor` out, which is the shape P22's own allowlist law refuses: a
/// comparison enumerating what its author thought of lets the ninth field
/// through. A mirror that dropped `exterior` — the flag that says which of a
/// building's doors is its front one — would have been invisible here, and the
/// way to keep it visible is to take the struct apart field by field, which is
/// what the pattern below does: a field ADDED to `DoorwaySlot` makes that
/// pattern non-exhaustive and this file stops compiling.
fn doorway_bits(built: &BuiltWorld) -> Vec<[u64; 10]> {
    volumes(built)
        .into_iter()
        .flat_map(|(_, v)| {
            v.doorways
                .into_iter()
                .map(|d| {
                    // Destructured, so a field ADDED to `DoorwaySlot` is a
                    // compile error here rather than a field this comparison
                    // silently stops making.
                    let inf_ecs::components::DoorwaySlot {
                        hinge,
                        closed_yaw_deg,
                        width_m,
                        height_m,
                        thickness_m,
                        inside_yaw_deg,
                        exterior,
                        floor,
                    } = d;
                    [
                        hinge.x.to_bits(),
                        hinge.y.to_bits(),
                        hinge.z.to_bits(),
                        closed_yaw_deg.to_bits(),
                        inside_yaw_deg.to_bits(),
                        width_m.to_bits(),
                        height_m.to_bits(),
                        thickness_m.to_bits(),
                        u64::from(exterior),
                        u64::from(floor),
                    ]
                })
                .collect::<Vec<_>>()
        })
        .collect()
}
