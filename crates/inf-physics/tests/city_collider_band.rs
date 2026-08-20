//! **IB-2a: a thousand-building city inside the streamed step budget.**
//!
//! The AAA-readiness certification's IB-2 is one arithmetic:
//!
//! > P19 town: **15 097 instances + 12 850 solids**, steady **4.663 ms/step**.
//! > Per collider: **0.363 µs/step**. 60 fps ceiling ≈ **46 000 colliders**.
//! > The town is **seven** buildings → ceiling ≈ **25 buildings**.
//!
//! Against `STREAMED_STEP_BUDGET_MS = 4.0` — the budget the phase-16 gate holds
//! a streamed world to — that is ~11 000 colliders, about seven buildings, for a
//! phase whose content is two cities and five towns.
//!
//! This file builds a real thousand-building city out of real grammar buildings
//! on real subdivided blocks, and measures what the band leaves in the physics
//! world. Everything here is a **function of sim state**: the band's anchors are
//! `StreamingSource` entities, the same set P16's cell activation reads. There is
//! no camera in this file.
//!
//! # What is asserted where
//!
//! The **functional** half asserts everywhere, on every machine: how many
//! colliders the band admits, that the near buildings are whole and enterable,
//! that a far building is exactly one box, that walking re-bands, and that two
//! processes agree. The **clock** half — the ms/step figure — is a measurement
//! this file prints and gates behind a calibration, per the standing rule that a
//! wall-clock assertion on a shared runner is a flake generator.

use glam::{DQuat, DVec2, DVec3};
use inf_ecs::components::{PcgVolume, ScatteredSolid, StreamingSource, Transform};
use inf_ecs::{EcsWorld, StructureGroup, Vec2d, Vec3d};
use inf_math::Tier;
use inf_pcg::building::{ArchetypeId, BuildingPass, LotRules};
use inf_pcg::grammar::{Ground, NoSplines, SpanSource};
use inf_pcg::height::FnHeight;
use inf_pcg::GrammarContext;
use inf_physics::d3::{pcg_shell_guid, pcg_structure_guid};
use inf_physics::PhysicsBridge3D;
use std::time::Instant;
use uuid::Uuid;

// ── the city ────────────────────────────────────────────────────────────────

/// Blocks on a side. 10 × 10 blocks × 10 lots = **1 000 buildings**.
const BLOCKS: u32 = 10;
/// Block footprint in metres — 100 × 60 cuts into 5 × 2 lots at 20 m of
/// frontage and 30 m of depth, which is a downtown block.
const BLOCK: DVec2 = DVec2::new(100.0, 60.0);
/// Block pitch: the block plus the street between two of them.
const PITCH: DVec2 = DVec2::new(140.0, 100.0);
const PLAYER: Uuid = Uuid::from_u128(0x1B2A_0001);
const BLOCK_BASE: u128 = 0x1B2A_1000;

fn block_guid(i: u32) -> Uuid {
    Uuid::from_u128(BLOCK_BASE + u128::from(i))
}

fn block_centre(i: u32) -> DVec2 {
    let (cx, cz) = (i % BLOCKS, i / BLOCKS);
    DVec2::new(
        (f64::from(cx) - f64::from(BLOCKS - 1) * 0.5) * PITCH.x,
        (f64::from(cz) - f64::from(BLOCKS - 1) * 0.5) * PITCH.y,
    )
}

/// One block's buildings, through the real IB-2c subdivision + IB-2b grouping.
fn block_population(
    i: u32,
) -> (
    Vec<inf_ecs::ScatteredInstance>,
    Vec<ScatteredSolid>,
    Vec<StructureGroup>,
    usize,
) {
    let c = block_centre(i);
    let ring: Vec<DVec3> = [
        (-BLOCK.x * 0.5, -BLOCK.y * 0.5),
        (BLOCK.x * 0.5, -BLOCK.y * 0.5),
        (BLOCK.x * 0.5, BLOCK.y * 0.5),
        (-BLOCK.x * 0.5, BLOCK.y * 0.5),
    ]
    .iter()
    .map(|&(x, z)| DVec3::new(c.x + x, 0.0, c.y + z))
    .collect();

    let pass = BuildingPass {
        name: "block".into(),
        layer: "city".into(),
        enabled: true,
        // One archetype per row, so the city is a city and not a hundred copies
        // of one street.
        archetype: ArchetypeId::ALL[(i as usize / BLOCKS as usize) % ArchetypeId::ALL.len()],
        seed: u64::from(i),
        floors: 2,
        furnish: false,
        size: DVec2::ZERO,
        lot: Some(SpanSource::Polyline {
            points: ring,
            closed: true,
        }),
        lots: Some(LotRules {
            frontage_m: 20.0,
            depth_m: 30.0,
            jitter: 0.1,
            setback_m: 1.5,
            min_area_m2: 40.0,
        }),
        ground: Ground::Span,
        altitude_offset: 0.0,
    };
    let cx = GrammarContext {
        entity: Some(block_guid(i)),
        center: DVec3::new(c.x, 0.0, c.y),
        extent: BLOCK * 0.5,
        seed_offset: u64::from(i),
    };
    let out = inf_pcg::evaluate_buildings(
        std::slice::from_ref(&pass),
        &NoSplines,
        &FnHeight::new(|_, _| Some(0.0)),
        &cx,
    );
    let buildings = out.groups.len();
    let composed = inf_pcg::compose_volume(Vec::new(), out);
    let instances = composed
        .instances
        .iter()
        .map(|p| inf_ecs::ScatteredInstance {
            position: p.pos,
            rotation: p.rotation,
            scale: p.scale,
            kind: p.kind_index,
        })
        .collect();
    let solids = composed
        .colliders
        .iter()
        .map(|s| ScatteredSolid {
            center: s.center,
            half_extents: s.half_extents,
            rotation: s.rotation,
        })
        .collect();
    let groups = composed
        .groups
        .iter()
        .map(|g| StructureGroup {
            shell: ScatteredSolid {
                center: g.shell.center,
                half_extents: g.shell.half_extents,
                rotation: g.shell.rotation,
            },
            start: g.start,
            len: g.len,
            inst_start: g.inst_start,
            inst_len: g.inst_len,
        })
        .collect();
    (instances, solids, groups, buildings)
}

struct City {
    world: EcsWorld,
    solids: usize,
    buildings: usize,
}

/// The city, with the player at `at`.
fn city(at: DVec3) -> City {
    let mut world = EcsWorld::new();
    let (mut solids, mut buildings) = (0usize, 0usize);
    for i in 0..(BLOCKS * BLOCKS) {
        let (inst, s, g, b) = block_population(i);
        solids += s.len();
        buildings += b;
        let c = block_centre(i);
        let e = world.spawn_with_guid(block_guid(i), &format!("Block {i}"), None);
        let mut vol = PcgVolume {
            extent: Vec2d::new(BLOCK.x * 0.5, BLOCK.y * 0.5),
            ..Default::default()
        };
        vol.set_population(inst, s, g);
        assert_eq!(
            vol.structure_groups.len(),
            b,
            "the population door refused a group — the fixture is not grouped"
        );
        world.world_mut().entity_mut(e).insert((
            Transform {
                translation: Vec3d::new(c.x, 0.0, c.y),
                ..Transform::IDENTITY
            },
            vol,
        ));
    }
    let p = world.spawn_with_guid(PLAYER, "Player", None);
    world.world_mut().entity_mut(p).insert((
        Transform {
            translation: Vec3d::new(at.x, at.y, at.z),
            ..Transform::IDENTITY
        },
        StreamingSource { radius_m: 256.0 },
    ));
    world.mark_dirty();
    world.propagate();
    City {
        world,
        solids,
        buildings,
    }
}

fn move_player(world: &mut EcsWorld, to: DVec3) {
    let e = world.entity_of(PLAYER).expect("the player exists");
    if let Some(mut t) = world.world_mut().get_mut::<Transform>(e) {
        t.translation = Vec3d::new(to.x, to.y, to.z);
    }
    world.mark_dirty();
    world.propagate();
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **THE HEADLINE.** A thousand-building city, and what the band leaves solid.
///
/// The `4.0 ms` is `inf_player::budget::STREAMED_STEP_BUDGET_MS`, restated here
/// because Ring 0 cannot name a Ring-2 constant; `runtime/inf-player/tests/
/// city_scale.rs` is where it is asserted against the real constant, on the
/// committed sample, through a real boot path.
#[test]
fn a_thousand_building_city_bands_to_a_budget_sized_collider_set() {
    let mut city = city(DVec3::ZERO);
    assert!(
        city.buildings >= 1_000,
        "the fixture is {} buildings, not a city",
        city.buildings
    );
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&city.world);
    let banded = bridge.body_count();
    let admitted = bridge.admitted_structures();
    assert_eq!(banded, admitted, "every body here is a structure body");

    // THE ALTERNATIVE, PRICED — the pre-I3 world, on the same city: with the
    // streaming source gone there is no band, and every box is described. (The
    // source is removed from *this* world rather than from a copy because a
    // thousand grammar buildings is a second of assembly and `EcsWorld` has no
    // clone; a second bridge over the mutated world is the same comparison.)
    {
        let e = city.world.entity_of(PLAYER).expect("the player");
        city.world
            .world_mut()
            .entity_mut(e)
            .remove::<StreamingSource>();
        city.world.mark_dirty();
    }
    let mut plain = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    plain.sync_from_world(&city.world);
    let unbanded = plain.body_count();
    assert_eq!(
        unbanded, city.solids,
        "the unbanded world must hold every solid"
    );

    let (near_m, far_m) = bridge.collider_band_radii();
    println!(
        "IB-2a: {} buildings / {} solids -> {} colliders banded at near {near_m} m / far {far_m} m \
         ({:.2}% of the unbanded {unbanded}); the certification's 0.363 us/collider makes that \
         {:.3} ms/step against {:.3} ms",
        city.buildings,
        city.solids,
        banded,
        100.0 * banded as f64 / unbanded as f64,
        banded as f64 * 0.363e-3,
        unbanded as f64 * 0.363e-3,
    );

    // The functional bound, asserted on every machine: the band must hold the
    // active set inside what a 4.0 ms step affords at the certification's
    // measured cost per collider.
    const BUDGET_MS: f64 = 4.0;
    const US_PER_COLLIDER: f64 = 0.363;
    let ceiling = (BUDGET_MS * 1000.0 / US_PER_COLLIDER) as usize;
    assert!(
        banded <= ceiling,
        "the band admits {banded} colliders; {ceiling} is what {BUDGET_MS} ms buys \
         at {US_PER_COLLIDER} us each"
    );
    // …and the alternative really is past it, or this arm is measuring nothing.
    assert!(
        unbanded > ceiling * 10,
        "the unbanded city is {unbanded} colliders, only {:.1}x the ceiling — \
         this fixture is not a city",
        unbanded as f64 / ceiling as f64
    );
}

/// **A near building is whole; a far building is one box.**
///
/// The claim IB-2b makes is not "fewer colliders" — it is that the *reduction is
/// a shell*, so a body cannot walk through a distant building, and that a
/// building you can reach is untouched.
#[test]
fn near_buildings_keep_every_part_and_far_buildings_keep_exactly_one() {
    let city = city(DVec3::ZERO);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&city.world);
    let (near_m, far_m) = bridge.collider_band_radii();
    let band = inf_ecs::SimBand::from_world(&city.world, near_m, far_m);

    let (mut whole, mut shelled, mut gone) = (0usize, 0usize, 0usize);
    for i in 0..(BLOCKS * BLOCKS) {
        let guid = block_guid(i);
        let e = city.world.entity_of(guid).expect("a block");
        let vol = city
            .world
            .world()
            .get::<PcgVolume>(e)
            .expect("the block's volume");
        for (gi, g) in vol.structure_groups.iter().enumerate() {
            let tier = band.tier(g.shell.center, g.shell.half_extents, g.shell.rotation);
            let parts = g
                .range()
                .filter(|i| bridge.collider_of(pcg_structure_guid(guid, *i)).is_some())
                .count();
            let shell = bridge.collider_of(pcg_shell_guid(guid, gi)).is_some();
            match tier {
                Tier::Near => {
                    assert_eq!(
                        parts,
                        g.len as usize,
                        "a NEAR building lost {} of its {} parts",
                        g.len as usize - parts,
                        g.len
                    );
                    assert!(!shell, "a NEAR building also grew a shell — it is sealed");
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
        "IB-2b tiers at near {near_m} m / far {far_m} m: {whole} whole, {shelled} shells, \
         {gone} out, of {} buildings",
        city.buildings
    );
    assert!(
        whole > 0,
        "no building is near the player — nothing enterable"
    );
    assert!(
        shelled > 0,
        "no building is shelled — the fixture does not exercise the far tier"
    );
    // The shell really is a barrier: a body dropped on a shelled building lands
    // on it rather than falling through the city.
    let far_block = (0..BLOCKS * BLOCKS)
        .find(|i| {
            let e = city.world.entity_of(block_guid(*i)).unwrap();
            let vol = city.world.world().get::<PcgVolume>(e).unwrap();
            vol.structure_groups.first().is_some_and(|g| {
                band.tier(g.shell.center, g.shell.half_extents, g.shell.rotation) == Tier::Far
            })
        })
        .expect("some block is in the far tier");
    let e = city.world.entity_of(block_guid(far_block)).unwrap();
    let shell = city
        .world
        .world()
        .get::<PcgVolume>(e)
        .unwrap()
        .structure_groups[0]
        .shell;
    let top = shell.center.y + shell.half_extents.y;
    let probe = bridge.world_mut().add_body(
        inf_physics::BodyKind3D::Dynamic,
        DVec3::new(shell.center.x, top + 3.0, shell.center.z),
        DQuat::IDENTITY,
    );
    bridge
        .world_mut()
        .try_add_collider(
            probe,
            inf_physics::ColliderDesc3D::new(inf_physics::ColliderShape3D::Sphere { radius: 0.4 }),
        )
        .expect("the probe gets a collider");
    for _ in 0..120 {
        bridge.step(1.0 / 60.0);
    }
    let rest = bridge.world().body_translation(probe).expect("live").y;
    assert!(
        rest > top - 1.0,
        "a body dropped on a far building's shell fell to {rest:.3} m; the shell's \
         top is {top:.3} m — a shell that is not a barrier is not a LOD, it is a hole"
    );
    println!("IB-2b: a probe rests at {rest:.3} m on a shell whose top is {top:.3} m");
}

/// **Content covered by no group is banded box by box** (IB-2a, I3 audit).
///
/// A fence, a `grammar.expand` run or a plain scatter has no shell to stand in
/// for it, so `structure_snaps_of` admits a loose solid only at [`Tier::Near`]
/// and drops it beyond. That is a ruling with a consequence — a fence seventy
/// metres away is walk-through — and it had **no fixture**: every solid in this
/// file's city and in `samples/phase30-city` belongs to a building, so the loose
/// branch could be widened to admit at any tier, or narrowed to admit at none,
/// with every arm in both files green.
///
/// The two posts are appended AFTER the block's grouped solids, which is the
/// shape the cursor walk handles last and therefore the one most likely to be
/// dropped by an off-by-one.
#[test]
fn a_solid_covered_by_no_group_is_banded_box_by_box() {
    let mut city = city(DVec3::ZERO);
    let post = |c: DVec3| ScatteredSolid {
        center: c,
        half_extents: DVec3::new(0.3, 1.2, 0.3),
        rotation: DQuat::IDENTITY,
    };
    let (near_post, far_post) = (DVec3::new(2.0, 1.2, 2.0), DVec3::new(520.0, 1.2, 380.0));

    let guid = block_guid(0);
    let e = city.world.entity_of(guid).expect("block 0");
    let (instances, mut solids, groups) = {
        let vol = city.world.world().get::<PcgVolume>(e).expect("a volume");
        (
            vol.evaluated.clone(),
            vol.structures.clone(),
            vol.structure_groups.clone(),
        )
    };
    let grouped = solids.len();
    solids.push(post(near_post));
    solids.push(post(far_post));
    {
        let mut vol = city
            .world
            .world_mut()
            .get_mut::<PcgVolume>(e)
            .expect("live");
        vol.set_population(instances, solids, groups);
        assert_eq!(
            vol.structures.len(),
            grouped + 2,
            "the door refused the loose posts"
        );
    }
    city.world.mark_dirty();

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&city.world);
    let (near_m, far_m) = bridge.collider_band_radii();
    let band = inf_ecs::SimBand::from_world(&city.world, near_m, far_m);

    // ANTI-VACUITY FIRST: the two posts must really be on opposite sides of the
    // band, or this arm is two assertions about one tier.
    let half = post(near_post).half_extents;
    assert_eq!(band.tier(near_post, half, DQuat::IDENTITY), Tier::Near);
    assert_ne!(band.tier(far_post, half, DQuat::IDENTITY), Tier::Near);

    assert!(
        bridge
            .collider_of(pcg_structure_guid(guid, grouped))
            .is_some(),
        "a loose post beside the player has no collider — an ungrouped solid \
         inside the band must be solid"
    );
    assert!(
        bridge
            .collider_of(pcg_structure_guid(guid, grouped + 1))
            .is_none(),
        "a loose post {far_post:?} away kept its collider — an ungrouped solid \
         has no shell to be replaced by, so past the band it is dropped"
    );
    println!(
        "IB-2a ungrouped: 2 loose posts on {grouped} grouped solids — the near \
         one is solid, the far one is not"
    );
}

/// **Walking re-bands, and the swap is atomic.** A building that was a shell
/// becomes whole when the player reaches it — never both, never neither.
#[test]
fn walking_across_the_city_promotes_and_demotes_whole_buildings() {
    let mut city = city(DVec3::new(-600.0, 0.0, 0.0));
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&city.world);

    // A block on the far east, three blocks away.
    let east = BLOCKS * BLOCKS - 1;
    let e = city.world.entity_of(block_guid(east)).unwrap();
    let (first_part, shell_of) = {
        let vol = city.world.world().get::<PcgVolume>(e).unwrap();
        let g = vol.structure_groups.last().expect("a building");
        (g.start as usize, vol.structure_groups.len() - 1)
    };
    assert!(
        bridge
            .collider_of(pcg_structure_guid(block_guid(east), first_part))
            .is_none(),
        "the far block's parts are resident before the player has walked there"
    );

    let target = block_centre(east);
    move_player(&mut city.world, DVec3::new(target.x, 0.0, target.y));
    bridge.sync_from_world(&city.world);

    assert!(
        bridge
            .collider_of(pcg_structure_guid(block_guid(east), first_part))
            .is_some(),
        "walking to a block did not make its buildings solid"
    );
    assert!(
        bridge
            .collider_of(pcg_shell_guid(block_guid(east), shell_of))
            .is_none(),
        "the promoted building kept its shell — that box is standing in its doorway"
    );
    // And the block the player left is no longer whole.
    let west = 0;
    let wfirst = {
        let e = city.world.entity_of(block_guid(west)).unwrap();
        let vol = city.world.world().get::<PcgVolume>(e).unwrap();
        vol.structure_groups[0].start as usize
    };
    assert!(
        bridge
            .collider_of(pcg_structure_guid(block_guid(west), wfirst))
            .is_none(),
        "the block the player left kept its parts — the band only ever grows"
    );
    println!(
        "IB-2a: walking 1 200 m re-banded to {} colliders",
        bridge.body_count()
    );
}

/// **The band is a pure function of sim state.** Two independently built worlds
/// with the player in the same place produce the *same* active set, and moving
/// only inside a lattice cell does not move it at all.
#[test]
fn the_active_set_is_a_function_of_the_world_and_nothing_else() {
    let active = |at: DVec3| -> Vec<Uuid> {
        let c = city(at);
        let mut b = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        b.sync_from_world(&c.world);
        let mut v: Vec<Uuid> = b.tracked_bodies().into_iter().map(|(g, _, _)| g).collect();
        v.sort_unstable();
        v
    };
    let a = active(DVec3::new(30.0, 1.0, -20.0));
    let b = active(DVec3::new(30.0, 1.0, -20.0));
    assert_eq!(a, b, "two builds of one world banded differently");
    assert!(!a.is_empty());

    // Inside one lattice cell: the same set, because the anchor is snapped.
    let same = active(DVec3::new(30.0, 900.0, -20.0));
    assert_eq!(a, same, "height moved the band — the distance is XZ only");
    let jiggle = active(DVec3::new(31.5, 1.0, -19.0));
    assert_eq!(a, jiggle, "a step inside a lattice cell re-banded");

    // Across one: a different set, or the lattice is too coarse to be a band.
    let moved = active(DVec3::new(30.0 + inf_ecs::BAND_LATTICE_M * 4.0, 1.0, -20.0));
    assert_ne!(a, moved, "walking 64 m did not change the active set");
    println!(
        "IB-2a: {} colliders active; a 64 m walk changes {} of them",
        a.len(),
        a.iter().filter(|g| !moved.contains(g)).count()
            + moved.iter().filter(|g| !a.contains(g)).count()
    );
}

/// **The radius sweep**: what each near radius costs, so the default is a
/// measurement rather than a preference.
///
/// The **clock** column is this machine's and is printed, never asserted — the
/// standing rule (`bridge_sync_scaling`'s precedent, and the reason
/// `dig_stall_bench` is `#[ignore]`d). The **collider** column is a function of
/// the world alone, so it is asserted everywhere: the shipped default must be
/// the largest radius on this table that stays inside the budget the
/// certification's own µs/collider implies.
#[test]
fn the_radius_sweep_states_what_the_default_buys() {
    const BUDGET_MS: f64 = 4.0;
    const US_PER_COLLIDER: f64 = 0.363;
    let ceiling = (BUDGET_MS * 1000.0 / US_PER_COLLIDER) as usize;

    let city = city(DVec3::ZERO);
    println!(
        " near_m | colliders |  cert ms | measured ms   ({} buildings, {} solids, ceiling {ceiling})",
        city.buildings, city.solids
    );
    let mut rows: Vec<(f64, usize)> = Vec::new();
    for near_m in [16.0f64, 32.0, 48.0, 64.0, 96.0, 128.0] {
        let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        bridge.set_collider_band_radii(near_m, inf_ecs::DEFAULT_COLLIDER_FAR_M);
        bridge.sync_from_world(&city.world);
        bridge.sync_from_world(&city.world); // one warm pass
        let n = bridge.body_count();

        const ITERS: u32 = 30;
        let t0 = Instant::now();
        for _ in 0..ITERS {
            bridge.sync_from_world(&city.world);
            bridge.step(1.0 / 60.0);
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3 / f64::from(ITERS);
        // ANTI-VACUITY: a timer that reports zero is measuring its own
        // resolution, not a fixed step over thousands of colliders.
        assert!(ms > 0.0, "the fixed step took no measurable time");
        println!(
            "{near_m:>7.0} | {n:>9} | {:>8.3} | {ms:>8.4}",
            n as f64 * US_PER_COLLIDER / 1000.0
        );
        rows.push((near_m, n));
    }

    // The shipped default is on this table and inside the ceiling.
    let default = inf_ecs::DEFAULT_COLLIDER_NEAR_M;
    let at_default = rows
        .iter()
        .find(|(r, _)| *r == default)
        .unwrap_or_else(|| panic!("the shipped default {default} m is not on the sweep"));
    assert!(
        at_default.1 <= ceiling,
        "the shipped near radius {default} m admits {} colliders, past the {ceiling} \
         a {BUDGET_MS} ms step buys",
        at_default.1
    );
    // …and it is the WIDEST such row, which is the claim the constant's own doc
    // makes (I3 audit). Without this the arm passes for 16 m as happily as for
    // 64 m, and "the radius is a measurement" would be a statement about the
    // table rather than about the number that ships: a narrower default is a
    // character walking into a building that is still a shell.
    for (r, n) in &rows {
        assert!(
            *r <= default || *n > ceiling,
            "the sweep offers {r} m at {n} colliders — inside the {ceiling} \
             budget and wider than the shipped {default} m default, so the \
             default is not the widest radius the budget buys"
        );
    }
    // …and the sweep really does climb, or it is not measuring a radius.
    let widest = rows.last().expect("rows").1;
    assert!(
        widest > rows[0].1 * 4,
        "the widest radius admits {widest} against the narrowest's {} — this sweep \
         is not varying anything",
        rows[0].1
    );
}

/// **WHERE THE SOLVER'S MILLISECONDS WENT** (island wave I4b) - the measurement
/// behind `PhysicsWorld3D::active_collision_types`.
///
/// Wave I4 measured the shipped player's fixed step over this city at
/// 13.0-14.9 ms and could not say where it went; I4b's phase clock said
/// **72.9 % of it was `bridge3d.step`** - the rapier call itself, over a world
/// whose only moving thing is one character. This arm is the why, and the price
/// of the alternative, on four configurations of one city.
///
/// A building's boxes **touch each other** (a wall meets a floor meets a stair)
/// and every one of them rests on the **ground**, which under a streamed terrain
/// is a heightfield of 32 768 triangles. Under
/// `ActiveCollisionTypes::all()` every one of those adjacencies is a manifold
/// the narrow phase computes and re-computes at 60 Hz for a pair no solver can
/// ever move. Measured here before the rule landed, on this machine, MIN of five
/// rounds of sixty steps:
///
/// | configuration | ms/step, before | after |
/// |---|---|---|
/// | the banded city alone | 0.555-0.600 | **0.283** |
/// | + the host's per-step sync | 1.369-1.413 | **1.044** |
/// | + one awake dynamic body | 0.571-0.593 | **0.285** |
/// | **+ 25 heightfield tiles under it** | **4.259** | **0.319** |
///
/// The ground is the whole story: 3 446 box-versus-heightfield pairs cost
/// **+3.704 ms/step** and now cost **+0.036**. In the shipped instrument's scene,
/// with its real terrain residency under it, the same mechanism was **9.2 ms of a
/// 12.7 ms fixed step** and is now 0.3 of 1.2.
///
/// Printed, never asserted as a clock (the standing rule); the **pair counts**
/// are asserted, because a pair is a pair on every machine.
#[test]
fn the_solver_pays_for_static_pairs_the_city_can_never_move() {
    let city = city(DVec3::ZERO);
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&city.world);
    let dt = 1.0 / 60.0;
    // Warm: the first steps build the broad-phase tree and seat every manifold.
    for _ in 0..30 {
        bridge.step(dt);
    }
    let (tracked, touching) = bridge.world().contact_pair_counts();
    let colliders = bridge.admitted_structures();
    // MIN of rounds, the instrument's own discipline one crate down.
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..60 {
            bridge.step(dt);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1000.0 / 60.0);
    }
    println!(
        "banded city: {colliders} structure colliders, {tracked} contact pairs \
         tracked ({touching} touching), solver {best:.3} ms/step — \
         {:.1} pairs per collider",
        tracked as f64 / colliders.max(1) as f64
    );
    // …and the same step with the HOST'S OWN SHAPE around it: a sync before
    // every step, which is what `RuntimeSim::fixed_step` and `SimSession` both
    // do. A step measured without its sync is a step no host runs.
    let mut synced = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for _ in 0..60 {
            bridge.sync_from_world(&city.world);
            bridge.step(dt);
        }
        synced = synced.min(t.elapsed().as_secs_f64() * 1000.0 / 60.0);
    }
    println!(
        "  with the host's per-step sync in front of it: {synced:.3} ms/step \
         ({:+.3} ms)",
        synced - best
    );
    // …and with ONE AWAKE BODY in it. The instrument's scene has a character;
    // this one has none, and a world where nothing is awake is a world where
    // rapier's narrow phase early-outs on every edge it owns.
    {
        let mut w = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        w.sync_from_world(&city.world);
        let b = w.world_mut().add_body(
            inf_physics::d3::BodyKind3D::Dynamic,
            DVec3::new(0.0, 40.0, 0.0),
            DQuat::IDENTITY,
        );
        let _ = w.world_mut().add_collider(
            b,
            inf_physics::d3::ColliderDesc3D::new(inf_physics::d3::ColliderShape3D::Sphere {
                radius: 0.5,
            }),
        );
        for _ in 0..30 {
            w.step(dt);
        }
        let (t2, _) = w.world().contact_pair_counts();
        let mut awake = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..60 {
                w.step(dt);
            }
            awake = awake.min(t.elapsed().as_secs_f64() * 1000.0 / 60.0);
        }
        println!(
            "  with ONE awake dynamic body among the {colliders} static ones: \
             {awake:.3} ms/step ({:+.3} ms), {t2} pairs",
            awake - best
        );
    }
    // …and with the GROUND under it. The instrument's scene has a streamed
    // terrain and this one is a city floating in space; a heightfield tile is
    // 128 x 128 m of 32 768 triangles, and every box in the tile's footprint
    // pairs with it.
    {
        let mut w = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        w.sync_from_world(&city.world);
        let span = 128.0f64;
        let res = 129u32;
        for tz in -2i32..=2 {
            for tx in -2i32..=2 {
                let b = w.world_mut().add_body(
                    inf_physics::d3::BodyKind3D::Static,
                    DVec3::new(f64::from(tx) * span, 0.0, f64::from(tz) * span),
                    DQuat::IDENTITY,
                );
                let _ = w.world_mut().add_collider(
                    b,
                    inf_physics::d3::ColliderDesc3D::new(
                        inf_physics::d3::ColliderShape3D::Heightfield {
                            samples_x: res,
                            samples_z: res,
                            heights: vec![0.0; (res * res) as usize],
                            removed_cells: Vec::new(),
                            span: DVec2::splat(span),
                        },
                    ),
                );
            }
        }
        for _ in 0..30 {
            w.step(dt);
        }
        let (t3, u3) = w.world().contact_pair_counts();
        let mut ground = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            for _ in 0..60 {
                w.step(dt);
            }
            ground = ground.min(t.elapsed().as_secs_f64() * 1000.0 / 60.0);
        }
        println!(
            "  with 25 heightfield tiles under it: {ground:.3} ms/step \
             ({:+.3} ms), {t3} pairs ({u3} touching)",
            ground - best
        );
    }
    assert!(
        tracked > colliders,
        "the narrow phase tracks {tracked} pairs over {colliders} colliders — \
         fewer pairs than colliders means the city's boxes are not touching each \
         other and this arm is measuring a different world than the one its \
         header describes"
    );
    // **AND NONE OF THEM IS A MANIFOLD.** This is the arm the rule dies at: put
    // `FIXED_FIXED` back on solids and a city of static boxes on static ground
    // starts computing twenty thousand contact manifolds a step again.
    assert_eq!(
        touching, 0,
        "{touching} of the city's {tracked} static pairs computed a contact \
         manifold — `ActiveCollisionTypes` is back to `all()` for solids, and \
         this city's fixed step has just gone back to paying nine milliseconds \
         for contacts no solver can act on"
    );
}
