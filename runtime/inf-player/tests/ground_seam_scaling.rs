//! **The `terrain.height_at` host seam does not scale with the world**
//! (Hardening Wave E).
//!
//! This is the function a Blueprint's `terrain.height_at` node dispatches to, and
//! it is designed to be called from an actor `Tick` handler — so its cost is
//! multiplied by the actor count on every fixed step. It used to walk **every
//! entity in the world**, with a `Guid` lookup and a `Terrain` lookup each, to
//! find the one or two entities that carry a terrain: measured at 0.0137 ms/call
//! over 1 000 entities, 0.0668 over 5 000 and **0.2032 over 15 000** — linear in
//! the world, for a query whose answer set is one.
//!
//! Two arms, and the first is the one that matters.
//!
//! * **The answer is identical at every world size.** The seam gathers its
//!   terrains by an explicit `Guid` sort, not by iteration order, so scoping the
//!   walk to the archetypes that actually carry a `Terrain` cannot move it.
//!   Exact, machine-independent, and the property that makes the change a repair
//!   rather than a policy. (Which terrain *answers* became position-aware in the
//!   island phase's IB-15 — topmost surface that answers, ties to the first in
//!   `Guid` order — and this fixture has one terrain, where the two rules are the
//!   same answer by construction.)
//! * **The cost does not grow with the world.** A ratio measured on one machine
//!   in one process — far more robust than an absolute millisecond, and
//!   deliberately generous (4x) against a regression that would be 15x. Each leg
//!   is the **minimum of several rounds**, because preemption only ever *adds*
//!   time, so the min is the round the scheduler left alone. It is reported on
//!   every machine and asserted everywhere **except paravirtual macOS CI**: a
//!   run there measured 4.75x on byte-untouched code from a ~0.45 ms window —
//!   the same runner class that invalidated both of `bridge_sync_scaling`'s
//!   controls in opposite directions on consecutive runs — which falsified this
//!   file's original claim that a 4x tripwire "cannot flake". Windows and Linux
//!   CI and every dev machine still assert it; the answer half asserts
//!   everywhere, exact.

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{MeshRef, Terrain, Transform};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_terrain::TerrainData;

const TERRAIN_GUID: u128 = 0x1E08_0001;
const PROP_BASE: u128 = 0x1E08_1000;
/// The height every sample of the fixture terrain sits at.
const GROUND_Y: f64 = 5.0;

/// A flat one-tile terrain plus `props` ordinary mesh entities.
fn sim_with(props: u32) -> RuntimeSim {
    let mut world = EcsWorld::new();
    let t = world.spawn_with_guid(Uuid::from_u128(TERRAIN_GUID), "Terrain", None);
    let mut data = TerrainData::new(33, 1.0);
    data.author_tile((0, 0), |_, _| GROUND_Y);
    world
        .world_mut()
        .entity_mut(t)
        .insert(Transform::default())
        .insert(Terrain {
            meters_per_sample: 1.0,
            tile_resolution: 33,
            data,
            ..Terrain::default()
        });
    for i in 0..props {
        let e = world.spawn_with_guid(Uuid::from_u128(PROP_BASE + u128::from(i)), "Prop", None);
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(DVec3::new(
                f64::from(i % 32),
                0.0,
                f64::from(i / 32),
            )))
            .insert(MeshRef::default());
    }
    world.propagate();
    let mut sim = RuntimeSim::new(world, Vec::new(), DVec2::ZERO, 60.0);
    sim.step_once(RuntimeInput::default());
    sim
}

#[test]
fn the_ground_seam_answers_the_same_and_does_not_scale_with_the_world() {
    const CALLS: u32 = 400;
    let mut timings: Vec<(u32, f64)> = Vec::new();
    for props in [1_000u32, 5_000, 15_000] {
        let mut sim = sim_with(props);

        // (a) THE ANSWER. Exact, and the same at every size.
        for probe in [0.0, 4.5, 16.0, 31.0] {
            let h = sim.terrain_height_at(probe, probe);
            assert!(
                (h - GROUND_Y).abs() < 1e-9,
                "the ground seam answered {h} over a flat {GROUND_Y} m terrain \
                 with {props} props in the world"
            );
        }
        // …and it still answers `0.0` (the "no ground" reading) off the tile,
        // so the arm above is not passing because everything answers 5.
        assert_eq!(sim.terrain_height_at(100_000.0, 0.0), 0.0);

        // Minimum of ROUNDS: preemption can only inflate a round, never deflate
        // it, so the min is the scheduler-quiet measurement the ratio needs.
        const ROUNDS: u32 = 5;
        let mut best = f64::INFINITY;
        for _ in 0..ROUNDS {
            let start = std::time::Instant::now();
            let mut sink = 0.0;
            for i in 0..CALLS {
                sink += sim.terrain_height_at(f64::from(i % 32), 4.0);
            }
            let per = start.elapsed().as_secs_f64() * 1000.0 / f64::from(CALLS);
            assert!(
                sink > 0.0,
                "the timed loop must actually sample the terrain"
            );
            best = best.min(per);
        }
        eprintln!("ground_seam: {props} props -> {best:.4} ms/call (min of {ROUNDS})");
        timings.push((props, best));
    }

    // (b) THE SCALING. A whole-world walk is 15x from the first row to the last;
    // an archetype-scoped query is flat. Four is the tripwire — loose enough that
    // a noisy machine cannot flake it, far below what the regression produces.
    let (small, small_ms) = timings[0];
    let (large, large_ms) = timings[timings.len() - 1];
    let ratio = large_ms / small_ms.max(1e-6);
    eprintln!(
        "ground_seam: {large} props costs {ratio:.2}x what {small} does \
         ({large_ms:.4} ms vs {small_ms:.4} ms)"
    );
    // Reported everywhere; not asserted on the runner class whose scheduler has
    // measurably manufactured multipliers out of microsecond windows (see the
    // module docs). Windows/Linux CI and every dev machine assert.
    if cfg!(target_os = "macos") && std::env::var_os("CI").is_some() {
        eprintln!(
            "ground_seam: ratio {ratio:.2}x REPORTED, not asserted — paravirtual \
             macOS CI (the answer half above was asserted at every size)"
        );
        return;
    }
    assert!(
        ratio < 4.0,
        "the ground seam cost {ratio:.2}x more over {large} entities than over \
         {small} — it is walking the world again rather than its own archetypes \
         ({large_ms:.4} ms vs {small_ms:.4} ms)"
    );
}
