//! **The parallel pose step on the shipped island is invariant to the pool**
//! (audit PERF1c, item (a'')).
//!
//! PERF1c moved `inf_ecs::pose::step_pose_evaluation`'s per-character
//! evaluation onto `bevy_tasks`' `ComputeTaskPool` (`par_chunk_map_mut`, a
//! chunk of `ceil(jobs / threads)` characters). The in-crate arm
//! (`pose::tests::a_crowd_poses_each_character_as_it_poses_alone`) holds that
//! on forty synthetic characters for twelve steps; this holds it on the
//! subject: the cooked island, 21:00 on the Harbour City strip, with the
//! society, the traffic and its seated riders, a crowd character ragdolled
//! through the gameplay door and the hero walking, sprinting, aiming down
//! sights and firing, pressing cover, jumping and crouching — 600 steps, five
//! pool sizes (1, 2, 3, 8, 32 workers: one chunk, two, three, eight, and more
//! workers than characters) and a second run at the largest, each in its own
//! process because the pool is a process-global `OnceLock` (see
//! `src/bin/pose_probe.rs`).
//!
//! READS, every step: the sim's `state_bytes` (the PIE / replay fold), the pose
//! store's bytes and the state-machine trace (every `AnimStateMachine` runtime
//! and every published `AnimBridgeRes::states` entry), folded into three
//! hashes, plus a hash mark every 50 steps (so a divergence names its window).
//! Engagement: every leg step must have had at least `PARALLEL_POSE_MIN`
//! posed characters (so the pool path, not the calling thread, ran), and the
//! leg must have engaged riders, a ragdoll and aim; cover / mantle counts are
//! printed (they depend on the street the leg crosses).
//!
//! Mutation (audit, by hand, recorded in the report): inside a chunk each job
//! starts from the machine runtime its predecessor just produced (a job that
//! reads what another job wrote) — RED, `state` differs on a pool of 2 from a
//! pool of 1. The same probe built over the base tree's serial pose step
//! printed the same three hashes as the parallel one (audit, by hand).
//!
//! Needs a cooked island pack (`INF_ISLAND_PACK`), like
//! `fps_instrument::the_shipped_island_by_the_hour`; prints its skip otherwise
//! (CI never cooks the island).

use std::collections::HashMap;
use std::process::Command;

use glam::DVec3;

fn strip_centre() -> DVec3 {
    let recipe = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../samples/island/island.toml");
    let recipe = inf_island::IslandRecipe::load(&recipe).expect("recipe");
    let design = inf_island::read_design(&recipe).expect("design");
    let plans = inf_editor_core::settlement::settlements(&design);
    let city = plans
        .iter()
        .find(|p| p.name == "Harbour City")
        .expect("Harbour City");
    let strip: Vec<DVec3> = city
        .blocks
        .iter()
        .filter(|b| b.archetype.is_venue())
        .map(|b| DVec3::new(b.centre.x, 0.0, b.centre.y))
        .collect();
    strip.iter().fold(DVec3::ZERO, |a, &b| a + b) / strip.len().max(1) as f64
}

fn probe(pack: &std::path::Path, threads: usize, at: DVec3) -> HashMap<String, String> {
    let out = Command::new(env!("CARGO_BIN_EXE_pose_probe"))
        .arg(threads.to_string())
        .arg(STEPS.to_string())
        .arg(pack)
        .arg("21")
        .arg(format!("{}", at.x))
        .arg(format!("{}", at.z))
        .output()
        .expect("spawn pose_probe");
    assert!(
        out.status.success(),
        "pose_probe (threads={threads}) failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("utf8")
        .lines()
        .filter_map(|l| l.split_once('=').map(|(k, v)| (k.into(), v.into())))
        .collect()
}

const STEPS: usize = 600;

fn count(run: &HashMap<String, String>, key: &str) -> usize {
    run.get(key).and_then(|v| v.parse().ok()).unwrap_or(0)
}

#[test]
fn the_island_poses_the_same_bytes_on_every_pool_size() {
    let Some(pack) = std::env::var_os("INF_ISLAND_PACK").map(std::path::PathBuf::from) else {
        println!("SKIP the_island_poses_the_same_bytes_on_every_pool_size: INF_ISLAND_PACK names no pack");
        return;
    };
    let at = strip_centre();
    println!("strip centre ({:.3}, {:.3})", at.x, at.z);
    let sizes = [1usize, 2, 3, 8, 32, 32];
    let runs: Vec<HashMap<String, String>> = sizes.iter().map(|n| probe(&pack, *n, at)).collect();
    for (n, run) in sizes.iter().zip(&runs) {
        println!("pool {n}: {run:?}");
        assert_eq!(
            run.get("threads").map(String::as_str),
            Some(n.to_string().as_str()),
            "the probe did not get the pool it asked for"
        );
    }
    for key in ["state", "pose", "trace", "marks"] {
        let reference = &runs[0][key];
        for (n, run) in sizes.iter().zip(&runs) {
            assert_eq!(
                &run[key], reference,
                "{key} differs on a pool of {n} from a pool of 1"
            );
        }
    }
    let r = &runs[0];
    assert_eq!(
        count(r, "parallel_steps"),
        STEPS,
        "every leg step must have posed enough characters to take the pool path: {r:?}"
    );
    for key in ["rider_steps", "ragdoll_steps", "aim_steps"] {
        assert!(count(r, key) > 0, "the leg engaged no {key}: {r:?}");
    }
    println!(
        "ENGAGED: posed max {}, rider-steps {}, ragdoll-steps {}, aim-steps {}, arc-steps {}, cover-steps {}, mantle-steps {}",
        count(r, "posed_max"),
        count(r, "rider_steps"),
        count(r, "ragdoll_steps"),
        count(r, "aim_steps"),
        count(r, "arc_steps"),
        count(r, "cover_steps"),
        count(r, "mantle_steps"),
    );
}
