//! Subprocess probe for the **pool-size invariance of the parallel pose step**
//! (audit PERF1c, item (a'')).
//!
//! Since PERF1c `inf_ecs::pose::step_pose_evaluation` evaluates its characters
//! on `bevy_tasks`' `ComputeTaskPool` (`par_chunk_map_mut`, a chunk size of
//! `ceil(jobs / pool threads)`), so unlike the destruction / voxel / water
//! probes this one guards a fixed-step path that REALLY runs concurrently: the
//! pool size decides how many chunks there are, which characters share a chunk
//! and which worker evaluates which. The pool is a process-global `OnceLock`
//! (the first init wins), so the launching test varies it across
//! **subprocesses**, exactly as `destruct_probe.rs` explains.
//!
//! The subject is the shipped island, not a fixture: the probe opens a cooked
//! island pack the way the shipped player does (cells + terrain streaming
//! attached), lets the society and the traffic populate about the hero, then
//! drives the hero through a scripted leg (walk, sprint, aim down sights and
//! fire, cover, jumps, crouch) while one crowd character is put into a ragdoll
//! through the gameplay door. Every step it folds
//!
//! * the sim's own `state_bytes` (the PIE / replay fold: world snapshot, pose
//!   store, cloth, hair, doors, items, weapons, health, crowd, traffic, ...),
//! * the pose store's bytes on their own, and
//! * the state-machine trace: every `AnimStateMachine` runtime and every
//!   published `AnimBridgeRes::states` entry, in `Guid` order,
//!
//! into three 128-bit hashes, and counts what the leg engaged (posed
//! characters, jobs over the parallel threshold, seated riders, ragdolls,
//! traversal arcs, aimers, cover holders, mantles) so the launching test can say
//! what the equality was taken over.
//!
//! Usage: `pose_probe <threads> <steps> <pack dir>`. Output (stdout, one
//! `key=value` per line): `threads=`, `steps=`, `state=`, `pose=`, `trace=`,
//! `first_divergence_marks=` (a per-50-step hash list), and the engagement
//! counters.

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::run();
}

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::PathBuf;

    use inf_ecs::anim_bridge::AnimBridgeRes;
    use inf_ecs::components::{
        AnimStateMachine, CharacterMovement, Guid, MovementMode, RotationMode,
    };
    use inf_ecs::pose::{PoseStoreRes, PARALLEL_POSE_MIN};
    use inf_player::level::PackLevelSource;
    use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

    /// The `destruct_probe` fold: integer mixing only, bit-portable.
    #[derive(Default)]
    struct Hasher(u128);

    impl Hasher {
        fn eat(&mut self, bits: u64) {
            self.0 ^= bits as u128;
            self.0 = self
                .0
                .wrapping_mul(0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645);
            self.0 ^= self.0 >> 47;
        }
        fn bytes(&mut self, b: &[u8]) {
            self.eat(b.len() as u64);
            for c in b.chunks(8) {
                let mut w = [0u8; 8];
                w[..c.len()].copy_from_slice(c);
                self.eat(u64::from_le_bytes(w));
            }
        }
    }

    /// Steps the society and the traffic get to populate before the leg.
    const WARMUP_STEPS: usize = 240;

    /// The hero's scripted input on leg step `s`.
    fn input(s: usize) -> RuntimeInput {
        let walk = |i: RuntimeInput| i.axis_at("move_y", 1.0);
        match s {
            0..=99 => walk(RuntimeInput::default()).axis_at("look_x", 0.15),
            100..=199 => walk(RuntimeInput::with_down(["sprint"])),
            200..=299 => RuntimeInput::with_down(["aim"]).axis_at("move_y", 0.5),
            300..=339 => RuntimeInput::with_down(["aim", "attack"]).axis_at("look_x", 0.3),
            340..=399 => {
                if s.is_multiple_of(20) {
                    RuntimeInput::with_down(["cover"])
                } else {
                    RuntimeInput::default().axis_at("move_x", 0.5)
                }
            }
            400..=499 => {
                if s.is_multiple_of(25) {
                    walk(RuntimeInput::with_down(["jump"]))
                } else {
                    walk(RuntimeInput::default())
                }
            }
            _ => walk(RuntimeInput::with_down(["crouch"])).axis_at("look_x", -0.2),
        }
    }

    fn open(pack: &std::path::Path) -> RuntimeSim {
        let source = PackLevelSource::open(pack).expect("the island pack opens");
        let mut built = inf_player::build_world_from_pack(&source).expect("the world builds");
        let partition = built.take_partition();
        let pcg = built.pcg_context();
        let mut sim = inf_player::sim_from_built(built);
        inf_player::attach_cell_streaming(&mut sim, &partition, pcg);
        inf_player::attach_terrain_streaming(
            &mut sim,
            &inf_player::TerrainContent::Pack(PackLevelSource::open(pack).expect("re-open")),
        );
        sim
    }

    /// Put the lowest-`Guid` posed, non-hero character with a movement
    /// component into a ragdoll through the gameplay door.
    fn ragdoll_someone(sim: &mut RuntimeSim) -> bool {
        let candidates: Vec<uuid::Uuid> = {
            let w = sim.world().world();
            let Some(store) = w.get_resource::<PoseStoreRes>() else {
                return false;
            };
            store
                .0
                .keys()
                .copied()
                .filter(|g| {
                    sim.world().entity_of(*g).is_some_and(|e| {
                        w.get::<CharacterMovement>(e)
                            .is_some_and(|m| !m.player_controlled)
                    })
                })
                .collect()
        };
        candidates
            .into_iter()
            .any(|g| inf_physics::d3::ragdoll_bridge::start_ragdoll(sim.world_mut(), g))
    }

    pub fn run() {
        let mut args = std::env::args().skip(1);
        let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(600);
        let pack = PathBuf::from(
            args.next()
                .expect("usage: pose_probe <threads> <steps> <pack> [hour] [x z]"),
        );
        let hour: Option<f64> = args.next().and_then(|s| s.parse().ok());
        let at: Option<(f64, f64)> = match (args.next(), args.next()) {
            (Some(x), Some(z)) => x.parse().ok().zip(z.parse().ok()),
            _ => None,
        };
        let actual = inf_ecs::schedule::init_ecs_task_pool(threads);
        println!("threads={actual}");
        println!("steps={steps}");
        let mut sim = open(&pack);
        if let Some(hour) = hour {
            // The hour arm's door (`fps_instrument::the_shipped_island_by_the_hour`).
            let w = sim.world_mut().world_mut();
            let mut q = w.query::<&mut inf_ecs::components::TimeOfDay>();
            for mut tod in q.iter_mut(w) {
                tod.seconds = (hour * 3600.0 - tod.longitude_deg * 240.0).rem_euclid(86_400.0);
                tod.rate = 0.0;
            }
            sim.world_mut().mark_dirty();
        }
        if let Some((x, z)) = at {
            // The hero is the streaming anchor; standing it on the strip puts
            // the night society around it (the hour arm's `set_hero`).
            let y = sim.terrain_height_at(x, z) + 2.0;
            let world = sim.world_mut();
            let hero = world.world().iter_entities().find_map(|e| {
                e.get::<CharacterMovement>()
                    .is_some_and(|m| m.player_controlled)
                    .then_some(e.id())
            });
            if let Some(e) = hero {
                if let Some(mut t) = world
                    .world_mut()
                    .get_mut::<inf_ecs::components::Transform>(e)
                {
                    t.translation = inf_ecs::math::Vec3d::new(x, y, z);
                }
            }
        }
        for _ in 0..WARMUP_STEPS {
            sim.step_once(RuntimeInput::default());
        }
        let (mut state, mut pose, mut trace) =
            (Hasher::default(), Hasher::default(), Hasher::default());
        let mut marks = Vec::new();
        let mut posed_max = 0usize;
        let mut parallel_steps = 0usize;
        let (mut riders, mut ragdolls, mut arcs, mut aimers, mut cover, mut mantles) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let mut ragdolled = false;
        for s in 0..steps {
            if s == 150 || (s > 150 && !ragdolled && s.is_multiple_of(10)) {
                ragdolled |= ragdoll_someone(&mut sim);
            }
            sim.step_once(input(s));
            state.bytes(&sim.state_bytes());
            let w = sim.world();
            pose.bytes(&inf_ecs::pose::pose_state_bytes(w));
            // The state-machine trace, in Guid order.
            let mut machines: Vec<(uuid::Uuid, String)> = Vec::new();
            let mut q = w.world().iter_entities();
            for e in q.by_ref() {
                if let (Some(g), Some(asm)) = (e.get::<Guid>(), e.get::<AnimStateMachine>()) {
                    machines.push((g.0, format!("{:?}", asm.runtime)));
                }
                if let Some(m) = e.get::<CharacterMovement>() {
                    aimers += usize::from(m.rotation_mode == RotationMode::Aiming);
                    cover += usize::from(m.runtime.cover.active);
                    mantles += usize::from(m.mode == MovementMode::Mantle);
                }
            }
            machines.sort();
            for (g, rt) in &machines {
                trace.bytes(g.as_bytes());
                trace.bytes(rt.as_bytes());
            }
            if let Some(b) = w.world().get_resource::<AnimBridgeRes>() {
                for (g, st) in &b.states {
                    trace.bytes(g.as_bytes());
                    trace.eat(st.index as u64);
                    trace.bytes(st.name.as_bytes());
                    trace.eat(st.time_s.to_bits());
                    trace.eat(u64::from(st.blending));
                }
                ragdolls += b.ragdoll_pose.len();
                arcs += b.traversal.len();
            }
            if let Some(r) = w
                .world()
                .get_resource::<inf_ecs::traffic::SeatedRidersRes>()
            {
                riders += r.riders.len();
            }
            let posed = w
                .world()
                .get_resource::<PoseStoreRes>()
                .map_or(0, |p| p.0.len());
            posed_max = posed_max.max(posed);
            parallel_steps += usize::from(posed >= PARALLEL_POSE_MIN);
            if (s + 1).is_multiple_of(50) {
                marks.push(format!(
                    "{:032x}",
                    state.0 ^ pose.0.rotate_left(1) ^ trace.0.rotate_left(2)
                ));
            }
        }
        println!("state={:032x}", state.0);
        println!("pose={:032x}", pose.0);
        println!("trace={:032x}", trace.0);
        println!("marks={}", marks.join(","));
        println!("posed_max={posed_max}");
        println!("parallel_steps={parallel_steps}");
        println!("rider_steps={riders}");
        println!("ragdoll_steps={ragdolls}");
        println!("ragdolled={ragdolled}");
        println!("arc_steps={arcs}");
        println!("aim_steps={aimers}");
        println!("cover_steps={cover}");
        println!("mantle_steps={mantles}");
    }
}
