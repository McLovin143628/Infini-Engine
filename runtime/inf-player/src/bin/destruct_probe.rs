//! Subprocess probe for the **pool-size-invariance** leg of the P22.4 gate.
//!
//! `bevy_ecs`'s `ComputeTaskPool` is a process-global `OnceLock`: the **first**
//! call wins and every later one is a no-op that reports the count already
//! chosen, so a single process can only ever run under one pool size and an
//! in-process "matrix" is the two-run determinism arm wearing a stronger name.
//! `crates/inf-runtime/src/bin/replay_probe.rs` reached that conclusion first,
//! `water_probe.rs` is the P20.2 twin and `voxel_probe.rs` the P21.4 one; this is
//! destruction's.
//!
//! # What this probe ACTUALLY guards — the premise, corrected
//!
//! An earlier version of this comment said the `destruct.*` calls "run from a
//! Blueprint tick, **which runs in the ECS schedule**". **That is false**, and the
//! P22.4 audit caught it. `RuntimeSim` runs no `SimSchedule` at all: its
//! `run_all_with_args` is a serial `for` loop over a `Vec` of `BTreeMap` keys,
//! `inf_physics::d3::fracture` names no job pool, and rapier's `parallel` feature
//! is off by rule. Four pool sizes therefore execute **the same serial program**,
//! and this probe cannot *discover* an ordering race because there is no
//! concurrency on the step path for one to live in.
//!
//! Saying otherwise would be the "inference dressed as measurement" mistake this
//! phase's own ledger records one law up: four green runs proving a property that
//! was never at risk, presented as evidence that it was.
//!
//! So the honest statement of what these runs are worth:
//!
//! * **A regression tripwire, with a real cost of zero.** `inf-ecs` *does* have a
//!   parallel schedule (`SimSchedule` in `ScheduleMode::Parallel`), and
//!   `runtime_sim`'s own module docs name moving the tick pass onto it as a
//!   documented follow-up. The day that lands, this arm becomes load-bearing —
//!   and it will already exist, already be wired into the phase gate, and already
//!   have a reference hash, instead of being something somebody has to remember
//!   to write.
//! * **A pin on the process-global pool being irrelevant here.** `bevy_ecs`'s
//!   `ComputeTaskPool` is a `OnceLock`: the first init wins and later ones are
//!   no-ops that report the count already chosen. So the pool a process gets IS
//!   decided by the `threads=` line this prints, and the trace being invariant to
//!   it is the statement "nothing on the fixed-step path has reached for that
//!   pool" — checked rather than assumed, which is the part that can change
//!   without anyone noticing.
//!
//! Subprocesses rather than `init_ecs_task_pool` calls, for the `OnceLock` reason
//! above: an in-process "matrix" runs every leg on one pool and is the two-run
//! determinism arm wearing a stronger name.
//!
//! The probe builds its scene from a **bare `EcsWorld`** — no editor crate,
//! because a `[[bin]]` cannot reach dev-dependencies — pins the pool to a
//! requested size, runs the trace, and prints hashes. The launching test spawns it
//! once per pool size and compares the outputs *to each other*, so the scene lives
//! in exactly one place and cannot drift between the probe and its caller.
//!
//! Usage: `destruct_probe [threads] [steps]`  (defaults: threads=0→auto, steps=180)
//! Output (stdout, one `key=value` per line):
//!
//! ```text
//! threads=<actual pool worker count>
//! steps=<n>
//! first=<hex 128-bit hash of the first frame>
//! last=<hex 128-bit hash of the last frame>
//! trace=<hex 128-bit hash of every frame>
//! field=<hex 128-bit hash of the FINAL fracture state of every actor>
//! ```

use std::sync::Arc;

use glam::{DVec2, DVec3};
use inf_blueprint::{
    BinOp, BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{Destructible, MeshRef, Terrain, Transform};
use inf_ecs::EcsWorld;
use inf_mesh::{Aabb, ChunkSection, FractureAsset, FractureChunk, MeshVertex};
use inf_physics::d3::{resolve_fracture_states, CRACK_OPENING_M};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use uuid::Uuid;

/// Four towers on four actors, so the handlers really can interleave: one actor
/// could not observe an ordering dependency even if there were one.
const TOWERS: u128 = 4;
const TOWER_BASE: u128 = 0x2204_0B00;
const MESH: u128 = 0x2204_0BEE;
const TERRAIN: u128 = 0x2204_0BFF;

/// Chunks per tower. Six, not four: a tower whose every chunk has exactly one or
/// two neighbours has almost no graph in it, and the ordering question this probe
/// asks is about the graph.
const TOWER_CHUNKS: u32 = 6;

const TILE_RES: u32 = 33;
const MPS: f64 = 1.0;
const GROUND_Y: f64 = 0.0;

/// A 128-bit fold over a stream of `u64`s. Integer mixing only, so the hash is
/// bit-portable and reflects the trace rather than the machine.
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
}

/// One axis-aligned unit box chunk, as render triangles and an `f64` hull set —
/// the `simulate_destruct` / `runtime_destruct` fixture shape, so a chunk here
/// means what a chunk means everywhere else in the phase's tests.
fn box_chunk(centre: DVec3, neighbors: Vec<u32>) -> FractureChunk {
    let h = 0.5;
    let mut hull = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                hull.push([centre.x + sx * h, centre.y + sy * h, centre.z + sz * h]);
            }
        }
    }
    let vertices: Vec<MeshVertex> = hull
        .iter()
        .map(|p| MeshVertex {
            position: [p[0] as f32, p[1] as f32, p[2] as f32],
            ..Default::default()
        })
        .collect();
    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        0, 1, 3, 0, 3, 2,  4, 7, 5, 4, 6, 7,
        0, 4, 5, 0, 5, 1,  2, 3, 7, 2, 7, 6,
        0, 2, 6, 0, 6, 4,  1, 5, 7, 1, 7, 3,
    ];
    let index_count = indices.len() as u32;
    FractureChunk {
        vertices,
        indices,
        sections: vec![ChunkSection {
            material_slot: 0,
            first_index: 0,
            index_count,
        }],
        hull_points: hull,
        volume_m3: 1.0,
        center_of_mass: centre.to_array(),
        neighbors,
    }
}

/// A stack of [`TOWER_CHUNKS`] unit cubes, each shared face exactly 1 m².
fn tower() -> Arc<FractureAsset> {
    let n = TOWER_CHUNKS;
    let chunks: Vec<FractureChunk> = (0..n)
        .map(|i| {
            let mut nb = Vec::new();
            if i > 0 {
                nb.push(i - 1);
            }
            if i + 1 < n {
                nb.push(i + 1);
            }
            box_chunk(DVec3::new(0.0, i as f64 + 0.5, 0.0), nb)
        })
        .collect();
    Arc::new(FractureAsset {
        schema_version: FractureAsset::CURRENT_VERSION,
        source_mesh: *Uuid::from_u128(MESH).as_bytes(),
        bounds: Aabb {
            min: [-0.5, 0.0, -0.5],
            max: [0.5, n as f32, 0.5],
        },
        seed: 0,
        requested_chunks: n,
        slots: vec!["Concrete".into(), "Fracture Interior".into()],
        interior_slot: 1,
        chunks,
    })
}

/// The charge, in joules: three and a half of the tower's 1 m² concrete bonds —
/// enough to take chunks off every tick without finishing the tower in one blow,
/// so the trace has a *sequence* in it rather than a single edge.
fn blow_energy_j() -> f64 {
    Destructible::default().strength * 1.0 * CRACK_OPENING_M * 3.5
}

/// The demolition Blueprint: blow up my own actor, then report what I see.
fn demolition_class() -> BlueprintClass {
    let entity = || Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str("entity".into()))],
    };
    let get = |name: &str| Expr::Call {
        path: vec!["vars".into(), "get".into()],
        args: vec![Expr::Lit(Lit::Str(name.into()))],
    };
    let set = |name: &str, value: Expr| {
        Stmt::ExprStmt(Expr::Call {
            path: vec!["vars".into(), "set".into()],
            args: vec![Expr::Lit(Lit::Str(name.into())), value],
        })
    };
    let slot = |name: &str| Variable {
        name: name.into(),
        ty: Ty::Float,
        default: Lit::Float(0.0),
        exposed: false,
    };
    let mut class = BlueprintClass::new("act:destruct-probe", "Probe Charge");
    class.variables = vec![slot("absorbed"), slot("total"), slot("intact")];
    class.events = vec![EventBinding {
        event: EventKind::Tick,
        body: BlueprintFn {
            id: "tick".into(),
            name: "tick".into(),
            params: vec![Param {
                name: "dt".into(),
                ty: Ty::Float,
            }],
            ret: Ty::Unit,
            body: vec![
                set(
                    "absorbed",
                    Expr::Call {
                        path: vec!["destruct".into(), "apply_damage".into()],
                        args: vec![entity(), Expr::Lit(Lit::Float(blow_energy_j()))],
                    },
                ),
                set(
                    "total",
                    Expr::Binary(
                        BinOp::Add,
                        Box::new(get("total")),
                        Box::new(get("absorbed")),
                    ),
                ),
                set(
                    "intact",
                    Expr::Call {
                        path: vec!["destruct".into(), "is_intact".into()],
                        args: vec![entity()],
                    },
                ),
            ],
        },
    }];
    class
}

fn build() -> RuntimeSim {
    let mut world = EcsWorld::new();

    let t = world.spawn_with_guid(Uuid::from_u128(TERRAIN), "Ground", None);
    world.world_mut().entity_mut(t).insert(Transform::IDENTITY);
    let mut data = inf_terrain::TerrainData::new(TILE_RES, MPS);
    for c in [(-1, -1), (-1, 0), (0, -1), (0, 0)] {
        data.author_tile(c, |_, _| GROUND_Y);
    }
    world.world_mut().entity_mut(t).insert(Terrain {
        meters_per_sample: MPS,
        tile_resolution: TILE_RES,
        data,
        ..Terrain::default()
    });

    let mut actors = Vec::new();
    for i in 0..TOWERS {
        let guid = Uuid::from_u128(TOWER_BASE + i);
        let e = world.spawn_with_guid(guid, "Tower", None);
        // Four towers, four metres apart, so their debris cannot land on each
        // other and turn "who ran first" into "whose rubble was in the way".
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(DVec3::new(
                i as f64 * 4.0,
                0.0,
                0.0,
            )));
        world.world_mut().entity_mut(e).insert(Destructible {
            chunk_count: TOWER_CHUNKS,
            ..Destructible::default()
        });
        world.world_mut().entity_mut(e).insert(MeshRef {
            asset: Some(Uuid::from_u128(MESH)),
            ..Default::default()
        });
        actors.push((guid, demolition_class()));
    }
    world.mark_dirty();
    world.propagate();

    let mut sim = RuntimeSim::new(world, actors, DVec2::new(0.0, -9.81), 60.0);
    let asset = tower();
    // Through the ONE Ring-0 resolver both hosts use, so the probe seeds what the
    // product seeds.
    let states = resolve_fracture_states(sim.world(), |mesh| {
        (mesh == Uuid::from_u128(MESH)).then(|| asset.clone())
    });
    assert_eq!(
        states.len(),
        TOWERS as usize,
        "the probe seeded no fractures"
    );
    sim.set_fractures(states);
    sim
}

fn var(sim: &RuntimeSim, guid: Uuid, name: &str) -> f64 {
    match sim.actor_var(guid, name) {
        Some(Value::Float(f)) => *f,
        Some(Value::Bool(b)) => *b as u8 as f64,
        Some(Value::Int(i)) => *i as f64,
        _ => f64::NAN,
    }
}

/// Fold one step's state into `h`, and return that frame's own hash.
///
/// Both halves: what each Blueprint *recorded* (the joules it spent) and what the
/// world *became* (the audit's collapse and budget counters). A trace of the
/// first alone could agree while the collapses underneath diverged.
fn frame(sim: &RuntimeSim, h: &mut Hasher) -> u128 {
    let mut per_frame = Hasher::default();
    for i in 0..TOWERS {
        let guid = Uuid::from_u128(TOWER_BASE + i);
        for name in ["absorbed", "total", "intact"] {
            let bits = var(sim, guid, name).to_bits();
            per_frame.eat(bits);
            h.eat(bits);
        }
    }
    let audit = sim.fracture_audit();
    for v in [
        audit.tracked as u64,
        audit.collapsed as u64,
        audit.expired as u64,
        audit.over_budget as u64,
        audit.live_debris as u64,
    ] {
        per_frame.eat(v);
        h.eat(v);
    }
    per_frame.0
}

/// A hash of the **final world**: every actor's whole fracture state, poses
/// included. The frame trace is what the Blueprints saw; this is what the world
/// actually became, and they are not the same claim.
fn field_hash(sim: &RuntimeSim) -> u128 {
    let mut h = Hasher::default();
    for (guid, state) in sim.fractures() {
        h.eat(guid.as_u128() as u64);
        h.eat((guid.as_u128() >> 64) as u64);
        for w in state.bits() {
            h.eat(w);
        }
    }
    h.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(180);

    // Pin the pool BEFORE anything touches bevy_tasks. First init wins, and this
    // process only ever does one.
    let actual = inf_ecs::init_ecs_task_pool(threads);

    let mut sim = build();
    let mut trace = Hasher::default();
    let mut first = 0u128;
    let mut last = 0u128;
    for i in 0..steps {
        sim.step_once(RuntimeInput::default());
        let f = frame(&sim, &mut trace);
        if i == 0 {
            first = f;
        }
        last = f;
    }

    println!("threads={actual}");
    println!("steps={steps}");
    println!("first={first:032x}");
    println!("last={last:032x}");
    println!("trace={:032x}", trace.0);
    println!("field={:032x}", field_hash(&sim));
}
