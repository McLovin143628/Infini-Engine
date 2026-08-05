//! Subprocess probe for the **pool-size-invariance** leg of the P21.4 voxel gate.
//!
//! `bevy_ecs`'s `ComputeTaskPool` is a process-global `OnceLock`
//! ([`inf_ecs::init_ecs_task_pool`] says so on the tin): the **first** call wins
//! and every later one is a no-op that reports the already-configured count. A
//! single process can therefore only ever run under one pool size, and an
//! in-process test that "varies" it by calling `init_ecs_task_pool(1)` then
//! `init_ecs_task_pool(4)` is running the same trace twice on the same pool — a
//! duplicate of the two-run determinism arm wearing a stronger name. That is the
//! trap `crates/inf-runtime/src/bin/replay_probe.rs` exists to avoid, and
//! `water_probe.rs` is the water twin; this is the voxel one.
//!
//! **The premise this file shipped with was wrong, and P22.4's audit corrected
//! it.** It said "the blueprint tick runs inside the ECS schedule". It does not:
//! `RuntimeSim::run_all_with_args` is a serial `for` loop over a `Vec` of
//! `BTreeMap` keys and no `SimSchedule` runs on the player's fixed-step path at
//! all, so four pool sizes execute the same serial program and this probe cannot
//! *discover* an ordering race — there is no concurrency here for one to live in.
//!
//! What the runs are actually worth is stated in full on `destruct_probe.rs`,
//! P22.4's twin of this file, and applies verbatim: a **regression tripwire**
//! against the day the tick pass moves onto `inf-ecs`'s parallel `SimSchedule`
//! (which `runtime_sim`'s own docs name as a follow-up), plus a **pin** that
//! nothing on the fixed-step path has reached for the process-global
//! `ComputeTaskPool` — checked rather than assumed, which is the half that can
//! change without anybody noticing.
//!
//! The probe builds its scene from a **bare `EcsWorld`** — no editor crate,
//! because a `[[bin]]` cannot reach dev-dependencies — pins the pool to a
//! requested size, runs the trace, and prints hashes. The launching test spawns it
//! once per pool size and compares the outputs *to each other*, so the scene lives
//! in exactly one place and cannot drift between the probe and its caller.
//!
//! Usage: `voxel_probe [threads] [steps]`  (defaults: threads=0→auto, steps=160)
//! Output (stdout, one `key=value` per line):
//!   threads=<actual pool worker count>
//!   steps=<n>
//!   first=<hex 128-bit hash of the first frame>
//!   last=<hex 128-bit hash of the last frame>
//!   trace=<hex 128-bit hash of every frame>
//!   field=<hex 128-bit hash of the FINAL voxel field + hole mask>

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use inf_blueprint::{
    BinOp, BlueprintClass, BlueprintFn, EventBinding, EventKind, Expr, Lit, Param, Stmt, Ty, Value,
    Variable,
};
use inf_ecs::components::{Terrain, Transform, VoxelVolume};
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use inf_voxel::{ChunkKey, VoxelChunk, VoxelData};
use uuid::Uuid;

/// Four borers on four volumes, so the handlers really can interleave: one actor
/// could not observe an ordering dependency even if there were one.
const BORERS: u128 = 4;
const BORER_BASE: u128 = 0x2104_0B00;
const TERRAIN: u128 = 0x2104_0BFF;

/// 33 samples at 1 m ⇒ one tile spanning `[0, 32]²`, flat at 24 m — the same
/// plane the rock's top surface sits on, so every carve that crosses it opens a
/// hole and the hole mask is part of what the probe hashes.
const TILE_RES: u32 = 33;
const MPS: f64 = 1.0;
const GROUND_Y: f64 = 24.0;

/// A 128-bit fold over a stream of `f64` bits. Integer mixing only, so the hash
/// is bit-portable and reflects the trace rather than the machine.
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

/// One borer's Blueprint: advance along `+X` and carve, exactly the shape the
/// committed sample's borer has (and for the same reason — a sub-voxel step makes
/// the per-tick volume vary).
fn borer_class(x0: f64, z: f64) -> BlueprintClass {
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
    let bin = |op: BinOp, a: Expr, b: Expr| Expr::Binary(op, Box::new(a), Box::new(b));
    let slot = |name: &str| Variable {
        name: name.into(),
        ty: Ty::Float,
        default: Lit::Float(0.0),
        exposed: false,
    };

    let mut class = BlueprintClass::new("act:voxel-probe-borer", "Probe Borer");
    class.variables = vec![slot("step"), slot("removed"), slot("total"), slot("ground")];
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
                    "removed",
                    Expr::Call {
                        path: vec!["voxel".into(), "carve_sphere".into()],
                        args: vec![
                            Expr::Call {
                                path: vec!["vars".into(), "get".into()],
                                args: vec![Expr::Lit(Lit::Str("entity".into()))],
                            },
                            bin(
                                BinOp::Add,
                                Expr::Lit(Lit::Float(x0)),
                                bin(BinOp::Mul, get("step"), Expr::Lit(Lit::Float(0.25))),
                            ),
                            // Centred on the surface, so the cut opens the ground
                            // too and the hole mask joins the trace.
                            Expr::Lit(Lit::Float(GROUND_Y)),
                            Expr::Lit(Lit::Float(z)),
                            Expr::Lit(Lit::Float(2.0)),
                        ],
                    },
                ),
                set("total", bin(BinOp::Add, get("total"), get("removed"))),
                set(
                    "step",
                    bin(BinOp::Add, get("step"), Expr::Lit(Lit::Float(1.0))),
                ),
                set(
                    "ground",
                    Expr::Call {
                        path: vec!["terrain".into(), "height_at".into()],
                        args: vec![Expr::Lit(Lit::Float(x0)), Expr::Lit(Lit::Float(z))],
                    },
                ),
            ],
        },
    }];
    class
}

/// One volume of rock whose top surface is [`GROUND_Y`], covering the chunk the
/// borer at `z` drives through.
fn rock() -> VoxelData {
    let mut v = VoxelData::new(1.0);
    for cx in 0..2 {
        for cz in 0..2 {
            let key = ChunkKey::new(cx, 1, cz);
            let base = key.base_sample();
            v.insert_chunk(
                key,
                VoxelChunk::from_fn(|_, j, _| (base[1] + j as i32) as f64 - GROUND_Y),
            );
        }
    }
    v.clear_dirty();
    v
}

fn build() -> RuntimeSim {
    let mut world = EcsWorld::new();

    let t = world.spawn_with_guid(Uuid::from_u128(TERRAIN), "Ground", None);
    world.world_mut().entity_mut(t).insert(Transform::IDENTITY);
    let mut data = inf_terrain::TerrainData::new(TILE_RES, MPS);
    data.author_tile((0, 0), |_, _| GROUND_Y);
    world.world_mut().entity_mut(t).insert(Terrain {
        meters_per_sample: MPS,
        tile_resolution: TILE_RES,
        data,
        ..Terrain::default()
    });

    let mut actors = Vec::new();
    let mut volumes = BTreeMap::new();
    for i in 0..BORERS {
        let guid = Uuid::from_u128(BORER_BASE + i);
        let e = world.spawn_with_guid(guid, "Borer", None);
        world.world_mut().entity_mut(e).insert(Transform::IDENTITY);
        world.world_mut().entity_mut(e).insert(VoxelVolume {
            voxel_size_m: 1.0,
            runtime_carve: true,
            ..VoxelVolume::default()
        });
        // Four parallel drifts, four metres apart.
        actors.push((guid, borer_class(4.0, 6.0 + i as f64 * 4.0)));
        volumes.insert(guid, rock());
    }
    world.mark_dirty();
    world.propagate();

    let mut sim = RuntimeSim::new(world, actors, DVec2::new(0.0, -9.81), 60.0);
    sim.set_voxel_volumes(volumes);
    sim
}

fn var(sim: &RuntimeSim, guid: Uuid, name: &str) -> f64 {
    match sim.actor_var(guid, name) {
        Some(Value::Float(f)) => *f,
        _ => f64::NAN,
    }
}

/// Fold one step's borer state into `h`, and return that frame's own hash.
fn frame(sim: &RuntimeSim, h: &mut Hasher) -> u128 {
    let mut per_frame = Hasher::default();
    for i in 0..BORERS {
        let guid = Uuid::from_u128(BORER_BASE + i);
        for name in ["removed", "total", "ground"] {
            let bits = var(sim, guid, name).to_bits();
            per_frame.eat(bits);
            h.eat(bits);
        }
    }
    per_frame.0
}

/// A hash of the **final world**: every volume's signed distances and materials,
/// plus the terrain's hole mask. The frame trace is what a Blueprint saw; this is
/// what the world actually became, and they are not the same claim.
fn field_hash(sim: &RuntimeSim) -> u128 {
    let mut h = Hasher::default();
    for (guid, data) in sim.voxel_volumes() {
        h.eat(guid.as_u128() as u64);
        h.eat((guid.as_u128() >> 64) as u64);
        for (key, chunk) in data.chunks() {
            h.eat(key.x as u32 as u64);
            h.eat(key.y as u32 as u64);
            h.eat(key.z as u32 as u64);
            for s in chunk.sdf() {
                h.eat(s.to_bits() as u64);
            }
            for m in chunk.materials() {
                h.eat(*m as u64);
            }
        }
    }
    let world = sim.world();
    if let Some(e) = world.entity_of(Uuid::from_u128(TERRAIN)) {
        if let Some(t) = world.world().get::<Terrain>(e) {
            for i in 0..TILE_RES {
                for j in 0..TILE_RES {
                    let holed = t
                        .data
                        .is_hole_at(glam::DVec2::new(i as f64 * MPS, j as f64 * MPS));
                    h.eat(holed as u64);
                }
            }
        }
    }
    let _ = DVec3::ZERO;
    h.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(160);

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
