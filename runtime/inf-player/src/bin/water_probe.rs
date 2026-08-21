//! Subprocess probe for the **pool-size-invariance** leg of the P20.2 water gate.
//!
//! `bevy_ecs`'s `ComputeTaskPool` is a process-global `OnceLock`
//! ([`inf_ecs::init_ecs_task_pool`] says so on the tin): the **first** call wins
//! and every later one is a no-op that reports the already-configured count. A
//! single process can therefore only ever run under one pool size, and an
//! in-process test that "varies" it by calling `init_ecs_task_pool(1)` then
//! `init_ecs_task_pool(4)` is running the same trace twice on the same pool — a
//! duplicate of the two-run determinism arm wearing a stronger name. That is
//! exactly the trap `crates/inf-runtime/src/bin/replay_probe.rs` exists to avoid,
//! and this is its water twin.
//!
//! The probe builds the floating-stack scene from a **bare `EcsWorld`** — no
//! editor crate, because a `[[bin]]` cannot reach dev-dependencies — pins the pool
//! to a requested size, runs the trace, and prints hashes. The launching test
//! spawns it once per pool size and compares the outputs *to each other*, so the
//! scene lives in exactly one place and cannot drift between the probe and its
//! caller.
//!
//! Usage: `water_probe [threads] [steps]`  (defaults: threads=0→auto, steps=240)
//! Output (stdout, one `key=value` per line):
//!
//! ```text
//! threads=<actual pool worker count>
//! steps=<n>
//! first=<hex 128-bit hash of the first frame>
//! last=<hex 128-bit hash of the last frame>
//! trace=<hex 128-bit hash of every frame>
//! ```

use glam::DVec2;
use inf_ecs::components::{
    BodyKind3D, Buoyancy, Collider3D, ColliderShape3DKind, RigidBody3D, SkyAtmosphere, TimeOfDay,
    Transform, WaterBody, WaterKind,
};
use inf_ecs::math::Vec3d;
use inf_ecs::EcsWorld;
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};
use uuid::Uuid;

/// Eight crates of ascending density: `0x2002_3010 ..= 0x2002_3017`.
const CRATE_BASE: u128 = 0x2002_3010;
const CRATES: u128 = 8;
const OCEAN: u128 = 0x2002_3001;
const SKY: u128 = 0x2002_3000;

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

fn build() -> RuntimeSim {
    let mut world = EcsWorld::new();

    // A clock that really runs, so a wave phase that ignored it would be obvious.
    let sky = world.spawn_with_guid(Uuid::from_u128(SKY), "Sky", None);
    world.world_mut().entity_mut(sky).insert((
        TimeOfDay {
            seconds: 0.0,
            rate: 240.0,
            ..TimeOfDay::default()
        },
        SkyAtmosphere {
            weather_enabled: true,
            ..SkyAtmosphere::default()
        },
        Transform::IDENTITY,
    ));

    let ocean = world.spawn_with_guid(Uuid::from_u128(OCEAN), "Ocean", None);
    world.world_mut().entity_mut(ocean).insert((
        WaterBody {
            kind: WaterKind::Ocean,
            level_m: 0.0,
            wave_amplitude_m: 0.7,
            wave_length_m: 22.0,
            wave_steepness: 0.45,
            wave_count: 5,
            wave_seed: 0x5EA_5A17,
            wind_from_weather: false,
            wind_x: 7.5,
            wind_z: -2.5,
            ..WaterBody::default()
        },
        Transform::IDENTITY,
    ));

    for i in 0..CRATES {
        let density = 350.0 + i as f64 * 50.0;
        let e = world.spawn_with_guid(Uuid::from_u128(CRATE_BASE + i), "Crate", None);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Dynamic,
                ..RigidBody3D::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::splat(0.5),
                density,
                ..Collider3D::default()
            },
            Buoyancy::of_density(density),
            Transform {
                translation: Vec3d::new(
                    (i as f64 - 3.5) * 1.4,
                    1.5 + i as f64 * 0.45,
                    (i as f64 % 3.0) * 0.9,
                ),
                ..Transform::IDENTITY
            },
        ));
    }
    world.mark_dirty();

    // The 3D solver's gravity flows from `gravity_2d.y` (see `RuntimeSim::new`).
    RuntimeSim::new(world, vec![], DVec2::new(0.0, -9.81), 60.0)
}

/// Fold one step's poses into `h`, and return that frame's own hash.
fn frame(sim: &RuntimeSim, h: &mut Hasher) -> u128 {
    let mut per_frame = Hasher::default();
    let world = sim.world();
    for i in 0..CRATES {
        let Some(e) = world.entity_of(Uuid::from_u128(CRATE_BASE + i)) else {
            continue;
        };
        let Some(t) = world.world().get::<Transform>(e) else {
            continue;
        };
        let q = t.quat();
        for bits in [
            t.translation.x.to_bits(),
            t.translation.y.to_bits(),
            t.translation.z.to_bits(),
            q.x.to_bits(),
            q.y.to_bits(),
            q.z.to_bits(),
            q.w.to_bits(),
        ] {
            per_frame.eat(bits);
            h.eat(bits);
        }
    }
    per_frame.0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(240);

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
}
