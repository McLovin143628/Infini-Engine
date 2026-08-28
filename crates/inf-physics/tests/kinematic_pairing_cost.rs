//! **Where a CROWD's milliseconds go in the fixed step** (island wave NPC1c) —
//! the measurement behind [`inf_physics::d3::ColliderPairing`].
//!
//! NPC1b put a thousand NPCs on the island and clocked the phases. The `solver`
//! row was the one nobody could explain:
//!
//! > `solver` 2.233 → **4.844 ms** (**+2.61**). 288 *moving kinematic* capsules
//! > among 17 823 bodies — **1.6 % more bodies for 117 % more solver.**
//!
//! `solver` is `bridge.step(dt); bridge3d.step(dt)` — the whole rapier pipeline,
//! broad phase and narrow phase included — so "the solver" is a phase name, not a
//! diagnosis. This file is the diagnosis, on a controlled fixture, and it is a
//! near-repeat of island wave I4b's finding one pairing over:
//!
//! * I4b: `FIXED_FIXED` meant a static box on a streamed **heightfield** had a
//!   manifold recomputed at 60 Hz for a pair no solver could act on. **+3.704
//!   ms/step** for 3 446 pairs. Removed.
//! * NPC1c: `KINEMATIC_FIXED` is the same geometry with the "moving" part
//!   switched on. A crowd agent is a kinematic capsule standing on that
//!   heightfield, and it **moves every step**, so its manifolds can never be
//!   re-used — they are recomputed from scratch, every one of them, for ever.
//!
//! # The two questions this file answers with numbers
//!
//! 1. **Is it the pairs, or is it the movement?** The same capsules standing
//!    still make the same pairs. If still and moving cost the same, the count is
//!    the story; if moving costs multiples of still, recomputation is.
//! 2. **Does narrowing the capsule fix it?** rapier tests `ActiveCollisionTypes`
//!    as a **union** (`!a.test() && !b.test()` is what skips a pair), so the
//!    obvious prescription — mark the crowd capsule `DynamicOnly` — may be a
//!    complete no-op. Measured before prescribed, per this tree's law.
//!
//! # What is asserted and what is printed
//!
//! **Manifold counts are asserted**: a manifold is a manifold on every machine,
//! and every claim this file makes about the *door* is a claim about manifolds.
//! `contact_pair_counts` reports two numbers and they answer different
//! questions — `tracked` is what the **broad** phase handed over, `touching` is
//! what the **narrow** phase actually computed — and `ActiveCollisionTypes` gates
//! only the second, so the assertions are on `touching` with `tracked` held equal
//! as the control. The **clocks are printed**, per the standing rule
//! (`city_collider_band.rs`'s precedent): a wall-clock assertion on a shared
//! runner is a flake generator.
//!
//! Every ms/step figure here is the MIN of five rounds of sixty steps, on this
//! crate's **dev** profile (`[profile.dev.package."*"] opt-level = 2`, so rapier
//! itself is optimised), Windows, RTX 4070 Ti dev box. The timer brackets
//! `world.step` **alone** — the per-step re-placement of the kinematic bodies
//! happens outside it, because in the shipped host that write is the
//! `physics3d sync` phase and this file is about `solver`.

use glam::{DQuat, DVec2, DVec3};
use inf_physics::d3::{
    BodyKind3D, ColliderDesc3D, ColliderPairing, ColliderShape3D, PhysicsWorld3D,
};
use std::time::Instant;

const DT: f64 = 1.0 / 60.0;

// ── the fixture ─────────────────────────────────────────────────────────────

/// The island's own tile: `tile_resolution = 257` at `meters_per_sample = 1.0`
/// (`samples/island/island.toml`), so 256 m of ground and 131 072 triangles per
/// tile. [`ground_side`] lays down as many as the block field needs — 2 × 2 in
/// the default fixture.
const TILE_RES: u32 = 257;
const TILE_SPAN: f64 = 256.0;

/// Building pitch, metres. 6 m of box and 1 m of gap is a dense downtown block
/// face — the shape the phase-30 city bands to.
const BUILDING_PITCH: f64 = 7.0;
/// Buildings on a side, in the default fixture: 43 × 43 = **1 849** static boxes
/// over 301 m. [`the_cost_of_one_agent_grows_with_the_world_around_it`] varies it.
const BUILDINGS: i32 = 43;
const BUILDING_HALF: DVec3 = DVec3::new(3.0, 4.0, 3.0);

/// Crowd agent proportions — `CrowdArchetype::humanoid`'s 1.8 m adult.
const AGENT_HALF_HEIGHT: f64 = 0.9;
const AGENT_RADIUS: f64 = 0.3;
/// Standing on flat ground: `0.9 + 0.3` puts the capsule's foot at y = 0.
const AGENT_Y: f64 = AGENT_HALF_HEIGHT + AGENT_RADIUS;
/// A walk, m/s — ALS's own walk speed band.
const AGENT_SPEED: f64 = 1.4;

/// The crowd size NPC1b measured: 1 000 agents, **288** of them posed and
/// materialised in the near tiers.
const CROWD: usize = 288;

fn building_centre(i: i32, side: i32) -> DVec3 {
    let (bx, bz) = (i % side, i / side);
    DVec3::new(
        (f64::from(bx) - f64::from(side - 1) * 0.5) * BUILDING_PITCH,
        BUILDING_HALF.y,
        (f64::from(bz) - f64::from(side - 1) * 0.5) * BUILDING_PITCH,
    )
}

/// The crowd lattice: 17 × 17 = 289 sites, of which the first [`CROWD`] are used.
const AGENT_SIDE: i32 = 17;

/// Agent `i`'s home: **touching a building's west face**, which is where a crowd
/// on a sidewalk actually walks. Placing them mid-street instead would leave
/// every capsule paired with the ground and nothing else, and would measure a
/// world with no buildings in it while claiming to have 1 849.
///
/// The lattice is **centred** in the block field whatever `side` is, so growing
/// the world grows what is *around* the crowd and never what the crowd stands on
/// — which is the one thing
/// [`the_cost_of_one_agent_grows_with_the_world_around_it`] varies.
fn agent_home(i: usize, side: i32) -> DVec3 {
    let (ax, az) = (i as i32 % AGENT_SIDE, i as i32 / AGENT_SIDE);
    let base = (side - AGENT_SIDE * 2) / 2;
    let b = building_centre((base + az * 2) * side + (base + ax * 2), side);
    DVec3::new(b.x + BUILDING_HALF.x + AGENT_RADIUS, AGENT_Y, b.z)
}

/// Heightfield tiles per side needed to cover a `side`-block field: 2 at the
/// default 43 (±150 m under ±256 m of ground), 4 at 130.
fn ground_side(side: i32) -> i32 {
    let half = f64::from(side - 1) * 0.5 * BUILDING_PITCH + BUILDING_HALF.x;
    ((2.0 * half / TILE_SPAN).ceil() as i32).max(2)
}

/// What world to build. Every field is something an arm varies.
#[derive(Clone, Copy)]
struct Spec {
    /// Blocks on a side; `side * side` static boxes.
    side: i32,
    /// Whether the heightfield tiles are present at all.
    ground: bool,
    /// The pairing the **ground and the buildings** are described with.
    scenery: ColliderPairing,
    /// How many kinematic capsules, and how they are described.
    agents: usize,
    agent_pairing: ColliderPairing,
    /// Whether the capsules are re-placed before every step.
    moving: bool,
}

impl Spec {
    /// The default world: 1 849 boxes on 2 × 2 tiles of ground, no crowd,
    /// nothing narrowed, nothing moving.
    fn new() -> Self {
        Self {
            side: BUILDINGS,
            ground: true,
            scenery: ColliderPairing::All,
            agents: 0,
            agent_pairing: ColliderPairing::All,
            moving: false,
        }
    }
    fn crowd(mut self, n: usize) -> Self {
        self.agents = n;
        self
    }
    fn moving(mut self, moving: bool) -> Self {
        self.moving = moving;
        self
    }
    fn ground(mut self, ground: bool) -> Self {
        self.ground = ground;
        self
    }
    fn side(mut self, side: i32) -> Self {
        self.side = side;
        self
    }
    /// Narrow the capsules only — the prescription the union rule kills.
    fn narrow_agents(mut self) -> Self {
        self.agent_pairing = ColliderPairing::DynamicOnly;
        self
    }
    /// Narrow both sides — the one that works.
    fn narrow_both(mut self) -> Self {
        self.scenery = ColliderPairing::DynamicOnly;
        self.agent_pairing = ColliderPairing::DynamicOnly;
        self
    }
}

struct Fixture {
    world: PhysicsWorld3D,
    spec: Spec,
    agents: Vec<inf_physics::d3::BodyId3D>,
    bodies: usize,
}

fn fixture(spec: Spec) -> Fixture {
    let mut world = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));

    if spec.ground {
        // Flat on purpose: a flat tile makes "touching" a fact rather than a
        // coincidence of where a slope happened to be, and parry indexes a
        // height-field cell arithmetically, so the cell walk under a capsule is
        // the same work whatever the heights hold.
        let heights = vec![0.0f32; (TILE_RES * TILE_RES) as usize];
        let n = ground_side(spec.side);
        for tz in 0..n {
            for tx in 0..n {
                let at = |k: i32| (f64::from(k) - f64::from(n - 1) * 0.5) * TILE_SPAN;
                let b = world.add_body(
                    BodyKind3D::Static,
                    DVec3::new(at(tx), 0.0, at(tz)),
                    DQuat::IDENTITY,
                );
                world
                    .try_add_collider(
                        b,
                        ColliderDesc3D::new(ColliderShape3D::Heightfield {
                            samples_x: TILE_RES,
                            samples_z: TILE_RES,
                            heights: heights.clone(),
                            removed_cells: Vec::new(),
                            span: DVec2::splat(TILE_SPAN),
                        })
                        .pairing(spec.scenery),
                    )
                    .expect("a terrain tile attaches");
            }
        }
    }

    for i in 0..(spec.side * spec.side) {
        let b = world.add_body(
            BodyKind3D::Static,
            building_centre(i, spec.side),
            DQuat::IDENTITY,
        );
        world
            .try_add_collider(
                b,
                ColliderDesc3D::new(ColliderShape3D::Box {
                    half_extents: BUILDING_HALF,
                })
                .pairing(spec.scenery),
            )
            .expect("a building attaches");
    }

    let mut ids = Vec::with_capacity(spec.agents);
    for i in 0..spec.agents {
        let b = world.add_body(
            BodyKind3D::Kinematic,
            agent_home(i, spec.side),
            DQuat::IDENTITY,
        );
        world
            .try_add_collider(
                b,
                ColliderDesc3D::new(ColliderShape3D::Capsule {
                    half_height: AGENT_HALF_HEIGHT,
                    radius: AGENT_RADIUS,
                })
                .pairing(spec.agent_pairing),
            )
            .expect("an agent attaches");
        ids.push(b);
    }

    // Read the WORLD rather than restating the recipe: if a collider ever
    // refuses, the printed body count says so instead of agreeing with the spec.
    let bodies = world.body_ids().len();
    Fixture {
        world,
        spec,
        agents: ids,
        bodies,
    }
}

/// Walk every agent along the frontage it started on — the shape a crowd route
/// has, and the one that keeps re-pairing a capsule with successive buildings
/// instead of parking it on one manifold for ever.
fn walk(f: &mut Fixture, step: u32) {
    let d = AGENT_SPEED * DT * f64::from(step);
    let side = f.spec.side;
    for (i, &b) in f.agents.iter().enumerate() {
        let home = agent_home(i, side);
        // A 12 m beat, so an agent stays on its own frontage and the fixture
        // does not slowly turn into a different world as the arm runs.
        let t = (d + i as f64 * 0.37) % 12.0 - 6.0;
        f.world
            .set_body_translation(b, DVec3::new(home.x, home.y, home.z + t));
    }
}

struct Measured {
    tracked: usize,
    touching: usize,
    bodies: usize,
    ms: f64,
}

/// MIN of `ROUNDS` rounds of `STEPS` steps, timing `step` alone.
fn measure(f: &mut Fixture) -> Measured {
    const WARM: u32 = 30;
    const ROUNDS: u32 = 5;
    const STEPS: u32 = 60;
    let moving = f.spec.moving;

    let mut clock = 0u32;
    for _ in 0..WARM {
        if moving {
            clock += 1;
            walk(f, clock);
        }
        f.world.step(DT);
    }
    let (tracked, touching) = f.world.contact_pair_counts();

    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let mut total = 0.0f64;
        for _ in 0..STEPS {
            if moving {
                clock += 1;
                walk(f, clock);
            }
            let t = Instant::now();
            f.world.step(DT);
            total += t.elapsed().as_secs_f64();
        }
        best = best.min(total * 1000.0 / f64::from(STEPS));
    }
    // ANTI-VACUITY: a timer reading zero measured its own resolution.
    assert!(best > 0.0, "the fixed step took no measurable time");

    Measured {
        tracked,
        touching,
        bodies: f.bodies,
        ms: best,
    }
}

fn run(spec: Spec) -> Measured {
    let mut f = fixture(spec);
    measure(&mut f)
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **THE HEADLINE.** What 288 moving kinematic capsules cost, against the same
/// capsules standing still, against no capsules at all — and what each of the two
/// narrowings buys.
///
/// The N sweep is here so the growth's *shape* is visible: if the cost is the
/// capsules' own manifolds it is linear in N, and if it is something about the
/// world having any moving thing in it at all it is a step at N = 1.
#[test]
fn a_moving_kinematic_crowd_costs_multiples_of_the_same_crowd_standing_still() {
    let control = run(Spec::new());
    println!(
        "\n{} static boxes + {} heightfield tiles of {}^2 samples, {} bodies, no capsules: \
         {} pairs ({} manifolds), {:.3} ms/step\n",
        BUILDINGS * BUILDINGS,
        ground_side(BUILDINGS) * ground_side(BUILDINGS),
        TILE_RES,
        control.bodies,
        control.tracked,
        control.touching,
        control.ms
    );
    println!(
        "{:>5} | {:<30} | pairs | manifolds |  ms/step | vs control |    us/agent",
        "N", "arm"
    );
    for n in [32usize, 128, CROWD] {
        let base = Spec::new().crowd(n);
        for (label, spec) in [
            ("kinematic, STILL", base.moving(false)),
            ("kinematic, MOVING", base.moving(true)),
            (
                "MOVING, capsules narrowed",
                base.moving(true).narrow_agents(),
            ),
            (
                "MOVING, BOTH sides narrowed",
                base.moving(true).narrow_both(),
            ),
        ] {
            let m = run(spec);
            println!(
                "{n:>5} | {label:<30} | {:>5} | {:>9} | {:>8.3} | {:>+10.3} | {:>10.2}",
                m.tracked,
                m.touching,
                m.ms,
                m.ms - control.ms,
                (m.ms - control.ms) * 1000.0 / n as f64,
            );
        }
    }
    println!();
}

/// **Is it the buildings or is it the GROUND?** The same crowd with and without
/// the four heightfield tiles under it.
///
/// I4b's answer for static boxes was "the ground is the whole story". This arm
/// asks the same question of a moving capsule, and it is the one that decides
/// whether NPC1c's door has to reach the terrain colliders or only the buildings.
#[test]
fn the_ground_is_where_a_moving_capsule_pays_for_its_pairing() {
    for ground in [false, true] {
        let base = Spec::new().ground(ground);
        let control = run(base);
        let still = run(base.crowd(CROWD).moving(false));
        let moving = run(base.crowd(CROWD).moving(true));
        let narrow = run(base.crowd(CROWD).moving(true).narrow_both());
        println!(
            "ground {:<7} | control {:>4} mf {:>7.3} ms | still {:>4} mf {:>7.3} ({:+.3}) \
             | moving {:>4} mf {:>7.3} ({:+.3}) | narrowed {:>4} mf {:>7.3} ({:+.3})   \
             [mf = manifolds, i.e. pairs the narrow phase computed]",
            if ground { "PRESENT" } else { "absent" },
            control.touching,
            control.ms,
            still.touching,
            still.ms,
            still.ms - control.ms,
            moving.touching,
            moving.ms,
            moving.ms - control.ms,
            narrow.touching,
            narrow.ms,
            narrow.ms - control.ms,
        );
    }
}

/// **THE MECHANISM, NAMED IN ONE LINE**, with its numbers — the sentence the
/// wave quotes.
#[test]
fn the_mechanism_states_itself() {
    let base = Spec::new().crowd(CROWD);
    let control = run(Spec::new());
    let still = run(base.moving(false));
    let moving = run(base.moving(true));
    let one = run(base.moving(true).narrow_agents());
    let both = run(base.moving(true).narrow_both());
    println!(
        "\nNPC1c: {CROWD} MOVING kinematic capsules over a heightfield + {} static boxes make \
         {} contact pairs a step of which {} are MANIFOLDS, and cost {:+.3} ms/step; the SAME \
         capsules standing still make {} manifolds and cost {:+.3}; narrowing the capsules \
         ALONE costs {:+.3} and removes {} manifolds (rapier's flag test is a union); narrowing \
         BOTH sides costs {:+.3} and removes {}.\n\
         THE MECHANISM IS RECOMPUTATION, NOT COUNT: the same {} manifolds cost {:.0}x more when \
         the capsule under them moves ({:+.3} still vs {:+.3} moving), because a moved pose \
         invalidates every one of them and a capsule-versus-heightfield manifold walks the \
         tile's cells underneath. Narrowing both sides recovers {:.3} of the {:.3} ms; the \
         {:.3} that remains is broad-phase and body bookkeeping for {CROWD} moving proxies, \
         which no pairing flag can reach.\n",
        BUILDINGS * BUILDINGS,
        moving.tracked,
        moving.touching,
        moving.ms - control.ms,
        still.touching,
        still.ms - control.ms,
        one.ms - control.ms,
        moving.touching as i64 - one.touching as i64,
        both.ms - control.ms,
        moving.touching as i64 - both.touching as i64,
        moving.touching - control.touching,
        (moving.ms - control.ms) / (still.ms - control.ms).max(1e-9),
        still.ms - control.ms,
        moving.ms - control.ms,
        moving.ms - both.ms,
        moving.ms - control.ms,
        both.ms - control.ms,
    );

    // The MANIFOLD claims, asserted — these are facts about rapier and this
    // world, not about this machine's clock. `tracked` is what the BROAD phase
    // handed over and `touching` is what the narrow phase computed;
    // `ActiveCollisionTypes` gates the second and never the first, which is why
    // every claim below is about `touching`. (Measured: the first spelling of
    // this arm asserted `tracked` and failed against a door that works.)
    assert!(
        still.touching > control.touching,
        "adding {CROWD} capsules added no manifolds at all ({} -> {}) — the crowd \
         is not standing on anything and this file measures nothing",
        control.touching,
        still.touching
    );
    assert_eq!(
        moving.touching, one.touching,
        "narrowing ONLY the capsules changed the manifold count ({} -> {}). \
         rapier's test is `!a.test() && !b.test()`, so the scenery's own \
         `all() - FIXED_FIXED` still carries every kinematic-vs-fixed pair; if that \
         is no longer true the union rule this wave's wiring instruction rests on \
         has changed and the instruction is wrong",
        moving.touching, one.touching
    );
    assert_eq!(
        both.touching, control.touching,
        "narrowing BOTH sides left {} manifolds against the no-capsule control's \
         {} — `DynamicOnly` on both halves must leave a kinematic-vs-fixed pair \
         with no flag to carry it",
        both.touching, control.touching
    );
    // …and the saving really is the NARROW phase: the broad phase still hands
    // over the same pairs. Without this the arm above could be satisfied by a
    // door that accidentally culled the proxies instead.
    assert_eq!(
        moving.tracked, both.tracked,
        "narrowing changed the BROAD-phase pair count ({} -> {}) — `ColliderPairing` \
         must gate the manifold and leave the proxy alone",
        moving.tracked, both.tracked
    );
}

/// **Why this fixture's per-agent number is smaller than the island's**, and
/// what part of the gap is world size.
///
/// NPC1b measured `+2.61 ms` for 288 agents on the island — **9.06 µs per agent
/// per step**. The default fixture here measures about a third of that, and the
/// obvious suspect is that the island is **17 823 bodies** to this fixture's
/// 2 141: a moving proxy has to be re-inserted into a broad-phase tree, and that
/// tree is eight times bigger there.
///
/// So this arm holds the crowd fixed at [`CROWD`] agents standing on the same
/// ground, and grows only *the world around them*. If the per-agent cost climbs
/// with the body count the suspicion is confirmed and this file's absolute
/// numbers are a floor for the island rather than an estimate of it; if it is
/// flat, the gap is something else and the wave should go looking.
///
/// Printed, not asserted — it is a clock, and its shape is the finding.
#[test]
fn the_cost_of_one_agent_grows_with_the_world_around_it() {
    println!(
        "\n side | boxes | bodies | control ms | +{CROWD} moving | us/agent | narrowed | us/agent"
    );
    for side in [BUILDINGS, 86, 130] {
        let base = Spec::new().side(side);
        let control = run(base);
        let moving = run(base.crowd(CROWD).moving(true));
        let narrow = run(base.crowd(CROWD).moving(true).narrow_both());
        println!(
            "{side:>5} | {:>5} | {:>6} | {:>10.3} | {:>+11.3} | {:>8.2} | {:>+8.3} | {:>8.2}",
            side * side,
            control.bodies,
            control.ms,
            moving.ms - control.ms,
            (moving.ms - control.ms) * 1000.0 / CROWD as f64,
            narrow.ms - control.ms,
            (narrow.ms - control.ms) * 1000.0 / CROWD as f64,
        );
    }
    println!();
}

// ── the door's own arms ─────────────────────────────────────────────────────

fn tiny(scenery: ColliderPairing, agent: ColliderPairing) -> (usize, usize) {
    let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let g = w.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    let floor = ColliderDesc3D::new(ColliderShape3D::Box {
        half_extents: DVec3::new(10.0, 1.0, 10.0),
    })
    .pairing(scenery);
    w.try_add_collider(g, floor).expect("the floor attaches");

    let k = w.add_body(
        BodyKind3D::Kinematic,
        DVec3::new(0.0, 1.0 + AGENT_HALF_HEIGHT + AGENT_RADIUS - 0.05, 0.0),
        DQuat::IDENTITY,
    );
    w.try_add_collider(
        k,
        ColliderDesc3D::new(ColliderShape3D::Capsule {
            half_height: AGENT_HALF_HEIGHT,
            radius: AGENT_RADIUS,
        })
        .pairing(agent),
    )
    .expect("the capsule attaches");

    for i in 0..8 {
        // Moving, so nothing sleeps its way to a false negative.
        w.set_body_translation(
            k,
            DVec3::new(
                f64::from(i) * 0.01,
                1.0 + AGENT_HALF_HEIGHT + AGENT_RADIUS - 0.05,
                0.0,
            ),
        );
        w.step(DT);
    }
    w.contact_pair_counts()
}

/// **`All` is exactly today's behaviour** — the default changes nothing.
///
/// A kinematic capsule resting on a static floor computes a manifold, as it
/// always has. Without this arm every other arm in this file could pass with the
/// default silently narrowed, and the whole tree would have quietly lost its
/// kinematic contacts on the commit that added a knob nobody called.
#[test]
fn the_default_pairing_keeps_a_kinematic_versus_static_manifold() {
    let (tracked, touching) = tiny(ColliderPairing::All, ColliderPairing::All);
    println!("default (All / All): {tracked} tracked, {touching} touching");
    assert_eq!(tracked, 1, "the broad phase must find the pair");
    assert_eq!(
        touching, 1,
        "a kinematic capsule standing on a static floor computed NO manifold under \
         the DEFAULT pairing — `ColliderPairing::All` has stopped meaning what the \
         tree had before this knob existed, and every kinematic contact in the \
         engine has just gone quiet"
    );
    // …and `ColliderDesc3D::new` really does default to `All`, rather than this
    // arm having asked for it explicitly.
    assert_eq!(
        ColliderDesc3D::new(ColliderShape3D::Sphere { radius: 1.0 }).pairing,
        ColliderPairing::All,
        "the descriptor's default pairing is not `All` — a caller that never heard \
         of this knob would get a narrowed collider"
    );
}

/// **Narrowing ONE side changes nothing** — the union rule, armed.
///
/// This is the hypothesis NPC1c's measurement killed, and it is worth an arm of
/// its own precisely because it is the prescription a reader would reach for: put
/// `DynamicOnly` on the crowd's capsule and stop there.
#[test]
fn narrowing_only_one_side_of_a_pair_removes_nothing() {
    let capsule_only = tiny(ColliderPairing::All, ColliderPairing::DynamicOnly);
    let floor_only = tiny(ColliderPairing::DynamicOnly, ColliderPairing::All);
    println!(
        "capsule narrowed only: {:?}; floor narrowed only: {:?}",
        capsule_only, floor_only
    );
    assert_eq!(
        capsule_only,
        (1, 1),
        "narrowing the CAPSULE alone dropped the manifold — the union rule \
         (`!a.test() && !b.test()`) says the floor's `all() - FIXED_FIXED` still \
         carries `KINEMATIC_FIXED`, and NPC1c's whole wiring instruction rests on it"
    );
    assert_eq!(
        floor_only,
        (1, 1),
        "narrowing the FLOOR alone dropped the manifold — same union rule, other side"
    );
}

/// **Narrowing BOTH sides removes the manifold** — the door does what it says.
#[test]
fn narrowing_both_sides_removes_the_kinematic_versus_static_manifold() {
    let (tracked, touching) = tiny(ColliderPairing::DynamicOnly, ColliderPairing::DynamicOnly);
    println!("both narrowed: {tracked} tracked, {touching} touching");
    assert_eq!(
        tracked, 1,
        "the broad phase still finds the pair — dropping it there would need \
         collision groups, which are the author's"
    );
    assert_eq!(
        touching, 0,
        "a kinematic-vs-fixed pair with `DynamicOnly` on BOTH colliders still \
         computed a manifold — `ColliderPairing::DynamicOnly` is not reaching \
         `active_collision_types`, and NPC1c's crowd saving does not exist"
    );
}

/// **A SENSOR keeps `all()` whatever its `pairing` says** — the union rail.
///
/// The reason the engine widened these flags in the first place is a trigger, and
/// a trigger over a moving kinematic body is the archetypal case. `pairing` must
/// not be able to take it away, so the sensor rule wins; this arm is the proof.
///
/// The measurement is the **event**, not [`PhysicsWorld3D::contact_pair_counts`]:
/// a sensor pair lives in rapier's *intersection* graph and never appears in the
/// contact graph that function walks, so a sensor reads `(0, 0)` there whether it
/// reports or not. (Measured — the first spelling of this arm asserted the wrong
/// graph and would have "passed" a narrowed sensor that had gone silent.)
#[test]
fn a_sensor_reports_a_kinematic_overlap_even_when_it_asks_to_be_narrowed() {
    let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
    let g = w.add_body(BodyKind3D::Static, DVec3::ZERO, DQuat::IDENTITY);
    let mut trigger = ColliderDesc3D::new(ColliderShape3D::Box {
        half_extents: DVec3::new(4.0, 4.0, 4.0),
    })
    .pairing(ColliderPairing::DynamicOnly);
    trigger.sensor = true;
    w.try_add_collider(g, trigger)
        .expect("the trigger attaches");

    let k = w.add_body(
        BodyKind3D::Kinematic,
        DVec3::new(-8.0, 0.0, 0.0),
        DQuat::IDENTITY,
    );
    w.try_add_collider(
        k,
        ColliderDesc3D::new(ColliderShape3D::Capsule {
            half_height: AGENT_HALF_HEIGHT,
            radius: AGENT_RADIUS,
        })
        .pairing(ColliderPairing::DynamicOnly),
    )
    .expect("the capsule attaches");

    let mut sensor_events = 0usize;
    for i in 0..64 {
        w.set_body_translation(k, DVec3::new(-8.0 + f64::from(i) * 0.25, 0.0, 0.0));
        w.step(DT);
        sensor_events += w.drain_contact_events().iter().filter(|e| e.sensor).count();
    }
    println!("sensor over a narrowed moving kinematic capsule: {sensor_events} sensor events");
    assert!(
        sensor_events > 0,
        "a SENSOR asking for `DynamicOnly` stopped reporting a kinematic overlap. \
         The sensor rule must win over `pairing`: the flags are a union, a trigger \
         is the case the engine widened them for, and a narrowed trigger is exactly \
         the configuration that goes quiet without anyone noticing"
    );
}

/// **Two narrowed kinematic capsules do not manifold each other** — the
/// `KINEMATIC_KINEMATIC` half, which the crowd fixture's own spacing never
/// exercises (its agents stand seven metres apart) and which is therefore the
/// one claim in this file that would otherwise be vacuous.
#[test]
fn two_narrowed_kinematic_capsules_stop_pairing_with_each_other() {
    let pair = |p: ColliderPairing| {
        let mut w = PhysicsWorld3D::new(DVec3::new(0.0, -9.81, 0.0));
        for x in [0.0f64, 0.4] {
            let b = w.add_body(
                BodyKind3D::Kinematic,
                DVec3::new(x, AGENT_Y, 0.0),
                DQuat::IDENTITY,
            );
            w.try_add_collider(
                b,
                ColliderDesc3D::new(ColliderShape3D::Capsule {
                    half_height: AGENT_HALF_HEIGHT,
                    radius: AGENT_RADIUS,
                })
                .pairing(p),
            )
            .expect("a capsule attaches");
        }
        for _ in 0..4 {
            w.step(DT);
        }
        w.contact_pair_counts()
    };
    let all = pair(ColliderPairing::All);
    let narrowed = pair(ColliderPairing::DynamicOnly);
    println!("capsule ↔ capsule: All {all:?}, DynamicOnly {narrowed:?}");
    assert_eq!(
        all,
        (1, 1),
        "two overlapping kinematic capsules do NOT compute a manifold under the \
         default — `KINEMATIC_KINEMATIC` has gone missing from `All`"
    );
    assert_eq!(
        narrowed,
        (1, 0),
        "two overlapping `DynamicOnly` kinematic capsules still compute a manifold"
    );
}
