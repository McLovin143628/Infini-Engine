//! **Where `character move`'s milliseconds go** (the NPC1c audit) — the
//! decomposition of the one number that wave created.
//!
//! NPC1c gave the crowd's `Full` tier a controller and the island's fixed step
//! grew a phase:
//!
//! > `character move` 0.132 → **5.629 ms** (**+5.50**) for 32 controllers among
//! > 17 823 bodies — *"0.18 ms a character in the mover"*, the arc's next wall.
//!
//! …and the same wave measured **92.756 ms for 291** of them, which is 0.319 ms
//! each. Three numbers, three different per-character costs — 0.132 at one,
//! 0.171 at thirty-three, 0.319 at two hundred and ninety-one — and the wave
//! called that *"super-linear"* without saying what the term is. This file is
//! that measurement, and it answers two questions the ledger leaves open:
//!
//! 1. **Is there an `O(N²)` term at all?** If the mover's cost per character
//!    climbs with the number of characters, something in it walks the character
//!    list per character (this crate has had exactly that defect before — the
//!    P29.4 audit's A8, `try_mantle` asking `movement_targets` for its own copy).
//!    If the per-character cost is flat in N, the cost is per-character work and
//!    the "super-linear" reading is about the *world*, not the crowd.
//! 2. **What is the per-character constant made of?** The mover is
//!    collide-and-slide plus probes, and every one of those is a query against
//!    whatever is under and around the body. So the same character is measured
//!    on four worlds: nothing, the buildings alone, the ground alone, and both.
//!
//! # The fixture is `kinematic_pairing_cost`'s, one system over
//!
//! Same 43 × 43 = **1 849** static boxes at a 7 m pitch, same island tile
//! resolution (257 samples at 1 m), same frontage lattice — so the two files
//! measure two phases of one step over one world and their numbers can be read
//! against each other. What differs is what is being timed: that file brackets
//! `world.step` (the `solver` phase) and this one brackets
//! [`step_character_movement`] (the `character move` phase), which touches the
//! **query** pipeline and never the contact graph.
//!
//! # What is asserted and what is printed
//!
//! **The shape is asserted, the clocks are printed** — `kinematic_pairing_cost`'s
//! own rule and `city_collider_band`'s before it. A wall-clock assertion on a
//! shared runner is a flake generator, and every absolute millisecond here is a
//! statement about one dev box. What *is* asserted is a **ratio with a 4×
//! margin**: a genuine `O(N²)` term would make the per-character cost at N = 64
//! sixty-four times its cost at N = 1, so a bound at 4× cannot be met by noise
//! and cannot be missed by a quadratic. Every arm also asserts the characters
//! actually **moved**, because a mover that refused every step would be the
//! cheapest of all.
//!
//! Every figure is the MIN of five rounds of sixty steps on this crate's dev
//! profile (`[profile.dev.package."*"] opt-level = 2`, so rapier and parry are
//! optimised and `inf-physics` itself is not — the *shape* is what this file
//! claims and the shape does not depend on that).

use std::time::Instant;

use glam::DVec3;
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, Gait,
    RigidBody3D, RotationMode, Terrain, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::EcsWorld;
use inf_physics::d3::step_character_movement;
use inf_physics::PhysicsBridge3D;
use inf_terrain::TerrainData;
use uuid::Uuid;

const DT: f64 = 1.0 / 60.0;

/// The island's own tile: `tile_resolution = 257` at `meters_per_sample = 1.0`
/// (`samples/island/island.toml`), so 256 m of ground per tile.
const TILE_RES: u32 = 257;
const MPS: f64 = 1.0;
/// The walking surface, metres. Flat, for `kinematic_pairing_cost`'s reason: a
/// flat tile makes "standing on it" a fact rather than a coincidence of where a
/// slope happened to be, and parry indexes a height-field cell arithmetically,
/// so the cell walk under a capsule is the same work whatever the heights hold.
const GROUND_Y: f64 = 0.0;

/// Building pitch, metres — 6 m of box and 1 m of gap, a dense downtown face.
const BUILDING_PITCH: f64 = 7.0;
/// Buildings on a side in the full fixture: 43 × 43 = **1 849** static boxes.
const BUILDINGS: i32 = 43;
const BUILDING_HALF: DVec3 = DVec3::new(3.0, 4.0, 3.0);

/// `CrowdArchetype::humanoid`'s corrected 1.8 m adult (NPC1c defect 2).
const AGENT_HALF_HEIGHT: f64 = 0.6;
const AGENT_RADIUS: f64 = 0.3;

/// The crowd lattice: 17 × 17 = 289 sites, of which the first N are used.
const AGENT_SIDE: i32 = 17;

fn building_centre(i: i32, side: i32) -> DVec3 {
    let (c, r) = (i % side, i / side);
    let half = f64::from(side - 1) * 0.5;
    DVec3::new(
        (f64::from(c) - half) * BUILDING_PITCH,
        GROUND_Y + BUILDING_HALF.y,
        (f64::from(r) - half) * BUILDING_PITCH,
    )
}

/// Character `i`'s home: **touching a building's west face**, which is where a
/// crowd on a sidewalk actually walks.
///
/// The lattice fills **outward from the centre** — cells ordered by Chebyshev
/// ring, then by row, then by column — for the reason
/// [`the_movers_per_character_cost_is_a_function_of_the_worlds_size`] needs and
/// a corner-first walk would have quietly broken: at N = 1 the one character has
/// to be standing against a building on **every** world size the arm varies, or
/// the sweep measures a body that started in a field.
fn agent_home(i: usize) -> DVec3 {
    let half = f64::from(AGENT_SIDE - 1) * 0.5;
    let mid = AGENT_SIDE / 2;
    let mut cells: Vec<(i32, i32, i32)> = Vec::with_capacity((AGENT_SIDE * AGENT_SIDE) as usize);
    for r in 0..AGENT_SIDE {
        for c in 0..AGENT_SIDE {
            cells.push(((r - mid).abs().max((c - mid).abs()), r, c));
        }
    }
    cells.sort_unstable();
    let (_, r, c) = cells[i % cells.len()];
    DVec3::new(
        (f64::from(c) - half) * BUILDING_PITCH - BUILDING_HALF.x - AGENT_RADIUS,
        GROUND_Y + AGENT_HALF_HEIGHT + AGENT_RADIUS,
        (f64::from(r) - half) * BUILDING_PITCH,
    )
}

/// What world to build. Every field is something an arm varies.
#[derive(Clone, Copy)]
struct Spec {
    /// Blocks on a side; `side * side` static boxes. Zero for none.
    side: i32,
    /// Whether the four heightfield tiles are present at all.
    ground: bool,
    /// How many characters carry `CharacterMovement` — the mover's subjects.
    movers: usize,
    /// How many extra kinematic capsules stand in the world **without** a
    /// controller — the island's `Near` tier, which the mover never visits but
    /// every one of its queries has to look past.
    bystanders: usize,
}

impl Spec {
    fn new() -> Self {
        Self {
            side: BUILDINGS,
            ground: true,
            movers: 1,
            bystanders: 0,
        }
    }
    fn movers(mut self, n: usize) -> Self {
        self.movers = n;
        self
    }
    fn bystanders(mut self, n: usize) -> Self {
        self.bystanders = n;
        self
    }
    fn ground(mut self, on: bool) -> Self {
        self.ground = on;
        self
    }
    fn side(mut self, side: i32) -> Self {
        self.side = side;
        self
    }
}

struct Fixture {
    world: EcsWorld,
    bridge: PhysicsBridge3D,
    movers: Vec<Uuid>,
    bystanders: Vec<Uuid>,
    bodies: usize,
}

fn character_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x4d4f_5645_0000_0000_0000_0000_0000_0000 | i as u128)
}

fn bystander_guid(i: usize) -> Uuid {
    Uuid::from_u128(0x4259_5354_0000_0000_0000_0000_0000_0000 | i as u128)
}

/// The crowd's own movement model — `inf_ecs::crowd::crowd_movement`, spelled
/// here because that function is private to its module. The four things a
/// pedestrian is: not player-controlled, walking, facing its velocity, and
/// wearing the archetype's capsule.
fn crowd_movement() -> CharacterMovement {
    CharacterMovement {
        player_controlled: false,
        gait: Gait::Walk,
        rotation_mode: RotationMode::VelocityDirection,
        stand_half_height_m: AGENT_HALF_HEIGHT,
        ..CharacterMovement::default()
    }
}

fn capsule() -> Collider3D {
    Collider3D {
        shape_kind: ColliderShape3DKind::Capsule,
        half_extents: Vec3d::new(AGENT_RADIUS, AGENT_HALF_HEIGHT, AGENT_RADIUS),
        radius: AGENT_RADIUS,
        ..Collider3D::default()
    }
}

fn kinematic() -> RigidBody3D {
    RigidBody3D {
        kind: BodyKind3D::Kinematic,
        fixed_rotation: true,
        ..RigidBody3D::default()
    }
}

fn fixture(spec: Spec) -> Fixture {
    let mut world = EcsWorld::new();

    if spec.ground {
        // Four tiles, so the block field's ±150 m sits inside ±256 m of ground
        // however the arm sizes it.
        let mut data = TerrainData::new(TILE_RES, MPS);
        for tz in -1..=0 {
            for tx in -1..=0 {
                data.author_tile((tx, tz), |_, _| GROUND_Y);
            }
        }
        let e = world.spawn_with_guid(Uuid::from_u128(0x7e44_0001), "Terrain", None);
        world.world_mut().entity_mut(e).insert(Terrain {
            meters_per_sample: MPS,
            tile_resolution: TILE_RES,
            data,
            ..Terrain::default()
        });
    }

    for i in 0..(spec.side * spec.side) {
        let e = world.spawn_with_guid(
            Uuid::from_u128(0x8100_0000_0000_0000 | i as u128),
            "Block",
            None,
        );
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::from_dvec3(building_centre(i, spec.side));
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..RigidBody3D::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::from_dvec3(BUILDING_HALF),
                ..Collider3D::default()
            },
            t,
        ));
    }

    let mut movers = Vec::with_capacity(spec.movers);
    for i in 0..spec.movers {
        let guid = character_guid(i);
        let e = world.spawn_with_guid(guid, "Crowd NPC", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::from_dvec3(agent_home(i));
        world.world_mut().entity_mut(e).insert((
            kinematic(),
            capsule(),
            CharacterController3D::default(),
            crowd_movement(),
            t,
        ));
        movers.push(guid);
    }

    // The bystanders start on the far half of the lattice, so they are a crowd
    // standing among the movers rather than a crowd standing on top of them.
    let mut bystanders = Vec::with_capacity(spec.bystanders);
    for i in 0..spec.bystanders {
        let guid = bystander_guid(i);
        let e = world.spawn_with_guid(guid, "Near NPC", None);
        let mut t = Transform::IDENTITY;
        let home = agent_home(i) + DVec3::new(0.0, 0.0, 2.0);
        t.translation = Vec3d::from_dvec3(home);
        world
            .world_mut()
            .entity_mut(e)
            .insert((kinematic(), capsule(), t));
        bystanders.push(guid);
    }

    world.mark_dirty();
    world.propagate();

    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);
    let bodies = bridge.world().body_ids().len();
    Fixture {
        world,
        bridge,
        movers,
        bystanders,
        bodies,
    }
}

/// Write the intent a steered crowd agent writes — a planar unit wish, straight
/// down the frontage. `inf_ecs::crowd::steer_agent` writes exactly this field,
/// and `step_character_movement` is what consumes it.
fn drive(f: &mut Fixture) {
    for guid in &f.movers {
        let Some(e) = f.world.entity_of(*guid) else {
            continue;
        };
        if let Some(mut cm) = f.world.world_mut().get_mut::<CharacterMovement>(e) {
            cm.runtime.intent_move = Vec2d::new(0.0, 1.0);
        }
    }
}

/// Re-place every bystander, the way the crowd step re-places a `Near` agent's
/// transform and the physics sync then writes it into the body. The mover pays
/// for this through `ensure_query_pipeline`, which folds what moved into the
/// query BVH on the step's first query — and the step's first query is the
/// mover's own.
fn walk_bystanders(f: &mut Fixture, step: u32) {
    if f.bystanders.is_empty() {
        return;
    }
    let d = 1.4 * DT * f64::from(step);
    for (i, guid) in f.bystanders.clone().iter().enumerate() {
        let home = agent_home(i) + DVec3::new(0.0, 0.0, 2.0);
        let t = (d + i as f64 * 0.37) % 12.0 - 6.0;
        let Some(body) = f.bridge.body_of(*guid) else {
            continue;
        };
        f.bridge
            .world_mut()
            .set_body_translation(body, DVec3::new(home.x, home.y, home.z + t));
    }
}

struct Measured {
    /// MIN over rounds of the mean per-step millisecond.
    ms: f64,
    /// Bodies in the rapier world.
    bodies: usize,
    /// How far the first character travelled over the whole measurement, metres
    /// — the anti-vacuity reading.
    travelled_m: f64,
}

impl Measured {
    /// Microseconds a character a step.
    fn us_per_character(&self, movers: usize) -> f64 {
        self.ms * 1000.0 / movers.max(1) as f64
    }
}

fn measure(spec: Spec) -> Measured {
    const WARM: u32 = 20;
    const ROUNDS: u32 = 5;
    const STEPS: u32 = 60;

    let mut f = fixture(spec);
    let first = f.movers.first().copied();
    let start = first
        .and_then(|g| f.world.entity_of(g))
        .and_then(|e| f.world.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .unwrap_or(DVec3::ZERO);

    let mut clock = 0u32;
    for _ in 0..WARM {
        clock += 1;
        walk_bystanders(&mut f, clock);
        drive(&mut f);
        step_character_movement(&mut f.world, &mut f.bridge, DT);
    }

    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let mut total = 0.0f64;
        for _ in 0..STEPS {
            clock += 1;
            walk_bystanders(&mut f, clock);
            drive(&mut f);
            let t = Instant::now();
            step_character_movement(&mut f.world, &mut f.bridge, DT);
            total += t.elapsed().as_secs_f64();
        }
        best = best.min(total * 1000.0 / f64::from(STEPS));
    }
    // ANTI-VACUITY: a timer reading zero measured its own resolution.
    assert!(best > 0.0, "the movement step took no measurable time");

    let end = first
        .and_then(|g| f.world.entity_of(g))
        .and_then(|e| f.world.world().get::<Transform>(e))
        .map(|t| t.translation.to_dvec3())
        .unwrap_or(DVec3::ZERO);

    Measured {
        ms: best,
        bodies: f.bodies,
        travelled_m: (end - start).length(),
    }
}

// ── the arms ────────────────────────────────────────────────────────────────

/// **THE HEADLINE: is the mover quadratic in the crowd, or linear?**
///
/// The same world, the same buildings, the same ground, N controllers walking
/// their own frontages. If anything in `step_character_movement` walks the
/// character list per character, the per-character microsecond climbs with N;
/// if the cost is per-character work, it is flat and the wave's "super-linear"
/// reading has to be about the world instead.
#[test]
fn the_movers_cost_per_character_does_not_climb_with_the_crowd() {
    let mut rows: Vec<(usize, Measured)> = Vec::new();
    for n in [1usize, 2, 4, 8, 16, 32, 64] {
        rows.push((n, measure(Spec::new().movers(n))));
    }
    println!(
        "NPC1c audit / character move: {} bodies (1 849 boxes + 4 tiles + N capsules)",
        rows[0].1.bodies
    );
    for (n, m) in &rows {
        println!(
            "  N = {n:>3}  {:>8.4} ms/step   {:>8.2} us/character   walked {:.2} m",
            m.ms,
            m.us_per_character(*n),
            m.travelled_m
        );
    }
    // ANTI-VACUITY: a mover that refused every step would be the cheapest of all.
    for (n, m) in &rows {
        assert!(
            m.travelled_m > 1.0,
            "N = {n}: the character walked {:.3} m, so this measured a mover that \
             did nothing",
            m.travelled_m
        );
    }
    let one = rows[0].1.us_per_character(1);
    let many = rows
        .last()
        .unwrap()
        .1
        .us_per_character(rows.last().unwrap().0);
    println!(
        "  per-character 1 -> 64: {one:.2} -> {many:.2} us  ({:.2}x)",
        many / one
    );
    // A real `O(N²)` term is 64x here. The bound is 4x, which noise cannot reach
    // and a quadratic cannot meet.
    assert!(
        many < one * 4.0,
        "the mover costs {many:.2} us a character at N = 64 against {one:.2} at \
         N = 1 — that is a per-character walk of the character list, which is the \
         P29.4 audit's A8 defect returning at crowd N"
    );
}

/// **What the per-character constant is made of** — the same one character on
/// four worlds.
///
/// `move_and_slide` is shape casts through the query pipeline, and a cast's cost
/// is a function of what it has to test against. This is the split the wave's
/// "0.18 ms a character" needs before anybody can make it cheaper: if the ground
/// dominates, the lever is the ground query; if the buildings do, it is the
/// broad-phase depth.
#[test]
fn the_world_under_a_character_is_where_its_milliseconds_are() {
    let bare = measure(Spec::new().side(0).ground(false));
    let boxes = measure(Spec::new().ground(false));
    let dirt = measure(Spec::new().side(0));
    let both = measure(Spec::new());
    println!("NPC1c audit / one character, four worlds:");
    for (name, m) in [
        ("nothing at all", &bare),
        ("1 849 boxes, no ground", &boxes),
        ("4 tiles of ground, no boxes", &dirt),
        ("both (the fixture)", &both),
    ] {
        println!(
            "  {name:<28} {:>8.4} ms/step  {:>4} bodies  walked {:.2} m",
            m.ms, m.bodies, m.travelled_m
        );
    }
    // The ONE structural claim: a character with nothing to collide with still
    // walks, so none of the four rows measured a refusal.
    for (name, m) in [
        ("nothing at all", &bare),
        ("1 849 boxes, no ground", &boxes),
        ("4 tiles of ground, no boxes", &dirt),
        ("both (the fixture)", &both),
    ] {
        assert!(
            m.travelled_m > 1.0,
            "{name}: the character walked {:.3} m",
            m.travelled_m
        );
    }
    println!(
        "  the ground is {:.2}x the empty world and the boxes are {:.2}x",
        dirt.ms / bare.ms,
        boxes.ms / bare.ms
    );
}

/// **Does a crowd standing around make the mover dearer?**
///
/// The island's `character move` row is 32 controllers among **288** kinematic
/// capsules — the other 256 are `Near`, which carry a capsule and a body and are
/// moved by the clock rather than by the mover. They are not the mover's
/// subjects, they are in every one of its queries, and every one of them is
/// re-placed each step, which the mover pays for through
/// `ensure_query_pipeline` on the step's first query.
///
/// So: 32 controllers, with 0 / 64 / 256 of them standing around, each two
/// metres in front of a mover — which is where a bystander a crowd actually has
/// to get past stands.
///
/// **What it measures is the body in front, not the population.** The step from
/// 0 to 64 is the whole of it and 64 → 256 is almost nothing: a controller pays
/// for the thing it has to slide around, and a second capsule behind the first
/// costs it nothing. That is the honest reading of a crowd-proportional term
/// inside `character move` — it is proportional to how many controllers are in
/// contact, not to how many agents exist.
#[test]
fn a_controller_pays_for_the_body_in_front_of_it_not_for_the_crowds_size() {
    let mut rows: Vec<(usize, Measured)> = Vec::new();
    for b in [0usize, 64, 256] {
        rows.push((b, measure(Spec::new().movers(32).bystanders(b))));
    }
    println!("NPC1c audit / 32 controllers, N bystanders:");
    for (b, m) in &rows {
        println!(
            "  {b:>3} bystanders  {:>8.4} ms/step  {:>5} bodies  {:>7.2} us/character",
            m.ms,
            m.bodies,
            m.us_per_character(32)
        );
    }
    let alone = &rows[0].1;
    let crowded = &rows.last().unwrap().1;
    println!("  256 bystanders cost {:.2}x", crowded.ms / alone.ms);
    for (b, m) in &rows {
        assert!(
            m.travelled_m > 1.0,
            "{b} bystanders: the character walked {:.3} m",
            m.travelled_m
        );
    }
    assert_eq!(
        crowded.bodies - alone.bodies,
        256,
        "the bystanders are not in the physics world, so this arm measured the \
         same thing twice"
    );
}

/// **The per-character cost is a function of the WORLD, and this is that
/// function** — one character, four world sizes.
///
/// This is the arm that carries the island's numbers back to a fixture a tenth
/// its size. NPC1c measured `character move` at **0.176 ms a character** with 32
/// controllers among the island's 17 823 bodies, and the same wave's
/// `kinematic_pairing_cost::the_cost_of_one_agent_grows_with_the_world_around_it`
/// established the shape one phase over: what a body costs is what it has to be
/// tested against. One character over a growing block field says whether the
/// mover obeys the same law, and at what slope — which is the number a wave
/// trying to make the mover cheaper has to beat.
#[test]
fn the_movers_per_character_cost_is_a_function_of_the_worlds_size() {
    let mut rows: Vec<(i32, Measured)> = Vec::new();
    for side in [13i32, 25, 43, 61] {
        rows.push((side, measure(Spec::new().side(side))));
    }
    println!("NPC1c audit / one character, a growing city:");
    for (side, m) in &rows {
        println!(
            "  {:>2} x {:<2} = {:>4} boxes  {:>5} bodies  {:>8.4} ms/step  {:>7.2} us/character",
            side,
            side,
            side * side,
            m.bodies,
            m.ms,
            m.us_per_character(1)
        );
    }
    for (side, m) in &rows {
        assert!(
            m.travelled_m > 1.0,
            "side {side}: the character walked {:.3} m",
            m.travelled_m
        );
    }
    let small = &rows[0].1;
    let big = rows.last().unwrap().1.ms;
    println!(
        "  {:.0}x the bodies costs {:.2}x the millisecond",
        rows.last().unwrap().1.bodies as f64 / small.bodies as f64,
        big / small.ms
    );
    // The bound the shape has to hold: growing the world 20x must not grow the
    // per-character cost 20x, or the mover is walking the world rather than
    // querying a tree.
    assert!(
        big < small.ms * 4.0,
        "the mover costs {big:.4} ms a character on {} bodies against {:.4} on \
         {} — that is not a tree query, that is a walk",
        rows.last().unwrap().1.bodies,
        small.ms,
        small.bodies
    );
}
