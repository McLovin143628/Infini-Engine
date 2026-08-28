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
//! # …and island wave NPC1e adds the third, which is the one the arc ended on
//!
//! The two questions above sweep **N** and **the world**, and their answers were
//! *"flat in N"* and *"the ground is where the milliseconds are"*. The island's
//! certification row then read `character move` at **84.100 ms for 170
//! controllers — 0.494 ms each — against 0.171 at thirty-three** on the same
//! island, which is three times a per-character cost this file held flat over a
//! 64× range in N. Neither question can see it, because **neither of them
//! sweeps DENSITY**: the bystander arm puts one capsule in front of each mover
//! and then adds more of them further out, so what it grows is the crowd's
//! extent. So there is a third question — *how much does a controller pay for
//! the capsules **inside its own sweep**?* — and its answer is
//! [`the_movers_cost_is_flat_in_a_crowds_size_and_not_in_its_density`]: the same
//! population packed into the twenty metres the controllers stand in costs
//! **2.6×** what it costs spread over the lattice.
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

use glam::{DQuat, DVec3};
use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind, Gait,
    RigidBody3D, RotationMode, Terrain, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::EcsWorld;
use inf_physics::d3::{step_character_movement, AutoStep3D, CharacterMover3D, ColliderShape3D};
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

/// **A bystander's home** — the NPC1c audit's queue, or island wave NPC1e's
/// rush hour.
///
/// Unpacked (`None`) it is the arrangement that arm was written with: one
/// capsule two metres in front of each mover, on the movers' own 7 m lattice
/// spreading outward. That measures how much a controller pays for **the body
/// in front of it**, and it saturates at 1.24x, which is the honest answer to
/// the question it asks.
///
/// Packed it is a square lattice at `pitch` metres about the origin, so two
/// hundred capsules occupy the twenty metres the movers themselves stand in.
/// That is what a town's morning commute looks like — the island's own
/// certification row measured **191 residents inside a 32 m ring** — and it is a
/// different variable from the one the arm above sweeps.
fn bystander_home(i: usize, pitch: Option<f64>) -> DVec3 {
    match pitch {
        None => agent_home(i) + DVec3::new(0.0, 0.0, 2.0),
        Some(p) => {
            let side = 24_usize;
            let (c, r) = (i % side, (i / side) % side);
            let half = (side - 1) as f64 * 0.5;
            DVec3::new(
                (c as f64 - half) * p,
                GROUND_Y + AGENT_HALF_HEIGHT + AGENT_RADIUS,
                (r as f64 - half) * p,
            )
        }
    }
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
    /// **How tightly the bystanders are packed**, metres of lattice pitch
    /// (island wave NPC1e).
    ///
    /// `None` is the NPC1c audit's arrangement: one bystander two metres in
    /// front of each mover, spreading outward on the movers' own 7 m lattice.
    /// `Some(pitch)` packs all of them into a square lattice at that pitch about
    /// the origin — a **rush hour** rather than a queue. See
    /// [`the_movers_cost_is_flat_in_a_crowds_SIZE_and_not_in_its_DENSITY`].
    packed_m: Option<f64>,
}

impl Spec {
    fn new() -> Self {
        Self {
            side: BUILDINGS,
            ground: true,
            movers: 1,
            bystanders: 0,
            packed_m: None,
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
    fn packed(mut self, pitch_m: f64) -> Self {
        self.packed_m = Some(pitch_m);
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
    /// The spec's packing, carried so `walk_bystanders` re-derives the same
    /// homes the fixture placed.
    packed_m: Option<f64>,
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
    // standing among the movers rather than a crowd standing on top of them --
    // unless the spec packs them, in which case they are a rush hour.
    let mut bystanders = Vec::with_capacity(spec.bystanders);
    for i in 0..spec.bystanders {
        let guid = bystander_guid(i);
        let e = world.spawn_with_guid(guid, "Near NPC", None);
        let mut t = Transform::IDENTITY;
        let home = bystander_home(i, spec.packed_m);
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
        packed_m: spec.packed_m,
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
    let packed = f.packed_m;
    for (i, guid) in f.bystanders.clone().iter().enumerate() {
        let home = bystander_home(i, packed);
        // A packed crowd shuffles a metre; a spread one paces twelve. Either
        // way it MOVES, which is what makes the query BVH pay for it.
        let span = if packed.is_some() { 1.0 } else { 12.0 };
        let t = (d + i as f64 * 0.37) % span - 0.5 * span;
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
    /// **Whether the first character was standing on anything at the end**
    /// (island wave NPC1e), and how far it had sunk.
    ///
    /// The NPC1c audit's four-world table read `travelled_m > 1.0` as its
    /// anti-vacuity clause, and a body in **free fall** travels further than a
    /// body that walks: on the `no ground` worlds the character ends at
    /// y = −1.334 m having never touched anything. Those rows therefore measure
    /// a *different program* from the ones with ground in them — an airborne
    /// step runs `predict_landing` and skips the ground-normal probe — so the
    /// "the boxes cost 24 µs and the ground 30" split is not two ablations of
    /// one walk. Printed here so the table cannot be read that way again.
    grounded: bool,
    /// The first character's Y at the end of the measurement, metres.
    end_y: f64,
    /// Scene queries the whole measurement asked of the world, per character per
    /// step — the countable half of the same question (`PhysicsWorld3D::queries`).
    queries_per_character_step: f64,
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

    let queries_before = f.bridge.world().queries();
    let mut grounded = false;
    let mut best = f64::INFINITY;
    for _ in 0..ROUNDS {
        let mut total = 0.0f64;
        for _ in 0..STEPS {
            clock += 1;
            walk_bystanders(&mut f, clock);
            drive(&mut f);
            let t = Instant::now();
            let out = step_character_movement(&mut f.world, &mut f.bridge, DT);
            total += t.elapsed().as_secs_f64();
            grounded = out.first().is_some_and(|o| o.grounded);
        }
        best = best.min(total * 1000.0 / f64::from(STEPS));
    }
    let asked = f.bridge.world().queries() - queries_before;
    let queries_per_character_step =
        asked as f64 / (f64::from(ROUNDS) * f64::from(STEPS) * spec.movers.max(1) as f64);
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
        grounded,
        end_y: end.y,
        queries_per_character_step,
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
            "  N = {n:>3}  {:>8.4} ms/step  {:>8.2} us/character  walked {:>6.2} m  {:.2} queries/character/step",
            m.ms,
            m.us_per_character(*n),
            m.travelled_m,
            m.queries_per_character_step
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
            "  {name:<28} {:>8.4} ms/step  {:>4} bodies  moved {:>7.2} m  ends {} at y = {:>8.3}  {:.2} queries/step",
            m.ms,
            m.bodies,
            m.travelled_m,
            if m.grounded { "GROUNDED" } else { "FALLING " },
            m.end_y,
            m.queries_per_character_step
        );
    }
    // The ONE structural claim: a character with nothing to collide with still
    // moves, so none of the four rows measured a refusal.
    for (name, m) in [
        ("nothing at all", &bare),
        ("1 849 boxes, no ground", &boxes),
        ("4 tiles of ground, no boxes", &dirt),
        ("both (the fixture)", &both),
    ] {
        assert!(
            m.travelled_m > 1.0,
            "{name}: the character moved {:.3} m",
            m.travelled_m
        );
    }
    println!(
        "  the ground is {:.2}x the empty world and the boxes are {:.2}x",
        dirt.ms / bare.ms,
        boxes.ms / bare.ms
    );
    // **AND THE CLAUSE THE `travelled_m > 1.0` ONE COULD NOT MAKE** (island wave
    // NPC1e). A body in **free fall** travels further than a body that walks, so
    // the anti-vacuity above is satisfied by a character that never touched
    // anything — which is exactly what the two ground-free rows are: they end at
    // y = −139 m having fallen for the whole measurement. Their step is a
    // different program from the grounded ones' (an airborne step runs
    // `traversal::predict_landing` and takes no ground-normal probe), so the two
    // halves are **not** ablations of one walk and the "boxes 24 µs, ground
    // 30 µs, additive to 4 %" reading the NPC1c audit published is a comparison
    // across that gap. Pinned as a shape, so the table cannot be read that way
    // again and so it fails the day a ground-free world starts holding a body up.
    assert!(
        both.grounded && dirt.grounded,
        "a world with ground under the character did not leave it grounded, so \
         these rows are not about a walk"
    );
    assert!(
        !bare.grounded && !boxes.grounded,
        "a world with NO ground left the character grounded — these two rows \
         measure a body in free fall (ending at y = {:.1} and {:.1}), which is \
         the whole point of printing the column",
        bare.end_y,
        boxes.end_y
    );
    println!(
        "  the two ground-free rows are a FALLING body (y = {:.1} and {:.1}) at \
         {:.2} queries a step, against a walking one at {:.2} — they are not \
         ablations of one program",
        bare.end_y, boxes.end_y, boxes.queries_per_character_step, both.queries_per_character_step,
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

// ── wave NPC1e: the primitives, so a lever can be aimed ─────────────────────

/// One primitive, timed the way [`measure`] times a step: MIN over rounds of the
/// mean per-call microsecond.
fn time_calls(rounds: u32, calls: u32, mut f: impl FnMut(u32)) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..rounds {
        let t = Instant::now();
        for i in 0..calls {
            f(i);
        }
        best = best.min(t.elapsed().as_secs_f64() * 1.0e6 / f64::from(calls));
    }
    best
}

/// **WAVE NPC1e — WHICH OF THE MOVER'S QUERIES THE MILLISECOND IS IN.**
///
/// The NPC1c audit decomposed `character move` down to *"96 % of it is the
/// collide-and-slide queries, and four heightfield tiles cost more than 1 849
/// buildings"*, and stopped there — its verdict named the lever as *"fewer or
/// cheaper queries per character, and the **ground** is where to aim it first"*
/// without saying which query. A step is not one query: it is
/// `PhysicsWorld3D::move_character` (rapier's sweep, its ground snap and its
/// autostep, which is three further casts every time the sweep meets a wall) plus
/// this crate's own **ground-normal probe**, a second downward shape cast taken
/// on every grounded step.
///
/// So each is timed on its own, on the fixture's own world, with the crowd's own
/// mover — and each on the two half-worlds as well, because "the ground is dear"
/// is a statement about a *cast*, not about a step.
///
/// # …and the NPC1e audit's correction: a HIT is not comparable to a MISS
///
/// The first cut of this arm published *"the probe costs 5.6 µs against the
/// ground and 0.07 µs against 1 849 building boxes — **75×**"*. It is not a
/// comparison between two surfaces. On the `no ground` world the character has
/// been **falling** since the settle loop (it is the same free fall
/// [`the_world_under_a_character_is_where_its_milliseconds_are`] was corrected
/// for one function above), so it stands far below every box and its probe
/// **hits nothing at all** — 0.07 µs is a broad phase saying "no".
///
/// Two things close it. `probe_hit` is asserted against the world's own
/// `ground` flag, so a row that hit nothing can never again be quoted beside a
/// row that hit; and a **box control** drops the identical probe onto a
/// building's roof *in the same world, the same tree and the same filter*,
/// where it really does hit a `Collider3D` box. That is the honest ratio, and
/// on this machine it is a **single digit**, not seventy-five.
///
/// **Everything here is printed. The only assertions are shapes**: that every
/// primitive took measurable time (a zero is a timer reading its own
/// resolution), that the probe hits exactly on the worlds that have ground under
/// the character, and that the box control hits.
#[test]
fn where_the_movers_queries_go() {
    const ROUNDS: u32 = 5;
    const CALLS: u32 = 600;

    // The crowd's own tuning, spelled from the components' defaults the way
    // `crowd_movement` above spells the movement component: `mover_for` builds
    // exactly this for a `CharacterController3D::default()` + the crowd's
    // `CharacterMovement`.
    let shape = ColliderShape3D::Capsule {
        half_height: AGENT_HALF_HEIGHT,
        radius: AGENT_RADIUS,
    };
    let cc = CharacterController3D::default();
    let cm = crowd_movement();
    let base = CharacterMover3D::new(shape)
        .up(DVec3::Y)
        .slide(true)
        .offset(cc.offset)
        .max_slope_climb_angle(cm.slope_limit_deg.to_radians())
        .min_slope_slide_angle(cm.slide_slope_deg.to_radians())
        .snap_to_ground(Some(cc.snap_to_ground))
        .autostep(Some(AutoStep3D {
            max_height: cm.step_height_m,
            min_width: cm.step_min_width_m,
            include_dynamic_bodies: true,
        }));
    let no_step = base.clone().autostep(None);
    let no_snap = base.clone().snap_to_ground(None);
    let neither = base.clone().autostep(None).snap_to_ground(None);

    // The probe `step_one` takes at its section 9, verbatim: a sphere of 0.9 of
    // the capsule's radius, swept down by `(half + radius) * 0.25 + 0.05 + half`.
    let probe_shape = ColliderShape3D::Sphere {
        radius: AGENT_RADIUS * 0.9,
    };
    let probe_len = (AGENT_HALF_HEIGHT + AGENT_RADIUS) * 0.25 + 0.05 + AGENT_HALF_HEIGHT;

    // One step of a walking crowd agent's motion: the gait forward, the grounded
    // gravity bias down. `crowd_movement`'s walk speed at 60 Hz.
    let motion = DVec3::new(0.0, -9.81 * DT * DT, 1.65 * DT);

    for (world_name, spec) in [
        ("both (the fixture)", Spec::new()),
        ("1 849 boxes, no ground", Spec::new().ground(false)),
        ("4 tiles of ground, no boxes", Spec::new().side(0)),
    ] {
        let mut f = fixture(spec);
        // Settle, so the sweeps below start from a body standing on the floor
        // rather than one still falling into it.
        for _ in 0..40 {
            drive(&mut f);
            step_character_movement(&mut f.world, &mut f.bridge, DT);
        }
        let guid = f.movers[0];
        let at = f
            .world
            .entity_of(guid)
            .and_then(|e| f.world.world().get::<Transform>(e))
            .map(|t| t.translation.to_dvec3())
            .expect("the mover has a transform");
        let own = f.bridge.collider_of(guid);
        let mut exclude = std::collections::BTreeSet::new();
        if let Some(c) = own {
            exclude.insert(c);
        }

        let sweep = |m: &CharacterMover3D, f: &mut Fixture| {
            time_calls(ROUNDS, CALLS, |_| {
                let r = f.bridge.world_mut().move_character(m, at, motion, own);
                std::hint::black_box(r.grounded);
            })
        };
        let full = sweep(&base, &mut f);
        let a_off = sweep(&no_step, &mut f);
        let s_off = sweep(&no_snap, &mut f);
        let both_off = sweep(&neither, &mut f);
        let probe = time_calls(ROUNDS, CALLS, |_| {
            let hit = f.bridge.world_mut().cast_shape(
                &probe_shape,
                at,
                DQuat::IDENTITY,
                -DVec3::Y,
                probe_len,
                &exclude,
            );
            std::hint::black_box(hit.is_some());
        });
        // **DID THE PROBE HIT ANYTHING?** (the NPC1e audit) — the anti-vacuity
        // clause this table shipped without, and the one
        // [`what_a_cast_against_the_ground_costs`] already carries: *a cast that
        // hits nothing is a measurement of a broad phase saying "no" and not of
        // the ground.* On the ground-free world the character has been falling
        // since the settle loop above, so it stands a hundred-odd metres under
        // every box and its probe reaches nothing at all. Without this column
        // the 5.6 µs ground row and the 0.07 µs "boxes" row read as a comparison
        // between two SURFACES, and they are a comparison between a **hit and a
        // miss** — which is the same defect
        // [`the_world_under_a_character_is_where_its_milliseconds_are`] was
        // corrected for one function above.
        let probe_hit = f
            .bridge
            .world_mut()
            .cast_shape(
                &probe_shape,
                at,
                DQuat::IDENTITY,
                -DVec3::Y,
                probe_len,
                &exclude,
            )
            .is_some();
        // **THE RAY**, from the same origin and reaching the same depth — the
        // cheapest alternative shape for the same question, and the row the
        // lever's remaining options are priced against. Measured here rather
        // than quoted: a primitive with no arm behind it is a number nobody can
        // re-run, and the ledger's first cut published a **0.18 µs** ray this
        // file did not contain.
        //
        // **The reach is `probe_len` PLUS the sphere's own radius**, and that is
        // the whole difference between a number and a miss: the shape cast's
        // sphere starts a radius below `at` and ends a radius below `at −
        // probe_len`, so a ray of exactly `probe_len` stops **above the ground
        // the sphere touches** and answers `None` for free. Two casts that reach
        // different depths are not two ways of asking one question.
        let ray_len = probe_len + AGENT_RADIUS * 0.9;
        let ray = time_calls(ROUNDS, CALLS, |_| {
            let hit = f
                .bridge
                .world_mut()
                .cast_ray_excluding(at, -DVec3::Y, ray_len, &exclude);
            std::hint::black_box(hit.is_some());
        });
        let ray_hit = f
            .bridge
            .world_mut()
            .cast_ray_excluding(at, -DVec3::Y, ray_len, &exclude)
            .is_some();
        // **THE BOX CONTROL, in this same world and this same broad phase**: the
        // identical probe dropped onto a building's ROOF, where it hits a
        // `Collider3D` box instead of a height-field tile. This is the only
        // honest way to say "the ground is dearer than the city at the
        // primitive", because it holds everything except the shape that answers
        // constant — the body count, the tree, the filter and the fact that
        // something is hit at all.
        let roof = (spec.side > 0).then(|| {
            let b = building_centre(0, spec.side)
                + DVec3::new(0.0, BUILDING_HALF.y + AGENT_HALF_HEIGHT + AGENT_RADIUS, 0.0);
            let hit = f
                .bridge
                .world_mut()
                .cast_shape(
                    &probe_shape,
                    b,
                    DQuat::IDENTITY,
                    -DVec3::Y,
                    probe_len,
                    &exclude,
                )
                .is_some();
            let us = time_calls(ROUNDS, CALLS, |_| {
                let h = f.bridge.world_mut().cast_shape(
                    &probe_shape,
                    b,
                    DQuat::IDENTITY,
                    -DVec3::Y,
                    probe_len,
                    &exclude,
                );
                std::hint::black_box(h.is_some());
            });
            (hit, us)
        });
        println!(
            "NPC1e / the mover's queries on {world_name} ({} bodies):",
            f.bodies
        );
        for (what, us) in [
            ("move_character (shipped: autostep + ground snap)", full),
            ("move_character, autostep OFF", a_off),
            ("move_character, ground snap OFF", s_off),
            ("move_character, both OFF", both_off),
            ("the section-9 ground-normal probe (one cast_shape)", probe),
            ("the same depth as a downward RAY", ray),
        ] {
            println!("  {what:<50} {us:>8.2} us");
        }
        println!(
            "  the character stands at y = {:.3} and its probe {} (the ray {}){}",
            at.y,
            if probe_hit {
                "HIT"
            } else {
                "MISSED EVERYTHING — the microseconds above are a broad phase \
                 saying \"no\", NOT a cast against this world's colliders"
            },
            if ray_hit { "hit" } else { "missed" },
            match roof {
                Some((true, us)) => format!(
                    "; the same probe onto a building ROOF, which really hits a \
                     box, costs {us:.3} us"
                ),
                Some((false, _)) => "; the roof control hit nothing".to_string(),
                None => String::new(),
            }
        );
        if let (true, Some((true, roof_us))) = (probe_hit, roof) {
            println!(
                "  HIT AGAINST HIT in one world: a height-field tile is {:.2}x a \
                 building box at the same primitive ({probe:.3} against {roof_us:.3} us)",
                probe / roof_us.max(1.0e-9)
            );
        }
        if probe_hit && ray_hit {
            println!(
                "  …and a RAY to the same depth answers the same ground {:.2}x \
                 cheaper than the swept sphere ({ray:.3} against {probe:.3} us)",
                probe / ray.max(1.0e-9)
            );
        }
        println!(
            "  a step is one sweep + one probe = {:.2} us; the probe is {:.1} % of it, \
             the autostep {:.1} %, the ground snap {:.1} %",
            full + probe,
            probe / (full + probe) * 100.0,
            (full - a_off) / (full + probe) * 100.0,
            (full - s_off) / (full + probe) * 100.0,
        );

        // SHAPES, not clocks.
        for (what, us) in [
            ("the sweep", full),
            ("the sweep without autostep", a_off),
            ("the sweep without the ground snap", s_off),
            ("the sweep with neither", both_off),
            ("the ground-normal probe", probe),
            ("the ray", ray),
        ] {
            assert!(
                us > 0.0,
                "{world_name}: {what} took no measurable time, so this row is a \
                 timer reading its own resolution"
            );
        }
        // **AND THE SHAPE THAT STOPS THE TABLE BEING READ AS A SURFACE
        // COMPARISON** (the NPC1e audit). A row whose probe hit nothing is a row
        // about an empty broad phase; it is kept, because "an airborne character
        // takes no ground probe" is itself the finding one function above, but it
        // is pinned so nobody can quote its microseconds beside a row that hit.
        assert_eq!(
            probe_hit,
            spec.ground,
            "{world_name}: the ground-normal probe {} from y = {:.3}. A world \
             with ground under the character must answer this probe and a world \
             without it must not — the day that stops being true, every ratio \
             taken off this table is a hit compared against a miss",
            if probe_hit { "HIT" } else { "MISSED" },
            at.y
        );
        // …and the ray reaches what the sphere reaches, which is the clause that
        // makes "a ray is N times cheaper" a statement about the SHAPE. A ray of
        // `probe_len` alone stops above the ground and answers `None`, and the
        // ledger's first cut priced exactly that.
        assert_eq!(
            ray_hit,
            probe_hit,
            "{world_name}: the ray {} where the swept sphere {} — they are not \
             asking the same question, so their microseconds are not comparable",
            if ray_hit { "hit" } else { "missed" },
            if probe_hit { "hit" } else { "missed" },
        );
        if let Some((hit, _)) = roof {
            assert!(
                hit,
                "{world_name}: the probe dropped onto a building's roof hit \
                 nothing, so the box control measures an empty broad phase and \
                 cannot be compared against the ground row"
            );
        }
    }
}

/// A world holding `tiles × tiles` heightfield tiles at `res` samples and nothing
/// else, with one probe capsule parked at the origin corner.
fn ground_only(tiles: i32, res: u32) -> PhysicsBridge3D {
    let mut world = EcsWorld::new();
    let mut data = TerrainData::new(res, MPS * f64::from(TILE_RES - 1) / f64::from(res - 1));
    let half = tiles / 2;
    for tz in -half..(tiles - half) {
        for tx in -half..(tiles - half) {
            data.author_tile((tx, tz), |_, _| GROUND_Y);
        }
    }
    let e = world.spawn_with_guid(Uuid::from_u128(0x7e44_0002), "Terrain", None);
    world.world_mut().entity_mut(e).insert(Terrain {
        meters_per_sample: MPS * f64::from(TILE_RES - 1) / f64::from(res - 1),
        tile_resolution: res,
        data,
        ..Terrain::default()
    });
    world.mark_dirty();
    world.propagate();
    let mut bridge = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
    bridge.sync_from_world(&world);
    bridge
}

/// **WAVE NPC1e — WHY A CAST AGAINST THE GROUND IS DEAR, AND WHAT IT SCALES
/// WITH.**
///
/// [`where_the_movers_queries_go`] prices the mover's section-9 probe against
/// the ground at a few microseconds, which is the NPC1c audit's *"four
/// heightfield tiles cost more than 1 849 buildings"* arriving at the primitive
/// it is about. What that does not say is **what the number is a function of**,
/// and a lever cannot be aimed without it: if it is the tile COUNT the answer is
/// that the fixture parks its character on a four-way corner and the island's
/// would touch one; if it is the RESOLUTION the cast is walking cells it does
/// not need, which is a statement about the shape rather than about the fixture.
///
/// One capsule, one downward sweep of the mover's own probe length, against a
/// growing grid of tiles at three resolutions. Printed; the assertions are
/// shapes.
///
/// # The slab is the control, and the NPC1e audit made it one
///
/// A flat 512 × 512 m **box** under the same capsule answers the same question
/// against the same tree, so it is the only thing on this table that isolates
/// the *shape*. It was measured and printed and never compared; the ratio is
/// now printed beside it, and it is the number *"the ground is dearer than the
/// city at the primitive"* is actually worth — **a single digit**, where the
/// figure the first ledger published (75×) came from dividing a cast that hit a
/// height field by one that hit nothing at all.
#[test]
fn what_a_cast_against_the_ground_costs() {
    const ROUNDS: u32 = 5;
    const CALLS: u32 = 400;
    let probe_shape = ColliderShape3D::Sphere {
        radius: AGENT_RADIUS * 0.9,
    };
    let probe_len = (AGENT_HALF_HEIGHT + AGENT_RADIUS) * 0.25 + 0.05 + AGENT_HALF_HEIGHT;
    // Above the corner the fixture's character stands on, by the capsule's own
    // centre height, so the sweep reaches the ground exactly as the step's does.
    let at = DVec3::new(0.0, GROUND_Y + AGENT_HALF_HEIGHT + AGENT_RADIUS, 0.0);
    let none = std::collections::BTreeSet::new();

    println!("NPC1e / one downward shape cast against the ground:");
    let mut rows: Vec<(i32, u32, f64)> = Vec::new();
    for res in [33u32, 129, 257] {
        for tiles in [1i32, 2, 4] {
            let mut b = ground_only(tiles, res);
            // ANTI-VACUITY, before the clock: a cast that hits nothing is a
            // measurement of a broad phase saying "no" and not of the ground.
            assert!(
                b.world_mut()
                    .cast_shape(
                        &probe_shape,
                        at,
                        DQuat::IDENTITY,
                        -DVec3::Y,
                        probe_len,
                        &none
                    )
                    .is_some(),
                "{tiles}x{tiles} at {res}: the probe hit nothing, so this world \
                 has no ground collider in it"
            );
            let us = time_calls(ROUNDS, CALLS, |_| {
                let hit = b.world_mut().cast_shape(
                    &probe_shape,
                    at,
                    DQuat::IDENTITY,
                    -DVec3::Y,
                    probe_len,
                    &none,
                );
                std::hint::black_box(hit.is_some());
            });
            let cells = (res as u64 - 1) * (res as u64 - 1) * (tiles as u64) * (tiles as u64);
            println!(
                "  {tiles}x{tiles} tiles at {res:>3} samples ({cells:>7} cells, \
                 {} bodies): {us:>8.3} us",
                b.world().body_ids().len()
            );
            rows.push((tiles, res, us));
        }
    }
    for (t, r, us) in &rows {
        assert!(
            *us > 0.0,
            "{t}x{t} at {r}: the cast took no measurable time"
        );
    }
    // THE SHAPE: at a fixed resolution the cost is a function of how many tiles
    // the query AABB overlaps, and a 4x4 grid whose corner the capsule sits on
    // overlaps four of them — never sixteen. A cast that walked every cell of
    // every tile would be quadratic in `tiles` and cubic across this table.
    // **AND THE SAME GROUND AS A BOX**, which is the control the ratio needs: if
    // a flat box under the same capsule answers the same question in a fraction
    // of the time, the cost is the SHAPE and not the fact that there is ground.
    //
    // **It is now compared as well as printed** (the NPC1e audit). This is the
    // only hit-against-hit ground/box ratio on the table, and the wave's ledger
    // published a 75× taken instead from a cast that hit nothing.
    let slab_us;
    {
        let mut world = EcsWorld::new();
        let e = world.spawn_with_guid(Uuid::from_u128(0x7e44_0003), "Slab", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(0.0, GROUND_Y - 8.0, 0.0);
        world.world_mut().entity_mut(e).insert((
            RigidBody3D {
                kind: BodyKind3D::Static,
                ..RigidBody3D::default()
            },
            Collider3D {
                shape_kind: ColliderShape3DKind::Box,
                half_extents: Vec3d::new(256.0, 8.0, 256.0),
                ..Collider3D::default()
            },
            t,
        ));
        world.mark_dirty();
        world.propagate();
        let mut b = PhysicsBridge3D::new(DVec3::new(0.0, -9.81, 0.0));
        b.sync_from_world(&world);
        assert!(
            b.world_mut()
                .cast_shape(
                    &probe_shape,
                    at,
                    DQuat::IDENTITY,
                    -DVec3::Y,
                    probe_len,
                    &none
                )
                .is_some(),
            "the slab world's probe hit nothing"
        );
        let us = time_calls(ROUNDS, CALLS, |_| {
            let hit = b.world_mut().cast_shape(
                &probe_shape,
                at,
                DQuat::IDENTITY,
                -DVec3::Y,
                probe_len,
                &none,
            );
            std::hint::black_box(hit.is_some());
        });
        println!("  one 512 x 512 m BOX under the same capsule: {us:>8.3} us");
        slab_us = us;
    }
    let one = rows
        .iter()
        .find(|(t, r, _)| *t == 1 && *r == 257)
        .unwrap()
        .2;
    let four = rows
        .iter()
        .find(|(t, r, _)| *t == 4 && *r == 257)
        .unwrap()
        .2;
    println!(
        "  16 tiles cost {:.2}x one tile at 257 samples (four of them are under the query)",
        four / one
    );
    assert!(
        four < one * 16.0,
        "a cast against a 4x4 grid cost {four:.3} us against {one:.3} for one \
         tile — that is 16x or worse, so the cast is walking every tile in the \
         world rather than the ones its AABB touches"
    );
    // **THE RATIO THAT IS ACTUALLY ABOUT THE SHAPE** (the NPC1e audit): one
    // 257-sample tile against one flat box, both HIT by the same probe, in two
    // one-body worlds. Single digits — and `where_the_movers_queries_go`'s own
    // in-fixture roof control says the same thing at the mover's own station.
    println!(
        "  a height-field TILE is {:.2}x a BOX at the same primitive, both hit \
         ({one:.3} against {slab_us:.3} us) — this is the shape's ratio, and it \
         is NOT the 75x a hit-against-a-miss produces",
        one / slab_us.max(1.0e-9)
    );
    assert!(
        slab_us > 0.0,
        "the slab control took no measurable time, so the ratio above is a timer \
         reading its own resolution"
    );
}

/// **WAVE NPC1e'S LEVER, AS A COUNT** — a walking character asks the world once
/// per step; a sprinting one asks twice.
///
/// `movement::reads_the_ground_normal` is the whole of the lever: section 9's
/// downward sweep produces a number whose only reader is `slide_friction`, so a
/// character that is neither sliding nor holding sprint no longer pays for it.
/// [`where_the_movers_queries_go`] prices that probe at 18 % of a step over the
/// island's own tile resolution; this arm holds the **shape**, which is what
/// survives a busy runner: `PhysicsWorld3D::queries` is bumped once per ask in
/// the one door every query goes through, so "the mover asks twice and now asks
/// once" is a subtraction rather than a stopwatch.
///
/// Mutation: making the predicate `true` puts the walker back at 2.
#[test]
fn a_walking_character_asks_the_world_once_and_a_sprinting_one_twice() {
    let mut f = fixture(Spec::new().movers(1));
    let guid = f.movers[0];
    // Settle first: the spawn step takes a settle sweep of its own, and the
    // claim is about the steady state.
    for _ in 0..40 {
        drive(&mut f);
        step_character_movement(&mut f.world, &mut f.bridge, DT);
    }

    let walking = {
        let before = f.bridge.world().queries();
        drive(&mut f);
        let out = step_character_movement(&mut f.world, &mut f.bridge, DT);
        assert!(
            out.iter().any(|o| o.guid == guid && o.grounded),
            "the walker is not on the ground, so this arm measured a fall"
        );
        f.bridge.world().queries() - before
    };

    // The same character, holding sprint — which is the only door into `Slide`
    // and therefore the only way its ground normal can ever be read.
    {
        let e = f.world.entity_of(guid).expect("the mover exists");
        let mut cm = f
            .world
            .world_mut()
            .get_mut::<CharacterMovement>(e)
            .expect("it has a movement component");
        cm.runtime.want_sprint = true;
    }
    let sprinting = {
        let before = f.bridge.world().queries();
        drive(&mut f);
        let out = step_character_movement(&mut f.world, &mut f.bridge, DT);
        assert!(
            out.iter().any(|o| o.guid == guid && o.grounded),
            "the sprinter is not on the ground"
        );
        f.bridge.world().queries() - before
    };

    println!(
        "NPC1e / queries a character a step: walking {walking}, sprinting \
         {sprinting}"
    );
    assert_eq!(
        walking, 1,
        "a walking character asked the world {walking} times — one sweep is all \
         it needs, and anything above it is a probe nothing reads"
    );
    assert_eq!(
        sprinting, 2,
        "a sprinting character asked the world {sprinting} times — it must keep \
         its ground-normal probe, because sprint is the one door into `Slide` \
         and `Slide` is the one reader of the normal"
    );
    // The ground normal a sprinter gets is the surface's, not the default: an
    // arm that only counted queries would pass on a probe whose answer was
    // thrown away.
    let e = f.world.entity_of(guid).unwrap();
    let n = f
        .world
        .world()
        .get::<CharacterMovement>(e)
        .unwrap()
        .runtime
        .ground_normal;
    assert!(
        (n.y - 1.0).abs() < 1.0e-6 && n.x.abs() < 1.0e-6,
        "the sprinter's probe answered {n:?} over flat ground"
    );
}

/// **ISLAND WAVE NPC1e: THE MOVER'S COST IS FLAT IN A CROWD'S *SIZE* AND NOT IN
/// ITS *DENSITY*, and that is the arc's closing law.**
///
/// [`a_controller_pays_for_the_body_in_front_of_it_not_for_the_crowds_size`] is
/// the NPC1c audit's answer to "does a crowd standing around make the mover
/// dearer", and it reads **1.24x, saturating between 64 and 256** — which the
/// audit summarised as *"a controller pays for the thing it has to slide around,
/// and a second capsule behind the first costs it nothing."* That arm is right
/// about the variable it sweeps: it puts **one** bystander two metres in front
/// of each mover and then adds more of them **further away**, so what it grows
/// is the crowd's *extent*, not what any one controller has to get past.
///
/// **The island's certification row is a different variable.** With the hero
/// standing where the ladder's `Full` radius holds the most of the town —
/// **191 residents inside a 32 m ring** — `character move` reads **84.100 ms for
/// 170 controllers, 0.494 ms each**, against **0.171 ms** at thirty-three
/// controllers on the same island. Three times the per-character cost, on a
/// station the NPC1c audit's own sweep held flat over a 64x range in N. The
/// difference is not N. It is how many capsules are inside one controller's
/// sweep.
///
/// So this sweeps the density instead: **32 controllers**, and 0 / 64 / 192 /
/// 384 bystanders **packed into the twenty metres the controllers themselves
/// stand in** rather than spread over the lattice. Same fixture, same world,
/// same movers — one variable changed, and it is the one the island moved.
///
/// **The clocks are printed and the SHAPE is asserted**, this file's own rule:
/// what is held is that the packed arrangement is **dearer than the spread one
/// at the same population**, which is a fact about the program (a query pays for
/// what its swept AABB overlaps) and not about the machine. An engine whose
/// mover really were flat in density would fail it — and so would this file's
/// own reading of its sibling arm.
#[test]
fn the_movers_cost_is_flat_in_a_crowds_size_and_not_in_its_density() {
    const MOVERS: usize = 32;
    /// A metre and a half of pitch is shoulder to shoulder for a 0.3 m capsule
    /// with room to walk — a pavement at rush hour, not a mosh pit.
    const PITCH_M: f64 = 1.5;

    let mut rows: Vec<(usize, Measured, Measured)> = Vec::new();
    for b in [0usize, 64, 192, 384] {
        let spread = measure(Spec::new().movers(MOVERS).bystanders(b));
        let packed = measure(Spec::new().movers(MOVERS).bystanders(b).packed(PITCH_M));
        rows.push((b, spread, packed));
    }

    println!(
        "NPC1e / {MOVERS} controllers, N bystanders SPREAD over the 7 m lattice \
         against PACKED at {PITCH_M} m:"
    );
    for (b, spread, packed) in &rows {
        println!(
            "  {b:>3} bystanders   spread {:>7.2} us/character   packed {:>7.2}   ({:.2}x)",
            spread.us_per_character(MOVERS),
            packed.us_per_character(MOVERS),
            packed.ms / spread.ms.max(1.0e-9),
        );
    }
    for (b, spread, packed) in &rows {
        // **The two anti-vacuity bounds are different on purpose**, and the
        // difference is the phenomenon rather than a concession: a body pressed
        // into a rush hour walks *less* — at 384 packed capsules the controller
        // covers **0.30 m** where the spread arrangement takes it 1.56 — because
        // the crowd is in its way. That is what a dense crowd does to a
        // collide-and-slide mover, and it is why the packed row is dearer per
        // character while travelling further from nowhere. Both numbers are
        // printed above; what each clause rules out is "the mover refused every
        // step", which is a body that moved *nothing at all*.
        assert!(
            spread.travelled_m > 1.0,
            "{b} bystanders: a controller moved {:.3} m spread, so this row \
             measured a mover that did nothing",
            spread.travelled_m
        );
        assert!(
            packed.travelled_m > 0.1,
            "{b} bystanders: a controller moved {:.3} m packed — not slowed by a \
             crowd but stopped dead, which is a refusal rather than a price",
            packed.travelled_m
        );
        assert_eq!(
            spread.bodies, packed.bodies,
            "{b} bystanders: the two arrangements are not the same population, \
             so this row compares two worlds rather than two densities"
        );
    }
    // THE SHAPE. At zero bystanders the two arrangements ARE the same world, so
    // they must agree; past it, packing the same population must cost more.
    let (_, s0, p0) = &rows[0];
    println!(
        "  at 0 bystanders the two arrangements are one world: {:.2} against \
         {:.2} us/character",
        s0.us_per_character(MOVERS),
        p0.us_per_character(MOVERS)
    );
    let (b, spread, packed) = rows.last().unwrap();
    assert!(
        packed.ms > spread.ms,
        "{b} bystanders packed into {PITCH_M} m cost {:.4} ms against {:.4} \
         spread over the lattice — the mover would then be flat in DENSITY, \
         and the island's certification row (0.494 ms a character at 170 \
         controllers among 191 residents inside 32 m, against 0.171 at 33) \
         says it is not",
        packed.ms,
        spread.ms
    );
}
