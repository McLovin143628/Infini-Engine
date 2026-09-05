//! **A character stands ON the footway and steps down at the kerb** (wave
//! ROAD1b, clause 2) — the same walk on both hosts, traced by the height of the
//! character's own feet.
//!
//! # What this is for
//!
//! Wave ROAD1 drew a 150 mm kerb and 2 m of concrete beside every settlement
//! street and gave neither a collider. The ROAD1 audit measured the consequence
//! over the island fixture's 5 344 walkable footway triangles: the concrete is
//! drawn **p50 0.1775 m, area-weighted mean 0.1915 m above the ground an agent
//! stands on** (carried 19), so an agent walks shin-deep *inside* the slab and a
//! car crosses a kerb as though it were paint.
//!
//! `inf_physics::d3::kerb` makes the slab solid. This is the arm that says a
//! **character** can tell: it walks one across a street's kerb and reads the
//! height of its feet, which is derived (`position − half_height − radius`, the
//! spelling `movement.rs` uses at four call sites) rather than stored.
//!
//! # Why it is a whole file and not a case in `movement_parity`
//!
//! That fixture is three boxes and a hero, deliberately: it is the *mirror* arm
//! for the two hosts' movement code and every byte of its trace is pinned. A
//! footway needs `PcgVolume` blocks with residents, which would give it streets,
//! parked cars, a crowd and a traffic derivation — a different fixture with a
//! different claim. This file borrows its shape (two hosts, one script, one
//! byte record) and nothing else.
//!
//! # PIE == shipping
//!
//! Both traces are compared byte for byte, so a footway that were solid in one
//! host and paint in the other fails here rather than in a screenshot. The
//! collider is described by `PhysicsBridge3D::sync_from_world_sim`, which both
//! hosts call — but they call it from their own step functions, and "both call
//! the same function" is exactly the claim a mirror arm exists to check rather
//! than assert.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};
use uuid::Uuid;

use inf_ecs::components::{
    BodyKind3D, CharacterController3D, CharacterMovement, Collider3D, ColliderShape3DKind,
    PcgVolume, ResidentSlot, RigidBody3D, SlotRole, StreamingSource, Transform,
};
use inf_ecs::math::{Vec2d, Vec3d};
use inf_ecs::EcsWorld;
use inf_editor_core::scene::SceneDoc;
use inf_editor_core::simulate::{SimInput, SimSession};
use inf_player::runtime_sim::{RuntimeInput, RuntimeSim};

const HZ: f64 = 60.0;
const GRAVITY: DVec2 = DVec2::new(0.0, -9.81);
const STEPS: u32 = 150;
const RADIUS: f64 = 0.3;

/// A 2×2 grid of 80 m blocks on a 100 m pitch — two 20 m street reserves, the
/// shape `inf_editor_core::settlement` plans for a city and the same one
/// `inf-physics`' own `traffic_3d` uses.
const PITCH: f64 = 100.0;
const STREET: f64 = 20.0;

const HERO: Uuid = Uuid::from_u128(0x524f_4144_1b00);
const GROUND: Uuid = Uuid::from_u128(0x524f_4144_1b01);

/// The street this walk crosses: the one along X, at `z = PITCH / 2`.
fn street_z() -> f64 {
    PITCH * 0.5
}

/// Where the hero starts — on the carriageway's crown, a little west of the
/// crossing so it is not standing in a junction.
fn start_xz() -> DVec2 {
    DVec2::new(PITCH * 0.5 - 30.0, street_z())
}

fn spawn_world(world: &mut EcsWorld) {
    // The ground: one big static box at y = 0, so "the ground" is a number this
    // file can state rather than a heightfield it has to sample.
    let e = world.spawn_with_guid(GROUND, "Ground", None);
    let mut t = Transform::IDENTITY;
    t.translation = Vec3d::new(PITCH * 0.5, -0.5, PITCH * 0.5);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Static,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(300.0, 0.5, 300.0),
            ..Default::default()
        },
    ));

    let half = (PITCH - STREET) * 0.5;
    for row in 0..2i32 {
        for col in 0..2i32 {
            let c = DVec2::new(f64::from(col) * PITCH, f64::from(row) * PITCH);
            let guid = Uuid::from_u64_pair(0x524f_4144_1b10, (row as u64) << 32 | col as u64);
            let e = world.spawn_with_guid(guid, "block", None);
            let mut t = Transform::IDENTITY;
            t.translation = Vec3d::new(c.x, 0.0, c.y);
            let mut v = PcgVolume {
                extent: Vec2d::new(half, half),
                ..Default::default()
            };
            // One resident, because `volume_sites` reads volumes that offer one
            // — the condition `streets_of`'s own doc records.
            v.residents = vec![ResidentSlot {
                role: SlotRole::Home,
                at: DVec3::new(c.x, 0.0, c.y),
                room: 0,
                building: 0,
                floor: 0,
                index: 0,
                node: 0,
                posture: inf_ecs::components::SlotPosture::Stand,
                shift: inf_ecs::components::SlotShift::Day,
                face: DVec3::ZERO,
            }];
            world.world_mut().entity_mut(e).insert((t, v));
        }
    }

    // The hero, facing +Z so a forward stick walks it across the kerb.
    let cm = CharacterMovement {
        player_controlled: true,
        ..Default::default()
    };
    let mut t = Transform::IDENTITY;
    let s = start_xz();
    t.translation = Vec3d::new(s.x, cm.stand_half_height_m + RADIUS, s.y);
    let e = world.spawn_with_guid(HERO, "Hero", None);
    world.world_mut().entity_mut(e).insert((
        t,
        RigidBody3D {
            kind: BodyKind3D::Kinematic,
            ..Default::default()
        },
        Collider3D {
            shape_kind: ColliderShape3DKind::Capsule,
            half_extents: Vec3d::new(RADIUS, cm.stand_half_height_m, RADIUS),
            radius: RADIUS,
            ..Default::default()
        },
        CharacterController3D::default(),
        cm,
        // The band's anchor: without one, every derived collider in the level —
        // the footway included — is out of band and the walk is over open
        // ground, which is the arm passing for the wrong reason.
        StreamingSource { radius_m: 256.0 },
    ));
    world.mark_dirty();
    world.propagate();
}

/// Walk forward the whole way. One axis, so the trace is about the kerb rather
/// than about a gait ladder.
fn script(_i: u32) -> (Vec<&'static str>, BTreeMap<String, f32>) {
    let mut axes = BTreeMap::new();
    axes.insert("move_y".to_string(), 1.0f32);
    (Vec::new(), axes)
}

/// **The height of the character's own feet**, world metres — `position` less
/// the capsule's half height and its radius, which is the spelling
/// `inf_physics::d3::movement` uses wherever it needs one.
fn feet_of(world: &EcsWorld) -> (f64, DVec2) {
    let e = world.entity_of(HERO).expect("the hero");
    let t = world
        .world()
        .get::<Transform>(e)
        .expect("a transform")
        .translation
        .to_dvec3();
    let cm = world
        .world()
        .get::<CharacterMovement>(e)
        .expect("a character");
    (
        t.y - cm.half_height_for(cm.mode) - RADIUS,
        DVec2::new(t.x, t.z),
    )
}

/// One step, as bytes — the feet and the plan position, which is what this arm
/// is about. Deliberately narrower than `movement_parity`'s 38-float record:
/// that file pins the movement mirror, this one pins the ground under it.
fn step_bytes(world: &EcsWorld) -> Vec<u8> {
    let (feet, xz) = feet_of(world);
    let mut out = Vec::with_capacity(24);
    for v in [feet, xz.x, xz.y] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

fn player_trace() -> Vec<Vec<u8>> {
    let mut world = EcsWorld::new();
    spawn_world(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), GRAVITY, HZ);
    (0..STEPS)
        .map(|i| {
            let (held, axes) = script(i);
            sim.step_once(RuntimeInput::with_down(held).with_axes(axes));
            step_bytes(sim.world())
        })
        .collect()
}

fn editor_trace() -> Vec<Vec<u8>> {
    let mut doc = SceneDoc::new();
    spawn_world(doc.world_mut());
    let mut session = SimSession::enter(&mut doc, Vec::new(), GRAVITY, HZ);
    let out = (0..STEPS)
        .map(|i| {
            let (held, axes) = script(i);
            session.step_once(&mut doc, SimInput::with_down(held).with_axes(axes));
            step_bytes(doc.world())
        })
        .collect();
    session.exit(&mut doc);
    out
}

/// The feet height and the plan position at each step, decoded.
fn decode(trace: &[Vec<u8>]) -> Vec<(f64, f64, f64)> {
    trace
        .iter()
        .map(|r| {
            let f = |k: usize| f64::from_le_bytes(r[k * 8..k * 8 + 8].try_into().unwrap());
            (f(0), f(1), f(2))
        })
        .collect()
}

/// **THE ARM** — the character walks off the carriageway, up the kerb, and onto
/// the footway, and its feet say so.
#[test]
fn a_character_steps_up_onto_the_footway_at_the_kerb() {
    let trace = decode(&player_trace());
    assert_eq!(trace.len() as u32, STEPS);

    let kerb = inf_ecs::traffic::street_kerb_offset_m(STREET);
    let back = kerb + inf_ecs::traffic::KERB_WIDTH_M + inf_ecs::society::PAVEMENT_M;
    let z0 = street_z();

    // The two populations: steps taken over the carriageway, and steps taken
    // over the footway slab. Measured by WHERE THE CHARACTER IS, not by when.
    let mut on_road: Vec<f64> = Vec::new();
    let mut on_slab: Vec<f64> = Vec::new();
    for (feet, _, z) in &trace {
        let across = (z - z0).abs();
        // A margin of one capsule radius each side of the kerb face, so the
        // step itself — where the capsule is touching both — belongs to neither.
        if across < kerb - RADIUS {
            on_road.push(*feet);
        } else if across > kerb + RADIUS && across < back - RADIUS {
            on_slab.push(*feet);
        }
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    println!(
        "ROAD1b KERB WALK | {} step(s) on the carriageway (feet {:.4} m), {} on the footway (feet {:.4} m); kerb at {kerb:.3} m, slab to {back:.3} m; walked {:.2} m across",
        on_road.len(),
        mean(&on_road),
        on_slab.len(),
        mean(&on_slab),
        (trace.last().unwrap().2 - z0).abs()
    );
    assert!(
        !on_road.is_empty() && !on_slab.is_empty(),
        "the walk never crossed the kerb: it covered {:.2} m across a street \
         whose footway starts at {kerb:.3} m",
        (trace.last().unwrap().2 - z0).abs()
    );
    // On the carriageway the ground is the ground, to the controller's own
    // skin: `move_shape` keeps a small offset between a capsule and whatever it
    // is standing on, and 0.0201 m is what that is here. The claim below is the
    // DIFFERENCE, which the skin cancels out of exactly.
    assert!(
        mean(&on_road).abs() < 0.05,
        "the character's feet are at {:.4} m over the carriageway, and the \
         ground there is at 0",
        mean(&on_road)
    );
    // On the footway they are one kerb higher — which is the whole clause.
    let rise = mean(&on_slab) - mean(&on_road);
    assert!(
        (rise - inf_ecs::traffic::KERB_HEIGHT_M).abs() < 0.02,
        "the character's feet rise {rise:.4} m crossing the kerb and the kerb is \
         {} m high — before wave ROAD1b it rose by nothing at all, because the \
         concrete had no collider and the character walked through it",
        inf_ecs::traffic::KERB_HEIGHT_M
    );
    // …and it kept walking. A kerb that stopped the character dead would
    // satisfy the rise and be a wall.
    assert!(
        on_slab.len() >= 10,
        "the character reached the footway and took {} step(s) on it — the \
         150 mm upstand is being climbed, not walked into",
        on_slab.len()
    );
}

/// **PIE == shipping**: the editor's Simulate and the shipped player walk the
/// same character over the same kerb, byte for byte.
#[test]
fn both_hosts_walk_the_same_kerb_byte_for_byte() {
    let player = player_trace();
    let editor = editor_trace();
    assert_eq!(player.len(), editor.len());
    // Anti-vacuity: a trace that never moved would compare equal to itself.
    let first = decode(&player[..1]);
    let last = decode(&player[player.len() - 1..]);
    assert!(
        (last[0].2 - first[0].2).abs() > 1.0,
        "the character did not walk: {:?} -> {:?}",
        first[0],
        last[0]
    );
    for (i, (p, e)) in player.iter().zip(editor.iter()).enumerate() {
        assert_eq!(
            p,
            e,
            "step {i}: the shipped player has the character at {:?} and the \
             editor's Simulate at {:?}",
            decode(std::slice::from_ref(p))[0],
            decode(std::slice::from_ref(e))[0]
        );
    }
}

/// **The footway is affordable** — the band admits a bounded number of slabs,
/// and the derivation is a cache.
#[test]
fn the_footway_costs_a_bounded_number_of_boxes() {
    let mut world = EcsWorld::new();
    spawn_world(&mut world);
    let mut sim = RuntimeSim::new(world, Vec::new(), GRAVITY, HZ);
    sim.step_once(RuntimeInput::default());
    let first = sim.bridge3d().kerb_collider_audit();
    for _ in 0..30 {
        sim.step_once(RuntimeInput::default());
    }
    let settled = sim.bridge3d().kerb_collider_audit();
    println!(
        "ROAD1b KERB COST | {} slab(s) described, {} culled, over {} street(s); settled at {} described",
        first.described, first.culled, first.streets, settled.described
    );
    assert!(
        first.streets > 0,
        "the fixture derived no streets, so the audit is vacuous"
    );
    assert!(first.described > 0, "no footway slab was described at all");
    // **The ceiling, read by name.** Two sides of two streets inside a 64 m
    // band, at `KERB_SLAB_M` a chunk, is a couple of dozen boxes; this is the
    // bound a settlement's footways may cost the step, and it is here rather
    // than in the prose so a wider band or a shorter chunk has to move it
    // deliberately.
    assert!(
        first.described <= FOOTWAY_SLABS_CEILING,
        "the footways described {} boxes against a ceiling of \
         {FOOTWAY_SLABS_CEILING}",
        first.described
    );
    // A settled level re-offers what it has rather than rebuilding: the stamp
    // is doing its job, which is what keeps this off the per-step cost.
    assert_eq!(
        settled.described, first.described,
        "the slab set moved on a level where nothing did"
    );
    assert_eq!(settled.culled, 0, "a settled pass re-tiered its slabs");
}

/// **How many footway slab colliders a settlement may cost one fixed step.**
///
/// Two streets crossing, both sides, chunked at
/// `inf_physics::d3::kerb::KERB_SLAB_M` (32 m) inside
/// `inf_ecs::DEFAULT_COLLIDER_NEAR_M` (64 m): four half-streets of two chunks
/// each, twice over, is sixteen, and the band's tier is measured against a
/// slab's whole box rather than its centre so the ones straddling the boundary
/// come too. Forty is past that with room and far short of the 2 200 the
/// island's 35 km of street would be unbanded.
///
/// It is a **ratchet**: this may only ever decrease.
const FOOTWAY_SLABS_CEILING: u32 = 40;
