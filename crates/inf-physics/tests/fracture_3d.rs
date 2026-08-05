//! **P22.3: runtime destruction — the swap, the solve, the budget.**
//!
//! These tests are about the **rule and the bridge**, not about the chunking. The
//! Voronoi cut, the watertight hulls and the adjacency graph all have their own
//! suite in `inf-mesh`. What is unfalsifiable without these is that damage spends
//! energy the way the strength memo says, that support really is decided by
//! contact with static geometry rather than assumed, that the intact→chunks swap
//! is atomic (never both, never neither), and that the collapse is a function of
//! the content and of nothing else.
//!
//! The fixtures are **hand-built** `FractureAsset`s — stacks and pairs of unit
//! cubes with exactly known shared faces — so a claim about energy bookkeeping is
//! arithmetic rather than a measurement of the Voronoi cut. One test
//! (`a_real_cook_asset_measures_its_own_bond_areas`) runs the whole thing on real
//! `fracture_mesh` output, which is what keeps the hand-built fixtures honest.

use std::collections::BTreeMap;
use std::sync::Arc;

use glam::{DAffine3, DVec3};
use inf_ecs::components::{
    AlwaysLoaded, Collider3D, ColliderShape3DKind, Destructible, MeshRef, Terrain, Transform,
};
use inf_ecs::{EcsWorld, Vec3d};
use inf_mesh::{Aabb, ChunkSection, FractureAsset, FractureChunk, MeshVertex};
use inf_physics::d3::fracture::{fracture_chunk_guid, resolve_fracture_states};
use inf_physics::d3::{BodyKind3D, DebrisBudget, DestructOutcome, FractureState, CRACK_OPENING_M};
use inf_physics::PhysicsBridge3D;
use inf_terrain::TerrainData;
use uuid::Uuid;

const WALL: Uuid = Uuid::from_u128(0x2203_1001);
const WALL_B: Uuid = Uuid::from_u128(0x2203_1002);
const GROUND: Uuid = Uuid::from_u128(0x2203_1003);
const PLINTH: Uuid = Uuid::from_u128(0x2203_1004);
const PLINTH_B: Uuid = Uuid::from_u128(0x2203_1005);
const MESH: Uuid = Uuid::from_u128(0x2203_1FFF);

const DT: f64 = 1.0 / 60.0;

fn gravity() -> DVec3 {
    DVec3::new(0.0, -9.81, 0.0)
}

// ── fixtures ────────────────────────────────────────────────────────────────

/// One axis-aligned unit-ish box chunk centred at `centre` with half-extents
/// `half`, as both render triangles and a hull point set.
fn box_chunk(centre: DVec3, half: DVec3, neighbors: Vec<u32>) -> FractureChunk {
    let mut hull = Vec::new();
    for sx in [-1.0, 1.0] {
        for sy in [-1.0, 1.0] {
            for sz in [-1.0, 1.0] {
                hull.push([
                    centre.x + sx * half.x,
                    centre.y + sy * half.y,
                    centre.z + sz * half.z,
                ]);
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
    // Index order matches the `box_chunk` corner order: bit 2 = x, 1 = y, 0 = z.
    #[rustfmt::skip]
    let indices: Vec<u32> = vec![
        // -x (0,1,2,3) and +x (4,5,6,7)
        0, 1, 3, 0, 3, 2,
        4, 7, 5, 4, 6, 7,
        // -y (0,1,4,5) and +y (2,3,6,7)
        0, 4, 5, 0, 5, 1,
        2, 3, 7, 2, 7, 6,
        // -z (0,2,4,6) and +z (1,3,5,7)
        0, 2, 6, 0, 6, 4,
        1, 5, 7, 1, 7, 3,
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
        volume_m3: 8.0 * half.x * half.y * half.z,
        center_of_mass: centre.to_array(),
        neighbors,
    }
}

/// A tower of `n` unit cubes stacked along +Y, cube `i` centred at
/// `y = i + 0.5`. Chunk `i` neighbours `i ± 1`, and each shared face is exactly
/// 1 m².
fn tower(n: u32) -> Arc<FractureAsset> {
    let chunks: Vec<FractureChunk> = (0..n)
        .map(|i| {
            let mut nb = Vec::new();
            if i > 0 {
                nb.push(i - 1);
            }
            if i + 1 < n {
                nb.push(i + 1);
            }
            box_chunk(DVec3::new(0.0, i as f64 + 0.5, 0.0), DVec3::splat(0.5), nb)
        })
        .collect();
    Arc::new(FractureAsset {
        schema_version: FractureAsset::CURRENT_VERSION,
        source_mesh: *MESH.as_bytes(),
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

/// The default material: 5 MPa masonry/concrete at 2400 kg/m³.
fn concrete() -> Destructible {
    Destructible::default()
}

/// The energy, joules, one of the tower's 1 m² concrete bonds costs — **as the
/// implementation measures it**, so an assertion about what a blow absorbed is an
/// exact equality rather than a tolerance.
///
/// It is not spelled `strength × 1.0 × CRACK_OPENING_M` because the shared-face
/// area is *measured* from the chunk geometry, and a measured square metre is
/// `1.0` to within a couple of ULPs rather than exactly `1.0` — so the
/// hand-written arithmetic and the implementation disagree in the last bit.
/// `the_measured_bond_matches_the_memos_arithmetic` is what pins the two together
/// to a real tolerance; everything else compares against this.
fn one_bond_j() -> f64 {
    FractureState::new(tower(2), concrete(), DAffine3::IDENTITY, false).bond_energy_j(0, 1)
}

/// **The arithmetic claim, at a real tolerance.** A shared face of exactly one
/// square metre must cost `strength × 1 m² × CRACK_OPENING_M` joules — the memo's
/// §4.2 contract times the one constant P22.3 adds. If the measurement drifted
/// (or fell through to the `min(volume)^(2/3)` fallback) this is what says so.
#[test]
fn the_measured_bond_matches_the_memos_arithmetic() {
    let want = concrete().strength * 1.0 * CRACK_OPENING_M;
    let got = one_bond_j();
    assert!(
        (got - want).abs() < want * 1e-12,
        "a 1 m² concrete bond measured {got} J, not {want} J"
    );
    // …and the ground bond of a 1 m³ chunk is the same number, because its
    // characteristic cross-section is `1^(2/3) = 1 m²`.
    let state = FractureState::new(tower(2), concrete(), DAffine3::IDENTITY, false);
    assert!((state.ground_bond_j(0) - want).abs() < want * 1e-12);
}

/// A world with a flat terrain at `y = 0` (so the tower's bottom chunk really is
/// standing on static geometry) and one destructible actor at `at`.
fn wall_world(entities: &[(Uuid, DVec3, bool)]) -> EcsWorld {
    let mut w = EcsWorld::new();
    let g = w.spawn_with_guid(GROUND, "Ground", None);
    let mut data = TerrainData::new(5, 4.0);
    data.author_tile((-1, -1), |_, _| 0.0);
    data.author_tile((0, 0), |_, _| 0.0);
    data.author_tile((-1, 0), |_, _| 0.0);
    data.author_tile((0, -1), |_, _| 0.0);
    w.world_mut().entity_mut(g).insert(Terrain {
        meters_per_sample: 4.0,
        tile_resolution: 5,
        data,
        ..Terrain::default()
    });
    for &(guid, at, anchored) in entities {
        let e = w.spawn_with_guid(guid, "Wall", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::from_dvec3(at);
        w.world_mut().entity_mut(e).insert(t);
        w.world_mut().entity_mut(e).insert(concrete());
        w.world_mut().entity_mut(e).insert(MeshRef {
            asset: Some(MESH),
            ..Default::default()
        });
        // The authored intact collider — what the swap replaces.
        w.world_mut().entity_mut(e).insert(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(0.5, 2.0, 0.5),
            offset: Vec3d::new(0.0, 2.0, 0.0),
            ..Default::default()
        });
        if anchored {
            w.world_mut().entity_mut(e).insert(AlwaysLoaded);
        }
    }
    w.mark_dirty();
    w.propagate();
    w
}

/// Seed states for every destructible in `world` from one shared asset.
fn seed(world: &EcsWorld, asset: Arc<FractureAsset>) -> BTreeMap<Uuid, FractureState> {
    resolve_fracture_states(world, |mesh| (mesh == MESH).then(|| asset.clone()))
}

/// Sync + step + write-back + advance, the way a host's fixed step does.
fn tick(
    bridge: &mut PhysicsBridge3D,
    world: &EcsWorld,
    fractures: &mut BTreeMap<Uuid, FractureState>,
) -> inf_physics::d3::FractureAudit {
    bridge.sync_from_world_sim(world, &BTreeMap::new(), fractures);
    bridge.step(DT);
    bridge.write_back_fractures(fractures);
    let (audit, _) = bridge.step_fractures(fractures, DT, DebrisBudget::default());
    audit
}

// ── refusals are values ─────────────────────────────────────────────────────

/// Every refusal answers `0.0` joules and names itself on the log, and none of
/// them mutates a thing.
#[test]
fn every_refusal_is_a_value() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut fractures = seed(&world, tower(4));
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    let before = fractures[&WALL].bits();

    // 1. Degenerate energy — before the permission check, because a NaN is a bug
    //    in the caller whichever way the gate is set.
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let r = bridge.runtime_destruct(&mut fractures, WALL, true, bad);
        assert_eq!(r.outcome, DestructOutcome::Invalid, "energy {bad}");
        assert_eq!(r.energy_absorbed_j, 0.0);
        assert_eq!(r.detached, 0);
    }
    // 2. The gate off.
    let r = bridge.runtime_destruct(&mut fractures, WALL, false, 1.0e9);
    assert_eq!(r.outcome, DestructOutcome::NotPermitted);
    assert_eq!(r.energy_absorbed_j, 0.0);
    // 3. No fracture state (an entity that was never seeded — what a BeginPlay
    //    handler sees).
    let r = bridge.runtime_destruct(&mut fractures, WALL_B, true, 1.0e9);
    assert_eq!(r.outcome, DestructOutcome::NoFracture);
    assert_eq!(r.energy_absorbed_j, 0.0);

    // Nothing moved.
    assert_eq!(
        fractures[&WALL].bits(),
        before,
        "a refusal mutated the state"
    );
    assert!(fractures[&WALL].is_intact());

    // Every refusal has a distinct, non-empty message; `Applied` has none.
    let msgs: Vec<String> = [
        DestructOutcome::NoDestructible,
        DestructOutcome::NotPermitted,
        DestructOutcome::NoFracture,
        DestructOutcome::Invalid,
    ]
    .iter()
    .map(|o| {
        o.refusal("destruct::apply_damage")
            .expect("a refusal message")
    })
    .collect();
    let mut sorted = msgs.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(
        sorted.len(),
        msgs.len(),
        "two refusals read the same: {msgs:?}"
    );
    assert!(DestructOutcome::Applied.refusal("x").is_none());

    // The positive control: with the gate on and a real energy, it DOES break.
    let r = bridge.runtime_destruct(&mut fractures, WALL, true, 1.0e9);
    assert!(r.outcome.applied());
    assert!(r.detached > 0, "the positive control broke nothing");
}

// ── the energy model ────────────────────────────────────────────────────────

/// **The bookkeeping, exactly.** A four-cube tower's top chunk is held by one
/// 1 m² bond. Spend a fraction under that and nothing comes off and nothing is
/// absorbed; spend exactly it and precisely one chunk detaches, absorbing
/// precisely that.
///
/// The BASE chunk also has exactly one neighbour, so without the ground bond it
/// would tie with the top and lose the tie-break to its lower index — one bond's
/// energy would take a tower's foundation out and drop the whole thing. That is
/// what the first cut of this module did, and it is what this test found.
#[test]
fn energy_bookkeeping_is_exact() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let bond = one_bond_j();
    assert!(bond > 0.0);

    // Just under: nothing.
    {
        let mut fractures = seed(&world, tower(4));
        let mut bridge = PhysicsBridge3D::new(gravity());
        bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
        let r = bridge.runtime_destruct(&mut fractures, WALL, true, bond * 0.999);
        assert!(r.outcome.applied());
        assert_eq!(r.detached, 0, "a sub-threshold blow broke something");
        assert_eq!(r.energy_absorbed_j, 0.0, "it absorbed energy anyway");
        assert!(fractures[&WALL].is_intact());
    }
    // Exactly one bond: exactly one chunk, and the top one (the cheapest — the
    // interior chunks each have two bonds and the bottom one is on the ground).
    {
        let mut fractures = seed(&world, tower(4));
        let mut bridge = PhysicsBridge3D::new(gravity());
        bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
        let r = bridge.runtime_destruct(&mut fractures, WALL, true, bond);
        assert!(r.outcome.applied());
        assert_eq!(r.detached, 1);
        assert_eq!(r.energy_absorbed_j, bond);
        let cs = fractures[&WALL].chunks();
        assert!(
            cs[3].detached,
            "the TOP chunk should go first: it is held by one bond, while the base              is held by one bond AND the ground"
        );
        assert!(!cs[0].detached && !cs[1].detached && !cs[2].detached);
    }
    // Two bonds' worth takes the top TWO: after the top goes, the next one is
    // itself a one-bond chunk. That recomputation is what makes a hole widen.
    {
        let mut fractures = seed(&world, tower(4));
        let mut bridge = PhysicsBridge3D::new(gravity());
        bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
        let r = bridge.runtime_destruct(&mut fractures, WALL, true, bond * 2.0);
        assert_eq!(r.detached, 2);
        assert_eq!(r.energy_absorbed_j, bond * 2.0);
        let cs = fractures[&WALL].chunks();
        assert!(cs[2].detached && cs[3].detached);
    }
}

/// **Leftover energy dissipates; it is not banked.** Ten sub-threshold taps do
/// nothing at all, because a stored remainder would be fatigue and nothing asked
/// for fatigue.
#[test]
fn leftover_energy_dissipates() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut fractures = seed(&world, tower(4));
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    let tap = one_bond_j() * 0.5;
    for _ in 0..10 {
        let r = bridge.runtime_destruct(&mut fractures, WALL, true, tap);
        assert_eq!(r.detached, 0);
        assert_eq!(r.energy_absorbed_j, 0.0);
    }
    assert!(
        fractures[&WALL].is_intact(),
        "taps accumulated into a break"
    );
}

/// `strength` is the per-actor knob and it multiplies straight through: doubling
/// it exactly doubles the energy a bond costs.
#[test]
fn strength_scales_the_threshold_linearly() {
    let mut w = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let e = w.entity_of(WALL).unwrap();
    {
        let mut d = w.world_mut().get_mut::<Destructible>(e).unwrap();
        d.strength = concrete().strength * 2.0;
    }
    w.mark_dirty();
    let mut fractures = seed(&w, tower(4));
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_sim(&w, &BTreeMap::new(), &fractures);
    // The one-bond energy at the old strength now breaks nothing…
    let r = bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j());
    assert_eq!(r.detached, 0);
    // …and exactly twice it breaks exactly one.
    let r = bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j() * 2.0);
    assert_eq!(r.detached, 1);
    assert_eq!(r.energy_absorbed_j, one_bond_j() * 2.0);
}

// ── the structural solve ────────────────────────────────────────────────────

/// **THE PROGRESSIVE-COLLAPSE GATE.** Two towers, each standing on its own
/// plinth. Take one tower's plinth away and THAT tower collapses, chunk by chunk,
/// while its neighbour a few metres off does not move.
///
/// # Why the cut is a plinth and not a blast
///
/// The roadmap phrases the deliverable as "support-chunk removal drives
/// progressive collapse", and the temptation is to blow the base chunk out. The
/// energy model will not do that, and it is right not to: a base chunk is bonded
/// to its neighbour **and** to the ground, so it is the most expensive chunk in
/// the tower and a blast takes the cheap top ones first. Removing what it *stands
/// on* is the real shape of the failure — a floor gives way, a foundation is
/// undermined, a lower storey is destroyed — and it is the case where the collapse
/// is genuinely **free**: nothing pays for a single bond, and the whole tower comes
/// down because it is no longer connected to anything static.
#[test]
fn removing_a_towers_support_collapses_it_and_not_its_neighbour() {
    let mut world = plinth_world();
    let mut fractures = seed(&world, tower(5));
    let mut bridge = PhysicsBridge3D::new(gravity());
    // Settle, then chip one chunk off each tower so both solves are live — a
    // fully intact actor is never solved, and comparing a solved tower against an
    // unsolved one would prove nothing.
    tick(&mut bridge, &world, &mut fractures);
    for wall in [WALL, WALL_B] {
        let r = bridge.runtime_destruct(&mut fractures, wall, true, one_bond_j());
        assert_eq!(
            r.detached, 1,
            "the chip took more than one chunk off {wall}"
        );
    }
    tick(&mut bridge, &world, &mut fractures);
    assert_eq!(attached_count(&fractures, WALL), 4);
    assert_eq!(attached_count(&fractures, WALL_B), 4);

    // Take tower A's plinth away. Nothing is spent, nothing is damaged: the
    // world simply stops holding it up.
    let plinth = world.entity_of(PLINTH).expect("plinth A");
    world.world_mut().despawn(plinth);
    world.mark_dirty();
    world.propagate();

    let mut steps = 0;
    while steps < 300 && attached_count(&fractures, WALL) > 0 {
        tick(&mut bridge, &world, &mut fractures);
        steps += 1;
    }
    assert_eq!(
        attached_count(&fractures, WALL),
        0,
        "tower A did not fully collapse after its support was removed ({steps} steps)"
    );
    assert_eq!(
        attached_count(&fractures, WALL_B),
        4,
        "tower B collapsed too — the support graph is not per structure"
    );
    // Detaching is a decision; falling is what a solver does with it. Give the
    // rubble time to land before asserting that it moved.
    for _ in 0..150 {
        tick(&mut bridge, &world, &mut fractures);
    }

    // **THE WORLD ASSERTION (anti-vacuity).** The chunks did not merely change a
    // flag: they MOVED, and downward. A collapse that leaves every pose where it
    // started is a bookkeeping change, not a collapse.
    let fell = fractures[&WALL]
        .chunks()
        .iter()
        .enumerate()
        .filter(|(i, c)| c.translation.y < PLINTH_TOP_Y + *i as f64 + 0.4)
        .count();
    assert!(
        fell >= 4,
        "only {fell} of 5 chunks actually fell — the collapse is a flag, not a fall"
    );
    // …and the untouched neighbour's attached chunks are exactly where they were.
    for (i, c) in fractures[&WALL_B].chunks().iter().enumerate() {
        if c.detached {
            continue;
        }
        let rest = DVec3::new(TOWER_B_X, PLINTH_TOP_Y + i as f64 + 0.5, 0.0);
        assert!(
            (c.translation - rest).length() < 1e-9,
            "an attached chunk of the untouched tower moved"
        );
    }
}

/// The [`AlwaysLoaded`] override, honoured in both directions, on a **floating**
/// tower so the ground plays no part: the marker is the only difference between
/// the two arms.
#[test]
fn the_always_loaded_anchor_override_is_honoured() {
    for anchored in [true, false] {
        let world = floating_wall(anchored);
        let mut fractures = seed(&world, tower(5));
        let mut bridge = PhysicsBridge3D::new(gravity());
        tick(&mut bridge, &world, &mut fractures);
        // Just over two bonds' worth: enough to take the cheapest chunk in
        // either arm and not enough for a second. (The slack is real rather than
        // cosmetic — a measured face area is `1.0` to within an ULP, so `2.0 ×`
        // exactly is a hair under the anchored arm's bond-plus-ground cost.)
        let r = bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j() * 2.1);
        assert!(r.outcome.applied());
        for _ in 0..120 {
            tick(&mut bridge, &world, &mut fractures);
        }
        let still_up = attached_count(&fractures, WALL);
        if anchored {
            assert_eq!(
                still_up, 4,
                "an AlwaysLoaded actor collapsed — the explicit anchor override \
                 was not honoured"
            );
        } else {
            assert_eq!(
                still_up, 0,
                "the un-anchored control did NOT collapse — the anchored arm above \
                 proves nothing"
            );
        }
    }
}

/// A tower with **no ground under it** collapses entirely the moment anything
/// comes off: nothing is in contact with static geometry, so nothing is an
/// anchor. The negative control for "support really is a contact query".
#[test]
fn without_static_geometry_nothing_is_supported() {
    let world = floating_wall(false);
    let mut fractures = seed(&world, tower(5));
    let mut bridge = PhysicsBridge3D::new(gravity());
    tick(&mut bridge, &world, &mut fractures);
    let r = bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j());
    assert_eq!(
        r.detached, 5,
        "a floating tower kept chunks attached: {r:?}"
    );
    assert_eq!(
        r.energy_absorbed_j,
        one_bond_j(),
        "the free collapse charged for bonds it did not break"
    );
}

// ── the swap ────────────────────────────────────────────────────────────────

/// **ATOMICITY: never both, never neither.** An intact actor has its own
/// collider and no chunk bodies; a broken one has chunk bodies and no collider of
/// its own. There is no step on which both or neither is true.
#[test]
fn the_intact_to_chunks_swap_is_atomic() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut fractures = seed(&world, tower(4));
    let mut bridge = PhysicsBridge3D::new(gravity());

    let chunk_bodies = |b: &PhysicsBridge3D| {
        (0..4)
            .filter(|i| b.body_of(fracture_chunk_guid(WALL, *i)).is_some())
            .count()
    };

    // Intact: the actor's own collider, no chunks.
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    assert!(
        bridge.collider_of(WALL).is_some(),
        "an intact destructible lost its authored collider"
    );
    assert_eq!(
        chunk_bodies(&bridge),
        0,
        "an intact destructible has chunk bodies"
    );

    // Break it, then sync: the swap.
    bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j() * 1.5);
    assert!(!fractures[&WALL].is_intact());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    assert!(
        bridge.collider_of(WALL).is_none(),
        "a broken destructible kept its intact collider — the world would collide \
         with BOTH the wall and its rubble"
    );
    assert_eq!(
        chunk_bodies(&bridge),
        4,
        "the swap produced the wrong number of chunk bodies"
    );

    // …and the swap survives repeated syncs, in both directions of the stamp.
    for _ in 0..5 {
        bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
        assert!(bridge.collider_of(WALL).is_none());
        assert_eq!(chunk_bodies(&bridge), 4);
    }
}

/// Attached chunks are **static** and detached ones are **dynamic** with a real
/// mass from the honest density field.
#[test]
fn attached_chunks_are_static_and_detached_ones_weigh_something() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut fractures = seed(&world, tower(4));
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);

    let kind = |i: u32| {
        let b = bridge.body_of(fracture_chunk_guid(WALL, i)).unwrap();
        bridge.world().body_kind(b).unwrap()
    };
    assert_eq!(kind(0), BodyKind3D::Static);
    assert_eq!(
        kind(3),
        BodyKind3D::Dynamic,
        "the detached chunk is not dynamic"
    );

    // A 1 m³ concrete cube is 2400 kg — from `density_kg_m3 × volume_m3`, not
    // from `Collider3D::density` (which defaults to rapier's 1.0 placeholder).
    let mass = bridge
        .world()
        .body_mass(bridge.body_of(fracture_chunk_guid(WALL, 3)).unwrap())
        .unwrap();
    assert!(
        (mass - 2400.0).abs() < 1.0,
        "the chunk weighs {mass} kg, not 2400 — the density field is not reaching \
         the collider"
    );
    assert_eq!(fractures[&WALL].chunk_mass_kg(3), 2400.0);
}

/// A reclaimed chunk leaves the physics world **by omission**, on the same
/// generation that removes it from the render set.
#[test]
fn a_reclaimed_chunk_leaves_the_world() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut fractures = seed(&world, tower(4));
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    // Three and a half bonds: the top three chunks, leaving the base standing
    // (it is bonded to the ground as well, so it is the most expensive of the
    // four). The half-bond of slack is the same ULP story as above.
    bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j() * 3.5);
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    assert_eq!(fractures[&WALL].live_debris(), 3);
    assert!(bridge.body_of(fracture_chunk_guid(WALL, 3)).is_some());

    // A lifetime of one step reclaims everything on the next advance.
    let budget = DebrisBudget {
        max_live: 256,
        lifetime_s: DT * 0.5,
    };
    bridge.step(DT);
    let (audit, _) = bridge.step_fractures(&mut fractures, DT, budget);
    assert!(audit.expired >= 1, "nothing expired: {audit:?}");
    assert_eq!(audit.live_debris, 0);
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    for i in 1..4 {
        assert!(
            bridge.body_of(fracture_chunk_guid(WALL, i)).is_none(),
            "reclaimed chunk {i} still has a body"
        );
    }
    // The base chunk is the standing remnant, not debris: it never detached (it
    // is bonded to the ground as well as to its neighbour, so it is the most
    // expensive chunk in the tower), so the lifetime does not touch it and it
    // keeps its static body. Half a wall left standing is the point.
    assert!(!fractures[&WALL].chunks()[0].detached);
    assert!(bridge.body_of(fracture_chunk_guid(WALL, 0)).is_some());
}

/// Over budget, the **oldest detached first** is reclaimed, deterministically.
#[test]
fn the_debris_cap_reclaims_oldest_first() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut fractures = seed(&world, tower(6));
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    // Break the top three, one call at a time so their orders differ.
    for _ in 0..3 {
        bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j());
    }
    let orders: Vec<u64> = fractures[&WALL]
        .chunks()
        .iter()
        .map(|c| c.detach_order)
        .collect();
    // Top-down: chunk 5 came off first (order 1), then 4, then 3. Attached
    // chunks carry no order at all.
    assert_eq!(
        orders,
        vec![0, 0, 0, 3, 2, 1],
        "detach order is not top-down"
    );

    let (audit, _) = bridge.step_fractures(
        &mut fractures,
        DT,
        DebrisBudget {
            max_live: 2,
            lifetime_s: 1.0e9,
        },
    );
    assert_eq!(audit.over_budget, 1);
    assert_eq!(audit.live_debris, 2);
    let cs = fractures[&WALL].chunks();
    assert!(
        cs[5].gone,
        "the OLDEST detached chunk was not the one reclaimed"
    );
    assert!(!cs[4].gone && !cs[3].gone);
}

// ── determinism ─────────────────────────────────────────────────────────────

/// **The same damage trace produces byte-identical state.** Two runs of the same
/// script, compared as raw bits — not as `PartialEq`, which is IEEE equality and
/// would call a NaN "different from itself" and `-0.0` "the same as `0.0`".
#[test]
fn the_same_damage_trace_is_byte_identical() {
    let run = || {
        let world = wall_world(&[
            (WALL, DVec3::ZERO, false),
            (WALL_B, DVec3::new(20.0, 0.0, 0.0), false),
        ]);
        let mut fractures = seed(&world, tower(5));
        let mut bridge = PhysicsBridge3D::new(gravity());
        for step in 0..180 {
            if step == 5 {
                bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j() * 2.5);
            }
            if step == 40 {
                bridge.radial_impulse(DVec3::new(0.0, 2.0, 0.0), 5.0e4, 6.0);
            }
            tick(&mut bridge, &world, &mut fractures);
        }
        (fractures[&WALL].bits(), fractures[&WALL_B].bits())
    };
    let a = run();
    let b = run();
    assert_eq!(
        a.0, b.0,
        "two identical runs produced different fracture state"
    );
    assert_eq!(a.1, b.1);

    // Anti-vacuity: the trace did something.
    let baseline = {
        let world = wall_world(&[
            (WALL, DVec3::ZERO, false),
            (WALL_B, DVec3::new(20.0, 0.0, 0.0), false),
        ]);
        let mut fractures = seed(&world, tower(5));
        let mut bridge = PhysicsBridge3D::new(gravity());
        for _ in 0..180 {
            tick(&mut bridge, &world, &mut fractures);
        }
        (fractures[&WALL].bits(), fractures[&WALL_B].bits())
    };
    assert_ne!(a.0, baseline.0, "the damaged run matches the untouched one");
    // **A no-damage run is byte-identical to the baseline** — and the untouched
    // neighbour proves the blast did not reach across the level.
    assert_eq!(
        a.1, baseline.1,
        "the far tower drifted without being touched"
    );
}

// ── radial impulse ──────────────────────────────────────────────────────────

/// The blast pushes what is in range and nothing that is not, reports how many,
/// and refuses degenerate parameters as a value.
#[test]
fn a_radial_impulse_moves_what_it_reaches() {
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut fractures = seed(&world, tower(4));
    let mut bridge = PhysicsBridge3D::new(gravity());
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);
    bridge.runtime_destruct(&mut fractures, WALL, true, one_bond_j() * 3.0);
    bridge.sync_from_world_sim(&world, &BTreeMap::new(), &fractures);

    // Degenerate parameters are values.
    for (j, r) in [(0.0, 5.0), (1.0e5, 0.0), (f64::NAN, 5.0), (1.0e5, f64::NAN)] {
        assert_eq!(bridge.radial_impulse(DVec3::ZERO, j, r), 0);
    }
    // Out of range: nothing.
    assert_eq!(
        bridge.radial_impulse(DVec3::new(500.0, 0.0, 0.0), 1.0e5, 1.0),
        0
    );

    // In range: the detached (dynamic) chunks only — static chunks, the terrain
    // tiles and the intact collider are not pushed.
    let hit = bridge.radial_impulse(DVec3::new(0.0, 3.0, 0.0), 1.0e6, 10.0);
    assert!(hit >= 1, "the blast reached nothing");
    let dynamic = (0..4)
        .filter(|i| {
            bridge
                .body_of(fracture_chunk_guid(WALL, *i))
                .and_then(|b| bridge.world().body_kind(b))
                == Some(BodyKind3D::Dynamic)
        })
        .count() as u32;
    assert_eq!(
        hit, dynamic,
        "the blast pushed something that is not debris"
    );

    // …and it really moved them.
    let before: Vec<DVec3> = fractures[&WALL]
        .chunks()
        .iter()
        .map(|c| c.translation)
        .collect();
    for _ in 0..10 {
        tick(&mut bridge, &world, &mut fractures);
    }
    let after: Vec<DVec3> = fractures[&WALL]
        .chunks()
        .iter()
        .map(|c| c.translation)
        .collect();
    assert!(
        before
            .iter()
            .zip(&after)
            .any(|(a, b)| (*a - *b).length() > 0.1),
        "the impulse changed nothing about where anything is"
    );
}

// ── the real cook output ────────────────────────────────────────────────────

/// **The hand-built fixtures are honest.** The same machinery, on a real
/// `fracture_mesh` asset: every listed bond gets a positive, finite energy, and
/// the shared-face measurement is what produced it (a fallback for every bond
/// would mean the measurement never fires).
#[test]
fn a_real_cook_asset_measures_its_own_bond_areas() {
    let mesh = cube_mesh(1.0); // a 2 m cube
    let asset = Arc::new(
        inf_mesh::fracture_mesh(
            &mesh,
            inf_asset::AssetId(MESH),
            inf_mesh::FractureParams {
                seed: 7,
                chunk_count: 8,
            },
        )
        .expect("a 2 m cube fractures"),
    );
    assert!(asset.chunks.len() >= 2);
    let pairs = asset.adjacency_pairs();
    assert!(!pairs.is_empty(), "a real fracture has no adjacency");

    let state = FractureState::new(asset.clone(), concrete(), DAffine3::IDENTITY, false);
    let mut positive = 0;
    for (a, b) in &pairs {
        let e = state.bond_energy_j(*a, *b);
        assert!(e.is_finite() && e > 0.0, "bond {a}-{b} costs {e} J");
        positive += 1;
    }
    assert_eq!(positive, pairs.len());

    // The measurement really fires: at least one bond's implied area differs
    // from the `min(volume)^(2/3)` fallback, which it could not if every bond
    // took the fallback path.
    let implied_area =
        |a: u32, b: u32| state.bond_energy_j(a, b) / (concrete().strength * CRACK_OPENING_M);
    let differs = pairs.iter().any(|(a, b)| {
        let fb = asset.chunks[*a as usize]
            .volume_m3
            .min(asset.chunks[*b as usize].volume_m3)
            .powf(2.0 / 3.0);
        (implied_area(*a, *b) - fb).abs() > fb * 0.01
    });
    assert!(
        differs,
        "every bond took the fallback — the shared-face measurement never ran"
    );

    // And it survives a full break + collapse without a panic or a NaN.
    let mut fractures = BTreeMap::from([(WALL, state)]);
    let world = wall_world(&[(WALL, DVec3::ZERO, false)]);
    let mut bridge = PhysicsBridge3D::new(gravity());
    let r = bridge.runtime_destruct(&mut fractures, WALL, true, 1.0e12);
    assert!(r.outcome.applied());
    assert_eq!(r.detached as usize, asset.chunks.len());
    for _ in 0..60 {
        tick(&mut bridge, &world, &mut fractures);
    }
    for c in fractures[&WALL].chunks() {
        assert!(c.translation.is_finite(), "a chunk pose went non-finite");
    }
}

/// The scale of the actor scales the bond: a wall placed at twice the size is
/// four times as hard to break through (an area is the two-thirds power of a
/// volume).
#[test]
fn actor_scale_scales_the_bond_area() {
    let asset = tower(3);
    let unit = FractureState::new(asset.clone(), concrete(), DAffine3::IDENTITY, false);
    let big = FractureState::new(
        asset,
        concrete(),
        DAffine3::from_scale(DVec3::splat(2.0)),
        false,
    );
    let ratio = big.bond_energy_j(0, 1) / unit.bond_energy_j(0, 1);
    assert!(
        (ratio - 4.0).abs() < 1e-9,
        "scaling by 2 changed the bond by {ratio}×, not 4×"
    );
    // …and the mass by eight.
    assert!((big.chunk_mass_kg(0) / unit.chunk_mass_kg(0) - 8.0).abs() < 1e-9);
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// How many of `entity`'s chunks are still attached.
fn attached_count(fractures: &BTreeMap<Uuid, FractureState>, entity: Uuid) -> usize {
    fractures[&entity]
        .chunks()
        .iter()
        .filter(|c| !c.detached)
        .count()
}

/// The plinth's top surface, and therefore the towers' base.
const PLINTH_TOP_Y: f64 = 2.0;
/// Where the second tower stands — far enough that neither tower's rubble can
/// reach the other.
const TOWER_B_X: f64 = 20.0;

/// Two towers, each on its own **static box plinth**, over the shared terrain.
/// Removing a plinth removes exactly one tower's support and nothing else.
fn plinth_world() -> EcsWorld {
    let mut w = wall_world(&[
        (WALL, DVec3::new(0.0, PLINTH_TOP_Y, 0.0), false),
        (WALL_B, DVec3::new(TOWER_B_X, PLINTH_TOP_Y, 0.0), false),
    ]);
    for (guid, x) in [(PLINTH, 0.0), (PLINTH_B, TOWER_B_X)] {
        let e = w.spawn_with_guid(guid, "Plinth", None);
        let mut t = Transform::IDENTITY;
        t.translation = Vec3d::new(x, PLINTH_TOP_Y * 0.5, 0.0);
        w.world_mut().entity_mut(e).insert(t);
        // Collider-only ⇒ the bridge's implicit-static-body rule.
        w.world_mut().entity_mut(e).insert(Collider3D {
            shape_kind: ColliderShape3DKind::Box,
            half_extents: Vec3d::new(2.0, PLINTH_TOP_Y * 0.5, 2.0),
            ..Default::default()
        });
    }
    w.mark_dirty();
    w.propagate();
    w
}

/// One destructible tower and **nothing else at all** — no terrain, no plinth. A
/// chunk here can only ever be anchored by the `AlwaysLoaded` override.
fn floating_wall(anchored: bool) -> EcsWorld {
    let mut w = EcsWorld::new();
    let e = w.spawn_with_guid(WALL, "Wall", None);
    w.world_mut().entity_mut(e).insert(Transform::IDENTITY);
    w.world_mut().entity_mut(e).insert(concrete());
    w.world_mut().entity_mut(e).insert(MeshRef {
        asset: Some(MESH),
        ..Default::default()
    });
    if anchored {
        w.world_mut().entity_mut(e).insert(AlwaysLoaded);
    }
    w.mark_dirty();
    w.propagate();
    w
}

/// The `cook_fracture` cube fixture: an axis-aligned box of half-extent `half`.
fn cube_mesh(half: f32) -> inf_mesh::MeshAsset {
    let c = |x: f32, y: f32, z: f32| MeshVertex {
        position: [x * half, y * half, z * half],
        ..Default::default()
    };
    inf_mesh::MeshAsset::new(
        vec![inf_mesh::SubMesh {
            name: "wall".into(),
            vertices: vec![
                c(-1.0, -1.0, -1.0),
                c(1.0, -1.0, -1.0),
                c(1.0, 1.0, -1.0),
                c(-1.0, 1.0, -1.0),
                c(-1.0, -1.0, 1.0),
                c(1.0, -1.0, 1.0),
                c(1.0, 1.0, 1.0),
                c(-1.0, 1.0, 1.0),
            ],
            indices: vec![
                0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 3, 2, 6, 3, 6, 7, 0, 3, 7, 0,
                7, 4, 1, 5, 6, 1, 6, 2,
            ],
            material_slot: Some(0),
            skin: Vec::new(),
        }],
        vec!["Concrete".into()],
    )
}
