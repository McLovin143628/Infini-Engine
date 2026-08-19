//! **Two adjacent terrains, at their shared border** — the island phase's IB-15.
//!
//! The AAA-readiness certification carried this forward as *"RELAYED, needs
//! verification"*: the P16 remainder says *"every terrain is anchored at zero, so
//! two terrains cannot yet share one pyramid across their boundary"*, Wave G
//! shipped a non-zero `.inf_terrain` origin so the premise *might* be retired,
//! and **no source read in that audit claimed the seam itself was closed**.
//!
//! This file verifies it, and where it does not hold it states the bound with its
//! number rather than describing it. Nothing here is inferred; every claim below
//! is a measurement this file makes.
//!
//! # What was found, and what changed
//!
//! **The `.inf_terrain` header origin is still not read at runtime.** Wave G
//! writes a value into it (`chunked.rs` `with_origin`), and the cook's own
//! advisory says the quiet part out loud: *"the header value round-trips through
//! a write-back and never places a tile"*. A terrain's world frame is its
//! entity's `Transform`, and `TerrainData`'s lattice is `coord · span` from its
//! own local zero. **The P16 remainder stands verbatim** and is re-carried, not
//! closed.
//!
//! **A worse thing was found beside it, and is fixed here.** The gameplay ground
//! query picked *the lowest-`Guid` non-empty terrain in the world, with no test
//! of whether it covers the query point*. Over a second terrain it sampled the
//! first one outside its extent, got `None`, and the host default `0.0` took
//! over. A character walking from terrain A onto terrain B fell to sea level at
//! the border. It survived because **no committed scene places two terrains near
//! each other**: the phase16 gate's two are 1 024 m apart *by design* (its own
//! comment says "clear of it in XZ so the two never overlap"), and the render
//! golden's are 40 m apart and 6 m high. `inf_voxel::ground_height_at` now takes
//! every terrain and returns the topmost surface that answers.
//!
//! # The three claims this file measures
//!
//! 1. **Heights agree across the border**, exactly, when both terrains are
//!    authored from the same world-space height function on the same grid. Not a
//!    tolerance: the shared edge samples are written from the same `f(x, z)` at
//!    the same world coordinates.
//! 2. **The ground query answers on both sides**, and does not fall to `0.0`
//!    anywhere on the walk across. This is the arm that would have caught the
//!    defect above.
//! 3. **Normals do NOT agree at the border**, and the disagreement is reported as
//!    a measured angle. `TerrainData::normal_at` falls back to the centre height
//!    for a missing neighbour, so at a terrain's *outer* edge the gradient is
//!    one-sided — each terrain sees a wall of nothing where the other one is.
//!    That is the seam, it is real, and its number is printed.

use glam::{DVec2, DVec3};
use inf_ecs::components::{Terrain, Transform};
use inf_ecs::world::EcsWorld;
use uuid::Uuid;

/// Both terrains share this grid, which is what makes their lattices
/// commensurable at all — nothing in the engine requires it, and two terrains
/// with different `meters_per_sample` could not share edge samples even if placed
/// perfectly (the committed phase16 pair is 129² @ 8 m against 9² @ 4 m).
const RESOLUTION: u32 = 17;
const METERS_PER_SAMPLE: f64 = 4.0;
/// One level-0 tile's world span — the stride between tile origins.
const TILE_SPAN: f64 = (RESOLUTION as f64 - 1.0) * METERS_PER_SAMPLE;
/// Each terrain is 2 × 2 tiles.
const TILES: f64 = 2.0;
/// The world X at which terrain A ends and terrain B begins.
const BORDER_X: f64 = TILE_SPAN * TILES;

const TERRAIN_A: Uuid = Uuid::from_u128(0x00A0);
const TERRAIN_B: Uuid = Uuid::from_u128(0x00B0);

/// One continuous world-space surface. **Portable trig** (the P14 law): this
/// decides sampled heights that the equality assertions below compare.
fn ground(x: f64, z: f64) -> f64 {
    30.0 + 6.0 * inf_math::psin64(x * 0.03) + 4.0 * inf_math::pcos64(z * 0.05)
}

/// A terrain authored from [`ground`] over `[origin, origin + extent]` in WORLD
/// space, then anchored at `origin` — so its local lattice sees `f(x - o.x, ..)`
/// and the two terrains describe one continuous surface.
fn terrain_at(origin: DVec3) -> Terrain {
    let mut t = Terrain::configured(RESOLUTION, METERS_PER_SAMPLE);
    let span = TILE_SPAN * TILES;
    t.data
        .write_region(DVec2::ZERO, DVec2::splat(span), |lx, lz| {
            ground(lx + origin.x, lz + origin.z) - origin.y
        });
    assert!(!t.data.is_empty());
    t
}

/// A world holding terrain A at the origin and terrain B directly east of it,
/// sharing the plane `x = BORDER_X`.
fn two_adjacent_terrains() -> (EcsWorld, DVec3, DVec3) {
    let a_origin = DVec3::ZERO;
    let b_origin = DVec3::new(BORDER_X, 0.0, 0.0);
    let mut world = EcsWorld::default();
    for (guid, origin) in [(TERRAIN_A, a_origin), (TERRAIN_B, b_origin)] {
        let e = world.spawn_with_guid(guid, "Terrain", None);
        world.world_mut().entity_mut(e).insert(terrain_at(origin));
        world
            .world_mut()
            .entity_mut(e)
            .insert(Transform::from_translation(origin));
    }
    world.propagate();
    (world, a_origin, b_origin)
}

/// The world-space heights each terrain reports at world `(x, z)`, or `None`
/// where it does not answer.
fn heights_at(world: &EcsWorld, x: f64, z: f64) -> (Option<f64>, Option<f64>) {
    let w = world.world();
    let read = |g: Uuid, o: DVec3| -> Option<f64> {
        let e = world.entity_of(g)?;
        w.get::<Terrain>(e)?
            .data
            .height_at(DVec2::new(x - o.x, z - o.z))
            .map(|h| h + o.y)
    };
    (
        read(TERRAIN_A, DVec3::ZERO),
        read(TERRAIN_B, DVec3::new(BORDER_X, 0.0, 0.0)),
    )
}

/// **Claim 1: the heights agree at the shared border, exactly.**
#[test]
fn two_adjacent_terrains_agree_on_the_heights_of_their_shared_border() {
    let (world, ..) = two_adjacent_terrains();
    let mut compared = 0usize;
    // Every sample along the shared edge, at the grid's own spacing.
    let mut z = 0.0;
    while z <= TILE_SPAN * TILES {
        let (a, b) = heights_at(&world, BORDER_X, z);
        let (a, b) = (
            a.unwrap_or_else(|| panic!("terrain A does not reach its own edge at z = {z}")),
            b.unwrap_or_else(|| panic!("terrain B does not reach its own edge at z = {z}")),
        );
        assert_eq!(
            a, b,
            "the two terrains disagree at (x = {BORDER_X}, z = {z}) by {} m — both \
             authored the same world-space function at the same world coordinate, \
             so this is a lattice or anchoring failure, not rounding",
            (a - b).abs()
        );
        compared += 1;
        z += METERS_PER_SAMPLE;
    }
    // ANTI-VACUITY: a loop that ran zero times, or over one point, proves nothing.
    assert!(
        compared >= RESOLUTION as usize,
        "compared only {compared} border samples"
    );
    eprintln!("IB-15 heights: {compared} shared-border samples, all bit-identical");
}

/// **Claim 2: the gameplay ground query answers across the whole walk** —
/// the arm that would have caught the lowest-`Guid` dispatch.
#[test]
fn walking_from_one_terrain_onto_the_other_never_falls_to_sea_level() {
    let (world, a_origin, b_origin) = two_adjacent_terrains();
    let w = world.world();
    let terrains: Vec<(&inf_ecs::TerrainData, DVec3)> = [(TERRAIN_A, a_origin), (TERRAIN_B, b_origin)]
        .iter()
        .map(|(g, o)| {
            let e = world.entity_of(*g).unwrap();
            (&w.get::<Terrain>(e).unwrap().data, *o)
        })
        .collect();
    let volumes: std::collections::BTreeMap<u8, inf_voxel::VoxelData> =
        std::collections::BTreeMap::new();

    let z = TILE_SPAN; // mid-span, away from the north/south edges
    let mut answered = 0usize;
    let mut on_b = 0usize;
    let mut x = 1.0;
    while x < BORDER_X * 2.0 - 1.0 {
        let got = inf_voxel::ground_height_at(&terrains, &volumes, x, z)
            .unwrap_or_else(|| panic!("no ground at x = {x} — the walk crosses a hole"));
        let want = ground(x, z);
        assert!(
            (got - want).abs() < 0.5,
            "at x = {x} the ground query answered {got:.4} m against the authored \
             {want:.4} m. Before IB-15 this returned 0.0 for every x past \
             {BORDER_X}, because the query picked the lowest-Guid terrain and \
             never asked whether it covered the point."
        );
        answered += 1;
        if x > BORDER_X {
            on_b += 1;
        }
        x += 2.0;
    }
    // ANTI-VACUITY: the walk really crossed onto the second terrain. A walk that
    // stayed on terrain A would satisfy every assertion above and prove nothing.
    assert!(
        on_b > 10,
        "the walk only reached {on_b} samples past the border"
    );
    eprintln!(
        "IB-15 ground query: {answered} samples across the border ({on_b} on the \
         second terrain), none at sea level"
    );
}

/// **Claim 3 — the measured bound: normals do NOT agree at the border.**
///
/// `TerrainData::normal_at` takes central differences and `unwrap_or(c)`s a
/// missing neighbour, so at a terrain's *outer* edge the gradient across that
/// axis is halved: each terrain sees flat nothing where the other one is. This is
/// the seam the P16 remainder is about, expressed at the query level rather than
/// at the pyramid level, and it is **open**.
///
/// Recorded as a number so that the day someone closes it (a neighbour-aware
/// normal, or a shared skirt) this test fails and the ledger gets rewritten —
/// the house's own "assert the bound" pattern.
#[test]
fn the_border_normal_seam_is_open_and_this_is_its_angle() {
    let (world, a_origin, b_origin) = two_adjacent_terrains();
    let w = world.world();
    let normal = |g: Uuid, o: DVec3, x: f64, z: f64| -> Option<DVec3> {
        let e = world.entity_of(g)?;
        w.get::<Terrain>(e)?
            .data
            .normal_at(DVec2::new(x - o.x, z - o.z))
    };

    let z = TILE_SPAN;
    let a = normal(TERRAIN_A, a_origin, BORDER_X, z).expect("A's edge normal");
    let b = normal(TERRAIN_B, b_origin, BORDER_X, z).expect("B's edge normal");
    let cos = a.dot(b).clamp(-1.0, 1.0);
    let deg = cos.acos().to_degrees();

    // The control: **in its interior** each terrain reproduces the continuous
    // surface's own normal, so the disagreement measured below is the edge and
    // not a broken fixture. Compared against the analytic normal at the SAME
    // world point, by the same central-difference formula `normal_at` uses.
    let analytic = |x: f64, z: f64| -> DVec3 {
        let e = METERS_PER_SAMPLE;
        let dhdx = (ground(x + e, z) - ground(x - e, z)) / (2.0 * e);
        let dhdz = (ground(x, z + e) - ground(x, z - e)) / (2.0 * e);
        DVec3::new(-dhdx, 1.0, -dhdz).normalize()
    };
    for (label, guid, origin, x) in [
        ("A", TERRAIN_A, a_origin, BORDER_X - TILE_SPAN),
        ("B", TERRAIN_B, b_origin, BORDER_X + TILE_SPAN),
    ] {
        let got = normal(guid, origin, x, z).expect("an interior normal");
        let want = analytic(x, z);
        let off = got.dot(want).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            off < 1.0,
            "terrain {label} disagrees with the continuous surface by {off:.4}° in \
             its own INTERIOR (x = {x}), so the fixture is not describing one \
             surface and the border number below would be noise"
        );
    }

    assert!(
        deg > 1.0,
        "the border normals now agree to within {deg:.4}° — if a neighbour-aware \
         normal or a shared skirt landed, DELETE this arm and retire IB-15's \
         normal bound in the ROADMAP. If they agree for another reason, the \
         fixture changed."
    );
    // The bound, pinned generously so a fixture tweak does not re-litigate it.
    assert!(
        deg < 90.0,
        "the border normals are {deg:.4}° apart, which is worse than the \
         one-sided-gradient explanation predicts — something else is wrong"
    );
    eprintln!(
        "IB-15 BOUND (open): adjacent terrains' shading normals differ by \
         {deg:.4}° at their shared border. Cause: `normal_at` clamps a missing \
         neighbour to the centre height, so each terrain's outer edge reads a \
         one-sided gradient. Heights are exact (claim 1); only the derivative is \
         seamed."
    );
}

/// **The collider bound, measured**: the two terrains' heightfield tiles abut
/// exactly, with no gap and no overlap — because their lattices are
/// commensurable and their origins are a whole number of tile spans apart.
///
/// Stated as a *conditional* result, which is the honest shape: nothing in the
/// engine enforces either condition. `Terrain` has no origin field at all (its
/// world frame is the entity `Transform`), `TerrainData`'s lattice is
/// `coord · span` from its own local zero, and no two terrains are required to
/// share `meters_per_sample`. So this passes because the fixture arranged it, and
/// what the arm proves is that the arrangement is *sufficient* — which is what an
/// island author needs to know.
#[test]
fn adjacent_terrain_tiles_abut_when_the_author_lands_them_on_the_grid() {
    let a = terrain_at(DVec3::ZERO);
    let b = terrain_at(DVec3::new(BORDER_X, 0.0, 0.0));
    assert_eq!(
        a.data.tile_span(),
        b.data.tile_span(),
        "incommensurable lattices — nothing in the engine forbids this, and two \
         terrains with different meters_per_sample cannot share edge samples"
    );
    // A's last tile column ends where B's first begins.
    let a_east_edge = a.data.tile_origin_xz((1, 0)).x + a.data.tile_span();
    let b_west_edge = BORDER_X + b.data.tile_origin_xz((0, 0)).x;
    assert_eq!(
        a_east_edge, b_west_edge,
        "A's eastern tile edge is at {a_east_edge} and B's western at \
         {b_west_edge} — a {} m gap in the collider surface",
        (a_east_edge - b_west_edge).abs()
    );
    // …and the offset really is a whole number of tile spans, which is the
    // condition, stated rather than assumed.
    assert_eq!(BORDER_X % a.data.tile_span(), 0.0);
    eprintln!(
        "IB-15 colliders: adjacent tile edges meet exactly at x = {a_east_edge} \
         (tile span {} m). CONDITION: equal grids + an origin offset that is a \
         whole number of tile spans. Neither is enforced anywhere.",
        a.data.tile_span()
    );
}

/// **The `.inf_terrain` header origin is still not read at runtime** — the P16
/// remainder, re-verified rather than relayed.
///
/// Wave G gave the container a non-zero `origin` and the certification wondered
/// whether that retired the multi-anchor premise. It does not: the value
/// round-trips through the writer and nothing places a tile with it. This arm
/// writes an asset with a non-zero origin, reads the header back, and shows that
/// the tile the reader produces is anchored at `coord · span` regardless.
#[test]
fn the_terrain_asset_origin_is_carried_and_never_placed() {
    let data = {
        let mut d = inf_terrain::TerrainData::new(RESOLUTION, METERS_PER_SAMPLE);
        d.write_region(DVec2::ZERO, DVec2::splat(TILE_SPAN), |x, z| ground(x, z));
        d
    };
    let opts = inf_terrain::PyramidOptions::default();
    let pyramid = inf_terrain::build_pyramid(&data, opts);
    let anchor = DVec3::new(1_000.0, 7.0, -250.0);
    // Built through the SHIPPING builder with a non-zero anchor — the door Wave G
    // gave `with_origin` its first real caller through.
    let asset = {
        let mut b = inf_terrain::TerrainAssetBuilder::new(
            data.tile_resolution(),
            data.meters_per_sample(),
        )
        .with_pyramid(opts)
        .with_origin(anchor);
        for (&coord, tile) in data.tiles() {
            b.insert(inf_terrain::TileKey::lod0(coord), tile).unwrap();
        }
        for level in &pyramid {
            for (&coord, tile) in &level.tiles {
                b.insert(inf_terrain::TileKey::new(level.lod, coord), tile)
                    .unwrap();
            }
        }
        b.build().expect("build")
    };
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().join("Anchored.inf_terrain");
    inf_terrain::write_terrain_asset(&path, &asset).expect("write");

    let store = inf_terrain::open_file_tile_store(&path).expect("open");
    let header = *store.header();
    assert_eq!(header.origin, anchor, "the header no longer carries its origin");

    let mut into = inf_terrain::TerrainData::new(RESOLUTION, METERS_PER_SAMPLE);
    let wants = inf_terrain::tile_range(0, (0, 0), (0, 0));
    into.request_tiles(&wants, &store);
    let tile = into
        .get_tile((0, 0))
        .expect("the level-0 tile paged in");
    assert_eq!(
        (tile.origin.x, tile.origin.z),
        (0.0, 0.0),
        "the tile was placed at the header's origin — if that is now true, the \
         P16 multi-anchor remainder is CLOSED and this arm should be rewritten"
    );
    eprintln!(
        "IB-15 REMAINDER (open): the .inf_terrain header carries origin \
         {:?} and the tile it serves is still anchored at (0, 0). A terrain's \
         world frame is its entity Transform; `with_origin` remains provenance, \
         not placement.",
        header.origin
    );
}
