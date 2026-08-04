//! **The hole↔volume coupling** (P21.2): which heightfield samples a carve
//! opens, and which a fill closes again.
//!
//! P21.2 gave [`TerrainTile`](inf_terrain::TerrainTile) a per-sample hole mask
//! and [`VoxelShape`] a signed distance in world metres. This is the one place
//! that says how the two meet: *a cut that reaches the heightfield surface takes
//! the surface away there*, so the cave the author dug out of the hillside has a
//! mouth instead of being sealed behind ground that is still drawn.
//!
//! # THE RULE, stated once and depended on everywhere
//!
//! A heightfield sample is **inside the cut** iff the shape's signed distance at
//! *that sample's own surface point* is strictly negative:
//!
//! ```text
//! P(i, j) = ( tile_origin.x + i·mps , tile.origin.y + tile.sample(i, j) , tile_origin.z + j·mps ) + terrain_origin
//! inside  = shape.distance_m(P) < 0
//! ```
//!
//! A **carve** opens exactly the inside samples; a **fill** closes exactly the
//! inside samples. Three properties fall out of that, and all three are load-
//! bearing rather than incidental:
//!
//! * **Deterministic.** The tile walk is a `BTreeMap` order, the sample walk is
//!   an integer range, the arithmetic is `f64`, and no `std` trig is called. Two
//!   runs — or two machines — produce the same [`SurfaceSample`] list.
//! * **A pure function of the cut and the tile grid.** Nothing here reads the
//!   voxel field, the previous mask, or anything else that a history could make
//!   differ. Which is what makes the last property true:
//! * **Exactly invertible.** Filling the same shape clears precisely the bits the
//!   carve set, so `carve → fill` returns every touched tile to bytes
//!   indistinguishable from one nothing ever carved (the mask heals; see
//!   [`TerrainTile::heal_holes`](inf_terrain::TerrainTile::heal_holes)).
//!
//! ## Why the sample's OWN height, and not `height_at`
//!
//! [`TerrainData::height_at`] is bilinear **and** poisoned: it returns `None`
//! across the whole cell of any holed corner. Feeding it back in here would make
//! the predicate depend on the mask it is computing — the second carve over the
//! same ground would see a different surface from the first, a fill would clear a
//! different set than the carve opened, and invertibility would be gone. The raw
//! per-sample height has no such feedback: it is a number in the tile's height
//! buffer that a carve never writes.
//!
//! ## The two boundary cases, written down rather than discovered
//!
//! * **Exactly on the brush surface** (`distance == 0`) the sample stays *closed*,
//!   while [`VoxelData::apply_op`](crate::VoxelData::apply_op) empties the voxel
//!   sample there (it writes `max(old, −0)`, and `0` is not solid). That
//!   asymmetry is a measure-zero set of lattice points, and it errs in the safe
//!   direction: the [poison rule](inf_terrain::TerrainData::height_at) already
//!   removes the surface from every cell touching an opened sample, so a boundary
//!   sample left closed is still inside the discarded region whenever any
//!   neighbour opened.
//! * **A cut finer than the sample spacing** can pass between two samples and open
//!   nothing. The mask's resolution is the terrain's — one bit per height sample —
//!   so a 10 cm bore through a 1 m heightfield cannot punch a mouth, and
//!   [`cut_crosses_surface`] honestly reports `false`. That is a statement about
//!   the container, not a bug to work around: a mouth an author can see needs a
//!   cut at least a sample across.
//!
//! # Frames and units
//!
//! World **metres**, `f64`, `y` up (architecture rules 3 and 6). A
//! [`TerrainData`] carries no world anchor of its own — its placement is the
//! terrain entity's transform — so every entry point takes an explicit
//! `terrain_origin`, exactly as [`crate::ground_height_at`] does and for exactly
//! the same reason.

use glam::DVec3;
use inf_terrain::{HoleDelta, HoleDeltaBuilder, TerrainData};

use crate::ops::VoxelShape;

/// One heightfield sample a cut crosses: its tile grid coordinate and its
/// in-tile indices.
///
/// Ordered so a `Vec` of them sorts into the walk order the functions below
/// emit — tile coordinate, then row, then column — which is what lets a test
/// compare a produced list against an independently gathered one without caring
/// how either was built.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct SurfaceSample {
    /// Tile grid coordinate (`TerrainData`'s `(tx, tz)`).
    pub coord: (i32, i32),
    /// Sample row index within the tile (`j`), listed first so the derived `Ord`
    /// matches the row-major walk.
    pub j: u32,
    /// Sample column index within the tile (`i`).
    pub i: u32,
}

/// Every heightfield sample whose surface point lies inside `shape` — THE RULE,
/// evaluated (see the module docs).
///
/// Walks only **authored** tiles: a tile that is not resident has no height
/// buffer to test, and inventing one would make the answer depend on what had
/// paged in. The consequence is an obligation on the caller, the same one the
/// sculpt brush already carries — page the cut's footprint into the working set
/// *before* asking, or a carve at the edge of the streamed ring opens a mouth on
/// the tiles that happened to be in memory and not on the ones that were not.
///
/// Samples on a shared tile edge belong to two tiles and are returned **twice**,
/// once per tile. That is deliberate: both tiles store the sample and both must
/// open it, or the seam between them would keep a one-sample strip of surface
/// across every cave mouth that straddles a tile boundary.
pub fn surface_cut_samples(
    terrain: &TerrainData,
    terrain_origin: DVec3,
    shape: &VoxelShape,
) -> Vec<SurfaceSample> {
    let mut out = Vec::new();
    for_each_cut_sample(terrain, terrain_origin, shape, |coord, i, j, _| {
        out.push(SurfaceSample { coord, j, i });
    });
    out
}

/// `true` when the cut reaches the heightfield surface anywhere — i.e. when
/// [`surface_cut_samples`] would return anything.
///
/// **The gate the carve tools ask before committing anything.** A cut that stays
/// below the surface (a cave under a hillside) or entirely above it (a brush
/// swung in mid-air) crosses nothing and needs no hole, which is what makes those
/// carves legal on a terrain that cannot persist a mask at all. Short-circuits on
/// the first sample rather than gathering the set.
pub fn cut_crosses_surface(
    terrain: &TerrainData,
    terrain_origin: DVec3,
    shape: &VoxelShape,
) -> bool {
    let mut hit = false;
    for_each_cut_sample(terrain, terrain_origin, shape, |_, _, _, _| hit = true);
    hit
}

/// One tile's changing samples, staged between the read pass and the write pass
/// of [`touch_surface_cut`]: `(tile coordinate, [(i, j), …])`.
type PendingTile = ((i32, i32), Vec<(u32, u32)>);

/// Open (`open = true`, a carve) or close (`open = false`, a fill) the samples
/// the cut crosses, recording the change into `builder`.
///
/// Returns how many samples actually **changed**, which is what a tool's readout
/// quotes; re-carving ground that is already holed changes nothing and returns
/// `0`. Only tiles with at least one changing sample are taken `&mut`, so an
/// idempotent dab neither bumps a tile version nor marks it for write-back.
///
/// Every touched tile is [healed](inf_terrain::TerrainTile::heal_holes) on the
/// way out, so the fill that closes the last hole leaves the tile serializing
/// byte-identically to one that was never carved rather than carrying a
/// `resolution²/8` block of zeros forever.
///
/// The `builder` is threaded in rather than returned so a **stroke** — many dabs
/// under one mouse-down — merges into a single reversible record whose `before`
/// is the state at the first touch. Exactly the contract
/// [`HoleDeltaBuilder`](inf_terrain::HoleDeltaBuilder) documents, and the reason
/// a stroke undoes in one step to where it started.
pub fn touch_surface_cut(
    terrain: &mut TerrainData,
    terrain_origin: DVec3,
    shape: &VoxelShape,
    open: bool,
    builder: &mut HoleDeltaBuilder,
) -> usize {
    let res = terrain.tile_resolution();
    // Two phases, because the predicate reads the tile and the write needs it
    // `&mut`: gather the changing samples per tile first, then write them.
    // Gathering only what *changes* is what keeps an idempotent dab from
    // dirtying a tile it did not alter.
    let mut pending: Vec<PendingTile> = Vec::new();
    let mut last: Option<(i32, i32)> = None;
    for_each_cut_sample(terrain, terrain_origin, shape, |coord, i, j, was_hole| {
        if was_hole == open {
            return;
        }
        if last != Some(coord) {
            pending.push((coord, Vec::new()));
            last = Some(coord);
        }
        if let Some((_, samples)) = pending.last_mut() {
            samples.push((i, j));
        }
    });

    let mut changed = 0;
    for (coord, samples) in pending {
        for &(i, j) in &samples {
            // `before` is the pre-edit flag, which is `!open` for every sample
            // that reaches here — but written as the read value rather than as
            // `!open`, because the builder's contract is "the flag at the first
            // touch" and a later dab must not be able to lie about it.
            builder.touch(coord, i, j, !open);
        }
        let Some(tile) = terrain.get_tile_mut(coord) else {
            continue;
        };
        for &(i, j) in &samples {
            tile.set_hole(res, i, j, open);
        }
        tile.heal_holes();
        changed += samples.len();
    }
    changed
}

/// [`touch_surface_cut`] for a single cut, returning the finished reversible
/// record — the one-shot form a spline tunnel or a scripted carve wants.
///
/// An empty [`HoleDelta`] means the cut crossed nothing (or changed nothing),
/// which callers treat as "there is no heightfield half to this edit" rather than
/// as a failure.
pub fn apply_surface_cut(
    terrain: &mut TerrainData,
    terrain_origin: DVec3,
    shape: &VoxelShape,
    open: bool,
) -> HoleDelta {
    let mut builder = HoleDeltaBuilder::new();
    touch_surface_cut(terrain, terrain_origin, shape, open, &mut builder);
    builder.finalize(terrain)
}

/// What a document is about to lose: holes carried by a terrain whose container
/// cannot persist them.
///
/// See [`inline_hole_advisory`] for the whole argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InlineHoleAdvisory {
    /// Resident level-0 tiles carrying at least one holed sample.
    pub tiles: usize,
}

impl InlineHoleAdvisory {
    /// The line to show the author. Names the loss and names the fix, because an
    /// advisory that only says "something is wrong" costs a support round trip.
    pub fn message(&self) -> String {
        format!(
            "This terrain has cave mouths on {} tile(s) but is stored INLINE in the level, \
             and an inline terrain cannot persist its hole mask — saving and reloading will \
             seal them. Convert the terrain to asset-backed (import or export it as a \
             .inf_terrain) to keep them.",
            self.tiles
        )
    }
}

/// Detect the one state a level can be in that silently loses geometry: an
/// **inline** terrain that is carrying holes.
///
/// The scene schema is frozen at v19, so `TerrainData`'s wire form is pinned at
/// tile generation 3 and an `.inf_lvl` drops the hole mask by construction (see
/// `TerrainTileFrozenV3`). The carve tools refuse to *create* that state; this
/// finds a document that arrived in it anyway — an older session, a hand-edited
/// file, a level whose terrain was converted back to inline after a carve — so
/// the author is told **before** a save quietly seals every cave mouth in the
/// level rather than after a reload.
///
/// `asset_backed` is the caller's answer to "does this terrain page from a
/// `.inf_terrain`?" (`Terrain.asset.is_some()` in the editor). It is a parameter
/// rather than a read because Ring 0 has no ECS: keeping the *rule* here and the
/// *lookup* at the call site is what lets the editor, a cook advisory and a test
/// all reach the same verdict.
///
/// `None` means "nothing to say" — an asset-backed terrain, or an inline one with
/// no holes, which is every level written before P21.2.
pub fn inline_hole_advisory(
    terrain: &TerrainData,
    asset_backed: bool,
) -> Option<InlineHoleAdvisory> {
    if asset_backed {
        return None;
    }
    let tiles = terrain.tiles().filter(|(_, t)| t.has_holes()).count();
    if tiles == 0 {
        return None;
    }
    Some(InlineHoleAdvisory { tiles })
}

/// The shared walk behind every entry point above: visit every authored sample
/// whose surface point is inside `shape`, in `(tile, row, column)` order, with
/// its current hole flag.
///
/// Written once because the four public functions differ only in what they do
/// with a hit — and because a second copy of THE RULE is exactly how a carve and
/// its fill would come to disagree about which samples they own.
fn for_each_cut_sample(
    terrain: &TerrainData,
    terrain_origin: DVec3,
    shape: &VoxelShape,
    mut visit: impl FnMut((i32, i32), u32, u32, bool),
) {
    if !shape.is_valid() {
        return;
    }
    let res = terrain.tile_resolution();
    let mps = terrain.meters_per_sample();
    // The cut's world AABB, moved into the terrain's own frame once. No band
    // margin: unlike the voxel op — which widens by `SDF_BAND` so the gradient
    // around the new surface is written — a hole is a predicate, and a sample
    // outside the shape is simply not inside it.
    let (lo_w, hi_w) = shape.aabb_m(0.0);
    if !(lo_w.is_finite() && hi_w.is_finite()) {
        return;
    }
    let lo = lo_w - terrain_origin;
    let hi = hi_w - terrain_origin;

    // Walk the authored tiles rather than the AABB's tile-coordinate range: the
    // resident set is bounded by residency, an AABB is bounded by whatever
    // radius a UI slider allowed, and a `BTreeMap` walk is deterministic either
    // way. Tiles whose span misses the box are skipped before any sample work.
    let span = terrain.tile_span();
    for (&coord, tile) in terrain.tiles() {
        // `tile_origin_xz` returns a `DVec2` whose `y` is the world **Z** — the
        // XZ-plane convention every terrain query here uses. The box is a
        // `DVec3`, so its third axis is `z`. Mixing the two (testing `lo.y`, the
        // *vertical* extent, against a tile's Z span) silently skips every tile
        // whose Z happens not to overlap the cut's height, which reads as "the
        // carve punched no mouth" and nothing else.
        let (ox, oz) = {
            let o = terrain.tile_origin_xz(coord);
            (o.x, o.y)
        };
        if ox + span < lo.x || ox > hi.x || oz + span < lo.z || oz > hi.z {
            continue;
        }
        // Clip the sample rectangle to the box. `floor`/`ceil` rather than a
        // round: the box is inclusive and a sample exactly on its face must be
        // tested, not dropped.
        let Some((i0, i1)) = sample_range(lo.x - ox, hi.x - ox, mps, res) else {
            continue;
        };
        let Some((j0, j1)) = sample_range(lo.z - oz, hi.z - oz, mps, res) else {
            continue;
        };
        let base_y = tile.origin.y + terrain_origin.y;
        for j in j0..=j1 {
            let wz = oz + j as f64 * mps + terrain_origin.z;
            for i in i0..=i1 {
                let wx = ox + i as f64 * mps + terrain_origin.x;
                let p = DVec3::new(wx, base_y + tile.sample(res, i, j) as f64, wz);
                if shape.distance_m(p) < 0.0 {
                    visit(coord, i, j, tile.is_hole(res, i, j));
                }
            }
        }
    }
}

/// Inclusive sample-index range covering the local-axis interval `[lo, hi]`
/// (already relative to the tile's origin), or `None` when it misses the tile.
fn sample_range(lo: f64, hi: f64, mps: f64, res: u32) -> Option<(u32, u32)> {
    let last = (res - 1) as f64;
    let a = (lo / mps).floor().clamp(0.0, last);
    let b = (hi / mps).ceil().clamp(0.0, last);
    if b < a {
        return None;
    }
    Some((a as u32, b as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_terrain::asset::encode_tile;

    /// The fixture's height field: a cheap deterministic ripple so the surface is
    /// **not a plane** (a plane would let a wrong height rule pass). Polynomial
    /// rather than trigonometric on purpose — this crate calls no `std` trig
    /// anywhere, tests included, and `f32`/`f64` `sin` is not bit-portable
    /// (the P14 law).
    fn height(x: f64, z: f64) -> f64 {
        let (u, v) = (x * 0.13, z * 0.11);
        4.0 + u - u * u * 0.25 + v * 0.7 + u * v * 0.15
    }

    /// A terrain of `n × n` tiles at 1 m spacing over [`height`].
    fn terrain(n: i32, res: u32) -> TerrainData {
        let mut t = TerrainData::new(res, 1.0);
        for tz in 0..n {
            for tx in 0..n {
                t.author_tile((tx, tz), height);
            }
        }
        t
    }

    /// Every tile's bytes — what "byte-identical" is actually asserted against.
    /// `encode_tile` and NOT `TerrainData`'s serde: the scene wire is pinned at
    /// tile generation 3 and drops the hole mask on purpose (schema v19 is
    /// frozen), so a round trip through it would "pass" by losing the very thing
    /// under test. `.inf_terrain` is the container that carries holes, and this
    /// is its encoder.
    fn image(t: &TerrainData) -> Vec<((i32, i32), Vec<u8>)> {
        t.tiles()
            .map(|(&c, tile)| (c, encode_tile(tile).unwrap()))
            .collect()
    }

    /// The independent predicate: brute-force every sample of every tile against
    /// THE RULE, without going near `for_each_cut_sample`.
    fn brute_force(t: &TerrainData, origin: DVec3, shape: &VoxelShape) -> Vec<SurfaceSample> {
        let res = t.tile_resolution();
        let mps = t.meters_per_sample();
        let mut out = Vec::new();
        for (&coord, tile) in t.tiles() {
            let o = t.tile_origin_xz(coord);
            for j in 0..res {
                for i in 0..res {
                    let p = DVec3::new(
                        o.x + i as f64 * mps + origin.x,
                        tile.origin.y + tile.sample(res, i, j) as f64 + origin.y,
                        o.y + j as f64 * mps + origin.z,
                    );
                    if shape.distance_m(p) < 0.0 {
                        out.push(SurfaceSample { coord, j, i });
                    }
                }
            }
        }
        out
    }

    /// A deterministic 64-bit LCG — the property tests' randomness. No `rand`
    /// dependency and no wall clock: a failure here has to be reproducible from
    /// the seed printed in the assertion, or it is not a gate.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
        /// Uniform-ish `f64` in `[lo, hi)`.
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (self.next() >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
        }
    }

    /// A random cut somewhere over a 3-tile terrain: sometimes a ball, sometimes
    /// a box, sometimes a capsule, sometimes above the surface, sometimes below.
    fn random_shape(rng: &mut Lcg) -> VoxelShape {
        let c = DVec3::new(
            rng.range(-4.0, 24.0),
            rng.range(0.0, 8.0),
            rng.range(-4.0, 24.0),
        );
        match rng.next() % 3 {
            0 => VoxelShape::Sphere {
                center: c,
                radius_m: rng.range(0.5, 6.0),
            },
            1 => VoxelShape::Box {
                center: c,
                half_extents: DVec3::new(
                    rng.range(0.5, 4.0),
                    rng.range(0.5, 4.0),
                    rng.range(0.5, 4.0),
                ),
            },
            _ => VoxelShape::Capsule {
                a: c,
                b: c + DVec3::new(
                    rng.range(-6.0, 6.0),
                    rng.range(-3.0, 3.0),
                    rng.range(-6.0, 6.0),
                ),
                radius_m: rng.range(0.5, 3.0),
            },
        }
    }

    /// **The rule gate.** The walk's answer is the independently-computed
    /// predicate's answer, exactly — same samples, same order, no extras, none
    /// missed — over 200 random cuts.
    #[test]
    fn the_covered_set_is_exactly_the_independent_predicate() {
        let t = terrain(3, 17);
        let origin = DVec3::new(2.0, 1.0, -3.0);
        let mut rng = Lcg(0x212C_0000_0001);
        let mut crossings = 0;
        for case in 0..200 {
            let shape = random_shape(&mut rng);
            let got = surface_cut_samples(&t, origin, &shape);
            let want = brute_force(&t, origin, &shape);
            assert_eq!(got, want, "case {case}: {shape:?}");
            assert_eq!(
                cut_crosses_surface(&t, origin, &shape),
                !want.is_empty(),
                "case {case}: the gate disagrees with the set it gates"
            );
            crossings += usize::from(!want.is_empty());
        }
        assert!(
            crossings > 20,
            "only {crossings}/200 random cuts reached the surface — the fixture is \
             not exercising the crossing case"
        );
    }

    /// Determinism: the same cut over the same terrain answers identically, and
    /// the answer does not depend on the order the tiles were authored in.
    #[test]
    fn the_covered_set_is_deterministic_and_authoring_order_free() {
        let shape = VoxelShape::Sphere {
            center: DVec3::new(8.0, 4.0, 8.0),
            radius_m: 3.0,
        };
        let a = terrain(3, 17);
        let first = surface_cut_samples(&a, DVec3::ZERO, &shape);
        assert!(!first.is_empty(), "the fixture must cross the surface");
        assert_eq!(first, surface_cut_samples(&a, DVec3::ZERO, &shape));

        // Same tiles, authored back to front.
        let mut b = TerrainData::new(17, 1.0);
        for tz in (0..3).rev() {
            for tx in (0..3).rev() {
                b.author_tile((tx, tz), height);
            }
        }
        assert_eq!(first, surface_cut_samples(&b, DVec3::ZERO, &shape));
    }

    /// **The invertibility gate, on bytes.** Carve then fill the same region and
    /// every tile is byte-identical to the terrain nobody ever touched — mask
    /// buffer gone, not merely zeroed.
    #[test]
    fn carve_then_fill_leaves_byte_identical_tiles() {
        let mut rng = Lcg(0x212C_0000_0002);
        let mut checked = 0;
        for case in 0..120 {
            let mut t = terrain(3, 17);
            let origin = DVec3::new(-1.0, 0.5, 2.0);
            let before = image(&t);
            let shape = random_shape(&mut rng);

            let carve = apply_surface_cut(&mut t, origin, &shape, true);
            if carve.is_empty() {
                continue; // the cut never reached the surface
            }
            checked += 1;
            assert!(t.has_holes(), "case {case}: the carve opened nothing");
            assert!(carve.net_opened() > 0, "case {case}");
            assert_ne!(image(&t), before, "case {case}");

            let fill = apply_surface_cut(&mut t, origin, &shape, false);
            assert_eq!(
                fill.net_opened(),
                -carve.net_opened(),
                "case {case}: the fill closed a different number of samples than the \
                 carve opened"
            );
            assert!(!t.has_holes(), "case {case}");
            assert_eq!(
                image(&t),
                before,
                "case {case}: carve → fill left a scar ({shape:?})"
            );

            // …and the record itself round-trips: reverting the carve is the same
            // as filling it.
            let mut u = terrain(3, 17);
            let carve2 = apply_surface_cut(&mut u, origin, &shape, true);
            assert_eq!(
                carve2, carve,
                "case {case}: the record is not deterministic"
            );
            u.revert_hole_delta(&carve2);
            assert_eq!(image(&u), before, "case {case}: undo left a scar");
            u.apply_hole_delta(&carve2);
            assert!(u.has_holes(), "case {case}: redo opened nothing");
        }
        assert!(
            checked > 15,
            "only {checked}/120 random cuts reached the surface — the fixture is not \
             exercising the round trip"
        );
    }

    /// A cut wholly below the surface — the cave under the hillside — crosses
    /// nothing, so it is legal on a terrain that cannot persist a mask. Likewise
    /// one wholly above it.
    #[test]
    fn a_cut_that_misses_the_surface_opens_nothing() {
        let mut t = terrain(2, 17);
        let before = image(&t);
        for (label, shape) in [
            (
                "buried",
                VoxelShape::Sphere {
                    center: DVec3::new(8.0, -20.0, 8.0),
                    radius_m: 5.0,
                },
            ),
            (
                "airborne",
                VoxelShape::Sphere {
                    center: DVec3::new(8.0, 40.0, 8.0),
                    radius_m: 5.0,
                },
            ),
            (
                "off to the side",
                VoxelShape::Box {
                    center: DVec3::new(-500.0, 4.0, 8.0),
                    half_extents: DVec3::splat(3.0),
                },
            ),
        ] {
            assert!(!cut_crosses_surface(&t, DVec3::ZERO, &shape), "{label}");
            assert!(
                surface_cut_samples(&t, DVec3::ZERO, &shape).is_empty(),
                "{label}"
            );
            assert!(
                apply_surface_cut(&mut t, DVec3::ZERO, &shape, true).is_empty(),
                "{label}"
            );
        }
        assert_eq!(image(&t), before, "a missing cut must not dirty a byte");
    }

    /// Re-carving ground that is already holed changes nothing, dirties nothing
    /// and records nothing — the idempotence a stroke's overlapping dabs rely on.
    #[test]
    fn a_second_identical_carve_changes_nothing() {
        let mut t = terrain(2, 17);
        let shape = VoxelShape::Sphere {
            center: DVec3::new(8.0, 4.0, 8.0),
            radius_m: 4.0,
        };
        let first = apply_surface_cut(&mut t, DVec3::ZERO, &shape, true);
        assert!(!first.is_empty());
        let settled = image(&t);
        t.clear_dirty();

        let mut builder = HoleDeltaBuilder::new();
        assert_eq!(
            touch_surface_cut(&mut t, DVec3::ZERO, &shape, true, &mut builder),
            0,
            "the second carve changed samples"
        );
        assert!(builder.is_empty());
        assert_eq!(image(&t), settled);
        assert!(
            t.dirty_tiles().is_empty(),
            "an idempotent dab marked tiles for write-back it did not change"
        );
    }

    /// A cut that straddles a shared tile edge opens the sample in **both** tiles
    /// — the seam rule. Without it every cave mouth crossing a tile boundary
    /// would keep a one-sample strip of surface across it.
    #[test]
    fn a_cut_on_a_shared_edge_opens_both_tiles() {
        let res = 9u32;
        let mut t = terrain(2, res);
        let span = t.tile_span();
        // Centred exactly on the tile-(0,0)/(1,0) seam.
        let shape = VoxelShape::Sphere {
            center: DVec3::new(span, 4.0, span * 0.5),
            radius_m: 2.5,
        };
        let hits = surface_cut_samples(&t, DVec3::ZERO, &shape);
        assert!(hits.iter().any(|s| s.coord == (0, 0) && s.i == res - 1));
        assert!(hits.iter().any(|s| s.coord == (1, 0) && s.i == 0));

        apply_surface_cut(&mut t, DVec3::ZERO, &shape, true);
        let left = t.get_tile((0, 0)).unwrap();
        let right = t.get_tile((1, 0)).unwrap();
        let mid = res / 2;
        assert!(
            left.is_hole(res, res - 1, mid),
            "the seam column of the left tile"
        );
        assert!(
            right.is_hole(res, 0, mid),
            "the seam column of the right tile"
        );
    }

    /// The terrain's world placement really moves the cut: the same shape over
    /// the same tiles opens a different set once the terrain is somewhere else.
    #[test]
    fn the_terrain_origin_is_honoured() {
        let t = terrain(2, 17);
        let shape = VoxelShape::Sphere {
            center: DVec3::new(8.0, 4.0, 8.0),
            radius_m: 3.0,
        };
        let here = surface_cut_samples(&t, DVec3::ZERO, &shape);
        assert!(!here.is_empty());
        // Move the terrain 1 km east: the cut is now over empty space.
        assert!(surface_cut_samples(&t, DVec3::new(1000.0, 0.0, 0.0), &shape).is_empty());
        // …and lifting the terrain out from under the ball also empties it.
        assert!(surface_cut_samples(&t, DVec3::new(0.0, 500.0, 0.0), &shape).is_empty());
    }

    /// A degenerate brush (a zero radius, a NaN dragged in from a UI) opens
    /// nothing rather than a plane of holes — the twin of
    /// `a_degenerate_shape_is_a_clean_no_op` one crate layer over.
    #[test]
    fn a_degenerate_shape_opens_nothing() {
        let mut t = terrain(2, 17);
        let before = image(&t);
        for shape in [
            VoxelShape::Sphere {
                center: DVec3::new(8.0, 4.0, 8.0),
                radius_m: 0.0,
            },
            VoxelShape::Sphere {
                center: DVec3::new(8.0, 4.0, 8.0),
                radius_m: f64::NAN,
            },
            VoxelShape::Box {
                center: DVec3::splat(f64::INFINITY),
                half_extents: DVec3::splat(1.0),
            },
        ] {
            assert!(!cut_crosses_surface(&t, DVec3::ZERO, &shape), "{shape:?}");
            assert!(apply_surface_cut(&mut t, DVec3::ZERO, &shape, true).is_empty());
        }
        assert_eq!(image(&t), before);
    }

    /// The defensive advisory fires exactly for the state the tools refuse to
    /// create: an inline terrain carrying holes.
    #[test]
    fn the_inline_hole_advisory_fires_only_on_the_lossy_state() {
        let mut t = terrain(2, 17);
        assert_eq!(
            inline_hole_advisory(&t, false),
            None,
            "no holes, nothing to say"
        );
        assert_eq!(inline_hole_advisory(&t, true), None);

        apply_surface_cut(
            &mut t,
            DVec3::ZERO,
            &VoxelShape::Sphere {
                center: DVec3::new(8.0, 4.0, 8.0),
                radius_m: 3.0,
            },
            true,
        );
        assert!(t.has_holes());
        assert_eq!(
            inline_hole_advisory(&t, true),
            None,
            "an asset-backed terrain persists its holes — no advisory"
        );
        let note = inline_hole_advisory(&t, false).expect("an inline terrain with holes");
        assert!(note.tiles > 0);
        let msg = note.message();
        assert!(
            msg.contains(".inf_terrain"),
            "the advisory must name the fix: {msg}"
        );
    }

    /// A sample exactly on the brush surface stays CLOSED — the documented
    /// boundary convention, pinned so a future `<=` cannot slip in unnoticed and
    /// silently break invertibility at the rim.
    #[test]
    fn a_sample_exactly_on_the_brush_surface_stays_closed() {
        let mut t = TerrainData::new(3, 1.0);
        t.author_tile((0, 0), |_, _| 0.0);
        // A unit ball centred one metre above sample (0, 0): its surface passes
        // exactly through that sample.
        let shape = VoxelShape::Sphere {
            center: DVec3::new(0.0, 1.0, 0.0),
            radius_m: 1.0,
        };
        let hits = surface_cut_samples(&t, DVec3::ZERO, &shape);
        assert!(
            !hits.iter().any(|s| s.i == 0 && s.j == 0),
            "the sample on the brush surface was opened"
        );
        // Sink the ball by a hair and it opens.
        let inside = VoxelShape::Sphere {
            center: DVec3::new(0.0, 0.999, 0.0),
            radius_m: 1.0,
        };
        assert!(surface_cut_samples(&t, DVec3::ZERO, &inside)
            .iter()
            .any(|s| s.i == 0 && s.j == 0));
    }
}
