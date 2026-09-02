//! The lowered runtime shape of a `building.plan` node, and its evaluation.
//!
//! A [`BuildingPass`] is to [`build`] what a
//! [`GrammarPass`] is to
//! [`expand_span`](crate::grammar::expand_span): everything the node authored,
//! resolved against the evaluating volume at run time and not before.
//!
//! # Where it rides, and why not in `PcgDocument`
//!
//! Exactly where P19.4's grammar passes ride — [`LoweredPcg::buildings`]
//! (crate::graph::LoweredPcg), beside the document rather than inside it. The
//! argument is P19.4's, unchanged: `PcgDocument` is the frozen v2 `.inf_pcg`
//! wire and bincode is positional, so growing it by one field makes every
//! committed graph fail to *decode*. Since P19.3 the authored graph JSON is the
//! source of truth and every evaluation site re-lowers it, so a pass reaching
//! the runtime this way is exactly as available as a rule.
//!
//! # The lot
//!
//! A building needs a rectangle. It gets one of three ways, in order:
//!
//! 1. a **span** connected to the node's `lot` pin — the XZ bounding box of
//!    whatever [`build_spans`] produces, so a
//!    spline-derived lot and a footprint-derived lot both work and neither
//!    needed a new concept. **This is the closure P19.4's remainder pointed at**
//!    ("a biome is a painted *region* and a grammar needs a *span*"): a region
//!    reaches a building through the same span seam a fence uses.
//! 2. the node's own `size_x`/`size_z`, when either is positive;
//! 3. the evaluating volume's own extent — the P19.4 footprint default, so a
//!    building dropped on a `PcgVolume` matches the box the author already
//!    sized.

use glam::DVec2;

use super::palettes::ArchetypeId;
use super::{build_in, BuildingParams, Rect2};
use crate::grammar::expand::{build_spans, GrammarContext, GrammarOutput, GrammarPass, Ground};
use crate::grammar::span::SplineSource;
use crate::grammar::SpanSource;
use crate::hash::Hash64;
use crate::height::HeightProvider;

/// Separates a building pass's seed space from the grammar's, so a
/// `grammar.expand` and a `building.plan` carrying the same authored seed on the
/// same volume do not draw the same numbers.
const BUILDING_PASS_SALT: u64 = 0x0062_6C64_5F70_6173; // "bld_pas"

/// One lowered building pass — the runtime shape of a `building.plan` node.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingPass {
    /// The pass name (the node's `name` param) — diagnostics and ordering.
    pub name: String,
    /// The layer this pass was lowered under, so a disabled layer disables its
    /// buildings exactly as it disables its rules and its grammars.
    pub layer: String,
    pub enabled: bool,
    pub archetype: ArchetypeId,
    pub seed: u64,
    /// Storey override; `0` draws from the archetype's own range.
    pub floors: u32,
    /// Populate rooms with furniture.
    pub furnish: bool,
    /// Explicit lot size in metres; a zero axis falls back to the volume's own
    /// extent.
    pub size: DVec2,
    /// A span whose XZ bounds become the lot, when the node's `lot` pin is
    /// connected.
    pub lot: Option<SpanSource>,
    /// **Subdivision rules** (IB-2c): when set, the span on the `lot` pin is a
    /// *block*, and this pass builds one building per lot the block is cut into
    /// rather than one building on the block.
    ///
    /// `None` is the pre-I3 behaviour exactly — one pass, one lot, one building —
    /// so every committed graph lowers to the population it always did.
    pub lots: Option<super::LotRules>,
    pub ground: Ground,
    pub altitude_offset: f64,
}

/// One lot a pass will build on, with the seed the building standing on it draws
/// from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedLot {
    pub lot: crate::building::OrientedLot,
    pub seed: u64,
}

/// The seed one pass runs under: its authored seed folded with the building salt
/// and the evaluating volume's own seed.
///
/// Stated in one place so the editor and the player cannot derive it
/// differently — the same discipline as
/// [`grammar::pass_seed`](crate::grammar::pass_seed).
pub fn pass_seed(pass_seed: u64, volume_seed: u64) -> u64 {
    Hash64::new(pass_seed)
        .mix_u64(BUILDING_PASS_SALT)
        .mix_u64(volume_seed)
        .finish()
}

/// The lot this pass builds on, as a world-axis-aligned rectangle.
///
/// **Kept for callers that want a bounding box and know it.** The lot a building
/// is actually planned on is [`oriented_lot_of`]; this is its world AABB, which
/// for an identity frame is the same rectangle.
pub fn lot_of(pass: &BuildingPass, splines: &dyn SplineSource, cx: &GrammarContext) -> Rect2 {
    let lot = oriented_lot_of(pass, splines, cx);
    if lot.frame.is_identity() {
        return lot.rect;
    }
    let mut min = DVec2::splat(f64::INFINITY);
    let mut max = DVec2::splat(f64::NEG_INFINITY);
    for c in lot.world_corners() {
        min = min.min(c);
        max = max.max(c);
    }
    Rect2 { min, max }
}

/// **The lot, in its own frame** (IB-6).
///
/// A span's points are hulled and fitted with an oriented minimum-area rectangle
/// ([`inf_math::min_area_rect`]), so a footprint that is not on the compass grid
/// gets the rectangle it actually is rather than the bounding box of its
/// corners. Vancouver's West End and downtown are both rotated; the axis-aligned
/// box of a 30 × 10 lot turned off the grid is **780 m² against 300**, measured
/// in `inf_math::obb2`'s own arm.
///
/// The volume-box fall-through — a `building.plan` node with **no** `lot` pin,
/// which is what every committed sample uses — keeps the world axes and the
/// identity frame, so nothing in the tree moves.
pub fn oriented_lot_of(
    pass: &BuildingPass,
    splines: &dyn SplineSource,
    cx: &GrammarContext,
) -> crate::building::OrientedLot {
    use crate::building::{LotFrame, OrientedLot};

    if let Some(source) = &pass.lot {
        let pts = span_points(pass, source, splines, cx);
        if let Some(mar) = inf_math::min_area_rect(&pts) {
            if mar.half.x > 0.0 && mar.half.y > 0.0 {
                return OrientedLot {
                    rect: Rect2 {
                        min: -mar.half,
                        max: mar.half,
                    },
                    frame: LotFrame::new(mar.center, mar.u),
                };
            }
        }
        // A span that resolved to nothing — or to a line with no area — falls
        // through to the volume's box rather than building a zero-size building
        // on top of the origin.
    }
    let sx = if pass.size.x > 0.0 {
        pass.size.x
    } else {
        cx.extent.x * 2.0
    };
    let sz = if pass.size.y > 0.0 {
        pass.size.y
    } else {
        cx.extent.y * 2.0
    };
    OrientedLot::axis_aligned(Rect2::from_center(
        DVec2::new(cx.center.x, cx.center.z),
        DVec2::new(sx, sz),
    ))
}

/// The world-XZ points a pass's `lot` span resolves to — the block polygon.
///
/// Hoisted out of [`oriented_lot_of`] so the *subdivider* fits the block's own
/// hull rather than re-fitting a rectangle that has already thrown the hull away:
/// [`min_area_rect`](inf_math::min_area_rect) is a lossy summary of a polygon,
/// and a lot-containment test run against it could never refuse a corner lot on
/// a chamfered block.
fn span_points(
    pass: &BuildingPass,
    source: &SpanSource,
    splines: &dyn SplineSource,
    cx: &GrammarContext,
) -> Vec<DVec2> {
    // `build_spans` needs a `GrammarPass` shell — only its `span` field is read.
    let shell = GrammarPass {
        name: pass.name.clone(),
        layer: pass.layer.clone(),
        enabled: true,
        seed: 0,
        grammar: Default::default(),
        axiom: String::new(),
        span: source.clone(),
        corner_module: String::new(),
        ground: pass.ground,
        altitude_offset: 0.0,
    };
    let set = build_spans(&shell, splines, cx);
    let mut pts: Vec<DVec2> = Vec::new();
    for span in &set.spans {
        for p in span.points() {
            pts.push(DVec2::new(p.x, p.z));
        }
    }
    for f in &set.corners {
        pts.push(DVec2::new(f.position.x, f.position.z));
    }
    pts
}

/// **Every lot this pass builds on** (IB-2c) — one, or one per subdivided lot.
///
/// This is the function that retires the certification's *"one `building.plan`
/// node is one building"*. With [`BuildingPass::lots`] unset it returns exactly
/// one lot with exactly the seed the pass has always drawn, so no committed
/// content moves; with it set, the `lot` span is treated as a **block** and cut
/// by [`subdivide_block`](crate::building::subdivide_block).
///
/// The per-lot seeds come from the subdivider, which folds the lot's `(col,
/// row)` into the block seed — `plan.rs` states that two buildings handed the
/// same seed *are* the same building, so a district of identical towers is
/// exactly what a shared seed would produce.
pub fn oriented_lots_of(
    pass: &BuildingPass,
    splines: &dyn SplineSource,
    cx: &GrammarContext,
) -> Vec<PlacedLot> {
    let seed = pass_seed(pass.seed, cx.seed_offset);
    let Some(rules) = pass.lots else {
        return vec![PlacedLot {
            lot: oriented_lot_of(pass, splines, cx),
            seed,
        }];
    };
    let Some(source) = &pass.lot else {
        // Subdivision with no block to subdivide: the volume's own box is the
        // block, which is the same fall-through the single-lot path takes.
        let lot = oriented_lot_of(pass, splines, cx);
        let block = lot.world_corners().to_vec();
        return placed(&block, &rules, seed);
    };
    let pts = span_points(pass, source, splines, cx);
    placed(&pts, &rules, seed)
}

fn placed(block: &[DVec2], rules: &super::LotRules, seed: u64) -> Vec<PlacedLot> {
    crate::building::subdivide_block(block, rules, seed)
        .lots
        .into_iter()
        .map(|l| PlacedLot {
            lot: l.lot,
            seed: l.seed,
        })
        .collect()
}

/// Everything one building needs, resolved against the world — the ONE
/// resolution [`evaluate_buildings_in`] and [`plans_of`] share.
///
/// Before I3 these two functions each spelled the lot resolution, the datum
/// lookup and the seed fold out for themselves, which is two authorities on
/// "what does this pass build" and exactly the shape
/// `plans_match_what_evaluation_builds` exists to catch after the fact. Now the
/// arm is a tautology on purpose.
struct BuildJob {
    params: BuildingParams,
    frame: crate::building::LotFrame,
    seed: u64,
    furnish: bool,
}

fn jobs_of(
    passes: &[BuildingPass],
    splines: &dyn SplineSource,
    height: &dyn HeightProvider,
    cx: &GrammarContext,
) -> Vec<BuildJob> {
    let mut jobs = Vec::new();
    for pass in passes.iter().filter(|p| p.enabled) {
        for placed in oriented_lots_of(pass, splines, cx) {
            let lot = placed.lot;
            if !lot.rect.is_positive() {
                continue;
            }
            // The datum is asked for at the lot's centre **in the world**, which
            // is where the ground is; the plan is built in the lot's frame.
            let c = lot.frame.to_world(lot.rect.center());
            let base = match pass.ground {
                Ground::Span => cx.center.y,
                // Fail closed, exactly like a scattered instance over a hole:
                // no ground under the footprint centre means no building, not a
                // building at y = 0.
                Ground::Terrain => match height.height(c.x, c.y) {
                    Some(h) => h,
                    None => continue,
                },
            } + pass.altitude_offset;
            jobs.push(BuildJob {
                params: BuildingParams {
                    archetype: pass.archetype,
                    footprint: lot.rect,
                    base_y: base,
                    seed: placed.seed,
                    floors: pass.floors,
                },
                frame: lot.frame,
                seed: placed.seed,
                furnish: pass.furnish,
            });
        }
    }
    jobs
}

/// Evaluate every enabled building pass on the process-wide job pool.
pub fn evaluate_buildings(
    passes: &[BuildingPass],
    splines: &dyn SplineSource,
    height: &dyn HeightProvider,
    cx: &GrammarContext,
) -> GrammarOutput {
    evaluate_buildings_in(inf_core::global(), passes, splines, height, cx)
}

/// [`evaluate_buildings`] on a caller-supplied pool — the seam the determinism
/// guard drives.
///
/// Passes are mapped through [`inf_core::parallel_map`] (a deterministic
/// in-order pure map) and concatenated in pass order, so the population is
/// byte-identical for any worker count. Each pass's own assembly is serial:
/// splitting one building across workers would nest pools, and a building is
/// already a small unit of work beside a hundred-thousand-instance scatter.
pub fn evaluate_buildings_in(
    pool: &inf_core::JobPool,
    passes: &[BuildingPass],
    splines: &dyn SplineSource,
    height: &dyn HeightProvider,
    cx: &GrammarContext,
) -> GrammarOutput {
    // The lot and the datum are resolved OUTSIDE the pool: `SplineSource` and
    // `HeightProvider` are `&dyn`, and the world walk behind them is the one
    // part of the path each host writes for itself.
    let jobs = jobs_of(passes, splines, height, cx);
    if jobs.is_empty() {
        return GrammarOutput::default();
    }
    // **The building ordinal is the job index** (NPC1d), zipped on before the
    // map rather than counted after it: `parallel_map` is a deterministic
    // in-order map, so a job's position IS its building's name for the whole
    // call, and a slot can carry that name without anything downstream having to
    // re-derive it from a list length.
    let indexed: Vec<(u32, BuildJob)> = jobs
        .into_iter()
        .enumerate()
        .map(|(i, j)| (i as u32, j))
        .collect();
    let per: Vec<GrammarOutput> = pool.parallel_map(indexed, |(ordinal, job)| {
        let out = build_in(&job.params, job.frame, job.seed, job.furnish);
        // **One building is one group** (IB-2b): the shell is derived here,
        // where the lot frame is in hand, and the ranges are local to this
        // chunk — `GrammarOutput::extend` re-bases them as the chunks fold.
        let groups = match super::lod::group_shell(job.frame, &out.colliders) {
            Some(shell) => vec![super::StructureGroup {
                shell,
                start: 0,
                len: out.colliders.len() as u32,
                inst_start: 0,
                inst_len: out.instances.len() as u32,
            }],
            None => Vec::new(),
        };
        // I6: the doors this building wants, derived from the plan **here**,
        // in the one pass that has one — `build_in` throws it away, so a
        // doorway that was not taken now can never be taken at all.
        let mut doorways = super::doorway::doorways_of(&out.plan);
        super::doorway::place_doorways_in_frame(&mut doorways, job.frame);
        // NPC1d: and the people, derived from the same plan in the same place —
        // a slot is already in world metres because `interior_nav` and
        // `slots_of` both go through the plan's own frame, which is why there is
        // no `place_slots_in_frame` beside the doorway call above.
        let salt = super::society::building_salt(cx.entity, ordinal);
        let mut slots = super::society::slots_of(&out.plan, ordinal, salt);
        // VEN1b: …and the places its own FURNITURE offers, which a plan cannot
        // name. Appended after the plan's own, so a room's slots stay in one
        // run and the pre-venue slot order is untouched by construction.
        slots.extend(super::society::station_slots(
            &out.plan,
            &out.stations,
            ordinal,
            salt,
        ));
        // A building nobody lives or works in contributes no interior: a
        // route's endpoints are slots, and carrying a warehouse's corridors
        // into a level's network is nodes nothing ever asks for.
        let interior = if slots.is_empty() {
            inf_nav::NavGraph::new()
        } else {
            out.plan.interior_nav_in(salt)
        };
        GrammarOutput {
            instances: out.instances,
            colliders: out.colliders,
            groups,
            doorways,
            slots,
            interior,
            // The stations, carried whole (VEN1b): the occupied ones are
            // already slots above, and `compose_volume` keeps the rest as the
            // volume's emitters. Carried rather than filtered here because a
            // `GrammarOutput` is what `extend` folds, and a list that meant one
            // thing before the fold and another after it is the shape of defect
            // `groups`' own re-basing exists to avoid.
            stations: out.stations,
            // The rig the assembler hung (VEN1a). `assemble_in` has already
            // moved every fixture into the lot's frame beside the instances
            // and the colliders, so these are world metres like everything
            // else here.
            lights: out.lights,
            // `build_in` has already folded the assembler's decoration tail
            // into `instances` (I8b), so a building's panes are inside its own
            // group's instance range and there is nothing left here.
            decor: Vec::new(),
        }
    });
    let mut out = GrammarOutput::default();
    for chunk in per {
        out.extend(chunk);
    }
    out
}

/// The [`BuildingPlan`](super::BuildingPlan)s a pass list would build — the same
/// resolution [`evaluate_buildings_in`] performs, without assembling anything.
///
/// This exists so a gate (and a future debug view) can assert the *plan*
/// invariants — connectivity, reachability, opening clearance — against exactly
/// the plans the shipped content builds, rather than against a re-derived guess.
pub fn plans_of(
    passes: &[BuildingPass],
    splines: &dyn SplineSource,
    height: &dyn HeightProvider,
    cx: &GrammarContext,
) -> Vec<super::BuildingPlan> {
    jobs_of(passes, splines, height, cx)
        .iter()
        .map(|job| super::plan::plan_building_in(&job.params, job.frame))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::span::{NoSplines, SplinePath};
    use crate::grammar::{FootprintMode, SplineInterp};
    use crate::height::FnHeight;
    use glam::DVec3;

    fn flat(y: f64) -> FnHeight<impl Fn(f64, f64) -> Option<f64> + Send + Sync> {
        FnHeight::new(move |_, _| Some(y))
    }

    fn pass() -> BuildingPass {
        BuildingPass {
            name: "b".into(),
            layer: "layer".into(),
            enabled: true,
            archetype: ArchetypeId::House,
            seed: 3,
            floors: 2,
            furnish: true,
            size: DVec2::new(20.0, 14.0),
            lot: None,
            lots: None,
            ground: Ground::Terrain,
            altitude_offset: 0.0,
        }
    }

    fn ctx() -> GrammarContext {
        GrammarContext {
            entity: None,
            center: DVec3::new(100.0, 7.0, -50.0),
            extent: DVec2::new(30.0, 25.0),
            seed_offset: 11,
        }
    }

    #[test]
    fn the_lot_falls_back_from_the_span_to_the_size_to_the_volume() {
        let cx = ctx();
        // 2. explicit size, centred on the volume.
        let sized = lot_of(&pass(), &NoSplines, &cx);
        assert_eq!(sized.size(), DVec2::new(20.0, 14.0));
        assert_eq!(sized.center(), DVec2::new(100.0, -50.0));
        // 3. the volume's own extent when both axes are zero.
        let bare = lot_of(
            &BuildingPass {
                size: DVec2::ZERO,
                ..pass()
            },
            &NoSplines,
            &cx,
        );
        assert_eq!(bare.size(), DVec2::new(60.0, 50.0));
        // 1. a connected footprint span wins over both.
        let spanned = lot_of(
            &BuildingPass {
                lot: Some(SpanSource::Footprint {
                    size: DVec2::new(8.0, 6.0),
                    mode: FootprintMode::Perimeter { corner_size: 0.0 },
                }),
                ..pass()
            },
            &NoSplines,
            &cx,
        );
        assert_eq!(spanned.size(), DVec2::new(8.0, 6.0));
        assert_eq!(spanned.center(), DVec2::new(100.0, -50.0));
    }

    /// **IB-6: a rotated lot builds a rotated building, and the world says so.**
    ///
    /// The claim is not that a number in a struct changed — it is that the
    /// *walls* run along the lot's edges. So the arm reads the assembled
    /// population and asserts, for every placed box:
    ///
    /// * its centre is inside the lot rectangle (not merely inside the lot's
    ///   bounding box, which is 2.6× larger and is exactly the wrong answer this
    ///   item exists to retire);
    /// * its rotation carries the lot's basis, so its faces are parallel to the
    ///   lot's edges.
    ///
    /// The axis-aligned alternative is priced in the test, in square metres.
    #[test]
    fn a_rotated_lot_builds_a_rotated_building() {
        // A 24 × 12 lot turned by the 3-4-5 rotation (cos 0.8, sin 0.6), which
        // is exact in binary and needs no trigonometry.
        let (c, s) = (0.8f64, 0.6f64);
        let centre = DVec2::new(140.0, -70.0);
        let turn = |p: DVec2| DVec2::new(p.x * c - p.y * s, p.x * s + p.y * c) + centre;
        let ring: Vec<DVec3> = [(-12.0, -6.0), (12.0, -6.0), (12.0, 6.0), (-12.0, 6.0)]
            .iter()
            .map(|&(x, z)| {
                let w = turn(DVec2::new(x, z));
                DVec3::new(w.x, 0.0, w.y)
            })
            .collect();

        let p = BuildingPass {
            lot: Some(SpanSource::Polyline {
                points: ring.clone(),
                closed: true,
            }),
            ground: Ground::Span,
            floors: 2,
            ..pass()
        };
        let cx = GrammarContext {
            entity: None,
            center: DVec3::new(centre.x, 0.0, centre.y),
            extent: DVec2::new(40.0, 40.0),
            seed_offset: 11,
        };

        let lot = oriented_lot_of(&p, &NoSplines, &cx);
        assert!(
            (lot.rect.size() - DVec2::new(24.0, 12.0)).length() < 1e-9,
            "the lot is 24 x 12 however it is turned; got {:?}",
            lot.rect.size()
        );
        assert!((lot.frame.origin - centre).length() < 1e-9);
        assert!(
            (lot.frame.u - DVec2::new(c, s)).length() < 1e-9,
            "the frame's +X is the lot's long side: {:?}",
            lot.frame.u
        );
        assert!(!lot.frame.is_identity());

        // THE ALTERNATIVE, PRICED: the axis-aligned box of the same lot.
        let aabb = lot_of(&p, &NoSplines, &cx);
        assert!(
            aabb.area() > lot.rect.area() * 1.5,
            "the axis-aligned lot is {:.1} m2 against the oriented {:.1} — if that \
             ratio is not large this fixture is not rotated",
            aabb.area(),
            lot.rect.area()
        );
        println!(
            "IB-6 lot: oriented {:.1} m2 vs axis-aligned {:.1} m2",
            lot.rect.area(),
            aabb.area()
        );

        let out = evaluate_buildings(std::slice::from_ref(&p), &NoSplines, &flat(0.0), &cx);
        assert!(out.colliders.len() > 20, "{} boxes", out.colliders.len());

        // (a) Every box is inside the LOT, not merely inside its bounding box.
        let mut outside_lot = 0usize;
        let mut outside_aabb = 0usize;
        for col in &out.colliders {
            let w = DVec2::new(col.center.x, col.center.z);
            let l = lot.frame.to_local(w);
            if l.x.abs() > lot.rect.max.x + 0.6 || l.y.abs() > lot.rect.max.y + 0.6 {
                outside_lot += 1;
            }
            if w.x < aabb.min.x - 0.6
                || w.x > aabb.max.x + 0.6
                || w.y < aabb.min.y - 0.6
                || w.y > aabb.max.y + 0.6
            {
                outside_aabb += 1;
            }
        }
        assert_eq!(outside_lot, 0, "{outside_lot} boxes escaped the lot");
        assert_eq!(outside_aabb, 0);

        // (b) **The walls are parallel to the lot's edges, not to the axes.**
        // Every placed box's rotation must map the lot's basis onto a world axis
        // of its own faces: rotating local +X by the collider's own quaternion
        // gives a direction that is (anti)parallel to `u` or to `v`.
        let u = DVec3::new(lot.frame.u.x, 0.0, lot.frame.u.y);
        let v3 = lot.frame.v();
        let v = DVec3::new(v3.x, 0.0, v3.y);
        let mut aligned_to_lot = 0usize;
        let mut aligned_to_axes = 0usize;
        for col in &out.colliders {
            let f = col.rotation * DVec3::X;
            let par = |d: DVec3| f.dot(d).abs() > 0.999_9;
            if par(u) || par(v) {
                aligned_to_lot += 1;
            }
            if par(DVec3::X) || par(DVec3::Z) {
                aligned_to_axes += 1;
            }
        }
        assert_eq!(
            aligned_to_lot,
            out.colliders.len(),
            "every box must be square to the LOT; {aligned_to_lot} of {} are",
            out.colliders.len()
        );
        assert_eq!(
            aligned_to_axes,
            0,
            "and NONE may be square to the world axes — {aligned_to_axes} of {} \
             still are, which is the defect IB-6 names",
            out.colliders.len()
        );

        // (c) The control: the same lot un-rotated builds the same building.
        let flat_ring: Vec<DVec3> = [(-12.0, -6.0), (12.0, -6.0), (12.0, 6.0), (-12.0, 6.0)]
            .iter()
            .map(|&(x, z)| DVec3::new(centre.x + x, 0.0, centre.y + z))
            .collect();
        let straight = BuildingPass {
            lot: Some(SpanSource::Polyline {
                points: flat_ring,
                closed: true,
            }),
            ..p.clone()
        };
        let plain = evaluate_buildings(&[straight], &NoSplines, &flat(0.0), &cx);
        assert_eq!(
            plain.colliders.len(),
            out.colliders.len(),
            "turning a lot must not change WHAT is built, only where it faces"
        );
        for (a, b) in plain.colliders.iter().zip(&out.colliders) {
            assert!(
                (a.half_extents - b.half_extents).length() < 1e-9,
                "the same building, turned: {:?} vs {:?}",
                a.half_extents,
                b.half_extents
            );
        }
    }

    /// A **spline** lot: the closure P19.4's remainder asked for — a curve's
    /// bounding box becomes a building's footprint, with no new concept.
    #[test]
    fn a_spline_span_becomes_a_lot() {
        let guid = uuid::Uuid::from_u128(9);
        let mut splines = std::collections::HashMap::new();
        splines.insert(
            guid,
            SplinePath {
                points: vec![
                    DVec3::new(0.0, 0.0, 0.0),
                    DVec3::new(30.0, 0.0, 0.0),
                    DVec3::new(30.0, 0.0, 18.0),
                ],
                closed: false,
                interp: SplineInterp::Linear,
            },
        );
        let lot = lot_of(
            &BuildingPass {
                lot: Some(SpanSource::Spline {
                    entity: Some(guid),
                    samples_per_segment: 4,
                }),
                ..pass()
            },
            &splines,
            &ctx(),
        );
        assert_eq!(lot.min, DVec2::ZERO);
        assert_eq!(lot.max, DVec2::new(30.0, 18.0));
        // An unresolvable spline falls back rather than building at the origin.
        let fallback = lot_of(
            &BuildingPass {
                lot: Some(SpanSource::Spline {
                    entity: Some(uuid::Uuid::from_u128(404)),
                    samples_per_segment: 4,
                }),
                ..pass()
            },
            &NoSplines,
            &ctx(),
        );
        assert_eq!(fallback.center(), DVec2::new(100.0, -50.0));
    }

    #[test]
    fn a_pass_evaluates_onto_the_terrain_and_fails_closed_over_a_hole() {
        let cx = ctx();
        let out = evaluate_buildings(&[pass()], &NoSplines, &flat(42.0), &cx);
        assert!(!out.is_empty());
        // The datum is the terrain under the footprint CENTRE, once.
        let lowest = out
            .colliders
            .iter()
            .map(|s| s.y_band().0)
            .fold(f64::INFINITY, f64::min);
        assert!((lowest - (42.0 - 0.2)).abs() < 1e-9, "lowest {lowest}");
        // No ground ⇒ no building.
        let hole = FnHeight::new(|_, _| None);
        assert!(evaluate_buildings(&[pass()], &NoSplines, &hole, &cx).is_empty());
        // `Ground::Span` takes the volume's own Y instead and needs no terrain.
        let spanned = evaluate_buildings(
            &[BuildingPass {
                ground: Ground::Span,
                ..pass()
            }],
            &NoSplines,
            &hole,
            &cx,
        );
        assert!(!spanned.is_empty());
        // A disabled pass builds nothing — the layer toggle's mechanism.
        assert!(evaluate_buildings(
            &[BuildingPass {
                enabled: false,
                ..pass()
            }],
            &NoSplines,
            &flat(0.0),
            &cx
        )
        .is_empty());
    }

    /// Pool-size invariance, the P7.0 guard applied to the building path.
    #[test]
    fn evaluation_is_invariant_under_pool_size() {
        let cx = ctx();
        let passes = vec![
            pass(),
            BuildingPass {
                archetype: ArchetypeId::Shop,
                seed: 9,
                ..pass()
            },
        ];
        let want = evaluate_buildings_in(
            &inf_core::JobPool::new(1),
            &passes,
            &NoSplines,
            &flat(3.0),
            &cx,
        );
        assert!(!want.is_empty());
        for workers in [2usize, 4, 8] {
            let got = evaluate_buildings_in(
                &inf_core::JobPool::new(workers),
                &passes,
                &NoSplines,
                &flat(3.0),
                &cx,
            );
            assert_eq!(want, got, "output moved at {workers} workers");
        }
    }

    /// The volume's own seed is folded in, so two volumes sharing one graph
    /// build different buildings — and the same volume rebuilds the same one.
    #[test]
    fn the_volume_seed_decorrelates_two_volumes() {
        assert_eq!(pass_seed(4, 7), pass_seed(4, 7));
        assert_ne!(pass_seed(4, 7), pass_seed(4, 8));
        assert_ne!(pass_seed(4, 7), pass_seed(5, 7));
        // And it is a different space from the grammar's, for the same inputs.
        assert_ne!(pass_seed(4, 7), crate::grammar::pass_seed(4, 7).finish());
    }

    /// **IB-2c: one `building.plan` node stops meaning one building.**
    ///
    /// The claim is not that a struct grew a field — it is that the *world* holds
    /// several separate buildings on separate lots, each with its own plan, each
    /// square to the block, none overlapping another. So the arm reads the
    /// assembled population and the resolved plans, and prices the alternative
    /// the wave retires (one building on the whole block) in square metres.
    #[test]
    fn a_block_pass_builds_one_building_per_lot() {
        let centre = DVec2::new(0.0, 0.0);
        let ring: Vec<DVec3> = [(-50.0, -30.0), (50.0, -30.0), (50.0, 30.0), (-50.0, 30.0)]
            .iter()
            .map(|&(x, z)| DVec3::new(centre.x + x, 0.0, centre.y + z))
            .collect();
        let block = BuildingPass {
            lot: Some(SpanSource::Polyline {
                points: ring,
                closed: true,
            }),
            ground: Ground::Span,
            floors: 2,
            size: DVec2::ZERO,
            ..pass()
        };
        let cx = GrammarContext {
            entity: None,
            center: DVec3::new(0.0, 0.0, 0.0),
            extent: DVec2::new(60.0, 40.0),
            seed_offset: 11,
        };

        // Without rules: one lot, the whole block — the pre-I3 answer, unmoved.
        let one = oriented_lots_of(&block, &NoSplines, &cx);
        assert_eq!(one.len(), 1);
        assert!((one[0].lot.rect.area() - 100.0 * 60.0).abs() < 1e-6);
        assert_eq!(one[0].seed, pass_seed(block.seed, cx.seed_offset));

        // With rules: 4 x 2 lots, each a building of its own.
        let sub = BuildingPass {
            lots: Some(crate::building::LotRules {
                frontage_m: 25.0,
                depth_m: 30.0,
                jitter: 0.0,
                setback_m: 2.0,
                min_area_m2: 0.0,
            }),
            ..block.clone()
        };
        let many = oriented_lots_of(&sub, &NoSplines, &cx);
        assert_eq!(many.len(), 8, "{} lots", many.len());

        let plans = plans_of(std::slice::from_ref(&sub), &NoSplines, &flat(0.0), &cx);
        assert_eq!(plans.len(), 8, "one plan per lot");
        for p in &plans {
            assert!(p.fully_reachable(), "every subdivided lot is enterable");
            assert!(!p.rooms.is_empty());
        }
        // Eight DIFFERENT buildings, not eight copies: the room counts differ.
        let mut shapes: Vec<usize> = plans.iter().map(|p| p.rooms.len()).collect();
        shapes.sort_unstable();
        shapes.dedup();
        assert!(
            shapes.len() > 1,
            "eight lots produced one building shape — the lot seeds are correlated"
        );

        let out = evaluate_buildings(&[sub], &NoSplines, &flat(0.0), &cx);
        assert_eq!(out.groups.len(), 8, "one structure group per building");
        // The groups partition the collider list exactly.
        let mut covered = 0usize;
        for (i, g) in out.groups.iter().enumerate() {
            assert_eq!(g.start as usize, covered, "group {i} is not contiguous");
            covered += g.len as usize;
        }
        assert_eq!(covered, out.colliders.len());
        let mut inst = 0usize;
        for g in &out.groups {
            assert_eq!(g.inst_start as usize, inst);
            inst += g.inst_len as usize;
        }
        assert_eq!(inst, out.instances.len());

        // **Every collider is inside its own group's shell — every CORNER of it,
        // on all three axes** (island wave I8b audit).
        //
        // This used to read `distance_xz_to_box(c.center, …)`, which is a
        // *centre* test in *two* axes: a part hanging a metre over the shell's
        // roof, or a slab whose own half-extent reached past the shell's rim,
        // both passed. It became load-bearing in wave I8b, which stopped the
        // parts casting shadows **because the shell contains them** — so the
        // sentence the caster clause rests on is now the sentence this arm
        // actually checks. `group_shell` takes the exact support bound, so the
        // corners are inside by construction and this is the arm that says so.
        for g in &out.groups {
            let inv = g.shell.rotation.inverse();
            for c in &out.colliders[g.range()] {
                let ch = c.half_extents;
                for k in 0..8u32 {
                    let s = DVec3::new(
                        if k & 1 == 0 { -1.0 } else { 1.0 },
                        if k & 2 == 0 { -1.0 } else { 1.0 },
                        if k & 4 == 0 { -1.0 } else { 1.0 },
                    );
                    let corner = c.center + c.rotation * (s * ch);
                    let local = inv * (corner - g.shell.center);
                    let over = local.abs() - g.shell.half_extents;
                    assert!(
                        over.max_element() < 1e-6,
                        "a part's corner sits {} m outside its own shell (local \
                         {local:?} against half-extents {:?})",
                        over.max_element(),
                        g.shell.half_extents
                    );
                }
            }
        }

        // THE ALTERNATIVE, PRICED: the whole block as one building.
        let whole = evaluate_buildings(&[block], &NoSplines, &flat(0.0), &cx);
        println!(
            "IB-2c: block -> {} buildings / {} solids / {} instances, against \
             1 building / {} solids for the same block",
            out.groups.len(),
            out.colliders.len(),
            out.instances.len(),
            whole.colliders.len()
        );
        assert_eq!(whole.groups.len(), 1);
    }

    /// The fan-out stays a pure map: any pool size, the same population.
    #[test]
    fn a_subdivided_block_is_invariant_under_pool_size() {
        let cx = GrammarContext {
            entity: None,
            center: DVec3::ZERO,
            extent: DVec2::new(50.0, 30.0),
            seed_offset: 3,
        };
        let sub = BuildingPass {
            size: DVec2::ZERO,
            lots: Some(crate::building::LotRules::default()),
            ground: Ground::Span,
            ..pass()
        };
        let want = evaluate_buildings_in(
            &inf_core::JobPool::new(1),
            std::slice::from_ref(&sub),
            &NoSplines,
            &flat(0.0),
            &cx,
        );
        assert!(want.groups.len() > 1, "{} groups", want.groups.len());
        for workers in [2usize, 4, 8] {
            let got = evaluate_buildings_in(
                &inf_core::JobPool::new(workers),
                std::slice::from_ref(&sub),
                &NoSplines,
                &flat(0.0),
                &cx,
            );
            assert_eq!(want, got, "output moved at {workers} workers");
        }
    }

    /// `plans_of` resolves exactly what `evaluate_buildings_in` assembles — the
    /// property that lets a gate assert plan invariants about shipped content.
    #[test]
    fn plans_match_what_evaluation_builds() {
        let cx = ctx();
        let passes = vec![pass()];
        let plans = plans_of(&passes, &NoSplines, &flat(5.0), &cx);
        assert_eq!(plans.len(), 1);
        let direct = crate::building::assemble(&plans[0], pass_seed(3, 11), true);
        let via = evaluate_buildings(&passes, &NoSplines, &flat(5.0), &cx);
        // The population is the same population. The **grouping** is not part of
        // that claim: `assemble` is handed a plan and knows nothing about lots,
        // so it emits no group — which is why the two are compared list by list
        // and the group is asserted against the lists rather than against
        // `direct`. Comparing the whole struct would have meant either a
        // vacuous group on both sides or a false failure here.
        assert_eq!(direct.instances, via.instances);
        assert_eq!(direct.colliders, via.colliders);
        assert!(direct.groups.is_empty(), "assemble does not group");
        assert_eq!(via.groups.len(), 1, "one pass, one lot, one group");
        assert_eq!(via.groups[0].range(), 0..via.colliders.len());
        assert_eq!(via.groups[0].instance_range(), 0..via.instances.len());
        assert!(plans[0].fully_reachable());
    }
}
