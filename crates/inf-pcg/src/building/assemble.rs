//! Turning a [`BuildingPlan`] into placed instances and solid boxes.
//!
//! # Walls are P19.4, unmodified
//!
//! Each wall run becomes a [`Span`] at its floor's own height and is handed to
//! [`expand_span`] with the archetype's grammar and `Ground::Span`. Everything
//! P19.4 proved about that path — the exact fill, the counter-hashed
//! alternatives, the truncation policy, the bit-portable orientation — applies
//! verbatim, because it is the same function.
//!
//! # An opening is a run that was never emitted
//!
//! A wall carrying openings is cut into the intervals *between* them, and each
//! interval is expanded independently:
//!
//! ```text
//!   |= = = = = = = = = = = = = = = = = = = = = = = = = = = = =|   wall run
//!   |= = = = = =|            |= = = = = = =|        |= = = = =|   emitted runs
//!               [   door    ]              [ window ]
//!                                          [parapet ]             below the sill
//!               [  lintel   ]              [ lintel ]             above the head
//! ```
//!
//! There is no subtraction, no boolean, and no "delete the modules that overlap
//! the door" pass — so a collider cannot survive inside a doorway by accident.
//! That is what makes [`BuildingPlan::opening_is_clear`] a check on arithmetic
//! rather than a hopeful assertion about a mesh operation.
//!
//! # A building levels its site
//!
//! Every Y in the output derives from [`BuildingPlan::base_y`], which the
//! *evaluation site* samples from the terrain once, at the footprint centre.
//! Slabs, walls and stairs therefore share one datum. Sampling the terrain per
//! module — the scatter idiom, and the right one for a fence — would make a
//! building's floors follow the hill, which is not a building.
//!
//! # Furniture is solid
//!
//! Furniture rides the same [`PcgCollider`] path as structure. A "fully
//! furnished, enterable building" whose desks you walk through is worse than one
//! whose doors you cannot, and the palettes already declare the boxes. The cost
//! is bounded by the per-room-type density knobs, and the benefit is that the
//! door-clearance rule below becomes **gated** rather than cosmetic: a piece of
//! furniture in a doorway would fail the same assertion a wall panel would.

use glam::{DVec2, DVec3};

use super::palettes::{archetype, BuildingArchetype, FurnitureDef, Placement};
use super::plan::BuildingParams;
use super::{BuildingPlan, Opening, OpeningKind, Rect2, Room, RoomType, Wall};
use crate::grammar::dsl::Grammar;
use crate::grammar::expand::{expand_span, GrammarOutput, GrammarPass, Ground, SpanSource};
use crate::grammar::span::{positive, yaw_onto, Span};
use crate::hash::Hash64;
use crate::height::HeightProvider;
use crate::scatter::PcgSurface;
use crate::scatter::{PcgCollider, PcgInstance};

/// Per-draw salts, so the wall runs, the furniture stations and the free-grid
/// jitter are decorrelated within one building.
const SALT_WALL: u64 = 0x5741_4C4C; // "WALL"
const SALT_FURN: u64 = 0x4655_524E; // "FURN"
const SALT_JIT_X: u64 = 0x4A49_5458; // "JITX"
const SALT_JIT_Z: u64 = 0x4A49_545A; // "JITZ"

/// Target riser height in metres — the step count is `floor_height / this`,
/// rounded, so a taller storey grows steps rather than stretching them.
const STEP_RISE: f64 = 0.18;
/// The most treads one flight may hold. A guard: at 0.18 m this is a 14 m storey.
const MAX_STEPS: u32 = 80;
/// The most pieces one room may hold, whatever the density asks for.
const MAX_FURNITURE_PER_ROOM: usize = 64;
/// The most stations a furniture walk visits in one room.
const MAX_STATIONS: usize = 512;
/// Metres of air a wall-aligned piece keeps between its back and the wall face.
///
/// Not cosmetic: without it a piece's back plane lands *exactly* on the void
/// boundary of any window above it, and `(a + b) - b == a` is not an identity in
/// binary floating point — a one-ulp overshoot would read as a blocked window.
/// Two centimetres is also what furniture does in the world.
const FURNITURE_WALL_GAP: f64 = 0.02;

/// **Walking room left on each side of a room-centre piece** (wave VEN1a).
///
/// A stage clamped to the room's full inner width is a floor, not a stage: the
/// patrons in the reference stand *around* the catwalk, and the benches at its
/// edge need somewhere to be. 1.2 m is a person and a shoulder-turn either side.
const CENTRE_MARGIN_M: f64 = 1.2;

/// **Wall left clear at each end of a continuous run** (wave VEN1a).
///
/// A counter that reaches the corners of the room it is in seals two walls,
/// which would put the bartender's own side out of reach of the doorway. 0.9 m
/// is a person's width -- the same number `door_width` hovers around.
const RUN_END_MARGIN_M: f64 = 0.9;

/// The shortest half-length a continuous run may be, metres (wave VEN1a).
///
/// Below this it is not a run any more: a 0.6 m counter is a station, which is
/// the thing `Placement::Run` exists to stop being. An edge whose longest clear
/// stretch cannot carry one is refused and the next edge tried.
const RUN_MIN_HALF_M: f64 = 1.0;

/// How much a room-centre piece shrinks per attempt when it fouls an opening
/// void or a door swing, and how many attempts it gets (wave VEN1a).
///
/// A centred piece cannot be *moved* — being centred is what it is — so the
/// only honest answers are "smaller" and "not at all", and three attempts at
/// 70 % take a 5 m catwalk to 1.7 m before giving up.
const CENTRE_SHRINK: f64 = 0.7;
const CENTRE_SHRINK_TRIES: u32 = 3;

/// How far a street-face festoon reaches past each jamb of the door it hangs
/// over, metres (wave VEN1a).
const FESTOON_OVERHANG_M: f64 = 0.6;
/// Half-height of a street-face festoon swag, metres.
const FESTOON_HALF_H_M: f64 = 0.14;
/// Half-depth of a street-face festoon swag, metres.
const FESTOON_HALF_D_M: f64 = 0.09;
/// How far above an entrance's head a festoon hangs, metres.
const FESTOON_ABOVE_HEAD_M: f64 = 0.35;

/// Where furniture may **not** stand, on one floor.
///
/// Both lists exist because they are different claims. `voids` is the
/// enterability invariant itself — literally the rectangles
/// [`BuildingPlan::opening_is_clear`] tests — so a piece rejected by it cannot
/// later fail that assertion. `swings` is an ergonomic clearance around a door,
/// which is larger and which no assertion depends on.
#[derive(Default)]
struct Blockers {
    voids: Vec<Rect2>,
    swings: Vec<Rect2>,
}

/// A building's assembled contents, with the plan that produced them.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingOutput {
    pub plan: BuildingPlan,
    pub instances: Vec<PcgInstance>,
    /// The solid boxes — walls, slabs, stairs, lintels, parapets and furniture.
    pub colliders: Vec<PcgCollider>,
}

/// The height seam expansion demands but a building never uses: every module is
/// placed with [`Ground::Span`], whose Y comes from the span itself.
struct Levelled;

impl HeightProvider for Levelled {
    fn height(&self, _x: f64, _z: f64) -> Option<f64> {
        None
    }
    fn normal(&self, _x: f64, _z: f64) -> Option<DVec3> {
        None
    }
}

/// Plan and assemble one building in a single call — the shape both evaluation
/// sites use.
pub fn build(params: &BuildingParams, seed: u64, furnish: bool) -> BuildingOutput {
    build_in(params, crate::building::LotFrame::IDENTITY, seed, furnish)
}

/// [`build`], on a lot with its own frame (IB-6).
///
/// `params.footprint` is read in `frame`'s coordinates; the returned
/// instances and colliders are in the world.
pub fn build_in(
    params: &BuildingParams,
    frame: crate::building::LotFrame,
    seed: u64,
    furnish: bool,
) -> BuildingOutput {
    let plan = crate::building::plan::plan_building_in(params, frame);
    let out = assemble(&plan, seed, furnish);
    BuildingOutput {
        plan,
        instances: out.instances,
        colliders: out.colliders,
    }
}

/// Assemble `plan` on the process-wide job pool.
pub fn assemble(plan: &BuildingPlan, seed: u64, furnish: bool) -> GrammarOutput {
    assemble_in(inf_core::global(), plan, seed, furnish)
}

/// [`assemble`] on a caller-supplied pool — the seam the determinism guard
/// drives, mirroring [`evaluate_grammars_in`](crate::grammar::evaluate_grammars_in).
///
/// Floors are mapped through [`inf_core::parallel_map`] (a deterministic in-order
/// pure map) and concatenated in floor order, then the roof and the stairs are
/// appended. The output is therefore byte-identical for any worker count.
pub fn assemble_in(
    pool: &inf_core::JobPool,
    plan: &BuildingPlan,
    seed: u64,
    furnish: bool,
) -> GrammarOutput {
    let arch = archetype(plan.archetype);
    let Ok(grammar) = arch.grammar() else {
        // A shipped palette that does not parse is a programming error, caught
        // by `palettes::tests::every_palette_parses`. Failing closed here keeps
        // it from being a panic in the shipped player.
        return GrammarOutput::default();
    };
    if plan.rooms.is_empty() {
        return GrammarOutput::default();
    }
    let hash = Hash64::new(seed);
    let ctx = Ctx {
        plan,
        arch,
        exterior: wall_pass(&grammar, arch.exterior_axiom),
        interior: wall_pass(&grammar, arch.interior_axiom),
        grammar: &grammar,
        hash,
    };

    let floors: Vec<u32> = (0..plan.floors).collect();
    let per: Vec<GrammarOutput> = pool.parallel_map(floors, |f| ctx.floor(f, furnish));
    let mut out = GrammarOutput::default();
    for chunk in per {
        out.extend(chunk);
    }
    ctx.roof(&mut out);
    ctx.stairs(&mut out);
    ctx.street_face(&mut out);
    // **The decoration tail, folded in exactly once** (island wave I8b). Every
    // instance up to this point has a collider beside it at the same index;
    // everything appended here has none. Doing it before `place_in_frame` is
    // what puts the panes in the lot's world frame with the rest.
    let decor = std::mem::take(&mut out.decor);
    out.instances.extend(decor);
    place_in_frame(&mut out, plan.frame);
    out
}

/// **The one place a lot's frame is applied** (IB-6).
///
/// Everything above this line planned and assembled in the lot's own
/// coordinates, where it is axis-aligned; this turns the finished output into
/// the world. One rotation per placed box, at one site — against an oriented
/// rectangle type through the slicer, the adjacency test, the wall builder, the
/// roof, the stairs and the furniture grid.
///
/// **The identity frame is skipped entirely**, so a level that already contains
/// grammar buildings is byte-identical by construction rather than by a
/// tolerance: `is_identity` is an exact comparison and the early return means
/// not one multiplication happens.
fn place_in_frame(out: &mut GrammarOutput, frame: crate::building::LotFrame) {
    if frame.is_identity() {
        return;
    }
    let yaw = frame.yaw();
    let map = |p: DVec3| {
        let xz = frame.to_world(glam::DVec2::new(p.x, p.z));
        DVec3::new(xz.x, p.y, xz.y)
    };
    for i in &mut out.instances {
        i.pos = map(i.pos);
        // The module's own rotation composes with the lot's: a wall run placed
        // by `expand_span` already carries a yaw onto its span, and that span
        // was in lot coordinates.
        i.rotation = yaw * i.rotation;
    }
    for c in &mut out.colliders {
        c.center = map(c.center);
        c.rotation = yaw * c.rotation;
    }
}

/// A synthesized pass: everything `expand_span` reads and nothing it does not.
fn wall_pass(grammar: &Grammar, axiom: &str) -> GrammarPass {
    GrammarPass {
        name: "wall".into(),
        layer: "building".into(),
        enabled: true,
        seed: 0,
        grammar: grammar.clone(),
        axiom: axiom.into(),
        // Unused: `expand_span` is handed a `Span` directly.
        span: SpanSource::Footprint {
            size: DVec2::ZERO,
            mode: super::super::grammar::FootprintMode::Perimeter { corner_size: 0.0 },
        },
        corner_module: String::new(),
        ground: Ground::Span,
        altitude_offset: 0.0,
    }
}

struct Ctx<'a> {
    plan: &'a BuildingPlan,
    arch: &'a BuildingArchetype,
    exterior: GrammarPass,
    interior: GrammarPass,
    grammar: &'a Grammar,
    hash: Hash64,
}

impl Ctx<'_> {
    /// Place one named module as a box of given world dimensions.
    ///
    /// Structural pieces — slabs, stairs, lintels — are dimensioned by the
    /// *plan*, not by the palette, so their collider is computed here and the
    /// module's own `collider` attribute is not consulted.
    ///
    /// **The standing P19.4 gap closes here** (island wave I8b). The instance
    /// carries the module's mesh GUID *and* the half-extents the plan just
    /// computed, so a 10 m slab is drawn as a 10 m slab. Before this it carried
    /// `scale: 1.0` and a uniform-scale primitive, which is why a settlement
    /// drew as a cloud of one-metre cubes whatever its colliders said.
    fn boxed(&self, out: &mut GrammarOutput, module: &str, center: DVec3, half: DVec3) {
        let Some(kind) = self.grammar.module_index(module) else {
            return;
        };
        if !(half.x > 0.0 && half.y > 0.0 && half.z > 0.0) {
            return;
        }
        out.instances
            .push(self.instance(kind, center, glam::DQuat::IDENTITY, half));
        out.colliders.push(PcgCollider {
            center,
            half_extents: half,
            rotation: glam::DQuat::IDENTITY,
        });
    }

    /// One placed instance of palette module `kind`, drawn at `half`.
    ///
    /// The one place the assembler builds a [`PcgInstance`], so the mesh, the
    /// extent and the glow are read off the module in one statement rather than
    /// at each of the five sites that place something.
    fn instance(
        &self,
        kind: u32,
        center: DVec3,
        rotation: glam::DQuat,
        half: DVec3,
    ) -> PcgInstance {
        self.instance_lit(kind, center, rotation, half, None)
    }

    /// [`instance`](Self::instance) with the module's family emission
    /// **overridden** (wave VEN1a).
    ///
    /// A family states the brightness a thing of its kind emits at and a
    /// palette states the hue: `Neon` is a bright plate in every archetype, and
    /// whether it is magenta or green is the strip club's business and the
    /// cocktail bar's. `None` keeps the family's own colour, which is what every
    /// structural placement passes.
    ///
    /// An override on a family that does **not** emit is honoured, deliberately:
    /// `FurnitureDef::emissive` is an author saying "this one glows", and
    /// refusing it silently would make an authored light invisible with no
    /// diagnostic. What it cannot do is dim a family to nothing by accident,
    /// because `None` and `Some([0,0,0])` are different values.
    fn instance_lit(
        &self,
        kind: u32,
        center: DVec3,
        rotation: glam::DQuat,
        half: DVec3,
        emissive: Option<[f32; 3]>,
    ) -> PcgInstance {
        let def = self.grammar.modules().get(kind as usize);
        let mut surface = def.map_or(PcgSurface::DEFAULT, |m| m.surface);
        if let Some(e) = emissive {
            surface.emissive = e;
        }
        PcgInstance {
            pos: center,
            rotation,
            scale: 1.0,
            kind_index: kind,
            mesh: def.and_then(|m| m.mesh),
            extent: Some([half.x as f32, half.y as f32, half.z as f32]),
            glow: def.map_or(0.0, |m| m.glow),
            // **Wave VEN1a**: the module's own surface, from the same
            // `ModuleDef` the mesh and the glow come from.
            surface,
        }
    }

    /// **THE STREET FACE** (wave VEN1a) — the neon over the door and the string
    /// lights across it, so a venue reads as a venue from the far pavement.
    ///
    /// Runs once for the whole building, not once per floor, because a building
    /// has one entrance. Both pieces go onto [`GrammarOutput::decor`] and
    /// **neither takes a collider**, for the reason
    /// [`super::palettes::BuildingArchetype::pane`] gives about the pane: a sign
    /// you cannot walk through is a wall, and `opening_is_clear` is an assertion
    /// about *solids*. That also keeps the alignment invariant intact -- the
    /// first `colliders.len()` instances are the solid ones and everything after
    /// them is decoration.
    ///
    /// # Why this is not a grammar rule
    ///
    /// A wall run's grammar places modules *in* the wall: every element of an
    /// alternative consumes span, so a `Bay -> Clad | Neon` would put a 1 m sign
    /// in a 4 m wall and leave the rest of that bay a **hole** -- a stretch of
    /// facade with no full-height solid, which is this engine's definition of a
    /// doorway. A sign is hung on a wall, not built into one.
    ///
    /// The plate stands out from the **outside** face of the entrance wall,
    /// which is the direction away from the room the door belongs to.
    /// `Wall::inside` names that room, so the outward normal is derived rather
    /// than authored -- a palette cannot know which way a lot happened to face.
    fn street_face(&self, out: &mut GrammarOutput) {
        let Some(sign) = self.arch.entrance_sign else {
            return;
        };
        let Some(wi) = self.plan.entrance else {
            return;
        };
        let Some(w) = self.plan.walls.get(wi) else {
            return;
        };
        let Some(op) = self
            .plan
            .openings
            .iter()
            .find(|o| o.wall == wi && o.kind == OpeningKind::Door)
        else {
            return;
        };
        let mid = w.point_at((op.start + op.end) * 0.5);
        let dir = w.direction();
        let along_x = dir.x.abs() >= dir.y.abs();
        // Outward: away from the room the wall belongs to. A degenerate case
        // (the room's centre exactly on the wall line) answers `+`, which is a
        // sign on one face rather than a sign nowhere.
        let inside_c = self
            .plan
            .rooms
            .get(w.inside)
            .map_or(mid, |r| r.rect.center());
        let outward = if along_x {
            if mid.y >= inside_c.y {
                1.0
            } else {
                -1.0
            }
        } else if mid.x >= inside_c.x {
            1.0
        } else {
            -1.0
        };
        let half_t = self.arch.wall_thickness * 0.5;
        let y = self.plan.floor_y(0);
        let mut place = |module: &str, half: DVec3, cy: f64, standoff: f64| {
            let Some(kind) = self.grammar.module_index(module) else {
                return;
            };
            if !(half.x > 0.0 && half.y > 0.0 && half.z > 0.0) {
                return;
            }
            let (h, p) = if along_x {
                (
                    half,
                    DVec2::new(mid.x, mid.y + outward * (half_t + standoff)),
                )
            } else {
                (
                    DVec3::new(half.z, half.y, half.x),
                    DVec2::new(mid.x + outward * (half_t + standoff), mid.y),
                )
            };
            let normal = if along_x {
                DVec3::new(0.0, 0.0, outward)
            } else {
                DVec3::new(outward, 0.0, 0.0)
            };
            out.decor.push(self.instance_lit(
                kind,
                DVec3::new(p.x, cy, p.y),
                yaw_onto(normal),
                h,
                Some(sign.colour),
            ));
        };
        place(
            sign.plate,
            DVec3::from(sign.half),
            y + sign.height_m,
            sign.half[2],
        );
        // The string lights hang across the whole doorway, a little above the
        // head -- the swag over the entrance in venues/0060.
        if let Some(f) = sign.festoon {
            let span = (op.width() * 0.5 + FESTOON_OVERHANG_M).min(w.length() * 0.5);
            place(
                f,
                DVec3::new(span, FESTOON_HALF_H_M, FESTOON_HALF_D_M),
                y + op.head + FESTOON_ABOVE_HEAD_M,
                FESTOON_HALF_D_M,
            );
        }
    }

    /// **The glazed leaf in a window void** (island wave I8b clause 3).
    ///
    /// Pushed onto [`GrammarOutput::decor`] and **not** beside a collider, for
    /// the reason [`super::palettes::BuildingArchetype::pane`] gives: a pane you
    /// cannot see through is a wall, and `opening_is_clear` is an assertion
    /// about solids. It fills the void exactly — the same rectangle the
    /// enterability invariant tests — so a window reads as glass rather than as
    /// a hole, and it is what carries the night glow.
    fn pane(&self, out: &mut GrammarOutput, op: &Opening) {
        if op.kind != OpeningKind::Window {
            return;
        }
        let Some(kind) = self.grammar.module_index(self.arch.pane) else {
            return;
        };
        let Some((rect, (y0, y1))) = self.plan.opening_void(op) else {
            return;
        };
        if !(rect.is_positive() && y1 > y0) {
            return;
        }
        let c = rect.center();
        out.decor.push(self.instance(
            kind,
            DVec3::new(c.x, (y0 + y1) * 0.5, c.y),
            glam::DQuat::IDENTITY,
            DVec3::new(rect.size_x() * 0.5, (y1 - y0) * 0.5, rect.size_z() * 0.5),
        ));
    }

    /// Everything on one storey: its slabs, its walls (with openings) and its
    /// furniture.
    fn floor(&self, floor: u32, furnish: bool) -> GrammarOutput {
        let mut out = GrammarOutput::default();
        let y = self.plan.floor_y(floor);
        let half_t = self.arch.slab_thickness * 0.5;

        // ── slabs ───────────────────────────────────────────────────────────
        for (_, room) in self.plan.rooms_on(floor) {
            // The stairwell is OPEN on every storey above the ground: that hole
            // is what a stair climbs through. Floor 0 keeps its slab — it is the
            // bottom of the well, not a shaft.
            if room.kind == RoomType::Stair && floor > 0 {
                continue;
            }
            let c = room.rect.center();
            self.boxed(
                &mut out,
                self.arch.slab,
                DVec3::new(c.x, y - half_t, c.y),
                DVec3::new(room.rect.size_x() * 0.5, half_t, room.rect.size_z() * 0.5),
            );
        }

        // ── walls ───────────────────────────────────────────────────────────
        for (wi, wall) in self.plan.walls.iter().enumerate() {
            if wall.floor != floor {
                continue;
            }
            self.wall(&mut out, wi, wall, y);
        }

        // ── furniture ───────────────────────────────────────────────────────
        //
        // The blockers are computed ONCE per floor, not per room: a piece in one
        // room can reach into a doorway on a wall it does not own (its footprint
        // is up to a metre deep), which is exactly the bug a per-room swing list
        // let through.
        if furnish {
            let blockers = self.blockers(floor);
            for (ri, room) in self.plan.rooms_on(floor) {
                self.furnish(&mut out, ri, room, y, &blockers);
            }
        }
        out
    }

    /// One wall: the solid runs between its openings, expanded by the grammar,
    /// plus a lintel over every opening and a parapet under every window.
    fn wall(&self, out: &mut GrammarOutput, wi: usize, wall: &Wall, y: f64) {
        let len = wall.length();
        if !positive(len) {
            return;
        }
        let pass = if wall.is_exterior() {
            &self.exterior
        } else {
            &self.interior
        };
        let base = self
            .hash
            .mix_u64(SALT_WALL)
            .mix_u64(wall.floor as u64)
            .mix_u64(wi as u64);

        // Openings on this wall, in run order. The plan guarantees they are
        // disjoint and inside `[0, len]` (gated by
        // `plan::tests::openings_are_disjoint_and_inside_their_runs`), so the
        // gaps below are a simple ordered walk rather than an interval merge.
        let mut ops: Vec<&Opening> = self.plan.openings.iter().filter(|o| o.wall == wi).collect();
        ops.sort_by(|a, b| {
            a.start
                .partial_cmp(&b.start)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cursor = 0.0f64;
        let mut run_index = 0u64;
        for op in &ops {
            self.run(out, wall, y, cursor, op.start, pass, base, &mut run_index);
            self.opening_trim(out, wall, y, op);
            cursor = op.end;
        }
        self.run(out, wall, y, cursor, len, pass, base, &mut run_index);
    }

    /// Expand one solid stretch `[from, to]` of a wall.
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        out: &mut GrammarOutput,
        wall: &Wall,
        y: f64,
        from: f64,
        to: f64,
        pass: &GrammarPass,
        base: Hash64,
        run_index: &mut u64,
    ) {
        if to - from <= 0.0 {
            return;
        }
        let a = wall.point_at(from);
        let b = wall.point_at(to);
        let span = Span::from_points([DVec3::new(a.x, y, a.y), DVec3::new(b.x, y, b.y)], false);
        let seed = base.mix_u64(*run_index);
        *run_index += 1;
        out.extend(expand_span(pass, &span, seed, &Levelled));
    }

    /// The lintel over an opening, and the parapet under a window.
    fn opening_trim(&self, out: &mut GrammarOutput, wall: &Wall, y: f64, op: &Opening) {
        let mid = wall.point_at((op.start + op.end) * 0.5);
        let dir = wall.direction();
        let half_len = op.width() * 0.5;
        let half_t = self.arch.wall_thickness * 0.5;
        // The wall's own axis: `dir` is `+X` or `+Z` for a v1 plan, so the box's
        // half-extents are just the two swapped.
        let along_x = dir.x.abs() >= dir.y.abs();
        let half_xz = |l: f64, t: f64| {
            if along_x {
                DVec3::new(l, 0.0, t)
            } else {
                DVec3::new(t, 0.0, l)
            }
        };

        // Lintel: from the opening's head up to the underside of the slab above.
        let head_to_ceiling = self.arch.floor_height - self.arch.slab_thickness - op.head;
        if head_to_ceiling > 0.0 {
            let h = half_xz(half_len, half_t);
            self.boxed(
                out,
                self.arch.lintel,
                DVec3::new(mid.x, y + op.head + head_to_ceiling * 0.5, mid.y),
                DVec3::new(h.x, head_to_ceiling * 0.5, h.z),
            );
        }
        // The pane, before the parapet, so the decoration tail is in opening
        // order whatever the trim does.
        self.pane(out, op);
        // Parapet: the solid wall below a window's sill.
        if op.kind == OpeningKind::Window && op.sill > 0.0 {
            let h = half_xz(half_len, half_t);
            self.boxed(
                out,
                self.arch.parapet,
                DVec3::new(mid.x, y + op.sill * 0.5, mid.y),
                DVec3::new(h.x, op.sill * 0.5, h.z),
            );
        }
    }

    /// The roof deck over the top storey — one slab across the whole footprint,
    /// because a stairwell that is open to the sky is a hole, not a feature.
    fn roof(&self, out: &mut GrammarOutput) {
        let top = self.plan.floor_y(self.plan.floors.saturating_sub(1)) + self.arch.floor_height;
        let c = self.plan.footprint.center();
        let half_t = self.arch.slab_thickness * 0.5;
        self.boxed(
            out,
            self.arch.roof,
            DVec3::new(c.x, top - half_t, c.y),
            DVec3::new(
                self.plan.footprint.size_x() * 0.5,
                half_t,
                self.plan.footprint.size_z() * 0.5,
            ),
        );
    }

    /// Every flight: a solid stepped ramp inside the stair core.
    ///
    /// Each tread is a block from the *lower* floor's surface up to its own top,
    /// so the flight is a solid stair rather than a set of floating slabs — the
    /// shape a character controller climbs without a ramp collider or a
    /// navmesh.
    fn stairs(&self, out: &mut GrammarOutput) {
        for flight in &self.plan.stairs {
            // **The flight is inset from the core**, and that is a correctness
            // requirement rather than a detail: a tread running wall-to-wall
            // stands in the doorway of every room that opens onto the stairwell,
            // which `BuildingPlan::opening_is_clear` would (rightly) call a
            // blockage. Insetting by the wall thickness leaves a landing strip
            // all the way round — which is also what a real stair has.
            let inset = self
                .arch
                .wall_thickness
                .min(flight.rect.size_x() * 0.2)
                .min(flight.rect.size_z() * 0.2);
            let rect = flight.rect.inset(inset);
            if !rect.is_positive() {
                continue;
            }
            let bottom = self.plan.floor_y(flight.from);
            let n = ((self.arch.floor_height / STEP_RISE).round() as u32).clamp(2, MAX_STEPS);
            let along_x = rect.size_x() >= rect.size_z();
            let (lo, hi, cross_half) = if along_x {
                (rect.min.x, rect.max.x, rect.size_z() * 0.5)
            } else {
                (rect.min.y, rect.max.y, rect.size_x() * 0.5)
            };
            let cross_c = if along_x {
                rect.center().y
            } else {
                rect.center().x
            };
            for k in 0..n {
                // Derived from `k`, never accumulated (the P17.4 rule): the last
                // tread's top is exactly the upper floor's surface.
                let top = bottom + self.arch.floor_height * (k as f64 + 1.0) / n as f64;
                let a = lo + (hi - lo) * (k as f64) / n as f64;
                let b = lo + (hi - lo) * (k as f64 + 1.0) / n as f64;
                let half_run = (b - a) * 0.5;
                let mid = (a + b) * 0.5;
                let half_h = (top - bottom) * 0.5;
                let (center, half) = if along_x {
                    (
                        DVec3::new(mid, bottom + half_h, cross_c),
                        DVec3::new(half_run, half_h, cross_half),
                    )
                } else {
                    (
                        DVec3::new(cross_c, bottom + half_h, mid),
                        DVec3::new(cross_half, half_h, half_run),
                    )
                };
                self.boxed(out, self.arch.step, center, half);
            }
        }
    }

    /// The door-swing rectangles on one room's floor: no furniture may stand in
    /// one.
    fn blockers(&self, floor: u32) -> Blockers {
        let reach = self.arch.door_width;
        let mut out = Blockers::default();
        for o in &self.plan.openings {
            let Some(w) = self.plan.walls.get(o.wall) else {
                continue;
            };
            if w.floor != floor {
                continue;
            }
            // **The void itself** — derived from the very function the gate
            // asserts with, so the placement rule and the invariant cannot drift
            // apart. Widened by the furniture gap so a piece that merely abuts
            // it is not resolved by a float comparison.
            if let Some((rect, _)) = self.plan.opening_void(o) {
                out.voids.push(rect.inset(-FURNITURE_WALL_GAP));
            }
            // The **swing** is a bigger, separate idea: room to open the door and
            // walk through, not merely to avoid standing in the hole. Doors only.
            if o.kind == OpeningKind::Door {
                let mid = w.point_at((o.start + o.end) * 0.5);
                out.swings
                    .push(Rect2::from_center(mid, DVec2::splat(reach * 2.0)));
            }
        }
        out
    }

    /// Populate one room from its type's furniture set.
    fn furnish(
        &self,
        out: &mut GrammarOutput,
        room_index: usize,
        room: &Room,
        y: f64,
        blockers: &Blockers,
    ) {
        let set = self.arch.furniture_for(room.kind);
        if set.is_empty() || !room.rect.is_positive() {
            return;
        }
        let base = self
            .hash
            .mix_u64(SALT_FURN)
            .mix_u64(room.floor as u64)
            .mix_u64(room_index as u64);
        // Placed centres, shared across the room's defs so two pieces never
        // occupy one spot.
        let mut placed: Vec<(DVec2, f64)> = Vec::new();
        for (di, def) in set.iter().enumerate() {
            let Some(kind) = self.grammar.module_index(def.module) else {
                continue;
            };
            let dh = base.mix_u64(di as u64);
            match def.place {
                Placement::Wall => {
                    self.wall_furniture(out, room, y, def, kind, dh, blockers, &mut placed, 0.0)
                }
                Placement::Mounted { height_m } => self.wall_furniture(
                    out,
                    room,
                    y,
                    def,
                    kind,
                    dh,
                    blockers,
                    &mut placed,
                    height_m,
                ),
                Placement::Free => {
                    self.free_furniture(out, room, y, def, kind, dh, blockers, &mut placed)
                }
                Placement::Centre => {
                    self.centre_furniture(out, room, y, def, kind, blockers, &mut placed)
                }
                Placement::Run => {
                    self.run_furniture(out, room, y, def, kind, blockers, &mut placed)
                }
            }
        }
    }

    /// **One piece at the room's centre** (wave VEN1a) — the stage on a dance
    /// floor, the catwalk in a strip club.
    ///
    /// Not density-driven and not hashed: a room has one stage in the middle of
    /// it, and "0.4 stages per 10 m², accepted on a hash" is a room that
    /// sometimes has none. The size is authored in metres and **clamped to the
    /// room**, so a 6 m catwalk in a 5 m room becomes a 4 m one rather than a
    /// solid that walls the room off.
    ///
    /// It registers its own footprint in `placed`, so the stools and benches
    /// that follow it in the same set keep out of it — which is how a bench
    /// ends up at the stage's EDGE rather than on top of it.
    #[allow(clippy::too_many_arguments)]
    fn centre_furniture(
        &self,
        out: &mut GrammarOutput,
        room: &Room,
        y: f64,
        def: &FurnitureDef,
        kind: u32,
        blockers: &Blockers,
        placed: &mut Vec<(DVec2, f64)>,
    ) {
        let inner = room
            .rect
            .inset(self.arch.wall_thickness * 0.5 + FURNITURE_WALL_GAP);
        if !inner.is_positive() {
            return;
        }
        // The clamp leaves a walking margin on each side, or the "stage" is a
        // floor and the room has no floor left to stand on.
        let half = DVec3::new(
            def.half[0].min(inner.size_x() * 0.5 - CENTRE_MARGIN_M),
            def.half[1],
            def.half[2].min(inner.size_z() * 0.5 - CENTRE_MARGIN_M),
        );
        if !(half.x > 0.0 && half.y > 0.0 && half.z > 0.0) {
            return;
        }
        let p = inner.center();
        // **A stage must keep out of a doorway too** (the audit this wave's own
        // gate performed on it). `wall_furniture` and `free_furniture` have
        // tested the blockers since P19.5 and the first cut of these two did
        // not, so a 5 m catwalk in a small room stood squarely in the one door
        // -- the building drew, the collider was solid, and `no_solid_ever_
        // blocks_a_doorway` went red. A centred piece cannot be *moved* (it is
        // centred), so the only honest answers are "smaller" and "not at all".
        let mut half = half;
        for _ in 0..CENTRE_SHRINK_TRIES {
            if self.clear_of_blockers(p, DVec2::new(half.x, half.z), blockers) {
                break;
            }
            half.x *= CENTRE_SHRINK;
            half.z *= CENTRE_SHRINK;
        }
        if !self.clear_of_blockers(p, DVec2::new(half.x, half.z), blockers)
            || !(half.x > 0.0 && half.z > 0.0)
        {
            return;
        }
        out.instances.push(self.instance_lit(
            kind,
            DVec3::new(p.x, y + half.y, p.y),
            glam::DQuat::IDENTITY,
            half,
            def.emissive,
        ));
        out.colliders.push(PcgCollider {
            center: DVec3::new(p.x, y + half.y, p.y),
            half_extents: half,
            rotation: glam::DQuat::IDENTITY,
        });
        // The clearance a follower must keep is the piece's own diagonal reach,
        // not an authored number: a 3 m stage that declared 0.6 m of clearance
        // would still get a stool standing on it.
        placed.push((p, half.x.hypot(half.z)));
    }

    /// **One continuous run along the room's longest inset edge** (wave VEN1a) —
    /// the bar counter.
    ///
    /// This is the clause the venue mandate names: a counter placed by
    /// [`Placement::Wall`] is a row of discrete 1.2 m boxes with hashed gaps
    /// between them, and a bar is one piece of joinery. The run's length is the
    /// edge's, clamped by the authored maximum; its back is on the wall and its
    /// `+Z` faces the room, exactly as a wall-stationed piece's does.
    #[allow(clippy::too_many_arguments)]
    fn run_furniture(
        &self,
        out: &mut GrammarOutput,
        room: &Room,
        y: f64,
        def: &FurnitureDef,
        kind: u32,
        blockers: &Blockers,
        placed: &mut Vec<(DVec2, f64)>,
    ) {
        let inner = room
            .rect
            .inset(self.arch.wall_thickness * 0.5 + FURNITURE_WALL_GAP);
        if !inner.is_positive() {
            return;
        }
        // **The counter takes the longest CLEAR STRETCH of the longest edge.**
        //
        // The first cut took the longest edge whole and refused it if anything
        // fouled it, which is wrong twice: a run spans a wall, and every door
        // swing reaches a metre out from the wall it is in, so a counter could
        // never share a wall with a door. On a bar room with doors on three
        // sides that produced **no counter at all** (measured: `Bar` at seed
        // 512 built one bar room and nothing to drink at), and a bar beside its
        // own door is what a bar looks like.
        //
        // So each edge is projected to an interval, the blockers that reach into
        // the counter's own depth band are subtracted from it, and the longest
        // surviving gap wins. Edges are tried longest first with a fixed index
        // tie-break; the result is a pure function of the room's geometry and
        // its openings, with no hash in it.
        let mid = inner.center();
        let edges = [
            // (back-line point, inward normal, the edge's own length)
            (DVec2::new(mid.x, inner.min.y), DVec2::Y, inner.size_x()),
            (DVec2::new(mid.x, inner.max.y), DVec2::NEG_Y, inner.size_x()),
            (DVec2::new(inner.min.x, mid.y), DVec2::X, inner.size_z()),
            (DVec2::new(inner.max.x, mid.y), DVec2::NEG_X, inner.size_z()),
        ];
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|a, b| edges[*b].2.total_cmp(&edges[*a].2).then(a.cmp(b)));
        let mut chosen: Option<(DVec2, DVec2, DVec3)> = None;
        for k in order {
            let (back, normal, _) = edges[k];
            let along_x = normal.x == 0.0;
            let depth = def.half[2] * 2.0;
            // The band the counter would occupy across the wall, and the span it
            // could occupy along it.
            let (lo, hi) = if along_x {
                (inner.min.x, inner.max.x)
            } else {
                (inner.min.y, inner.max.y)
            };
            let (b0, b1) = {
                let base = if along_x { back.y } else { back.x };
                let far = base + if normal.x == 0.0 { normal.y } else { normal.x } * depth;
                (base.min(far), base.max(far))
            };
            // Every blocker that reaches into that band, as a forbidden
            // interval along the edge.
            let mut cuts: Vec<(f64, f64)> = Vec::new();
            for r in blockers.voids.iter().chain(blockers.swings.iter()) {
                let (across0, across1, along0, along1) = if along_x {
                    (r.min.y, r.max.y, r.min.x, r.max.x)
                } else {
                    (r.min.x, r.max.x, r.min.y, r.max.y)
                };
                if across1 > b0 && across0 < b1 {
                    cuts.push((along0, along1));
                }
            }
            cuts.sort_by(|a, b| a.0.total_cmp(&b.0));
            // The longest gap between the cuts, inside the edge's own margins.
            let (mut cursor, mut best) = (lo + RUN_END_MARGIN_M, (0.0f64, 0.0f64, 0.0f64));
            let end = hi - RUN_END_MARGIN_M;
            let consider = |a: f64, b: f64, best: &mut (f64, f64, f64)| {
                if b - a > best.2 {
                    *best = (a, b, b - a);
                }
            };
            for (c0, c1) in cuts {
                if c0 > cursor {
                    consider(cursor, c0.min(end), &mut best);
                }
                cursor = cursor.max(c1);
            }
            if end > cursor {
                consider(cursor, end, &mut best);
            }
            let half_len = (best.2 * 0.5).min(def.half[0]);
            let half = DVec3::new(half_len, def.half[1], def.half[2]);
            if !(half.x >= RUN_MIN_HALF_M && half.y > 0.0 && half.z > 0.0) {
                continue;
            }
            let centre = (best.0 + best.1) * 0.5;
            let back_at = if along_x {
                DVec2::new(centre, back.y)
            } else {
                DVec2::new(back.x, centre)
            };
            let p = back_at + normal * half.z;
            chosen = Some((p, normal, half));
            break;
        }
        let Some((p, normal, half)) = chosen else {
            return;
        };
        let rot = yaw_onto(DVec3::new(normal.x, 0.0, normal.y));
        out.instances.push(self.instance_lit(
            kind,
            DVec3::new(p.x, y + half.y, p.y),
            rot,
            half,
            def.emissive,
        ));
        out.colliders.push(PcgCollider {
            center: DVec3::new(p.x, y + half.y, p.y),
            half_extents: half,
            rotation: rot,
        });
        placed.push((p, half.x.hypot(half.z)));
    }

    /// Station a wall-aligned piece along the room's inset perimeter, facing in.
    #[allow(clippy::too_many_arguments)]
    fn wall_furniture(
        &self,
        out: &mut GrammarOutput,
        room: &Room,
        y: f64,
        def: &FurnitureDef,
        kind: u32,
        hash: Hash64,
        blockers: &Blockers,
        placed: &mut Vec<(DVec2, f64)>,
        // **Wave VEN1a**: the centre height above the walking surface, or `0.0`
        // for a piece that STANDS on it (which is every `Placement::Wall`).
        // `Placement::Mounted` is otherwise this function exactly -- a neon
        // plate and a locker are both stationed along the perimeter facing in,
        // and differ only in whether their feet are on the floor.
        mount_m: f64,
    ) {
        let half = DVec3::from(def.half);
        let inner = room
            .rect
            .inset(self.arch.wall_thickness * 0.5 + FURNITURE_WALL_GAP);
        if !inner.is_positive() {
            return;
        }
        // The four inset edges, each with the inward normal the piece faces.
        let edges = [
            (inner.min, DVec2::new(inner.max.x, inner.min.y), DVec2::Y),
            (
                DVec2::new(inner.max.x, inner.min.y),
                inner.max,
                DVec2::NEG_X,
            ),
            (
                inner.max,
                DVec2::new(inner.min.x, inner.max.y),
                DVec2::NEG_Y,
            ),
            (DVec2::new(inner.min.x, inner.max.y), inner.min, DVec2::X),
        ];
        let step = (2.0 * half.x + def.clearance).max(0.5);
        let want = (def.per_10m2 * room.rect.area() / 10.0).max(0.0);
        let perimeter: f64 = edges.iter().map(|(a, b, _)| (*b - *a).length()).sum();
        let stations_total = (perimeter / step).floor().max(1.0);
        let accept = (want / stations_total).clamp(0.0, 1.0);

        let mut station = 0u64;
        for (a, b, normal) in edges {
            let d = b - a;
            let len = d.length();
            if !positive(len) {
                continue;
            }
            let dir = d / len;
            let n = (len / step).floor() as usize;
            for k in 0..n.min(MAX_STATIONS) {
                station += 1;
                if placed.len() >= MAX_FURNITURE_PER_ROOM {
                    return;
                }
                let s = hash.mix_u64(station);
                if s.unit() >= accept {
                    continue;
                }
                // The piece's back is on the wall: push in by its own depth.
                let along = len * (k as f64 + 0.5) / n as f64;
                let p = a + dir * along + normal * half.z;
                // The piece faces along `normal`, so its footprint is its own
                // half-extents swapped when the wall runs along Z.
                let foot = if normal.x == 0.0 {
                    DVec2::new(half.x, half.z)
                } else {
                    DVec2::new(half.z, half.x)
                };
                if !self.accept_spot(p, foot, def, &inner, blockers, placed) {
                    continue;
                }
                placed.push((p, def.clearance));
                let rot = yaw_onto(DVec3::new(normal.x, 0.0, normal.y));
                // **The instance sits on the collider's centre now** (I8b).
                // It used to sit on the FLOOR while the box was centred half a
                // height above it, so a desk was drawn with its middle at ankle
                // level — invisible while the drawn thing was a unit cube and
                // very visible the moment it is the size of the desk.
                let cy = if mount_m > 0.0 {
                    y + mount_m
                } else {
                    y + half.y
                };
                out.instances.push(self.instance_lit(
                    kind,
                    DVec3::new(p.x, cy, p.y),
                    rot,
                    half,
                    def.emissive,
                ));
                out.colliders.push(PcgCollider {
                    center: DVec3::new(p.x, cy, p.y),
                    half_extents: half,
                    rotation: rot,
                });
            }
        }
    }

    /// Scatter a free-standing piece on a jittered grid over the room's
    /// interior — the scatter kernel's candidate scheme, at room scale.
    #[allow(clippy::too_many_arguments)]
    fn free_furniture(
        &self,
        out: &mut GrammarOutput,
        room: &Room,
        y: f64,
        def: &FurnitureDef,
        kind: u32,
        hash: Hash64,
        blockers: &Blockers,
        placed: &mut Vec<(DVec2, f64)>,
    ) {
        let half = DVec3::from(def.half);
        let margin = half.x.max(half.z) + self.arch.wall_thickness * 0.5 + FURNITURE_WALL_GAP;
        let inner = room.rect.inset(margin);
        if !inner.is_positive() {
            return;
        }
        let cell = (2.0 * half.x.max(half.z) + def.clearance).max(0.5);
        let nx = (inner.size_x() / cell).floor().max(1.0) as usize;
        let nz = (inner.size_z() / cell).floor().max(1.0) as usize;
        let want = (def.per_10m2 * room.rect.area() / 10.0).max(0.0);
        let accept = (want / (nx * nz) as f64).clamp(0.0, 1.0);
        for j in 0..nz.min(MAX_STATIONS) {
            for i in 0..nx.min(MAX_STATIONS) {
                if placed.len() >= MAX_FURNITURE_PER_ROOM {
                    return;
                }
                let slot = (j * nx + i) as u64;
                let s = hash.mix_u64(slot);
                if s.unit() >= accept {
                    continue;
                }
                let jx = s.mix_u64(SALT_JIT_X).unit() - 0.5;
                let jz = s.mix_u64(SALT_JIT_Z).unit() - 0.5;
                let p = DVec2::new(
                    inner.min.x + inner.size_x() * (i as f64 + 0.5 + jx * 0.6) / nx as f64,
                    inner.min.y + inner.size_z() * (j as f64 + 0.5 + jz * 0.6) / nz as f64,
                );
                if !self.accept_spot(p, DVec2::new(half.x, half.z), def, &inner, blockers, placed) {
                    continue;
                }
                placed.push((p, def.clearance));
                out.instances.push(self.instance_lit(
                    kind,
                    DVec3::new(p.x, y + half.y, p.y),
                    glam::DQuat::IDENTITY,
                    half,
                    def.emissive,
                ));
                out.colliders.push(PcgCollider {
                    center: DVec3::new(p.x, y + half.y, p.y),
                    half_extents: half,
                    rotation: glam::DQuat::IDENTITY,
                });
            }
        }
    }

    /// **Is this footprint out of every opening void and every door swing?**
    /// (wave VEN1a) — the half of [`accept_spot`](Self::accept_spot) that the
    /// two singular placements need and the other half of which they do not.
    ///
    /// A centred stage and a counter run are not density placements: they have
    /// no room inset to be inside of (they derive their own) and no `placed`
    /// list to be clear of (they go first). What they DO have to respect is the
    /// enterability invariant, which is about voids and swings — so the test is
    /// factored here rather than duplicated, which is how the first cut of them
    /// came to skip it entirely and put a catwalk in a doorway.
    fn clear_of_blockers(&self, p: DVec2, foot: DVec2, blockers: &Blockers) -> bool {
        let box_ = Rect2 {
            min: p - foot,
            max: p + foot,
        };
        !blockers.voids.iter().any(|v| v.overlaps(&box_))
            && !blockers.swings.iter().any(|s| s.overlaps(&box_))
    }

    /// The four placement rules, in one predicate: inside the room, **out of
    /// every opening void**, out of every door swing, and clear of everything
    /// already placed.
    ///
    /// The first three test the piece's **footprint**, not its centre. A centre
    /// test is the bug this batch's audit found: a bed is a metre deep, so its
    /// centre can clear a doorway by more than the door is wide while its back
    /// stands squarely in the hole.
    fn accept_spot(
        &self,
        p: DVec2,
        foot: DVec2,
        def: &FurnitureDef,
        inner: &Rect2,
        blockers: &Blockers,
        placed: &[(DVec2, f64)],
    ) -> bool {
        if p.x < inner.min.x || p.x > inner.max.x || p.y < inner.min.y || p.y > inner.max.y {
            return false;
        }
        let box_ = Rect2 {
            min: p - foot,
            max: p + foot,
        };
        if blockers.voids.iter().any(|v| v.overlaps(&box_)) {
            return false;
        }
        if blockers.swings.iter().any(|s| s.overlaps(&box_)) {
            return false;
        }
        !placed
            .iter()
            .any(|(q, c)| (p - *q).length() < c.max(def.clearance))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::palettes::{archetypes, ArchetypeId};
    use crate::building::plan::plan_building;
    use crate::building::solid_bounds;

    fn lot(w: f64, h: f64) -> Rect2 {
        Rect2::new(DVec2::ZERO, DVec2::new(w, h))
    }

    fn built(id: ArchetypeId, floors: u32, seed: u64) -> BuildingOutput {
        build(
            &BuildingParams {
                floors,
                ..BuildingParams::new(id, lot(34.0, 23.0), 4.0, seed)
            },
            seed,
            true,
        )
    }

    /// **THE VENUE ANCHOR ASSERTION** (wave VEN1a): a venue's largest ground
    /// room is the type its archetype names, not one a hash chose.
    ///
    /// The sweep is over lot sizes and seeds *deliberately*: the defect this
    /// closes is probabilistic. A weighted draw puts a dance floor in the big
    /// room roughly a third of the time, so an arm that built one nightclub
    /// would pass on a third of the seeds it could have been written with.
    #[test]
    fn a_venues_largest_ground_room_is_the_room_it_is_named_for() {
        let mut checked = 0;
        for id in ArchetypeId::ALL {
            let arch = archetype(id);
            if arch.ground_anchors.is_empty() {
                continue;
            }
            for seed in [1u64, 7, 44, 91, 203, 700] {
                for (w, h) in [(34.0, 23.0), (26.0, 26.0), (44.0, 18.0)] {
                    let out = build(
                        &BuildingParams {
                            floors: 2,
                            ..BuildingParams::new(id, lot(w, h), 4.0, seed)
                        },
                        seed,
                        true,
                    );
                    let mut ground: Vec<&Room> =
                        out.plan.rooms.iter().filter(|r| r.floor == 0).collect();
                    // Structural rooms are not products of the anchor rule, and
                    // a single-storey plan's stair rectangle IS drawn -- so the
                    // claim is about the rooms the anchor could have reached.
                    ground.retain(|r| !matches!(r.kind, RoomType::Stair | RoomType::Corridor));
                    ground.sort_by(|a, b| {
                        b.rect
                            .area()
                            .total_cmp(&a.rect.area())
                            .then(a.kind.name().cmp(b.kind.name()))
                    });
                    for (k, want) in arch.ground_anchors.iter().enumerate() {
                        let Some(got) = ground.get(k) else { continue };
                        assert_eq!(
                            got.kind,
                            *want,
                            "{} seed {seed} lot {w}x{h}: ground room #{k} by area is {} \
                             and the archetype says {}",
                            arch.display,
                            got.kind.name(),
                            want.name()
                        );
                        checked += 1;
                    }
                }
            }
        }
        assert!(
            checked >= 50,
            "only {checked} anchored rooms were examined -- the sweep found no venues"
        );
    }

    /// **The anchor rule is INERT for the seven that predate it** (wave VEN1a).
    ///
    /// The arm above says the venues get what they ask for; this one says
    /// nothing else moved. Without it, an anchor loop that ran on every
    /// archetype and happened to pick the same kinds would look identical.
    #[test]
    fn an_archetype_with_no_anchors_keeps_its_weighted_draw() {
        for id in ArchetypeId::ALL {
            let arch = archetype(id);
            if !arch.ground_anchors.is_empty() {
                continue;
            }
            let out = built(id, 3, 44);
            for r in out.plan.rooms.iter().filter(|r| r.floor == 0) {
                if matches!(r.kind, RoomType::Stair | RoomType::Corridor) {
                    continue;
                }
                assert!(
                    arch.ground_rooms.iter().any(|w| w.kind == r.kind),
                    "{}: ground room {} is not in its own table",
                    arch.display,
                    r.kind.name()
                );
            }
        }
    }

    /// **THE COUNTER IS ONE PIECE** (wave VEN1a) — the limitation the venue
    /// mandate names, falling.
    ///
    /// A bar room holds exactly one `BarRun` instance, and it is *long*: the
    /// claim is not "a counter exists" (a `Placement::Wall` counter would
    /// satisfy that eleven times over) but "the counter is a single solid
    /// several metres long", which is what a discrete-station placer cannot
    /// produce at all.
    #[test]
    fn a_bar_rooms_counter_is_one_continuous_run() {
        let mut seen = 0;
        for id in [
            ArchetypeId::Bar,
            ArchetypeId::Nightclub,
            ArchetypeId::StripClub,
        ] {
            let g = archetype(id).grammar().expect("parses");
            let run_kind = g.module_index("BarRun").expect("declared");
            for seed in [3u64, 44, 512] {
                let out = build(
                    &BuildingParams {
                        floors: 1,
                        ..BuildingParams::new(id, lot(34.0, 23.0), 4.0, seed)
                    },
                    seed,
                    true,
                );
                let bars: Vec<&PcgInstance> = out
                    .instances
                    .iter()
                    .filter(|i| i.kind_index == run_kind)
                    .collect();
                let rooms = out
                    .plan
                    .rooms
                    .iter()
                    .filter(|r| r.kind == RoomType::BarRoom)
                    .count();
                assert_eq!(
                    bars.len(),
                    rooms,
                    "{:?} seed {seed}: {} counters for {rooms} bar room(s)",
                    id,
                    bars.len()
                );
                for b in &bars {
                    let e = b.extent.expect("a counter is drawn at its own size");
                    let long = f64::from(e[0]).max(f64::from(e[2]));
                    assert!(
                        long >= 2.0,
                        "{:?} seed {seed}: a {} m half-length counter is a station, not a run",
                        id,
                        long
                    );
                    seen += 1;
                }
            }
        }
        assert!(seen > 0, "no bar room was built at all");
    }

    /// **A stage carries a pole, and the pole stands ON it** (wave VEN1a).
    ///
    /// Two centred pieces in one room is the case `centre_furniture`
    /// deliberately does not reject: the pole's footprint is inside the stage's,
    /// and a `placed`-clearance test would have thrown the pole away.
    #[test]
    fn a_stage_room_gets_its_platform_and_its_pole() {
        for id in [ArchetypeId::Nightclub, ArchetypeId::StripClub] {
            let g = archetype(id).grammar().expect("parses");
            let pole = g.module_index("Pole").expect("declared");
            let mut found = 0;
            for seed in [3u64, 44, 512, 900] {
                let out = build(
                    &BuildingParams {
                        floors: 1,
                        ..BuildingParams::new(id, lot(34.0, 23.0), 4.0, seed)
                    },
                    seed,
                    true,
                );
                let stages = out
                    .plan
                    .rooms
                    .iter()
                    .filter(|r| matches!(r.kind, RoomType::Stage | RoomType::DanceFloor))
                    .count();
                if stages == 0 {
                    continue;
                }
                let poles: Vec<&PcgInstance> = out
                    .instances
                    .iter()
                    .filter(|i| i.kind_index == pole)
                    .collect();
                assert_eq!(
                    poles.len(),
                    stages,
                    "{:?} seed {seed}: {} poles for {stages} stage room(s)",
                    id,
                    poles.len()
                );
                // Chrome, and it says so on the instance rather than in a table
                // somewhere -- this is the arm that says the surface reached
                // the placement.
                for p in &poles {
                    assert_eq!(p.surface.metallic, 1.0, "{id:?}: a plastic pole");
                    assert!(p.surface.roughness < 0.2);
                }
                found += 1;
            }
            assert!(found > 0, "{id:?}: no stage room in any seed");
        }
    }

    /// **A venue emits, and the emission is coloured** (wave VEN1a).
    ///
    /// The claim the whole wave rests on: after assembly a venue's instance
    /// list contains real, saturated, *authored* emission — not the warm-white
    /// window glow every building has had since I8b.
    #[test]
    fn a_venue_carries_authored_coloured_emission() {
        for id in ArchetypeId::ALL {
            let arch = archetype(id);
            let out = built(id, 2, 44);
            let emitters: Vec<&PcgInstance> =
                out.instances.iter().filter(|i| i.surface.emits()).collect();
            if !id.is_venue() {
                assert!(
                    emitters.is_empty(),
                    "{}: a non-venue emits {} authored colours",
                    arch.display,
                    emitters.len()
                );
                continue;
            }
            assert!(
                emitters.len() >= 4,
                "{}: only {} authored emitters",
                arch.display,
                emitters.len()
            );
            // SATURATED: at least one emitter whose brightest channel is more
            // than twice its dimmest. A warm white would satisfy "emits".
            let coloured = emitters.iter().any(|i| {
                let e = i.surface.emissive;
                let hi = e[0].max(e[1]).max(e[2]);
                let lo = e[0].min(e[1]).min(e[2]);
                hi > 1.0 && hi > lo * 2.0
            });
            assert!(
                coloured,
                "{}: every emitter is near-white -- that is a lit window, not neon",
                arch.display
            );
            // And exactly one kind of thing breathes.
            assert!(
                emitters.iter().any(|i| i.surface.pulse_hz > 0.0),
                "{}: nothing in this venue pulses",
                arch.display
            );
        }
    }

    /// **A venue's rooms are reachable and its people have somewhere to be**
    /// (wave VEN1a).
    ///
    /// The interior nav graph is built ONLY for a building with slots
    /// (`pass.rs`: "a building nobody lives or works in contributes no
    /// interior"), so a venue whose rooms held no slot would have every room
    /// orphaned and no route into it — silently, with no failure anywhere. This
    /// is the arm that says the new room types earned their occupancy.
    #[test]
    fn a_venue_has_slots_so_its_interior_is_not_orphaned() {
        for id in [
            ArchetypeId::Bar,
            ArchetypeId::Nightclub,
            ArchetypeId::StripClub,
        ] {
            let out = built(id, 2, 44);
            let slots = crate::building::society::slots_of(&out.plan, 0, 9);
            assert!(
                !slots.is_empty(),
                "{id:?}: no slots, so `pass.rs` hands this building an EMPTY nav graph"
            );
            let nav = out.plan.interior_nav_in(9);
            assert!(nav.len() > 1, "{id:?}: an interior of {} nodes", nav.len());
            // Every slot stands on a node the graph actually holds.
            for s in &slots {
                assert!(
                    nav.contains(s.node),
                    "{id:?}: a slot in room {} stands on no node",
                    s.room
                );
            }
            // A venue is somewhere the town GOES, not only somewhere it works.
            assert!(
                slots
                    .iter()
                    .any(|s| s.role == crate::building::society::SlotRole::Errand),
                "{id:?}: nowhere to visit"
            );
        }
    }

    /// **THE ENTERABILITY ASSERTION**, at unit scale: for every archetype, no
    /// solid — wall, slab, lintel, parapet, stair or furniture — intrudes into
    /// any door's void.
    #[test]
    fn no_solid_ever_blocks_a_doorway() {
        for arch in archetypes() {
            let out = built(arch.id, 3, 77);
            let doors: Vec<&Opening> = out
                .plan
                .openings
                .iter()
                .filter(|o| o.kind == OpeningKind::Door)
                .collect();
            assert!(!doors.is_empty(), "{}: no doors at all", arch.display);
            for (i, d) in doors.iter().enumerate() {
                assert!(
                    out.plan.opening_is_clear(d, &out.colliders, 0.02),
                    "{}: door {i} on wall {} is blocked",
                    arch.display,
                    d.wall
                );
            }
        }
    }

    /// **THE ANTI-TAUTOLOGY GATE.** `opening_is_clear` must be able to say *no*.
    ///
    /// This exists because the first implementation could not: it built the void
    /// from a `thickness` of `0.0` (a 2 µm band), then shrank it by the margin on
    /// **every** axis, which inverted the thin one; `Rect2::intersection`'s
    /// `max > min` test then never succeeded and every solid read as clear. The
    /// assertions above passed for a building that was one solid block.
    ///
    /// So: drop a slab over the entire building and require **every** opening to
    /// report blocked. Any future change that makes the void degenerate — a zero
    /// thickness, a margin applied across the wall, a swapped comparison — fails
    /// here rather than silently disarming the enterability arm.
    #[test]
    fn a_block_over_the_whole_building_blocks_every_opening() {
        for arch in archetypes() {
            let out = built(arch.id, 3, 77);
            let f = out.plan.footprint;
            let top = out.plan.floor_y(out.plan.floors) + arch.floor_height;
            let block = vec![PcgCollider {
                center: DVec3::new(f.center().x, (out.plan.base_y + top) * 0.5, f.center().y),
                half_extents: DVec3::new(
                    f.size_x(),
                    (top - out.plan.base_y) * 0.5 + 1.0,
                    f.size_z(),
                ),
                rotation: glam::DQuat::IDENTITY,
            }];
            assert!(
                !out.plan.openings.is_empty(),
                "{}: no openings",
                arch.display
            );
            for (i, o) in out.plan.openings.iter().enumerate() {
                assert!(
                    !out.plan.opening_is_clear(o, &block, 0.02),
                    "{}: opening {i} ({:?}) reads CLEAR through a solid building — \
                     the predicate is vacuous",
                    arch.display,
                    o.kind
                );
            }
            // …and the same predicate still says *yes* for an empty world, so it
            // is not merely always-false.
            for o in &out.plan.openings {
                assert!(out.plan.opening_is_clear(o, &[], 0.02));
            }
        }
    }

    /// The void is a real, positive rectangle as deep as the wall — the property
    /// the tautology violated, asserted directly rather than only through its
    /// consequences.
    #[test]
    fn an_opening_void_is_as_deep_as_the_wall() {
        for arch in archetypes() {
            let out = built(arch.id, 2, 4);
            for o in &out.plan.openings {
                let (rect, (y0, y1)) = out.plan.opening_void(o).expect("wall exists");
                assert!(rect.is_positive(), "{}: an inverted void", arch.display);
                let w = &out.plan.walls[o.wall];
                let (along, across) = if w.direction().x.abs() > 0.5 {
                    (rect.size_x(), rect.size_z())
                } else {
                    (rect.size_z(), rect.size_x())
                };
                assert!(
                    (across - arch.wall_thickness).abs() < 1e-9,
                    "{}: void is {across} deep, wall is {}",
                    arch.display,
                    arch.wall_thickness
                );
                assert!(
                    (along - o.width()).abs() < 1e-9,
                    "{}: void length",
                    arch.display
                );
                assert!(y1 > y0, "{}: inverted Y band", arch.display);
            }
        }
    }

    /// A window's *band* is clear too — the parapet under it must stop at the
    /// sill and the lintel must start at the head.
    #[test]
    fn no_solid_ever_blocks_a_window_band() {
        for arch in archetypes() {
            let out = built(arch.id, 2, 5);
            for w in out
                .plan
                .openings
                .iter()
                .filter(|o| o.kind == OpeningKind::Window)
            {
                assert!(
                    out.plan.opening_is_clear(w, &out.colliders, 0.02),
                    "{}: a window band is blocked",
                    arch.display
                );
            }
        }
    }

    /// Every floor gets a walking surface under every room it has, and the
    /// stairwell is open above the ground floor so a stair can climb through.
    #[test]
    fn floors_are_slabs_and_the_stairwell_is_a_void() {
        let out = built(ArchetypeId::Office, 4, 3);
        let arch = archetype(ArchetypeId::Office);
        let slab_kind = out.plan.rooms.len();
        assert!(slab_kind > 0);
        for f in 0..out.plan.floors {
            let y = out.plan.floor_y(f);
            for (_, room) in out.plan.rooms_on(f) {
                let c = room.rect.center();
                let under = out.colliders.iter().any(|s| {
                    let (a, b) = s.y_band();
                    (b - y).abs() < arch.slab_thickness && a < y && solid_bounds(s).contains(c)
                });
                if room.kind == RoomType::Stair && f > 0 {
                    assert!(!under, "floor {f}: the stairwell is capped");
                } else {
                    assert!(under, "floor {f}: a room has no floor under it");
                }
            }
        }
    }

    /// A flight's last tread reaches **exactly** the upper floor's surface — a
    /// step short is a stair you cannot finish climbing, and the derive-from-k
    /// rule is what makes it exact rather than nearly.
    #[test]
    fn a_flight_lands_exactly_on_the_floor_above() {
        for arch in archetypes() {
            let out = built(arch.id, 3, 12);
            if out.plan.stairs.is_empty() {
                continue;
            }
            for flight in &out.plan.stairs {
                let target = out.plan.floor_y(flight.to);
                let tops: Vec<f64> = out
                    .colliders
                    .iter()
                    .map(|s| s.y_band().1)
                    .filter(|t| (t - target).abs() < 1e-9)
                    .collect();
                assert!(
                    !tops.is_empty(),
                    "{}: no tread tops out at floor {}",
                    arch.display,
                    flight.to
                );
            }
        }
    }

    /// Assembly is a pure function of `(plan, seed)` — and independent of the
    /// worker count, which is the P7.0 guard applied to the building path.
    #[test]
    fn assembly_is_pure_and_pool_size_invariant() {
        let plan = plan_building(&BuildingParams {
            floors: 3,
            ..BuildingParams::new(ArchetypeId::Apartment, lot(30.0, 21.0), 0.0, 8)
        });
        let a = assemble(&plan, 8, true);
        assert_eq!(a, assemble(&plan, 8, true));
        assert_ne!(a, assemble(&plan, 9, true));
        for workers in [1usize, 2, 4, 8] {
            let pool = inf_core::JobPool::new(workers);
            let b = assemble_in(&pool, &plan, 8, true);
            assert_eq!(
                a.instances.len(),
                b.instances.len(),
                "instance count moved at {workers} workers"
            );
            for (x, y) in a.instances.iter().zip(b.instances.iter()) {
                assert_eq!(
                    x.pos.to_array().map(f64::to_bits),
                    y.pos.to_array().map(f64::to_bits)
                );
            }
            assert_eq!(
                a.colliders, b.colliders,
                "colliders moved at {workers} workers"
            );
        }
    }

    #[test]
    fn unfurnished_is_a_subset_of_furnished() {
        let plan = plan_building(&BuildingParams {
            floors: 2,
            ..BuildingParams::new(ArchetypeId::Hotel, lot(38.0, 22.0), 0.0, 4)
        });
        let bare = assemble(&plan, 4, false);
        let full = assemble(&plan, 4, true);
        assert!(
            bare.instances.len() < full.instances.len(),
            "nothing furnished"
        );
        assert!(
            !bare.is_empty(),
            "an unfurnished building is still a building"
        );
        // The structure is identical: furniture only ever adds.
        for s in &bare.colliders {
            assert!(full.colliders.contains(s), "furnishing moved the structure");
        }
    }

    /// Furniture stays out of door swings — the rule that keeps the enterability
    /// assertion from being satisfied by luck.
    #[test]
    fn furniture_keeps_out_of_door_swings() {
        for arch in archetypes() {
            let out = built(arch.id, 2, 30);
            // Furniture is whatever furnishing ADDED — the structure is
            // identical either way (pinned by
            // `unfurnished_is_a_subset_of_furnished`), and assembly interleaves
            // the two per floor, so this is a set difference rather than a tail
            // slice.
            let bare = assemble(&out.plan, 30, false).colliders;
            let furniture: Vec<PcgCollider> = out
                .colliders
                .iter()
                .filter(|s| !bare.contains(s))
                .copied()
                .collect();
            assert!(!furniture.is_empty(), "{}: nothing furnished", arch.display);
            let furniture = &furniture[..];
            for o in out
                .plan
                .openings
                .iter()
                .filter(|o| o.kind == OpeningKind::Door)
            {
                let Some(w) = out.plan.walls.get(o.wall) else {
                    continue;
                };
                let mid = w.point_at((o.start + o.end) * 0.5);
                // Only this door's own storey: a piece standing directly above a
                // doorway on the floor below shares its XZ and blocks nothing.
                let y0 = out.plan.floor_y(w.floor);
                let y1 = y0 + arch.floor_height;
                for f in furniture
                    .iter()
                    .filter(|f| f.center.y >= y0 && f.center.y < y1)
                {
                    let d = (DVec2::new(f.center.x, f.center.z) - mid).length();
                    assert!(
                        d > arch.door_width * 0.5,
                        "{}: furniture {d} m from a doorway on floor {}",
                        arch.display,
                        w.floor
                    );
                }
            }
        }
    }

    /// Every instance a building emits stands inside its own footprint (plus the
    /// wall thickness a module can reach across a boundary). That is the
    /// property the P16.5 cell binning relies on: a building's contents are
    /// where its volume entity says they are.
    #[test]
    fn everything_stands_inside_the_footprint() {
        for arch in archetypes() {
            let out = built(arch.id, 3, 55);
            let slack = arch.wall_thickness + 0.5;
            let bounds = out.plan.footprint.inset(-slack);
            for inst in &out.instances {
                let p = DVec2::new(inst.pos.x, inst.pos.z);
                assert!(
                    p.x >= bounds.min.x
                        && p.x <= bounds.max.x
                        && p.y >= bounds.min.y
                        && p.y <= bounds.max.y,
                    "{}: an instance at {p:?} escaped {:?}",
                    arch.display,
                    out.plan.footprint
                );
            }
            let (lo, hi) = (
                out.plan.base_y - arch.slab_thickness - 0.01,
                out.plan.base_y + out.plan.floors as f64 * arch.floor_height + 0.01,
            );
            for inst in &out.instances {
                assert!(
                    inst.pos.y >= lo && inst.pos.y <= hi,
                    "{}: an instance at y={} left the stack [{lo}, {hi}]",
                    arch.display,
                    inst.pos.y
                );
            }
        }
    }

    /// A building is a substantial object, not three boxes — a smoke test on the
    /// order of magnitude, so a silently-empty assembly cannot pass the
    /// property tests above by having nothing to check.
    #[test]
    fn a_building_has_substance() {
        for arch in archetypes() {
            let out = built(arch.id, 3, 2);
            assert!(
                out.instances.len() > 100,
                "{}: only {} instances",
                arch.display,
                out.instances.len()
            );
            assert!(
                out.colliders.len() > 60,
                "{}: only {} solids",
                arch.display,
                out.colliders.len()
            );
            assert!(
                out.colliders.len() <= out.instances.len(),
                "{}: more solids than instances",
                arch.display
            );
        }
    }

    /// **THE ZERO-PLACEHOLDER ARM, at the source** (island wave I8b).
    ///
    /// Every instance a building emits names a mesh and states the size of the
    /// box it occupies. A projector cannot draw authored geometry for something
    /// that names none, so this is where "a shipped city draws zero placeholder
    /// batches" begins — and it is a claim about *all* of them, which is what
    /// makes it falsifiable: dropping the stamp from one shape family, or the
    /// extent from one of the assembler's placement sites, fails here.
    #[test]
    fn every_building_instance_names_a_mesh_and_its_own_size() {
        for arch in archetypes() {
            let out = built(arch.id, 3, 91);
            assert!(!out.instances.is_empty(), "{}: nothing built", arch.display);
            for (i, inst) in out.instances.iter().enumerate() {
                assert!(
                    inst.mesh.is_some(),
                    "{}: instance {i} (module {}) draws a placeholder",
                    arch.display,
                    inst.kind_index
                );
                let e = inst
                    .extent
                    .unwrap_or_else(|| panic!("{}: instance {i} has no extent", arch.display));
                assert!(
                    e.iter().all(|c| c.is_finite() && *c > 0.0),
                    "{}: instance {i} extent {e:?}",
                    arch.display
                );
            }
        }
    }

    /// **The drawn box IS the solid box**, for every module that has one — the
    /// defect this wave exists to close, asserted directly rather than through
    /// its consequences. Before I8b the instance carried `scale: 1.0` and no
    /// extent, so a 10 m slab and a 0.3 m mullion drew identically.
    #[test]
    fn the_drawn_extent_matches_the_collider_it_was_placed_with() {
        for arch in archetypes() {
            let out = built(arch.id, 2, 12);
            let n = out.colliders.len();
            assert!(n > 0);
            assert!(
                out.instances.len() >= n,
                "{}: {} instances for {n} colliders — the aligned prefix is gone",
                arch.display,
                out.instances.len()
            );
            for (i, (inst, solid)) in out.instances.iter().zip(&out.colliders).enumerate() {
                let e = inst.extent.expect("an extent");
                let h = solid.half_extents;
                for (k, want) in [h.x, h.y, h.z].into_iter().enumerate() {
                    assert!(
                        (f64::from(e[k]) - want).abs() < 1e-3,
                        "{}: instance {i} axis {k} drawn {} vs solid {want}",
                        arch.display,
                        e[k]
                    );
                }
                assert!(
                    (inst.pos - solid.center).length() < 1e-9,
                    "{}: instance {i} is not on its own box",
                    arch.display
                );
            }
        }
    }

    /// **The decoration tail** (I8b): a pane per window, after the aligned
    /// prefix, glowing, and carrying no collider — so the enterability
    /// invariant is untouched by it.
    #[test]
    fn every_window_gets_a_pane_and_no_pane_is_solid() {
        for arch in archetypes() {
            let out = built(arch.id, 3, 44);
            let windows = out
                .plan
                .openings
                .iter()
                .filter(|o| o.kind == OpeningKind::Window)
                .count();
            assert!(windows > 0, "{}: no windows at all", arch.display);
            let decor = &out.instances[out.colliders.len()..];
            // **The tail is panes THEN the street face** (wave VEN1a). A venue
            // hangs a sign and a festoon over its entrance, and both are decor
            // for the same reason a pane is: a sign you cannot walk through is
            // a wall. `assemble_in` runs the floors before `street_face`, so
            // the panes are a prefix and the signage is a suffix -- which is a
            // stronger claim than the count this arm used to make alone.
            let signage = arch
                .entrance_sign
                .map_or(0, |s| 1 + usize::from(s.festoon.is_some()));
            assert_eq!(
                decor.len(),
                windows + signage,
                "{}: {} decorations for {windows} windows and {signage} sign piece(s)",
                arch.display,
                decor.len()
            );
            let g = archetype(arch.id).grammar().expect("parses");
            let pane_kind = g
                .module_index(arch.pane)
                .expect("the palette declares its pane");
            for p in &decor[..windows] {
                assert_eq!(p.kind_index, pane_kind, "{}: not a pane", arch.display);
                assert!(p.glow > 0.0, "{}: a pane that does not glow", arch.display);
            }
            if let Some(sign) = arch.entrance_sign {
                let plate = g
                    .module_index(sign.plate)
                    .expect("the palette declares its sign plate");
                let tail = &decor[windows..];
                assert_eq!(
                    tail[0].kind_index, plate,
                    "{}: the street face's first piece is not the plate",
                    arch.display
                );
                // The plate burns the palette's OWN colour, not the family's --
                // this is the arm that says `FurnitureDef`-style overriding
                // reached the street.
                assert_eq!(
                    tail[0].surface.emissive, sign.colour,
                    "{}: the street sign did not take its authored colour",
                    arch.display
                );
                if let Some(f) = sign.festoon {
                    let fk = g.module_index(f).expect("the palette declares its festoon");
                    assert_eq!(tail[1].kind_index, fk, "{}: no festoon", arch.display);
                }
            }
            // …and nothing in the aligned prefix glows, so "the windows light
            // up" is a statement about windows.
            for s in &out.instances[..out.colliders.len()] {
                let is_glazed = super::super::modules::shape_of(
                    &archetype(arch.id).grammar().expect("parses").modules()[s.kind_index as usize]
                        .name,
                )
                .is_some_and(super::super::modules::ModuleShape::is_glazing);
                assert_eq!(
                    s.glow > 0.0,
                    is_glazed,
                    "{}: a solid module's glow disagrees with its family",
                    arch.display
                );
            }
        }
    }

    /// A pane fills its window void exactly — the same rectangle
    /// `opening_is_clear` tests, so a window is glazed rather than approximately
    /// glazed.
    #[test]
    fn a_pane_fills_the_void_it_was_hung_in() {
        let out = built(ArchetypeId::House, 2, 7);
        let decor = &out.instances[out.colliders.len()..];
        let windows: Vec<&Opening> = out
            .plan
            .openings
            .iter()
            .filter(|o| o.kind == OpeningKind::Window)
            .collect();
        assert_eq!(decor.len(), windows.len());
        for (pane, w) in decor.iter().zip(windows) {
            let (rect, (y0, y1)) = out.plan.opening_void(w).expect("a void");
            let e = pane.extent.expect("an extent");
            assert!((f64::from(e[0]) - rect.size_x() * 0.5).abs() < 1e-6);
            assert!((f64::from(e[1]) - (y1 - y0) * 0.5).abs() < 1e-6);
            assert!((f64::from(e[2]) - rect.size_z() * 0.5).abs() < 1e-6);
            assert!((pane.pos.y - (y0 + y1) * 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn a_degenerate_plan_assembles_to_nothing() {
        let plan = plan_building(&BuildingParams::new(
            ArchetypeId::Shop,
            lot(0.0, 0.0),
            0.0,
            1,
        ));
        assert!(assemble(&plan, 1, true).is_empty());
    }
}
