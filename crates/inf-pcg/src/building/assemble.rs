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

use super::palettes::{archetype, BuildingArchetype, FurnitureDef};
use super::plan::BuildingParams;
use super::{BuildingPlan, Opening, OpeningKind, Rect2, Room, RoomType, Wall};
use crate::grammar::dsl::Grammar;
use crate::grammar::expand::{expand_span, GrammarOutput, GrammarPass, Ground, SpanSource};
use crate::grammar::span::{positive, yaw_onto, Span};
use crate::hash::Hash64;
use crate::height::HeightProvider;
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
        let def = self.grammar.modules().get(kind as usize);
        PcgInstance {
            pos: center,
            rotation,
            scale: 1.0,
            kind_index: kind,
            mesh: def.and_then(|m| m.mesh),
            extent: Some([half.x as f32, half.y as f32, half.z as f32]),
            glow: def.map_or(0.0, |m| m.glow),
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
            if def.against_wall {
                self.wall_furniture(out, room, y, def, kind, dh, blockers, &mut placed);
            } else {
                self.free_furniture(out, room, y, def, kind, dh, blockers, &mut placed);
            }
        }
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
                out.instances.push(self.instance(
                    kind,
                    DVec3::new(p.x, y + half.y, p.y),
                    rot,
                    half,
                ));
                out.colliders.push(PcgCollider {
                    center: DVec3::new(p.x, y + half.y, p.y),
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
                out.instances.push(self.instance(
                    kind,
                    DVec3::new(p.x, y + half.y, p.y),
                    glam::DQuat::IDENTITY,
                    half,
                ));
                out.colliders.push(PcgCollider {
                    center: DVec3::new(p.x, y + half.y, p.y),
                    half_extents: half,
                    rotation: glam::DQuat::IDENTITY,
                });
            }
        }
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
            assert_eq!(
                decor.len(),
                windows,
                "{}: {} panes for {windows} windows",
                arch.display,
                decor.len()
            );
            let pane_kind = archetype(arch.id)
                .grammar()
                .expect("parses")
                .module_index(arch.pane)
                .expect("the palette declares its pane");
            for p in decor {
                assert_eq!(p.kind_index, pane_kind, "{}: not a pane", arch.display);
                assert!(p.glow > 0.0, "{}: a pane that does not glow", arch.display);
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
