//! **Buildings & interiors** (P19.5): a footprint becomes a floor stack, the
//! floors become rooms, the rooms become walls with real openings, and the walls
//! are expanded by the P19.4 1-D grammar.
//!
//! ```text
//!   footprint rect ─▶ floor stack ─▶ per-floor PARTITION ─▶ walls
//!                          │              │                  │
//!                          │              ├─ doors (spanning) ┤ ← the connectivity invariant
//!                          │              └─ windows          │
//!                          ├─ STAIR CORE (one rect, every floor)
//!                          └─ furniture per room type ────────┘
//! ```
//!
//! # The dimensional split, restated
//!
//! P19.4's grammar is **one-dimensional**: a rule text expands along an arc
//! length. A building is **two-dimensional in plan and one-dimensional per
//! wall**, and this module is exactly that split — the 2-D half is a
//! deterministic slice-tree ([`partition`]), the 1-D half is P19.4 verbatim
//! ([`assemble`] hands each wall run to [`expand_span`](crate::grammar::expand::
//! expand_span)). Nothing here re-implements a layout: a wall *is* a
//! [`Span`](crate::grammar::Span), and an opening *is* the absence of wall
//! modules on it.
//!
//! # THE ENTERABILITY INVARIANT
//!
//! A building is enterable when three statements hold, and all three are
//! properties of [`BuildingPlan`] that it can answer about itself:
//!
//! 1. **[`BuildingPlan::rooms_connected`]** — every room on a floor is reachable
//!    from every other through door openings. Guaranteed by construction (see
//!    [`partition::connect`]'s proof) and asserted anyway.
//! 2. **[`BuildingPlan::floors_reachable`]** — every floor is reachable from
//!    *outside* through the entrance door and the stair cores. This is the graph
//!    walk the Phase 19 gate performs: outside → ground floor → up.
//! 3. **[`BuildingPlan::opening_is_clear`]** — no structural collider intersects
//!    an opening's void. Assembly *derives* the wall runs from the openings
//!    rather than cutting them afterwards, so this is a check on arithmetic, not
//!    on a boolean operation.
//!
//! An opening is **never a boolean cut**. A wall carrying a door is emitted as
//! the two (or more) runs *beside* the door plus a lintel above it; the door's
//! void is simply a place no run covers. That is why a doorway cannot end up with
//! a collider in it by accident — there is no subtraction to get wrong.
//!
//! # Determinism
//!
//! Every draw is `Hash64(building seed) · salt · index` — the counter-hash
//! doctrine, with the *floor index* folded in for anything per-floor and
//! deliberately **not** folded in for anything that must align across floors (the
//! stair core and the corridor band are drawn from the building hash alone, so
//! they are the same rectangle on every storey **by construction** rather than by
//! search). Nothing walks a stateful RNG and nothing depends on iteration order.
//!
//! # v1 scope, stated
//!
//! * **Footprints are axis-aligned rectangles.** The whole plan — rooms, walls,
//!   openings, stair cores — is rectangle arithmetic on the world XZ plane, which
//!   is the same restriction [`footprint_perimeter`](crate::footprint_perimeter)
//!   already has. A rotated or L-shaped lot needs an oriented rect type
//!   throughout and was not taken for a v1.
//! * **Modules draw their shape family** (island wave I8b). The seven palettes
//!   still declare no mesh GUID and still need no imported art; what a module
//!   draws is a derived, unit-space mesh chosen by its name (see [`modules`]),
//!   scaled onto the half-extents the palette or the plan gives it. An authored
//!   `module Panel = mesh <guid> …` still overrides it.

pub mod assemble;
// I6: where a building's DOORS go - the openings the grammar already plans,
// turned into hinges a door system can hang a leaf on.
pub mod doorway;
pub mod lod;
// I8b: what a module LOOKS like -- the shape families, their content-derived
// GUIDs and the unit-space meshes both hosts register under them.
pub mod modules;
pub mod palettes;
pub mod partition;
pub mod pass;
pub mod plan;
// NPC1d: who a building HOLDS -- the residents and workers its own rooms imply.
pub mod society;
pub mod subdivide;

use glam::{DVec2, DVec3};

use crate::grammar::span::positive;
use crate::scatter::PcgCollider;

pub use assemble::{assemble, assemble_in, build, build_in, BuildingOutput};
pub use doorway::{doorways_of, place_doorways_in_frame, PcgDoorway};
pub use lod::{StructureGroup, StructureTier, DEFAULT_STRUCTURE_LOD_M};
pub use palettes::{
    archetype, archetypes, ArchetypeId, BuildingArchetype, FurnitureDef, RoomWeight,
};
pub use partition::{connect, partition_floor, walls_of, Adjacency};
pub use pass::{
    evaluate_buildings, evaluate_buildings_in, lot_of, oriented_lot_of, oriented_lots_of,
    pass_seed, plans_of, BuildingPass,
};
pub use plan::{plan_building, plan_building_in, BuildingParams, MAX_FLOORS};
pub use subdivide::{subdivide_block, BlockLot, BlockSubdivision, LotRules, MAX_LOTS_PER_AXIS};

/// **The lot's own frame on the XZ plane** (IB-6): where its origin is and
/// which way its long side runs.
///
/// # Plan in the lot's frame, place in the world's
///
/// Every rectangle in this module — the plate, the rooms, the core, the stair
/// flights — is axis-aligned, and making each of them oriented would mean an OBB
/// type through the slicer, the adjacency test, the wall builder, the roof, the
/// stairs and the furniture grid: ten methods on [`Rect2`] and a dozen
/// world-axis comparisons, every one of which is *correct* in a local frame.
///
/// So the plan is built in the **lot's** coordinates, where it is axis-aligned
/// by construction and every existing rule reads the same way it always did, and
/// the finished output is transformed into the world at exactly one place
/// ([`assemble_in`]). A rotated lot costs one
/// rotation per placed box, and the adjacency test — whose world-axis
/// `same_line` comparison would silently find **zero** doors between rotated
/// rooms — never sees a rotation at all.
///
/// The identity frame is bit-exact: `to_world` multiplies by `1.0` and `0.0`,
/// and `assemble_in` skips the pass entirely when
/// [`is_identity`](LotFrame::is_identity) holds, so nothing a level already
/// contains moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LotFrame {
    /// Where the lot's local origin sits in world XZ.
    pub origin: DVec2,
    /// Unit direction of the lot's local `+X` in world XZ.
    pub u: DVec2,
}

impl Default for LotFrame {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl LotFrame {
    /// World axes: local == world.
    pub const IDENTITY: Self = Self {
        origin: DVec2::ZERO,
        u: DVec2::X,
    };

    /// A frame from an oriented rectangle's centre and basis.
    pub fn new(origin: DVec2, u: DVec2) -> Self {
        let u = if u.length_squared() > 0.0 && u.is_finite() {
            u.normalize()
        } else {
            DVec2::X
        };
        Self { origin, u }
    }

    /// The lot's local `+Z` direction in world XZ.
    #[inline]
    pub fn v(&self) -> DVec2 {
        DVec2::new(-self.u.y, self.u.x)
    }

    /// Whether this frame is the world's own.
    #[inline]
    pub fn is_identity(&self) -> bool {
        self.origin == DVec2::ZERO && self.u == DVec2::X
    }

    /// A lot-frame XZ position in world XZ.
    #[inline]
    pub fn to_world(&self, local: DVec2) -> DVec2 {
        self.origin + self.u * local.x + self.v() * local.y
    }

    /// A world XZ position in the lot's frame.
    #[inline]
    pub fn to_local(&self, world: DVec2) -> DVec2 {
        let d = world - self.origin;
        DVec2::new(d.dot(self.u), d.dot(self.v()))
    }

    /// The yaw-only rotation taking a lot-frame direction into the world.
    ///
    /// Built through [`crate::grammar::span::yaw_onto`] — the crate's one
    /// trig-free yaw door — because a lot's rotation reaches committed content
    /// and `DQuat::from_axis_angle` is `sin_cos` inside glam (the P14 law).
    pub fn yaw(&self) -> glam::DQuat {
        if self.is_identity() {
            return glam::DQuat::IDENTITY;
        }
        // Local `+Z` is `v` in world XZ, and `yaw_onto` takes `+Z` onto a
        // direction — so the frame's rotation is exactly `yaw_onto(v)`.
        let v = self.v();
        crate::grammar::span::yaw_onto(DVec3::new(v.x, 0.0, v.y))
    }
}

/// A lot, in its own frame: an axis-aligned rectangle plus where that frame sits
/// in the world. See [`LotFrame`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrientedLot {
    /// The lot rectangle, in [`frame`](Self::frame)'s coordinates.
    pub rect: Rect2,
    pub frame: LotFrame,
}

impl OrientedLot {
    /// A lot on the world axes — what every pre-IB-6 caller produced.
    pub fn axis_aligned(rect: Rect2) -> Self {
        Self {
            rect,
            frame: LotFrame::IDENTITY,
        }
    }

    /// The lot's four corners in world XZ, counter-clockwise from `min`.
    pub fn world_corners(&self) -> [DVec2; 4] {
        let (a, b) = (self.rect.min, self.rect.max);
        [
            self.frame.to_world(a),
            self.frame.to_world(DVec2::new(b.x, a.y)),
            self.frame.to_world(b),
            self.frame.to_world(DVec2::new(a.x, b.y)),
        ]
    }
}

/// An axis-aligned rectangle on the world **XZ** plane (Y is the floor's own
/// height, carried by whatever owns the rect).
///
/// `min` is componentwise ≤ `max` for every rectangle this module produces;
/// [`Rect2::new`] enforces it so a caller cannot build an inverted one.
///
/// **On an oriented lot these are the LOT's axes, not the world's** — see
/// [`LotFrame`]. Nothing in the plan changes; the frame is applied once, on the
/// way out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect2 {
    pub min: DVec2,
    pub max: DVec2,
}

impl Rect2 {
    /// A rectangle from two corners, normalized so `min ≤ max` on both axes.
    pub fn new(a: DVec2, b: DVec2) -> Self {
        Self {
            min: a.min(b),
            max: a.max(b),
        }
    }

    /// A rectangle from a centre and a **full** size.
    pub fn from_center(center: DVec2, size: DVec2) -> Self {
        let h = (size * 0.5).abs();
        Self {
            min: center - h,
            max: center + h,
        }
    }

    /// Size along world X.
    #[inline]
    pub fn size_x(&self) -> f64 {
        self.max.x - self.min.x
    }

    /// Size along world Z.
    #[inline]
    pub fn size_z(&self) -> f64 {
        self.max.y - self.min.y
    }

    /// Full size on both axes.
    #[inline]
    pub fn size(&self) -> DVec2 {
        DVec2::new(self.size_x(), self.size_z())
    }

    /// The centre point.
    #[inline]
    pub fn center(&self) -> DVec2 {
        (self.min + self.max) * 0.5
    }

    /// Floor area in m².
    #[inline]
    pub fn area(&self) -> f64 {
        self.size_x().max(0.0) * self.size_z().max(0.0)
    }

    /// `true` when both axes carry positive length.
    #[inline]
    pub fn is_positive(&self) -> bool {
        self.size_x() > 0.0 && self.size_z() > 0.0
    }

    /// This rectangle shrunk by `d` metres on every side — **collapsing** to its
    /// own centre rather than inverting when `d` is too large, and expanding for
    /// a negative `d`.
    ///
    /// Collapsing rather than swapping matters: a swapped rectangle passes
    /// `is_positive` and would silently be a *larger* room than the one that did
    /// not fit, which is how an inset turns into a bug that only shows up on a
    /// small floor plate.
    pub fn inset(&self, d: f64) -> Self {
        let c = self.center();
        Self {
            min: (self.min + DVec2::splat(d)).min(c),
            max: (self.max - DVec2::splat(d)).max(c),
        }
    }

    /// `true` when `p` is inside (half-open on `max`, so abutting rectangles do
    /// not both claim a boundary point).
    pub fn contains(&self, p: DVec2) -> bool {
        p.x >= self.min.x && p.x < self.max.x && p.y >= self.min.y && p.y < self.max.y
    }

    /// The overlap of two rectangles, or `None` when they do not overlap with
    /// positive area.
    pub fn intersection(&self, other: &Rect2) -> Option<Rect2> {
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        (max.x > min.x && max.y > min.y).then_some(Rect2 { min, max })
    }

    /// The two rectangles overlap with **positive area** (touching faces do not
    /// count — that is adjacency, not intersection).
    pub fn overlaps(&self, other: &Rect2) -> bool {
        self.intersection(other).is_some()
    }

    /// This rectangle as an axis-aligned solid centred at world height `y` with
    /// half-height `half_y` — the bridge from plan space to a
    /// [`PcgCollider`].
    pub fn to_solid(&self, y: f64, half_y: f64) -> PcgCollider {
        let c = self.center();
        PcgCollider {
            center: DVec3::new(c.x, y, c.y),
            half_extents: DVec3::new(self.size_x() * 0.5, half_y, self.size_z() * 0.5),
            rotation: glam::DQuat::IDENTITY,
        }
    }
}

/// The axis-aligned XZ rectangle a solid occupies, conservatively (its rotated
/// box's bounds).
///
/// Used by the opening-clearance check: an opening is clear when no solid's
/// bounds overlap its void, and using *bounds* makes that check conservative in
/// the safe direction — it can report a blockage that is not there, never miss
/// one that is. The half-extents come from
/// [`PcgCollider::xz_half_extents`], which is exact for a v1 building's
/// axis-aligned walls and uses no trigonometry at all.
pub fn solid_bounds(solid: &PcgCollider) -> Rect2 {
    let (ex, ez) = solid.xz_half_extents();
    Rect2 {
        min: DVec2::new(solid.center.x - ex, solid.center.z - ez),
        max: DVec2::new(solid.center.x + ex, solid.center.z + ez),
    }
}

/// What a room is for. Drives the wall grammar chosen for its walls, whether it
/// takes windows, and which furniture set populates it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RoomType {
    /// The circulation spine. Always connected to everything that touches it.
    Corridor,
    /// The stair core. One rectangle, identical on every floor **by
    /// construction** — see [`plan`].
    Stair,
    Lobby,
    Office,
    Meeting,
    /// Plant, risers, WCs — the service core.
    Service,
    Living,
    Bedroom,
    Kitchen,
    Bath,
    Retail,
    Storage,
    Workshop,
    Guest,
}

impl RoomType {
    /// A stable short name for diagnostics and gate traces.
    pub fn name(self) -> &'static str {
        match self {
            RoomType::Corridor => "corridor",
            RoomType::Stair => "stair",
            RoomType::Lobby => "lobby",
            RoomType::Office => "office",
            RoomType::Meeting => "meeting",
            RoomType::Service => "service",
            RoomType::Living => "living",
            RoomType::Bedroom => "bedroom",
            RoomType::Kitchen => "kitchen",
            RoomType::Bath => "bath",
            RoomType::Retail => "retail",
            RoomType::Storage => "storage",
            RoomType::Workshop => "workshop",
            RoomType::Guest => "guest",
        }
    }
}

/// One room on one floor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Room {
    /// 0-based storey.
    pub floor: u32,
    pub rect: Rect2,
    pub kind: RoomType,
}

/// A straight wall run on one floor: the boundary between two rooms, or between
/// a room and the outside.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wall {
    pub floor: u32,
    /// Start, in world XZ. Canonically the end with the smaller `(x, z)`, so a
    /// wall's direction is a pure function of its geometry.
    pub a: DVec2,
    /// End, in world XZ.
    pub b: DVec2,
    /// The room on each side; `None` is the outside. A wall always has at least
    /// `inside`.
    pub inside: usize,
    pub outside: Option<usize>,
}

impl Wall {
    /// Run length in metres.
    #[inline]
    pub fn length(&self) -> f64 {
        (self.b - self.a).length()
    }

    /// `true` when this wall faces the outside world.
    #[inline]
    pub fn is_exterior(&self) -> bool {
        self.outside.is_none()
    }

    /// The world XZ point `s` metres along the run.
    pub fn point_at(&self, s: f64) -> DVec2 {
        let len = self.length();
        if !positive(len) {
            return self.a;
        }
        let t = (s / len).clamp(0.0, 1.0);
        self.a * (1.0 - t) + self.b * t
    }

    /// The unit direction of travel (`+X` for a degenerate run, so nothing
    /// downstream sees a NaN).
    pub fn direction(&self) -> DVec2 {
        let d = self.b - self.a;
        let len = d.length();
        if len > 0.0 {
            d / len
        } else {
            DVec2::X
        }
    }
}

/// Whether an opening is walked through or looked through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpeningKind {
    /// A doorway: void from the floor to its head. **This is the edge of the
    /// room graph.**
    Door,
    /// A window: void from its sill to its head, with solid wall below.
    Window,
}

/// A void in a wall run.
///
/// `start`/`end` are metres along [`Wall`]'s own run, `sill`/`head` are metres
/// above the floor's walking surface. The wall assembly emits runs *around*
/// `[start, end]` rather than cutting the interval out afterwards — see the
/// module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Opening {
    pub kind: OpeningKind,
    /// Index into [`BuildingPlan::walls`].
    pub wall: usize,
    pub start: f64,
    pub end: f64,
    /// Metres above the floor. `0` for a door.
    pub sill: f64,
    /// Metres above the floor.
    pub head: f64,
}

impl Opening {
    /// Width in metres.
    #[inline]
    pub fn width(&self) -> f64 {
        self.end - self.start
    }
}

/// One flight connecting two adjacent storeys, occupying the building's stair
/// core.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stair {
    /// The core rectangle — the **same** on every flight, which is what makes
    /// the stack align.
    pub rect: Rect2,
    /// Lower storey.
    pub from: u32,
    /// Upper storey (always `from + 1`).
    pub to: u32,
}

/// The complete, deterministic plan of one building. Pure data: no world, no
/// entities, no GPU.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildingPlan {
    pub archetype: ArchetypeId,
    /// The lot the building stands on, in [`frame`](Self::frame)'s XZ — which
    /// **is** world XZ whenever the frame is the identity, i.e. for every lot
    /// this engine produced before IB-6.
    pub footprint: Rect2,
    /// Where this plan's own axes sit in the world. `LotFrame::IDENTITY` for an
    /// axis-aligned lot; [`assemble_in`] applies
    /// it to the finished output and skips the pass entirely when it is the
    /// identity.
    pub frame: LotFrame,
    /// World Y of the ground floor's walking surface.
    pub base_y: f64,
    pub floors: u32,
    pub floor_height: f64,
    /// Every room of every floor, ordered floor-major then partition order.
    pub rooms: Vec<Room>,
    pub walls: Vec<Wall>,
    pub openings: Vec<Opening>,
    pub stairs: Vec<Stair>,
    /// The stair core rectangle, `None` for a single-storey building (which
    /// needs no stair and gets an ordinary room there instead).
    pub core: Option<Rect2>,
    /// Index into [`walls`](Self::walls) of the entrance — the one exterior door
    /// on the ground floor. `None` only for a degenerate plan.
    pub entrance: Option<usize>,
}

impl BuildingPlan {
    /// The rooms on `floor`, as `(index, &Room)` pairs in plan order.
    pub fn rooms_on(&self, floor: u32) -> impl Iterator<Item = (usize, &Room)> {
        self.rooms
            .iter()
            .enumerate()
            .filter(move |(_, r)| r.floor == floor)
    }

    /// World Y of `floor`'s walking surface.
    #[inline]
    pub fn floor_y(&self, floor: u32) -> f64 {
        self.base_y + floor as f64 * self.floor_height
    }

    /// The stair room's index on `floor`, if the plan has a core.
    pub fn stair_room(&self, floor: u32) -> Option<usize> {
        self.rooms_on(floor)
            .find(|(_, r)| r.kind == RoomType::Stair)
            .map(|(i, _)| i)
    }

    /// The undirected room-to-room edges the **door** openings create, as
    /// `(room, room)` index pairs, ascending and deduplicated.
    ///
    /// A door on an exterior wall has only one room and contributes no edge —
    /// it is the entrance, and [`floors_reachable`](Self::floors_reachable)
    /// treats it as the link to the outside instead.
    pub fn door_edges(&self) -> Vec<(usize, usize)> {
        let mut out: std::collections::BTreeSet<(usize, usize)> = Default::default();
        for o in &self.openings {
            if o.kind != OpeningKind::Door {
                continue;
            }
            let Some(w) = self.walls.get(o.wall) else {
                continue;
            };
            if let Some(other) = w.outside {
                out.insert((w.inside.min(other), w.inside.max(other)));
            }
        }
        out.into_iter().collect()
    }

    /// The undirected room-to-room edges the **stairs** create: the two stair
    /// rooms every [`Stair`] joins, ascending and deduplicated.
    ///
    /// The twin of [`door_edges`](Self::door_edges) for the other axis — a door
    /// joins two rooms across a wall, a stair joins two across a slab — and
    /// together they are [`room_links`](Self::room_links), which is the graph
    /// every walk over this plan uses.
    fn stair_links(&self) -> Vec<(usize, usize)> {
        let mut out: std::collections::BTreeSet<(usize, usize)> = Default::default();
        for s in &self.stairs {
            if let (Some(a), Some(b)) = (self.stair_room(s.from), self.stair_room(s.to)) {
                if a != b {
                    out.insert((a.min(b), a.max(b)));
                }
            }
        }
        out.into_iter().collect()
    }

    /// **Every edge of the room graph**: [`door_edges`](Self::door_edges) within
    /// a floor plus `stair_links` between them, ascending and deduplicated.
    ///
    /// One derivation with three readers — [`reachable_rooms`](Self::reachable_rooms),
    /// [`room_path`](Self::room_path) and the arm that checks
    /// they agree. The reachability walk carried its own inline copy of the
    /// stair rule until the room-path search needed the same edges, and two
    /// copies of one rule is the pair this tree keeps watching drift apart (the
    /// P22 "one door for three paths" ruling).
    ///
    /// An edge naming a room this plan does not have is dropped rather than
    /// indexed: `Wall::inside`/`outside` are indices, a plan that produced a bad
    /// one is a builder defect, and a walk that panicked on it would take the
    /// whole population pass down with it.
    fn room_links(&self) -> Vec<(usize, usize)> {
        let n = self.rooms.len();
        let mut out: std::collections::BTreeSet<(usize, usize)> = Default::default();
        for (a, b) in self.door_edges().into_iter().chain(self.stair_links()) {
            if a < n && b < n && a != b {
                out.insert((a.min(b), a.max(b)));
            }
        }
        out.into_iter().collect()
    }

    /// **Invariant 1**: every room on `floor` is reachable from every other
    /// through doors.
    pub fn rooms_connected(&self, floor: u32) -> bool {
        let ids: Vec<usize> = self.rooms_on(floor).map(|(i, _)| i).collect();
        if ids.len() <= 1 {
            return true;
        }
        let edges = self.door_edges();
        let mut seen = vec![ids[0]];
        let mut stack = vec![ids[0]];
        while let Some(cur) = stack.pop() {
            for &(a, b) in &edges {
                let next = if a == cur {
                    b
                } else if b == cur {
                    a
                } else {
                    continue;
                };
                if !seen.contains(&next) {
                    seen.push(next);
                    stack.push(next);
                }
            }
        }
        ids.iter().all(|i| seen.contains(i))
    }

    /// **Invariant 2**: the set of rooms reachable from **outside**, walking
    /// through the entrance, through door openings within a floor, and up or
    /// down through stair cores.
    ///
    /// This is the graph the Phase 19 gate walks. Returns one flag per room, in
    /// [`rooms`](Self::rooms) order.
    ///
    /// The edges are `room_links` — doors within a floor, stairs between them —
    /// so this and [`room_path`](Self::room_path) walk one graph and cannot
    /// disagree about which rooms a building offers.
    pub fn reachable_rooms(&self) -> Vec<bool> {
        let mut seen = vec![false; self.rooms.len()];
        // Seed: whichever room the entrance door's wall belongs to.
        let Some(start) = self
            .entrance
            .and_then(|w| self.walls.get(w))
            .map(|w| w.inside)
            .filter(|i| *i < seen.len())
        else {
            return seen;
        };
        let edges = self.room_links();
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(cur) = stack.pop() {
            for &(a, b) in &edges {
                let next = if a == cur {
                    b
                } else if b == cur {
                    a
                } else {
                    continue;
                };
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        seen
    }

    /// **Invariant 2, as the headline predicate**: every floor has at least one
    /// room reachable from outside — i.e. you can walk in and get to every
    /// storey.
    pub fn floors_reachable(&self) -> bool {
        let seen = self.reachable_rooms();
        (0..self.floors).all(|f| {
            self.rooms_on(f)
                .any(|(i, _)| seen.get(i).copied().unwrap_or(false))
        })
    }

    /// The strongest form: every *room* of every floor is reachable from
    /// outside.
    pub fn fully_reachable(&self) -> bool {
        !self.rooms.is_empty() && self.reachable_rooms().iter().all(|&b| b)
    }

    /// **The way from one room to another, as a sequence** — the room indices
    /// to walk, `from` first and `to` last.
    ///
    /// [`reachable_rooms`](Self::reachable_rooms) answers *whether* a room can
    /// be got to; this answers *how*, over exactly the same graph
    /// (`room_links`: doors within a floor, stairs between
    /// them), so the two can never disagree — the arm
    /// `a_room_path_agrees_with_reachability` asserts that room by room on a
    /// real plan.
    ///
    /// Breadth-first, so the answer is the route through the fewest **doors**
    /// rather than the shortest walk. That is the right measure for this plan:
    /// a room here is one hop wide and has no interior circulation of its own,
    /// so metres inside a room are not a thing the plan knows. When metres are
    /// what a caller wants, [`interior_nav`](Self::interior_nav) prices the same
    /// interior in them and `inf_nav::route` searches it.
    ///
    /// # Determinism: ties break on the lower room index
    ///
    /// A floor plate is symmetric by construction — a corridor with rooms either
    /// side offers two equally short ways round — so a search that resolved ties
    /// on an insertion order would hand two hosts two different routes through
    /// one building. The frontier is a `BTreeSet` expanded a level at a time in
    /// ascending index order, and each room's neighbours are ascending too, so a
    /// room is first reached from the **lowest-indexed** room of the previous
    /// level and the answer is a function of the plan alone. Same rule, one
    /// crate up, that `inf_nav::route` states about its `(cost, id)` frontier.
    ///
    /// `Some(vec![from])` for a room to itself; `None` for a room this one
    /// cannot reach and for an index this plan does not have. A refusal is a
    /// value (the P21 ruling), never a panic — "can this NPC get to the back
    /// room" is a question with a legitimate no.
    pub fn room_path(&self, from: usize, to: usize) -> Option<Vec<usize>> {
        if from >= self.rooms.len() || to >= self.rooms.len() {
            return None;
        }
        if from == to {
            return Some(vec![from]);
        }
        // Adjacency once, ascending both ways — a `BTreeMap` of `BTreeSet`s, so
        // neither the neighbour order nor the frontier order can depend on how
        // the plan was built.
        let mut adj: std::collections::BTreeMap<usize, std::collections::BTreeSet<usize>> =
            Default::default();
        for (a, b) in self.room_links() {
            adj.entry(a).or_default().insert(b);
            adj.entry(b).or_default().insert(a);
        }
        let mut prev: std::collections::BTreeMap<usize, usize> = Default::default();
        let mut seen = vec![false; self.rooms.len()];
        seen[from] = true;
        let mut frontier: std::collections::BTreeSet<usize> = [from].into_iter().collect();
        while !frontier.is_empty() && !seen[to] {
            let mut next: std::collections::BTreeSet<usize> = Default::default();
            for cur in &frontier {
                for n in adj.get(cur).into_iter().flatten() {
                    if !seen[*n] {
                        seen[*n] = true;
                        prev.insert(*n, *cur);
                        next.insert(*n);
                    }
                }
            }
            frontier = next;
        }
        if !seen[to] {
            return None;
        }
        // Unwind. `prev` is a tree rooted at `from`, so this ends in at most one
        // step per room; the bound is a loop guard rather than a trust, exactly
        // as `inf_nav::route`'s own unwind states — a corrupted predecessor map
        // must not hang a fixed step.
        let mut out = vec![to];
        let mut cur = to;
        for _ in 0..self.rooms.len() {
            if cur == from {
                break;
            }
            let p = *prev.get(&cur)?;
            out.push(p);
            cur = p;
        }
        if cur != from {
            return None;
        }
        out.reverse();
        Some(out)
    }

    /// **The interior as an [`inf_nav::NavGraph`]** (NPC1c): a node standing in
    /// every room, a node in every doorway, and the stair between the storeys.
    ///
    /// # A doorway is a node, not an edge label
    ///
    /// A wall lets a body through at exactly one place, and a room-to-room edge
    /// that named only the two rooms would draw a straight line between their
    /// centres — through the wall. So every door opening becomes a node at its
    /// own threshold and the room graph goes `room → doorway → room`, which is
    /// the geometry the assembly already built: an opening is a place no wall
    /// run covers (the module's own "never a boolean cut" doctrine), and its
    /// threshold is [`opening_void`](Self::opening_void)'s rectangle centre —
    /// the one derivation of an opening's world rectangle, reused rather than
    /// written a second time.
    ///
    /// An **exterior** door — the entrance — gets a doorway node with only one
    /// room on it. That leaf is the point an outside caller welds to: a street
    /// graph and an interior graph meet on the geometry they already agree
    /// about, which is what `NavGraph::weld` is for, and neither has to be
    /// taught about the other.
    ///
    /// # Positions
    ///
    /// Every node is placed through [`LotFrame::to_world`] and lifted to
    /// [`floor_y`](Self::floor_y), because the whole plan is computed in the
    /// lot's own frame and turned into the world at one place (the IB-6 ruling).
    /// A room's node is its rectangle's **centre**, which is an honest limit
    /// worth naming: a route through this graph is centre to centre and knows
    /// nothing about the furniture the same plan scatters into the room. Avoiding
    /// a table is a body's problem, not a plan's.
    ///
    /// # A stair costs its own rise
    ///
    /// The two stair rooms of a flight share one rectangle — that is what makes
    /// the stack line up — so the link between them is vertical and `link`
    /// prices it at exactly one `floor_height`. It is deliberately *not* marked
    /// up: `NavGraph::link_with_cost` is the door for saying a metre of stair is
    /// dearer than a metre of corridor, and nothing in this tree has measured
    /// what that multiplier should be. An unmeasured prescription can be
    /// backwards (the P25 law), so the geometry stands until somebody measures.
    ///
    /// # The id layout
    ///
    /// | bits | meaning |
    /// |---|---|
    /// | 60–63 | [`inf_nav::domain::BUILDING`] — who minted the id |
    /// | 59 | the class: `0` a room, `1` a doorway |
    /// | 20–58 | the **salt**: which building (NPC1d; `0` here) |
    /// | 0–19 | the index into [`rooms`](Self::rooms) or [`openings`](Self::openings) |
    ///
    /// See [`room_node_id_in`] and [`doorway_node_id_in`], which are the only
    /// two places the layout is written. This plan does not know which building
    /// it is, and inventing an identity here would be a second opinion about a
    /// thing the level already knows — so a caller folding several buildings
    /// into one network passes its own salt through
    /// [`interior_nav_in`](Self::interior_nav_in). This entry point is that call
    /// with a salt of zero, which is right for the one-building case and
    /// **wrong, silently, for two** — which is why it is not the only door.
    pub fn interior_nav(&self) -> inf_nav::NavGraph {
        self.interior_nav_in(0)
    }

    /// **This building's interior as a graph, in `salt`'s own namespace**
    /// (NPC1d).
    ///
    /// Identical to [`interior_nav`](Self::interior_nav) except that every node
    /// id carries `salt` in bits 20–58, so two buildings' graphs
    /// [`absorb`](inf_nav::NavGraph::absorb) into one network without room 5 of
    /// the first becoming room 5 of the second. Before this wave that hazard was
    /// written into a doc and a carried list and armed nowhere; the arm is
    /// `two_buildings_absorb_into_one_network_only_when_they_are_salted_apart`.
    ///
    /// The salt is masked to thirty-nine bits and otherwise uninterpreted: a
    /// dense ordinal and a hash are equally welcome, and the width is
    /// thirty-nine bits, which is chosen so a hash is safe (see the private
    /// `NAV_SALT_MASK` for the arithmetic).
    pub fn interior_nav_in(&self, salt: u64) -> inf_nav::NavGraph {
        let mut g = inf_nav::NavGraph::new();
        for (i, r) in self.rooms.iter().enumerate() {
            let c = self.frame.to_world(r.rect.center());
            g.add_node(
                room_node_id_in(salt, i),
                DVec3::new(c.x, self.floor_y(r.floor), c.y),
                inf_nav::NavKind::Room,
            );
        }
        for (i, o) in self.openings.iter().enumerate() {
            if o.kind != OpeningKind::Door {
                continue;
            }
            let (Some(w), Some((void, _band))) = (self.walls.get(o.wall), self.opening_void(o))
            else {
                continue;
            };
            let c = self.frame.to_world(void.center());
            let door = doorway_node_id_in(salt, i);
            g.add_node(
                door,
                DVec3::new(c.x, self.floor_y(w.floor), c.y),
                inf_nav::NavKind::Doorway,
            );
            // A link naming a room this plan does not have is ignored by the
            // graph itself, which is why there is no second filter here.
            g.link(
                room_node_id_in(salt, w.inside),
                door,
                inf_nav::NavKind::Doorway,
                Vec::new(),
            );
            if let Some(other) = w.outside {
                g.link(
                    door,
                    room_node_id_in(salt, other),
                    inf_nav::NavKind::Doorway,
                    Vec::new(),
                );
            }
        }
        for f in &self.stairs {
            let (Some(a), Some(b)) = (self.stair_room(f.from), self.stair_room(f.to)) else {
                continue;
            };
            if a == b {
                continue;
            }
            // **A stair edge walks the FLIGHT, not the shaft** (NPC1c). Both
            // stair rooms are the same rectangle on two storeys, so a link
            // between their centres is a *vertical line*: a pure-pursuit
            // follower reading a target directly above its own feet has no
            // direction at all, and the NPC stands at the bottom of the stairs
            // for ever. Measured exactly that way on the island's own town walk.
            //
            // So the edge carries the two ends of the run, at their own storeys'
            // heights, on the SAME axis the assembler lays its treads along
            // (`along_x = size_x >= size_z`) -- a nav path that climbed the other
            // one would walk a body into the side of its own staircase.
            //
            // **ONE waypoint, at the TOP of the run**, and the ordering is the
            // whole lesson. A flight fills its core and its door opens onto the
            // SIDE of the run, so a body enters part-way up it; a via that named
            // the bottom of the run first sent a follower that had already
            // climbed 3.20 m of a 3.60 m storey back DOWN the treads to reach a
            // waypoint behind it, and it oscillated there until the walk gave
            // up. Measured exactly that way on the island's own town walk.
            //
            // The top of the flight is where a climb ends whichever tread it
            // starts on, so the edge names that and nothing else, and the climb
            // is monotone from any entry.
            let (_, hi) = flight_run(f.rect);
            let via = vec![self.floor_point(hi, f.to.max(f.from))];
            let (from, to) = if f.from < f.to { (a, b) } else { (b, a) };
            g.link(
                room_node_id_in(salt, from),
                room_node_id_in(salt, to),
                inf_nav::NavKind::Stair,
                via,
            );
        }
        g
    }

    /// A point on `floor`'s walking surface, in world metres.
    fn floor_point(&self, xz: glam::DVec2, floor: u32) -> DVec3 {
        let w = self.frame.to_world(xz);
        DVec3::new(w.x, self.floor_y(floor), w.y)
    }

    /// The world-space void an opening carves: its XZ rectangle (widened by
    /// `thickness` across the wall, so a wall solid straddling the line counts)
    /// and its world-Y band.
    pub fn opening_void(&self, o: &Opening) -> Option<(Rect2, (f64, f64))> {
        let w = self.walls.get(o.wall)?;
        let floor_y = self.floor_y(w.floor);
        let thickness = palettes::archetype(self.archetype).wall_thickness;
        Some((
            wall_band(w, o.start, o.end, thickness),
            (floor_y + o.sill, floor_y + o.head),
        ))
    }

    /// **Invariant 3**: no solid in `solids` intrudes into this opening's void.
    ///
    /// # The margin shrinks ALONG the run and in Y — never across the wall
    ///
    /// This distinction is the whole predicate, and getting it wrong makes the
    /// check vacuous rather than merely loose. The void is a **thin** rectangle:
    /// as long as the opening but only as deep as the wall. Shrinking it on
    /// *every* axis eats the thin one first, inverts it
    /// (`min.y > max.y`), and then [`Rect2::intersection`]'s `max > min` test can
    /// never succeed — so every solid reads as clear and the assertion passes for
    /// a building made of one solid block. It is shrunk only where a *legitimate*
    /// touch happens:
    ///
    /// * **along the run**, because the wall runs beside an opening are derived
    ///   from its bounds and end exactly at the jamb;
    /// * **in Y**, because the floor slab's top face is exactly the door's sill
    ///   and the lintel's underside is exactly its head.
    ///
    /// Across the wall there is no legitimate touch: anything there is *in the
    /// doorway*, which is the thing being ruled out.
    pub fn opening_is_clear(&self, o: &Opening, solids: &[PcgCollider], margin: f64) -> bool {
        let Some(w) = self.walls.get(o.wall) else {
            return true;
        };
        let (a, b) = (o.start + margin, o.end - margin);
        let floor_y = self.floor_y(w.floor);
        let (y0, y1) = (floor_y + o.sill + margin, floor_y + o.head - margin);
        // A margin wider than the opening leaves nothing to test. Vacuous rather
        // than false — but it can only happen for an opening narrower than two
        // margins, which `plan` never produces.
        if !positive(b - a) || !positive(y1 - y0) {
            return true;
        }
        let rect = wall_band(w, a, b, palettes::archetype(self.archetype).wall_thickness);
        debug_assert!(rect.is_positive(), "the void must never invert");
        !solids.iter().any(|s| {
            let (lo, hi) = s.y_band();
            hi > y0 && lo < y1 && solid_bounds(s).overlaps(&rect)
        })
    }
}

/// The class bit that separates a **doorway** node's id from a room's — bit 59.
/// See [`BuildingPlan::interior_nav`] for the whole layout.
///
/// A room takes class `0`, so an unsalted room node's id is the domain tag and
/// its index and nothing else. That is not a shortcut: it means the low twenty
/// bits of an unsalted room node are its `rooms` index verbatim, which is what
/// makes a node id readable in a trace without a decoder.
///
/// **NPC1d moved this bit from 40 to 59** to open the salt field between it and
/// the index. Nothing persists a nav node id — the graph is derived on demand in
/// every host — so the move is invisible outside a single process, and the
/// *relative* order of ids is unchanged (every doorway still sorts above every
/// room of the same building), which is what a Dijkstra tie-break reads.
const NAV_DOORWAY_CLASS: u64 = 1 << 59;

/// The index field of an interior node's id — the low twenty bits.
///
/// A million rooms or openings in one building, against a
/// [`MAX_FLOORS`](plan::MAX_FLOORS) of 400; the mask is here so a caller that
/// hands in a nonsense index corrupts its own node rather than the salt, the
/// class or the domain tag above it.
const NAV_INDEX_MASK: u64 = (1 << 20) - 1;

/// How far up an interior node's id the **salt** starts — bit 20, immediately
/// above the index.
const NAV_SALT_SHIFT: u32 = 20;

/// The salt field of an interior node's id — thirty-nine bits, between the index
/// and the class bit.
///
/// Wide on purpose. The salt exists so several buildings' graphs can be folded
/// into one network, and a caller with no dense ordinal to hand will hash
/// something into it — at 2³⁹ the birthday collision probability over the
/// island's own thousand buildings is about one in a million, where a
/// nineteen-bit field would have collided about once per island. A collision is
/// not a slow route, it is a bedroom welded to a bedroom in a different
/// building, so the field is sized for that rather than for the counter.
const NAV_SALT_MASK: u64 = (1 << 39) - 1;

/// **The two ends of a flight's run**, in the plan's own XZ, quarter-inset.
///
/// The axis is chosen by the same rule the assembler lays treads with -- the
/// core's longer side -- because a nav waypoint on the other axis would send a
/// body into the side of its own staircase. The quarter inset keeps both ends on
/// treads rather than on the landing lip at either end.
fn flight_run(rect: Rect2) -> (glam::DVec2, glam::DVec2) {
    let c = rect.center();
    let along_x = rect.size_x() >= rect.size_z();
    if along_x {
        let q = rect.size_x() * 0.25;
        (
            glam::DVec2::new(c.x - q, c.y),
            glam::DVec2::new(c.x + q, c.y),
        )
    } else {
        let q = rect.size_z() * 0.25;
        (
            glam::DVec2::new(c.x, c.y - q),
            glam::DVec2::new(c.x, c.y + q),
        )
    }
}

/// **The nav-graph id of room `i`** in the unsalted namespace. See
/// [`BuildingPlan::interior_nav`] for the bit layout and why the namespace
/// exists.
///
/// Equivalent to [`room_node_id_in(0, i)`](room_node_id_in), and kept as the
/// short spelling because a caller that folds exactly one building into a
/// network needs no salt.
pub fn room_node_id(i: usize) -> inf_nav::NavNodeId {
    room_node_id_in(0, i)
}

/// **The nav-graph id of the doorway standing in opening `i`** in the unsalted
/// namespace.
///
/// The index is into [`BuildingPlan::openings`] — *all* of them, windows
/// included — rather than into the doors alone, so an opening's node id does not
/// move when a window is added beside it. Only doors get a node; a window is an
/// opening you look through.
pub fn doorway_node_id(i: usize) -> inf_nav::NavNodeId {
    doorway_node_id_in(0, i)
}

/// **The nav-graph id of room `i` of the building `salt` names** (NPC1d).
///
/// The salt is the caller's word for *which building this is*; a plan does not
/// know, and inventing an identity inside it would be a second opinion about a
/// thing the level already holds. See [`BuildingPlan::interior_nav_in`].
pub fn room_node_id_in(salt: u64, i: usize) -> inf_nav::NavNodeId {
    inf_nav::domain::BUILDING
        | ((salt & NAV_SALT_MASK) << NAV_SALT_SHIFT)
        | (i as u64 & NAV_INDEX_MASK)
}

/// **The nav-graph id of the doorway in opening `i` of the building `salt`
/// names** (NPC1d).
pub fn doorway_node_id_in(salt: u64, i: usize) -> inf_nav::NavNodeId {
    inf_nav::domain::BUILDING
        | NAV_DOORWAY_CLASS
        | ((salt & NAV_SALT_MASK) << NAV_SALT_SHIFT)
        | (i as u64 & NAV_INDEX_MASK)
}

/// **Which building an interior node belongs to** — the salt back out of an id.
///
/// The inverse of the field [`room_node_id_in`] writes, so a caller holding a
/// route can say which building each of its interior nodes was in without
/// carrying a second map.
pub fn node_salt(id: inf_nav::NavNodeId) -> u64 {
    (id >> NAV_SALT_SHIFT) & NAV_SALT_MASK
}

/// The XZ rectangle a stretch `[from, to]` of `w` occupies at `thickness`
/// metres of wall — the band an opening's void is, and the band a wall module
/// straddling the room boundary fills.
fn wall_band(w: &Wall, from: f64, to: f64, thickness: f64) -> Rect2 {
    let dir = w.direction();
    let normal = DVec2::new(-dir.y, dir.x);
    let half_t = (thickness * 0.5).max(1e-6);
    let (p0, p1) = (w.point_at(from), w.point_at(to));
    let corners = [
        p0 + normal * half_t,
        p0 - normal * half_t,
        p1 + normal * half_t,
        p1 - normal * half_t,
    ];
    let mut min = corners[0];
    let mut max = corners[0];
    for c in corners.iter().skip(1) {
        min = min.min(*c);
        max = max.max(*c);
    }
    Rect2 { min, max }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rects_normalize_and_measure() {
        let r = Rect2::new(DVec2::new(4.0, 9.0), DVec2::new(-1.0, 2.0));
        assert_eq!(r.min, DVec2::new(-1.0, 2.0));
        assert_eq!(r.max, DVec2::new(4.0, 9.0));
        assert_eq!(r.size(), DVec2::new(5.0, 7.0));
        assert_eq!(r.area(), 35.0);
        assert_eq!(r.center(), DVec2::new(1.5, 5.5));
        assert!(r.is_positive());

        let c = Rect2::from_center(DVec2::ZERO, DVec2::new(4.0, 6.0));
        assert_eq!(c.min, DVec2::new(-2.0, -3.0));
        assert_eq!(c.max, DVec2::new(2.0, 3.0));
    }

    /// Touching faces are **adjacency**, not intersection — the distinction the
    /// whole partition rests on, since a tiling's rooms all touch.
    #[test]
    fn touching_rects_do_not_overlap() {
        let a = Rect2::new(DVec2::ZERO, DVec2::new(4.0, 4.0));
        let b = Rect2::new(DVec2::new(4.0, 0.0), DVec2::new(8.0, 4.0));
        assert!(!a.overlaps(&b));
        assert!(a.intersection(&b).is_none());
        let c = Rect2::new(DVec2::new(3.9, 0.0), DVec2::new(8.0, 4.0));
        assert!(a.overlaps(&c));
        // `contains` is half-open on max, so a shared corner belongs to one rect.
        assert!(a.contains(DVec2::ZERO));
        assert!(!a.contains(DVec2::new(4.0, 4.0)));
    }

    #[test]
    fn inset_collapses_rather_than_inverting() {
        let r = Rect2::new(DVec2::ZERO, DVec2::new(2.0, 2.0));
        assert_eq!(r.inset(0.5).size(), DVec2::new(1.0, 1.0));
        let tiny = r.inset(5.0);
        assert_eq!(tiny.size(), DVec2::ZERO, "an over-inset rect must collapse");
        assert_eq!(tiny.min, r.center());
        assert!(!tiny.is_positive());
        // A negative inset expands, which is how a tolerance band is built.
        assert_eq!(r.inset(-1.0).size(), DVec2::new(4.0, 4.0));
    }

    /// `Rect2::new` normalizes, so "negative sizes" are not degenerate at all —
    /// only a directly-built inverted rect is, and `is_positive` is what says so.
    #[test]
    fn only_a_directly_built_inverted_rect_is_degenerate() {
        assert!(Rect2::new(DVec2::ZERO, DVec2::splat(-4.0)).is_positive());
        assert!(!Rect2 {
            min: DVec2::splat(1.0),
            max: DVec2::ZERO,
        }
        .is_positive());
        assert!(!Rect2::new(DVec2::ZERO, DVec2::new(0.0, 5.0)).is_positive());
    }

    /// The bounds of an unrotated solid are **exactly** its half-extents — no
    /// trig, no epsilon — and a quarter turn swaps them exactly too, because the
    /// quarter-turn quaternion's components make `1 − 2s²` and `2sc` exact.
    #[test]
    fn solid_bounds_are_exact_without_trigonometry() {
        let b = PcgCollider {
            center: DVec3::new(0.0, 1.5, 0.0),
            half_extents: DVec3::new(0.1, 1.5, 2.0),
            rotation: glam::DQuat::IDENTITY,
        };
        let bounds = solid_bounds(&b);
        assert_eq!(bounds.size_x(), 0.2);
        assert_eq!(bounds.size_z(), 4.0);
        assert_eq!(b.y_band(), (0.0, 3.0));
        // A quarter turn about +Y swaps the extents. `yaw_onto(+X)` is the
        // grammar's own construction, so this is the real wall case.
        let turned = PcgCollider {
            rotation: crate::grammar::yaw_onto(DVec3::X),
            ..b
        };
        let tb = solid_bounds(&turned);
        assert!((tb.size_x() - 4.0).abs() < 1e-12, "{}", tb.size_x());
        assert!((tb.size_z() - 0.2).abs() < 1e-12, "{}", tb.size_z());
    }

    #[test]
    fn a_wall_measures_and_points_exactly_at_its_ends() {
        let w = Wall {
            floor: 0,
            a: DVec2::new(1.0, 2.0),
            b: DVec2::new(1.0, 8.0),
            inside: 0,
            outside: None,
        };
        assert_eq!(w.length(), 6.0);
        assert!(w.is_exterior());
        assert_eq!(w.point_at(0.0), w.a);
        assert_eq!(w.point_at(6.0), w.b);
        assert_eq!(w.point_at(3.0), DVec2::new(1.0, 5.0));
        assert_eq!(w.direction(), DVec2::new(0.0, 1.0));
        // Degenerate walls answer without NaN.
        let d = Wall { b: w.a, ..w };
        assert_eq!(d.length(), 0.0);
        assert_eq!(d.point_at(1.0), d.a);
        assert!(d.direction().is_finite());
    }

    // ── the interior as a route (NPC1c) ─────────────────────────────────────

    /// **A real three-storey plan**, built through the same door `doorway.rs`'s
    /// own arms use — so what these tests walk is the interior the generator
    /// actually produces, not a hand-drawn graph that agrees with itself.
    fn storeys() -> BuildingPlan {
        plan_building(&BuildingParams {
            archetype: ArchetypeId::Apartment,
            footprint: Rect2::new(DVec2::new(0.0, 0.0), DVec2::new(24.0, 16.0)),
            base_y: 0.0,
            seed: 7,
            floors: 3,
        })
    }

    /// The room the entrance door opens into — the seed of every walk.
    fn entrance_room(plan: &BuildingPlan) -> usize {
        plan.entrance
            .and_then(|w| plan.walls.get(w))
            .map(|w| w.inside)
            .expect("a plan with an entrance")
    }

    /// **A room path is a sequence of REAL doors and stairs**, not a list of
    /// rooms that happen to be reachable.
    #[test]
    fn a_room_path_walks_only_doors_and_stairs() {
        let plan = storeys();
        assert_eq!(plan.floors, 3);
        assert!(plan.fully_reachable(), "the fixture is not enterable");
        let from = entrance_room(&plan);
        let (to, _) = plan
            .rooms_on(1)
            .find(|(i, _)| *i != from)
            .expect("a room on the first floor");

        let path = plan
            .room_path(from, to)
            .expect("the first floor is reachable from the front door");
        assert_eq!(path.first(), Some(&from));
        assert_eq!(path.last(), Some(&to));
        let doors = plan.door_edges();
        let stairs = plan.stair_links();
        let mut used_stair = 0usize;
        for w in path.windows(2) {
            let pair = (w[0].min(w[1]), w[0].max(w[1]));
            let is_door = doors.contains(&pair);
            let is_stair = stairs.contains(&pair);
            assert!(
                is_door || is_stair,
                "the path steps {pair:?}, which is neither a door nor a stair"
            );
            if is_stair {
                used_stair += 1;
            }
        }
        assert!(
            used_stair > 0,
            "a route to the first floor that climbed no stair walked through a slab"
        );
        // A room to itself is a stand, and an index this plan does not have is a
        // refusal rather than a panic.
        assert_eq!(plan.room_path(from, from), Some(vec![from]));
        assert_eq!(plan.room_path(from, plan.rooms.len()), None);
        assert_eq!(plan.room_path(plan.rooms.len(), from), None);
        // …and the answer is a function of the plan, not of a hash order.
        for _ in 0..16 {
            assert_eq!(plan.room_path(from, to).as_deref(), Some(path.as_slice()));
        }
        println!(
            "NPC1c interior: {} rooms over {} floors, {} door edges + {} stair \
             links; entrance room {from} -> room {to} on floor 1 is {} rooms \
             ({used_stair} of them a stair): {path:?}",
            plan.rooms.len(),
            plan.floors,
            doors.len(),
            stairs.len(),
            path.len()
        );
    }

    /// **The path and the reachability answer one graph** — every room a path
    /// reaches is reachable and every reachable room has a path. Two walks over
    /// two copies of one rule is the drift the shared `room_links` exists to
    /// stop, and this is the arm that would see it.
    #[test]
    fn a_room_path_agrees_with_reachability() {
        for arch in ArchetypeId::ALL {
            let plan = plan_building(&BuildingParams {
                archetype: arch,
                footprint: Rect2::new(DVec2::new(0.0, 0.0), DVec2::new(26.0, 18.0)),
                base_y: 12.5,
                seed: 3,
                floors: 2,
            });
            let from = entrance_room(&plan);
            let seen = plan.reachable_rooms();
            assert_eq!(seen.len(), plan.rooms.len());
            for (i, reachable) in seen.iter().enumerate() {
                assert_eq!(
                    plan.room_path(from, i).is_some(),
                    *reachable,
                    "{arch:?}: room {i} is reachable={reachable} and has a path={}",
                    plan.room_path(from, i).is_some()
                );
            }
            assert!(plan.fully_reachable(), "{arch:?} is not fully enterable");
        }
    }

    /// **The interior routes up the stairs**, in metres, over the same rooms
    /// `room_path` names.
    #[test]
    fn the_interior_nav_graph_routes_between_storeys() {
        let plan = storeys();
        let g = plan.interior_nav();
        let doors = plan
            .openings
            .iter()
            .filter(|o| o.kind == OpeningKind::Door)
            .count();
        assert_eq!(
            g.len(),
            plan.rooms.len() + doors,
            "a node per room and a node per door opening"
        );
        for n in g.nodes() {
            assert_eq!(inf_nav::domain::of(n.id), inf_nav::domain::BUILDING);
            assert!(n.position.is_finite());
        }
        // A doorway node really stands in its own opening's threshold, at the
        // walking surface of the floor the wall is on.
        for (i, o) in plan.openings.iter().enumerate() {
            if o.kind != OpeningKind::Door {
                continue;
            }
            let node = g.node(doorway_node_id(i)).expect("every door is a node");
            let w = &plan.walls[o.wall];
            assert_eq!(node.kind, inf_nav::NavKind::Doorway);
            assert_eq!(node.position.y, plan.floor_y(w.floor));
            let (void, _) = plan.opening_void(o).expect("a door has a void");
            let want = plan.frame.to_world(void.center());
            assert!((node.position.x - want.x).abs() < 1e-12);
            assert!((node.position.z - want.y).abs() < 1e-12);
        }

        let from = entrance_room(&plan);
        let (to, _) = plan
            .rooms_on(1)
            .find(|(i, _)| *i != from)
            .expect("a room on the first floor");
        let r = inf_nav::route(&g, room_node_id(from), room_node_id(to))
            .route()
            .expect("the first floor is reachable from the front door");
        let kinds = inf_nav::route::kinds_of(&g, &r.nodes);
        assert!(
            kinds.contains(&inf_nav::NavKind::Stair),
            "a route between storeys that walked no stair went through a slab: \
             {kinds:?}"
        );
        assert!(kinds.contains(&inf_nav::NavKind::Doorway));
        assert_eq!(r.nodes.first(), Some(&room_node_id(from)));
        assert_eq!(r.nodes.last(), Some(&room_node_id(to)));
        // The route really climbs: it ends on the first floor's walking surface.
        assert_eq!(
            r.path.points()[r.path.points().len() - 1].y,
            plan.floor_y(1)
        );
        assert!(r.cost_m >= plan.floor_height, "{} m", r.cost_m);
        println!(
            "NPC1c interior: {} nodes / {} directed edges ({} rooms + {doors} \
             doorways); entrance -> floor 1 is {} nodes, {:.3} m, kinds {kinds:?}",
            g.len(),
            g.edge_count(),
            plan.rooms.len(),
            r.nodes.len(),
            r.cost_m
        );
    }

    /// **A sealed room is a refusal, not a route** — `None` from the sequence
    /// and a `Disconnected` verdict from the search, both naming the same fact.
    #[test]
    fn a_sealed_room_answers_a_refusal_rather_than_a_route() {
        let plan = storeys();
        let from = entrance_room(&plan);
        // A room on the ground floor that is not the entrance room and not the
        // stair core — sealing a stair room would only prove that a stair is a
        // stair.
        let (target, _) = plan
            .rooms_on(0)
            .find(|(i, r)| *i != from && r.kind != RoomType::Stair)
            .expect("a ground-floor room besides the entrance and the core");
        assert!(
            plan.room_path(from, target).is_some(),
            "the fixture is sealed already"
        );

        // Brick up every door on a wall that touches it.
        let mut sealed = plan.clone();
        sealed.openings.retain(|o| {
            let Some(w) = plan.walls.get(o.wall) else {
                return true;
            };
            o.kind != OpeningKind::Door || (w.inside != target && w.outside != Some(target))
        });
        assert!(
            sealed.openings.len() < plan.openings.len(),
            "the fixture's own room had no doors to brick up"
        );
        assert_eq!(sealed.room_path(from, target), None);
        assert!(!sealed.reachable_rooms()[target]);

        let g = sealed.interior_nav();
        let verdict = inf_nav::route(&g, room_node_id(from), room_node_id(target));
        assert_eq!(
            verdict,
            inf_nav::NavVerdict::Disconnected {
                from: room_node_id(from),
                to: room_node_id(target),
            },
            "a sealed room answered {}",
            verdict.reason()
        );
        // …and the room is still a node: it is unreachable, not absent, which is
        // the distinction `OffGraph` exists to keep.
        assert!(g.contains(room_node_id(target)));
        println!(
            "NPC1c interior: sealing room {target} removed {} door openings and \
             left it a node with no route: \"{}\"",
            plan.openings.len() - sealed.openings.len(),
            verdict.reason()
        );
    }
}

#[cfg(test)]
mod nav_namespace_tests {
    use super::*;
    use crate::building::plan::{plan_building, BuildingParams};

    fn plan_of(seed: u64) -> BuildingPlan {
        let footprint = Rect2::new(DVec2::new(-9.0, -7.0), DVec2::new(9.0, 7.0));
        let mut params =
            BuildingParams::new(palettes::ArchetypeId::Apartment, footprint, 0.0, seed);
        params.floors = 3;
        plan_building(&params)
    }

    /// **The one-namespace blocker, armed** (NPC1d). NPC1c's carried item 6 and
    /// the NPC1c audit's carried item 15 both say a plan has ONE id namespace and
    /// that folding two of them welds a bedroom to a bedroom; nothing failed the
    /// day somebody did it. This does.
    #[test]
    fn two_buildings_absorb_into_one_network_only_when_they_are_salted_apart() {
        let a = plan_of(0xA11);
        let b = plan_of(0xB22);
        assert!(a.rooms.len() > 3 && b.rooms.len() > 3, "thin fixture");

        // Unsalted, and the damage is worse than "the second one is lost":
        // `absorb` keeps the FIRST graph's node record but pushes the second
        // graph's EDGES, so B's doors are hung on A's rooms. The fused graph is
        // A's metres wearing A's and B's connectivity — a bedroom with a door
        // into a corridor in another building.
        let (mut fused, ga, gb) = (a.interior_nav(), a.interior_nav(), b.interior_nav());
        fused.absorb(&gb);
        // Printed because the wave ledger quotes these four numbers, and a
        // measurement no arm reproduces is a sentence (NPC1d audit).
        println!(
            "NPC1d fused interiors: {} + {} nodes -> {}; {} + {} directed edges \
             -> {}",
            ga.len(),
            gb.len(),
            fused.len(),
            ga.edge_count(),
            gb.edge_count(),
            fused.edge_count()
        );
        assert!(
            fused.len() < ga.len() + gb.len(),
            "two unsalted interiors are supposed to collide id for id: {} + {} \
             fused to {}",
            ga.len(),
            gb.len(),
            fused.len()
        );
        assert_eq!(
            fused.edge_count(),
            ga.edge_count() + gb.edge_count(),
            "the fused graph should carry BOTH buildings' edges on ONE \
             building's rooms -- that is the defect, and it is what makes the \
             collision a wrong route rather than a missing one"
        );

        // And the collision is not benign: the node that answers for B's room 0
        // stands where A's room 0 stands, metres away in another building.
        let a0 = a
            .interior_nav()
            .node(room_node_id(0))
            .expect("a room")
            .position;
        let b0 = b
            .interior_nav()
            .node(room_node_id(0))
            .expect("a room")
            .position;
        let fused0 = fused.node(room_node_id(0)).expect("a room").position;
        assert_eq!(fused0, a0);
        // (Same footprint, different seed -- the partition differs, so the two
        // room 0s are not the same point. If they ever were this arm is vacuous.)
        assert_ne!(a0, b0, "the two fixtures partition identically");

        // Salted: every node of both survives, and B's room 0 is B's own place.
        let mut apart = a.interior_nav_in(1);
        let n_a = apart.len();
        apart.absorb(&b.interior_nav_in(2));
        assert_eq!(
            apart.len(),
            n_a + b.interior_nav_in(2).len(),
            "a salted absorb lost nodes"
        );
        assert_eq!(
            apart.node(room_node_id_in(1, 0)).expect("a room").position,
            a0
        );
        assert_eq!(
            apart.node(room_node_id_in(2, 0)).expect("a room").position,
            b0
        );
    }

    /// The salt is recoverable, the fields do not overlap, and a salt of zero is
    /// the pre-NPC1d id verbatim.
    #[test]
    fn an_interior_node_id_decomposes_into_its_own_fields() {
        assert_eq!(room_node_id(7), room_node_id_in(0, 7));
        assert_eq!(doorway_node_id(7), doorway_node_id_in(0, 7));
        assert_eq!(room_node_id(7), inf_nav::domain::BUILDING | 7);
        assert_eq!(node_salt(room_node_id(7)), 0);

        for salt in [1u64, 2, 1_000, 1 << 20, NAV_SALT_MASK] {
            for i in [0usize, 1, 4095, NAV_INDEX_MASK as usize] {
                let r = room_node_id_in(salt, i);
                let d = doorway_node_id_in(salt, i);
                assert_eq!(inf_nav::domain::of(r), inf_nav::domain::BUILDING);
                assert_eq!(inf_nav::domain::of(d), inf_nav::domain::BUILDING);
                assert_eq!(node_salt(r), salt, "the salt did not survive its own id");
                assert_eq!(node_salt(d), salt);
                assert_eq!(r & NAV_INDEX_MASK, i as u64);
                assert_eq!(d & NAV_INDEX_MASK, i as u64);
                assert_ne!(r, d, "a room and a doorway share an id");
                assert!(d > r, "a doorway must sort above its own room");
            }
        }
    }

    /// A salted graph is the unsalted one with every id shifted and **nothing
    /// else** — same node count, same edge count, same positions, same kinds.
    #[test]
    fn salting_moves_the_ids_and_no_metres() {
        let p = plan_of(0xC33);
        let zero = p.interior_nav();
        let salted = p.interior_nav_in(0x7F_FFFF_FFFF);
        assert_eq!(zero.len(), salted.len());
        assert_eq!(zero.edge_count(), salted.edge_count());
        for n in zero.nodes() {
            let mate = salted
                .node(n.id | (0x7F_FFFF_FFFF << NAV_SALT_SHIFT))
                .expect("every node has a salted twin");
            assert_eq!(mate.position, n.position);
            assert_eq!(mate.kind, n.kind);
            assert_eq!(
                salted.edges_from(mate.id).len(),
                zero.edges_from(n.id).len(),
                "the salted twin of {} has a different degree",
                n.id
            );
        }
    }
}
