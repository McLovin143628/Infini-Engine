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
//! * **Modules are primitives.** The seven palettes declare modules with no mesh
//!   GUID, so a building needs no imported art to exist: it is boxes with real
//!   dimensions and real colliders. A palette entry gains a mesh the day the
//!   project has one, with no code change (`module Panel = mesh <guid> …`).

pub mod assemble;
pub mod palettes;
pub mod partition;
pub mod pass;
pub mod plan;

use glam::{DVec2, DVec3};

use crate::grammar::span::positive;
use crate::scatter::PcgCollider;

pub use assemble::{assemble, assemble_in, build, build_in, BuildingOutput};
pub use palettes::{
    archetype, archetypes, ArchetypeId, BuildingArchetype, FurnitureDef, RoomWeight,
};
pub use partition::{connect, partition_floor, walls_of, Adjacency};
pub use pass::{
    evaluate_buildings, evaluate_buildings_in, lot_of, oriented_lot_of, pass_seed, plans_of,
    BuildingPass,
};
pub use plan::{plan_building, plan_building_in, BuildingParams, MAX_FLOORS};

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
/// ([`assemble_in`](crate::building::assemble_in)). A rotated lot costs one
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
    /// axis-aligned lot; [`assemble_in`](crate::building::assemble_in) applies
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
    pub fn reachable_rooms(&self) -> Vec<bool> {
        let mut seen = vec![false; self.rooms.len()];
        // Seed: whichever room the entrance door's wall belongs to.
        let Some(start) = self
            .entrance
            .and_then(|w| self.walls.get(w))
            .map(|w| w.inside)
        else {
            return seen;
        };
        let edges = self.door_edges();
        let mut stack = vec![start];
        if start < seen.len() {
            seen[start] = true;
        }
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
            // A stair links the two stair rooms of the storeys it joins.
            if self.rooms[cur].kind == RoomType::Stair {
                let f = self.rooms[cur].floor;
                for s in &self.stairs {
                    let other = if s.from == f {
                        self.stair_room(s.to)
                    } else if s.to == f {
                        self.stair_room(s.from)
                    } else {
                        None
                    };
                    if let Some(n) = other {
                        if !seen[n] {
                            seen[n] = true;
                            stack.push(n);
                        }
                    }
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
}
