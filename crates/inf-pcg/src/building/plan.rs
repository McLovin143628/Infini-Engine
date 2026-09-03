//! From a footprint to a [`BuildingPlan`]: the floor stack, the core strip, the
//! rooms, the walls, the openings and the stairs.
//!
//! # The core strip, and why it is drawn before anything else
//!
//! A stair that does not line up between storeys is not a stair. The obvious
//! implementation — partition each floor, then look for the room on the floor
//! above that overlaps the stair below — needs a tolerance, fails when the
//! partitions disagree, and produces a different answer for a different seed.
//!
//! So the core strip is carved **first**, from the *building* hash with no floor
//! index folded in, and every storey is partitioned around the same rectangle.
//! Alignment is then a property of the arithmetic rather than of a search, and
//! the stairwell, the corridor and the risers are all vertically continuous for
//! free. A single-storey building needs none of it, so its core rectangle
//! becomes an ordinary room and [`BuildingPlan::core`] is `None`.
//!
//! # The entrance
//!
//! "Enterable" starts outside. Exactly one exterior door is placed on the ground
//! floor — the longest exterior run belonging to a room that is not the stair
//! core, ties broken by wall index — and [`BuildingPlan::reachable_rooms`] seeds
//! its walk from the room behind it. Without this the room graph could be
//! perfectly connected and the building still sealed, which is the failure mode
//! a "rooms are connected" assertion alone would miss.

use glam::DVec2;

use super::palettes::{archetype, ArchetypeId};
use super::partition::{connect, core_fraction, partition_floor, room_type, walls_of};
use super::{BuildingPlan, Opening, OpeningKind, Rect2, Room, RoomType, Stair};
use crate::hash::Hash64;

/// Separates a building's draws from every other consumer of the counter hash —
/// the same role [`GRAMMAR_SALT`](crate::grammar::expand) plays one layer down.
const BUILDING_SALT: u64 = 0x6275_696C_6469_6E67; // "building"
const SALT_FLOORS: u64 = 0x464C_5253; // "FLRS"
/// Per-floor decorrelation, so two storeys with the same slabs still differ.
const SALT_FLOOR: u64 = 0x464C_4F52; // "FLOR"

/// Metres of solid wall kept at each end of an **exterior** run before an
/// opening may start.
///
/// Same figure and same reason as [`DOOR_JAMB`](super::partition::DOOR_JAMB): a
/// module on the perpendicular wall reaches its own `collider.x` along this run,
/// and no shipped palette declares one wider than `0.4`.
const JAMB_MARGIN: f64 = super::partition::DOOR_JAMB;

/// The most storeys a plan will build, whatever the params ask for. A guard: at
/// the tallest archetype's 3.1 m storey this is still over a kilometre.
pub const MAX_FLOORS: u32 = 400;

/// What a caller asks for. Everything else comes from the archetype.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildingParams {
    pub archetype: ArchetypeId,
    /// The lot, in world XZ.
    pub footprint: Rect2,
    /// World Y of the ground floor's walking surface.
    pub base_y: f64,
    /// The building's own seed. Two buildings with the same seed *and* the same
    /// footprint are the same building — which is the point.
    pub seed: u64,
    /// Storey count override; `0` draws from the archetype's range.
    pub floors: u32,
}

impl BuildingParams {
    /// A plan request with the archetype's own storey range.
    pub fn new(archetype: ArchetypeId, footprint: Rect2, base_y: f64, seed: u64) -> Self {
        Self {
            archetype,
            footprint,
            base_y,
            seed,
            floors: 0,
        }
    }
}

/// The building's root counter-hash state.
///
/// **Seed distinctness is the caller's job, and is already arranged.** This
/// folds one salt into one seed; two buildings handed the same `seed` are the
/// same building, by design (that is what makes a plan reproducible). What keeps
/// two *different* lots apart is [`pass_seed`](super::pass_seed), which mixes the
/// pass's authored seed with the evaluating volume's own — so a graph shared by
/// seven volumes builds seven different buildings without anybody typing seven
/// seeds. Pinned by `pass::tests::the_volume_seed_decorrelates_two_volumes`.
fn building_hash(seed: u64) -> Hash64 {
    Hash64::new(seed).mix_u64(BUILDING_SALT)
}

/// One side of a [`cut`]: an interval, or `None` when it would be empty.
type Piece = Option<(f64, f64)>;

/// A one-dimensional cut of `[lo, hi]` at `lo + len`, returning `(first, rest)`
/// and `None` for the part that would be empty.
fn cut(lo: f64, hi: f64, len: f64) -> (Piece, Piece) {
    let at = (lo + len).clamp(lo, hi);
    let a = (at - lo > 0.0).then_some((lo, at));
    let b = (hi - at > 0.0).then_some((at, hi));
    (a, b)
}

/// A sub-rectangle of `plate` spanning `[from, to]` on one axis and the whole
/// plate on the other.
fn band(plate: Rect2, vertical: bool, from: f64, to: f64) -> Rect2 {
    if vertical {
        Rect2 {
            min: DVec2::new(from, plate.min.y),
            max: DVec2::new(to, plate.max.y),
        }
    } else {
        Rect2 {
            min: DVec2::new(plate.min.x, from),
            max: DVec2::new(plate.max.x, to),
        }
    }
}

/// The width of the core strip, in metres.
///
/// A corridor archetype's strip carries a corridor, which is *meant* to be
/// narrower than a room. Everything else's strip leaves an ordinary room behind
/// the stair, so it must clear `min_room`.
fn strip_width(arch: &super::BuildingArchetype) -> f64 {
    if arch.corridor {
        arch.corridor_width.max(arch.stair_size.1)
    } else {
        arch.min_room.max(arch.stair_size.1)
    }
}

/// The carved core strip: the stair rectangle, the hall beside it, and the two
/// slabs left over. Any of them may be absent on a tight plate.
struct Core {
    stair: Option<Rect2>,
    hall: Option<Rect2>,
    slabs: Vec<Rect2>,
}

/// Carve `plate` into `[slab | strip | slab]` and the strip into
/// `[stair | hall]`, entirely from the **building** hash.
fn carve_core(plate: Rect2, arch: &super::BuildingArchetype, hash: Hash64) -> Core {
    let mut out = Core {
        stair: None,
        hall: None,
        slabs: Vec::new(),
    };
    if !plate.is_positive() {
        return out;
    }
    // The strip runs along the LONGER axis (so a corridor serves many rooms) and
    // is measured across the shorter one. **A square lot ties to X** — `>=`, not
    // `>` — which is arbitrary but must be *stable*: a tie broken by a hash, or
    // by a comparison somebody later flips, would rotate every square building's
    // whole plan and move its stair core. Pinned by
    // `tests::a_square_lot_runs_its_strip_along_x`.
    let along_x = plate.size_x() >= plate.size_z();
    // `vertical` describes the *banding* axis: the strip is a band of constant
    // extent on the axis it is measured across.
    let banding_vertical = !along_x;
    let (cross_lo, cross_hi) = if banding_vertical {
        (plate.min.x, plate.max.x)
    } else {
        (plate.min.y, plate.max.y)
    };
    let cross = cross_hi - cross_lo;
    // How wide the strip is depends on what its LEFTOVER becomes. For a corridor
    // archetype the leftover is a corridor and may legitimately be narrower than
    // a room — that is what a corridor is. For everything else it is an ordinary
    // room (a hall, a landing) drawn from the room table, so it has to clear
    // `min_room` like any other; sizing the strip by the stairwell alone gave an
    // industrial floor a 3.2 m "workshop" against an 8 m minimum.
    let strip_w = strip_width(arch);
    if cross < strip_w {
        // No room for a strip at all: the whole plate is one slab and the
        // building will be single-storey (the caller sees `stair: None`).
        out.slabs.push(plate);
        return out;
    }

    // Where the strip sits across the plate, from the building hash — no floor
    // index, so every storey agrees.
    //
    // **A slab is never left below `min_room`.** The plate narrows through three
    // regimes, and it is the strip's POSITION that gives, never the slabs' size:
    //
    // * `slack >= 2*min_room` — **double-loaded**: a slab either side, the strip
    //   free to wander inside the band that keeps both of them legal.
    // * `min_room <= slack < 2*min_room` — **single-loaded**: the strip goes hard
    //   against one edge (which one is hashed, so it is not always the same
    //   side), leaving exactly one slab carrying the whole slack.
    // * `slack < min_room` — **strip only**: the leftover cannot be a room at
    //   all, so the strip swallows it and the floor is just `[stair | hall]`.
    //
    // Splitting the slack evenly whatever its size — a naive
    // `min(min_room, slack/2)` pad — puts *two* sub-minimum slivers on any
    // ordinary narrow lot (a 12 x 6 m house gave two 1.90 m slabs against a
    // 2.6 m minimum). Halving a room that will not fit is the one move that is
    // always wrong.
    let slack = cross - strip_w;
    let strip = if slack >= 2.0 * arch.min_room {
        let at = cross_lo + arch.min_room + core_fraction(hash, 0) * (slack - 2.0 * arch.min_room);
        (at, at + strip_w)
    } else if slack >= arch.min_room {
        if core_fraction(hash, 2) < 0.5 {
            (cross_lo, cross_lo + strip_w)
        } else {
            (cross_hi - strip_w, cross_hi)
        }
    } else {
        (cross_lo, cross_hi)
    };

    if strip.0 - cross_lo > 0.0 {
        out.slabs
            .push(band(plate, banding_vertical, cross_lo, strip.0));
    }
    if cross_hi - strip.1 > 0.0 {
        out.slabs
            .push(band(plate, banding_vertical, strip.1, cross_hi));
    }
    let strip_rect = band(plate, banding_vertical, strip.0, strip.1);

    // Split the strip along its own length into [stair | hall].
    let (len_lo, len_hi) = if banding_vertical {
        (strip_rect.min.y, strip_rect.max.y)
    } else {
        (strip_rect.min.x, strip_rect.max.x)
    };
    let strip_len = len_hi - len_lo;
    let stair_len = arch.stair_size.0.min(strip_len);
    // The stair goes to whichever end the building hash picks — deterministic,
    // and again floor-independent.
    let from_min = core_fraction(hash, 1) < 0.5;
    if strip_len - stair_len < arch.min_room {
        // The strip is only long enough for the stair: no hall.
        out.stair = Some(strip_rect);
        return out;
    }
    let (first, rest) = if from_min {
        cut(len_lo, len_hi, stair_len)
    } else {
        let (a, b) = cut(len_lo, len_hi, strip_len - stair_len);
        (b, a)
    };
    out.stair = first.map(|(a, b)| band(strip_rect, !banding_vertical, a, b));
    out.hall = rest.map(|(a, b)| band(strip_rect, !banding_vertical, a, b));
    out
}

/// Build the complete plan for one building.
///
/// Pure in `params`: same params ⇒ same plan, bit for bit, on any machine and at
/// any thread count (nothing here is parallel; the parallelism lives one level
/// up, over passes and spans).
pub fn plan_building(params: &BuildingParams) -> BuildingPlan {
    plan_building_in(params, crate::building::LotFrame::IDENTITY)
}

/// [`plan_building`], on a lot with its own frame (IB-6).
///
/// `params.footprint` is read in `frame`'s coordinates, so **the plan itself is
/// unchanged** — every rule in this file, in `partition.rs` and in `assemble.rs`
/// is axis-aligned in the lot's frame and stays that way. The frame rides on the
/// plan and is applied once, to the finished output, by
/// [`assemble_in`](crate::building::assemble_in).
///
/// Two buildings with the same seed and the same *local* footprint are the same
/// building in two places — which is what the frame means, and why it is a
/// placement rather than a design parameter on [`BuildingParams`].
pub fn plan_building_in(params: &BuildingParams, frame: crate::building::LotFrame) -> BuildingPlan {
    let arch = archetype(params.archetype);
    let hash = building_hash(params.seed);
    let plate = params.footprint;

    let core = carve_core(plate, arch, hash);

    // Storey count: the override, else a draw inside the archetype's range —
    // and never more than one when there is no stair core to climb.
    let (fmin, fmax) = arch.floors;
    let drawn = if fmax > fmin {
        let u = hash.mix_u64(SALT_FLOORS).unit();
        fmin + ((u * (fmax - fmin + 1) as f64) as u32).min(fmax - fmin)
    } else {
        fmin
    };
    let mut floors = if params.floors > 0 {
        params.floors
    } else {
        drawn
    }
    .clamp(1, MAX_FLOORS);
    // A floor you cannot reach is not a floor, and a building you cannot enter is
    // not a building. Two ways that happens, both on lots too small for the
    // archetype, and both answered the same way — drop to a single storey, which
    // turns the core rectangle back into an ordinary room:
    //
    // 1. no stair core at all (the plate is narrower than the strip);
    // 2. a core and *nothing else* — the whole plate became the stairwell, so the
    //    ground floor has no room to enter into and `choose_entrance` (which
    //    refuses to put the front door into a stairwell) would find nowhere.
    if core.stair.is_none() || (core.hall.is_none() && core.slabs.is_empty()) {
        floors = 1;
    }
    let multi = floors > 1;
    // A single-storey building has no stairwell, so the strip should be ONE room
    // rather than a stair rectangle demoted to an ordinary one. Demoting it
    // leaves a room sized by `stair_size` — which a palette may legitimately set
    // below `min_room` — masquerading as a normal room; merging is both the
    // honest plan and the one that keeps the minimum.
    let core = if multi {
        core
    } else {
        let merged = match (core.stair, core.hall) {
            (Some(a), Some(b)) => Some(Rect2::new(a.min.min(b.min), a.max.max(b.max))),
            (a, b) => a.or(b),
        };
        Core {
            stair: None,
            hall: merged,
            slabs: core.slabs,
        }
    };

    let mut plan = BuildingPlan {
        archetype: params.archetype,
        footprint: plate,
        frame,
        base_y: params.base_y,
        floors,
        floor_height: arch.floor_height,
        rooms: Vec::new(),
        walls: Vec::new(),
        openings: Vec::new(),
        stairs: Vec::new(),
        // A single-storey building has no *core*: its stair rectangle is just a
        // room, and saying so keeps `stairs` and `core` from disagreeing.
        core: multi.then_some(core.stair).flatten(),
        entrance: None,
    };
    if !plate.is_positive() {
        return plan;
    }

    for floor in 0..floors {
        let fhash = hash.mix_u64(SALT_FLOOR).mix_u64(floor as u64);
        let first_room = plan.rooms.len();
        let first_wall = plan.walls.len();

        // ── the rooms of this floor, in a fixed spatial order ───────────────
        let mut rects: Vec<(Rect2, Option<RoomType>)> = Vec::new();
        if let Some(slab) = core.slabs.first() {
            rects.extend(
                partition_floor(*slab, arch, fhash.mix_u64(0))
                    .into_iter()
                    .map(|r| (r, None)),
            );
        }
        if let Some(stair) = core.stair {
            rects.push((stair, multi.then_some(RoomType::Stair)));
        }
        if let Some(hall) = core.hall {
            rects.push((hall, arch.corridor.then_some(RoomType::Corridor)));
        }
        if let Some(slab) = core.slabs.get(1) {
            rects.extend(
                partition_floor(*slab, arch, fhash.mix_u64(1))
                    .into_iter()
                    .map(|r| (r, None)),
            );
        }
        // **THE DETERMINISTIC ANCHORS** (wave VEN1a). On the ground floor, the
        // archetype's `ground_anchors` claim the largest un-forced rooms in
        // descending area order *before* the weighted draw runs, so a venue's
        // main room IS its dance floor rather than whichever room a hash
        // happened to pick. Ties break on the room's index in the plan's own
        // fixed spatial order, so the answer holds no hash at all.
        //
        // `f64::total_cmp` rather than `partial_cmp`: a degenerate rect's area
        // can be a NaN, and a sort that panics inside a level load is worse than
        // one that puts the NaN somewhere definite.
        //
        // Empty for the seven archetypes that predate the venues — and for no
        // others, since wave EMS1's four institutions each anchor their ground
        // floor too — so for those seven this loop does nothing at all and
        // their plans are byte-identical.
        if floor == 0 && !arch.ground_anchors.is_empty() {
            let mut order: Vec<usize> =
                (0..rects.len()).filter(|k| rects[*k].1.is_none()).collect();
            order.sort_by(|a, b| {
                rects[*b]
                    .0
                    .area()
                    .total_cmp(&rects[*a].0.area())
                    .then(a.cmp(b))
            });
            for (anchor, k) in arch.ground_anchors.iter().zip(order) {
                rects[k].1 = Some(*anchor);
            }
        }
        for (i, (rect, forced)) in rects.iter().enumerate() {
            plan.rooms.push(Room {
                floor,
                rect: *rect,
                kind: forced.unwrap_or_else(|| room_type(arch, floor, i, fhash)),
            });
        }

        // ── the walls ───────────────────────────────────────────────────────
        let indexed: Vec<(usize, Rect2)> = plan.rooms[first_room..]
            .iter()
            .enumerate()
            .map(|(i, r)| (first_room + i, r.rect))
            .collect();
        plan.walls.extend(walls_of(floor, plate, &indexed));

        // ── the doors ───────────────────────────────────────────────────────
        let floor_rooms: Vec<(usize, Rect2, RoomType)> = plan.rooms[first_room..]
            .iter()
            .enumerate()
            .map(|(i, r)| (first_room + i, r.rect, r.kind))
            .collect();
        let (doors, _connected) = connect(&floor_rooms, &plan.walls[first_wall..], arch);
        plan.openings.extend(doors.into_iter().map(|mut o| {
            o.wall += first_wall;
            o
        }));

        // ── the entrance (ground floor only), then the windows ──────────────
        if floor == 0 {
            plan.entrance = choose_entrance(&plan, first_wall);
            if let Some(w) = plan.entrance {
                let run = plan.walls[w].length();
                let width = arch.door_width.min(run - 2.0 * JAMB_MARGIN);
                if width > 0.0 {
                    let start = (run - width) * 0.5;
                    plan.openings.push(Opening {
                        kind: OpeningKind::Door,
                        wall: w,
                        start,
                        end: start + width,
                        sill: 0.0,
                        head: arch.door_height,
                    });
                } else {
                    plan.entrance = None;
                }
            }
        }
        let windows = place_windows(&plan, first_wall, arch);
        plan.openings.extend(windows);
    }

    if let Some(core_rect) = plan.core {
        plan.stairs = (0..floors.saturating_sub(1))
            .map(|f| Stair {
                rect: core_rect,
                from: f,
                to: f + 1,
            })
            .collect();
    }
    plan
}

/// The ground floor's entrance wall: the longest exterior run whose room is not
/// the stair core, ties broken by the lower wall index.
fn choose_entrance(plan: &BuildingPlan, first_wall: usize) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (i, w) in plan.walls.iter().enumerate().skip(first_wall) {
        if !w.is_exterior() {
            continue;
        }
        if plan.rooms.get(w.inside).map(|r| r.kind) == Some(RoomType::Stair) {
            continue;
        }
        let len = w.length();
        if len <= 2.0 * JAMB_MARGIN {
            continue;
        }
        match best {
            Some((_, b)) if b >= len => {}
            _ => best = Some((i, len)),
        }
    }
    best.map(|(i, _)| i)
}

/// Windows on this floor's exterior runs, at the archetype's façade pitch,
/// skipping anything that would clash with an opening already placed.
fn place_windows(
    plan: &BuildingPlan,
    first_wall: usize,
    arch: &super::BuildingArchetype,
) -> Vec<Opening> {
    let mut out = Vec::new();
    for (wi, w) in plan.walls.iter().enumerate().skip(first_wall) {
        if !w.is_exterior() {
            continue;
        }
        // A stairwell and a service riser keep their walls solid, which also
        // keeps the façade from reading as one uniform grid.
        if matches!(
            plan.rooms.get(w.inside).map(|r| r.kind),
            Some(RoomType::Stair) | Some(RoomType::Service)
        ) {
            continue;
        }
        let run = w.length();
        let usable = run - 2.0 * JAMB_MARGIN;
        if usable < arch.window_width {
            continue;
        }
        // How many windows fit at the authored pitch, at least one.
        let n = ((usable / arch.window_pitch).floor() as u32).max(1);
        for k in 0..n {
            // Derived from `k`, never accumulated — the P17.4 exact-linear rule.
            let center = JAMB_MARGIN + usable * (k as f64 + 0.5) / n as f64;
            let start = center - arch.window_width * 0.5;
            let end = center + arch.window_width * 0.5;
            let clash = plan
                .openings
                .iter()
                .chain(out.iter())
                .any(|o| o.wall == wi && o.start < end && start < o.end);
            if clash {
                continue;
            }
            out.push(Opening {
                kind: OpeningKind::Window,
                wall: wi,
                start,
                end,
                sill: arch.window_sill,
                head: arch.window_head,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::palettes::archetypes;

    fn lot(w: f64, h: f64) -> Rect2 {
        Rect2::new(DVec2::ZERO, DVec2::new(w, h))
    }

    fn plan_of(id: ArchetypeId, w: f64, h: f64, seed: u64) -> BuildingPlan {
        plan_building(&BuildingParams::new(id, lot(w, h), 0.0, seed))
    }

    /// **The headline invariant, over all seven archetypes.** Every floor's
    /// rooms are connected, every floor is reachable from outside, and in fact
    /// every *room* is.
    #[test]
    fn every_archetype_plans_a_connected_enterable_building() {
        for arch in archetypes() {
            for seed in [1u64, 7, 64, 4321] {
                let p = plan_building(&BuildingParams::new(arch.id, lot(34.0, 24.0), 12.5, seed));
                assert!(p.floors >= 1, "{}: no storeys", arch.display);
                assert!(!p.rooms.is_empty(), "{}: no rooms", arch.display);
                assert!(
                    p.entrance.is_some(),
                    "{} seed {seed}: sealed — no entrance",
                    arch.display
                );
                for f in 0..p.floors {
                    assert!(
                        p.rooms_connected(f),
                        "{} seed {seed}: floor {f} is not connected",
                        arch.display
                    );
                }
                assert!(
                    p.floors_reachable(),
                    "{} seed {seed}: a floor cannot be reached from outside",
                    arch.display
                );
                assert!(
                    p.fully_reachable(),
                    "{} seed {seed}: a room cannot be reached from outside",
                    arch.display
                );
            }
        }
    }

    /// **The alignment guarantee.** Every storey's stair room is the *same*
    /// rectangle — asserted on `to_bits()`, because "approximately the same
    /// stairwell" is not a stairwell.
    #[test]
    fn the_stair_core_is_bit_identical_on_every_storey() {
        for arch in archetypes() {
            let p = plan_building(&BuildingParams {
                floors: 4,
                ..BuildingParams::new(arch.id, lot(40.0, 26.0), 0.0, 9)
            });
            let Some(core) = p.core else {
                panic!("{} has no core at 4 storeys", arch.display)
            };
            for f in 0..p.floors {
                let idx = p
                    .stair_room(f)
                    .unwrap_or_else(|| panic!("{} floor {f} has no stair room", arch.display));
                let r = p.rooms[idx].rect;
                assert_eq!(r.min.x.to_bits(), core.min.x.to_bits());
                assert_eq!(r.min.y.to_bits(), core.min.y.to_bits());
                assert_eq!(r.max.x.to_bits(), core.max.x.to_bits());
                assert_eq!(r.max.y.to_bits(), core.max.y.to_bits());
            }
            assert_eq!(
                p.stairs.len(),
                3,
                "{}: 4 storeys need 3 flights",
                arch.display
            );
            for (i, s) in p.stairs.iter().enumerate() {
                assert_eq!((s.from, s.to), (i as u32, i as u32 + 1));
                assert_eq!(s.rect, core);
            }
        }
    }

    /// **No planned room falls below `min_room`** — swept over lot sizes, not
    /// just the comfortable one the other tests use.
    ///
    /// `partition::tests::no_room_falls_below_the_minimum` only checks the
    /// *splitter*; it never sees the slabs the core carve hands it, and the carve
    /// was where the bug was: an even `slack/2` pad put **two** sub-minimum
    /// slivers on any ordinary narrow lot. The three sizes marked below are the
    /// reproduced cases.
    ///
    /// The stair and corridor are exempt by design — see
    /// [`BuildingArchetype::min_room`](super::palettes::BuildingArchetype::min_room)
    /// for why a 2.2 m stairwell in a house is right and not a violation.
    #[test]
    fn no_planned_room_falls_below_min_room() {
        for arch in archetypes() {
            for (w, h) in [
                (12.0, 6.0), // reproduced: two 1.90 m slabs vs a 2.6 m minimum
                (10.0, 5.5), // reproduced
                (14.0, 7.0), // reproduced
                (9.0, 9.0),
                (34.0, 24.0),
                (60.0, 18.0),
                (18.0, 60.0),
                (7.0, 30.0),
                (100.0, 45.0),
            ] {
                for seed in [1u64, 2, 3, 40, 41] {
                    let p = plan_building(&BuildingParams::new(arch.id, lot(w, h), 0.0, seed));
                    // `min_room` bounds SUBDIVISION, not the lot: a plate already
                    // narrower than the minimum stays one room rather than
                    // becoming zero rooms, so the bound is against the lot's own
                    // extent too.
                    let floor_min = (arch.min_room.min(w), arch.min_room.min(h));
                    for (i, r) in p.rooms.iter().enumerate() {
                        if matches!(r.kind, RoomType::Stair | RoomType::Corridor) {
                            continue;
                        }
                        assert!(
                            r.rect.size_x() >= floor_min.0 - 1e-9
                                && r.rect.size_z() >= floor_min.1 - 1e-9,
                            "{} {w}x{h} seed {seed}: room {i} is {:.3} x {:.3}, \
                             below the {:.3} x {:.3} floor",
                            arch.display,
                            r.rect.size_x(),
                            r.rect.size_z(),
                            floor_min.0,
                            floor_min.1
                        );
                    }
                    // The plan is still a building: connected, enterable, tiling.
                    if p.rooms.is_empty() {
                        continue;
                    }
                    assert!(
                        p.fully_reachable(),
                        "{} {w}x{h} seed {seed}: not enterable",
                        arch.display
                    );
                    for f in 0..p.floors {
                        let sum: f64 = p.rooms_on(f).map(|(_, r)| r.rect.area()).sum();
                        assert!(
                            (sum - p.footprint.area()).abs() < 1e-6,
                            "{} {w}x{h} floor {f}: does not tile",
                            arch.display
                        );
                    }
                }
            }
        }
    }

    /// The three carve regimes are all reachable, and each leaves the slab count
    /// its name implies — so the branch is exercised rather than merely present.
    #[test]
    fn the_core_carve_has_three_regimes() {
        let arch = archetype(ArchetypeId::House);
        let strip_w = super::strip_width(arch);
        // Double-loaded: slack >= 2*min_room on the cross axis.
        let wide = plan_building(&BuildingParams::new(
            ArchetypeId::House,
            lot(30.0, strip_w + 2.0 * arch.min_room + 4.0),
            0.0,
            1,
        ));
        // Single-loaded: min_room <= slack < 2*min_room (the 12x6 case).
        let narrow = plan_building(&BuildingParams::new(
            ArchetypeId::House,
            lot(12.0, 6.0),
            0.0,
            1,
        ));
        // Strip only: slack < min_room.
        let tight = plan_building(&BuildingParams::new(
            ArchetypeId::House,
            lot(12.0, strip_w + 0.5),
            0.0,
            1,
        ));
        for (what, p) in [("wide", &wide), ("narrow", &narrow), ("tight", &tight)] {
            assert!(!p.rooms.is_empty(), "{what}: no rooms");
            assert!(p.fully_reachable(), "{what}: not enterable");
        }
        // The narrow lot is single-loaded: the strip is hard against one edge, so
        // exactly one side of it carries rooms.
        let strip_edge = narrow.rooms_on(0).any(|(_, r)| {
            r.rect.min.y == narrow.footprint.min.y || r.rect.max.y == narrow.footprint.max.y
        });
        assert!(strip_edge, "the narrow lot's strip is not against an edge");
    }

    /// The rooms of one floor tile the footprint — the property that makes the
    /// derived walls a complete envelope.
    #[test]
    fn each_floor_tiles_the_footprint() {
        for arch in archetypes() {
            let p = plan_building(&BuildingParams {
                floors: 2,
                ..BuildingParams::new(arch.id, lot(28.0, 19.0), 0.0, 3)
            });
            for f in 0..p.floors {
                let rooms: Vec<Rect2> = p.rooms_on(f).map(|(_, r)| r.rect).collect();
                let sum: f64 = rooms.iter().map(|r| r.area()).sum();
                assert!(
                    (sum - p.footprint.area()).abs() < 1e-6,
                    "{} floor {f}: {sum} != {}",
                    arch.display,
                    p.footprint.area()
                );
            }
        }
    }

    /// **The square-lot tie-break is stable.** `size_x() >= size_z()` sends a
    /// square lot's strip along X. Arbitrary, but a tie broken by a hash — or by
    /// a `>` somebody later prefers — would rotate every square building's plan
    /// and move its stair core, which is a content change disguised as a
    /// refactor.
    #[test]
    fn a_square_lot_runs_its_strip_along_x() {
        for arch in archetypes() {
            let p = plan_building(&BuildingParams {
                floors: 2,
                ..BuildingParams::new(arch.id, lot(30.0, 30.0), 0.0, 6)
            });
            let Some(core) = p.core else { continue };
            // The strip runs along X, so the core is a band measured across Z:
            // its X extent is the stair's own length, its Z extent the strip's
            // width — i.e. the core does NOT span the full X of the plate.
            assert!(
                core.size_x() < p.footprint.size_x(),
                "{}: the strip did not run along X on a square lot",
                arch.display
            );
            // And it is stable: same lot, same seed, same rect.
            let again = plan_building(&BuildingParams {
                floors: 2,
                ..BuildingParams::new(arch.id, lot(30.0, 30.0), 0.0, 6)
            });
            assert_eq!(again.core, p.core, "{}: the tie-break moved", arch.display);
        }
    }

    /// A plan is a pure function of its params, and a different seed is a
    /// different building.
    #[test]
    fn plans_are_pure_and_seed_sensitive() {
        let a = plan_of(ArchetypeId::Office, 30.0, 22.0, 12);
        assert_eq!(a, plan_of(ArchetypeId::Office, 30.0, 22.0, 12));
        assert_ne!(a, plan_of(ArchetypeId::Office, 30.0, 22.0, 13));
        assert_ne!(a, plan_of(ArchetypeId::Office, 31.0, 22.0, 12));
        assert_ne!(a, plan_of(ArchetypeId::Hotel, 30.0, 22.0, 12));
    }

    /// A lot too small for a stair core cannot be multi-storey, whatever the
    /// caller asks for — a floor you cannot reach is not a floor.
    #[test]
    fn a_lot_too_small_for_a_core_is_single_storey() {
        let p = plan_building(&BuildingParams {
            floors: 6,
            ..BuildingParams::new(ArchetypeId::Estate, lot(1.5, 1.5), 0.0, 2)
        });
        assert_eq!(p.floors, 1);
        assert!(p.core.is_none());
        assert!(p.stairs.is_empty());
        assert!(!p.rooms.is_empty(), "it is still a building");
        assert!(p.floors_reachable());
        // No stair room is drawn when there is nothing to climb.
        assert!(p.rooms.iter().all(|r| r.kind != RoomType::Stair));
    }

    #[test]
    fn a_degenerate_lot_plans_nothing_rather_than_panicking() {
        // `Rect2::new` normalizes, so a "negative size" is a real lot; the
        // degenerate cases are the zero-area ones and a directly-built
        // inverted rect.
        let inverted = Rect2 {
            min: DVec2::splat(2.0),
            max: DVec2::ZERO,
        };
        for lot in [
            Rect2::new(DVec2::ZERO, DVec2::new(0.0, 10.0)),
            Rect2::new(DVec2::ZERO, DVec2::new(10.0, 0.0)),
            inverted,
        ] {
            let p = plan_building(&BuildingParams::new(ArchetypeId::House, lot, 0.0, 1));
            assert!(p.rooms.is_empty());
            assert!(p.walls.is_empty());
            assert!(p.openings.is_empty());
            assert!(p.entrance.is_none());
            assert!(!p.fully_reachable(), "an empty plan is not enterable");
        }
    }

    /// Openings never overlap each other on one wall, never touch a corner, and
    /// always stay inside their run — the preconditions wall assembly relies on
    /// to derive its runs by subtraction-free arithmetic.
    #[test]
    fn openings_are_disjoint_and_inside_their_runs() {
        for arch in archetypes() {
            let p = plan_building(&BuildingParams {
                floors: 3,
                ..BuildingParams::new(arch.id, lot(36.0, 25.0), 0.0, 17)
            });
            for (i, o) in p.openings.iter().enumerate() {
                let w = &p.walls[o.wall];
                assert!(o.start >= 0.0, "{}: opening before the run", arch.display);
                assert!(
                    o.end <= w.length() + 1e-9,
                    "{}: opening {i} runs past its wall",
                    arch.display
                );
                assert!(o.width() > 0.0, "{}: zero-width opening", arch.display);
                assert!(o.sill < o.head, "{}: inverted opening band", arch.display);
                assert!(
                    o.head <= arch.floor_height,
                    "{}: an opening reaches the slab above",
                    arch.display
                );
                for other in p.openings.iter().skip(i + 1) {
                    if other.wall != o.wall {
                        continue;
                    }
                    assert!(
                        other.start >= o.end || o.start >= other.end,
                        "{}: two openings overlap on wall {}",
                        arch.display,
                        o.wall
                    );
                }
            }
        }
    }

    /// Doors sit between two rooms; windows always face outside.
    #[test]
    fn doors_join_rooms_and_windows_face_out() {
        let p = plan_of(ArchetypeId::Apartment, 32.0, 20.0, 5);
        let mut interior_doors = 0;
        let mut exterior_doors = 0;
        for o in &p.openings {
            let w = &p.walls[o.wall];
            match o.kind {
                OpeningKind::Door if w.is_exterior() => exterior_doors += 1,
                OpeningKind::Door => interior_doors += 1,
                OpeningKind::Window => {
                    assert!(w.is_exterior(), "a window on an interior wall")
                }
            }
        }
        assert_eq!(exterior_doors, 1, "exactly one entrance");
        assert!(interior_doors >= p.rooms.len() - p.floors as usize);
    }

    /// A storey override is honoured, and the stack grows the flights to match.
    #[test]
    fn the_storey_override_is_honoured() {
        for n in [1u32, 2, 5, 9] {
            let p = plan_building(&BuildingParams {
                floors: n,
                ..BuildingParams::new(ArchetypeId::Hotel, lot(40.0, 22.0), 0.0, 1)
            });
            assert_eq!(p.floors, n);
            assert_eq!(p.stairs.len(), (n - 1) as usize);
            assert_eq!(p.core.is_some(), n > 1);
            for f in 0..n {
                assert!(p.rooms_on(f).count() > 0, "floor {f} is empty");
                assert_eq!(p.floor_y(f), f as f64 * p.floor_height);
            }
        }
        // The drawn count always lands inside the archetype's own range.
        for arch in archetypes() {
            for seed in 0..24u64 {
                let p = plan_building(&BuildingParams::new(arch.id, lot(40.0, 26.0), 0.0, seed));
                assert!(
                    p.floors >= arch.floors.0 && p.floors <= arch.floors.1,
                    "{}: drew {} storeys, range is {:?}",
                    arch.display,
                    p.floors,
                    arch.floors
                );
            }
        }
    }

    /// Corridor archetypes really do get a corridor, and every room on a
    /// corridor floor opens onto it.
    #[test]
    fn corridor_archetypes_open_every_room_onto_the_spine() {
        for arch in archetypes().into_iter().filter(|a| a.corridor) {
            let p = plan_building(&BuildingParams {
                floors: 3,
                ..BuildingParams::new(arch.id, lot(44.0, 24.0), 0.0, 21)
            });
            for f in 0..p.floors {
                let corridor = p
                    .rooms_on(f)
                    .find(|(_, r)| r.kind == RoomType::Corridor)
                    .map(|(i, _)| i);
                assert!(
                    corridor.is_some(),
                    "{} floor {f} has no corridor",
                    arch.display
                );
            }
        }
    }
}
