//! Cutting a floor plate into rooms, deriving its walls, and choosing the doors
//! that make the result **connected**.
//!
//! # The one algorithm
//!
//! Every floor of every archetype is cut the same way, and the archetype's
//! parameters are the only difference:
//!
//! ```text
//!   floor rect
//!     ├── slab A          ─ recursive split
//!     ├── CORE STRIP      ─ [ stair core | corridor/hall ]
//!     └── slab B          ─ recursive split
//! ```
//!
//! The core strip is placed from the **building** hash, never the floor hash, so
//! it is the same rectangle on every storey. That is what makes the stair align
//! across floors **by construction** — no search, no "find the nearest room on
//! the floor above", no tolerance. It also means the corridor is continuous
//! vertically, which is what a hotel or an office actually is.
//!
//! A slab is split recursively: split the **longer** axis (never a hashed axis —
//! hashing the axis produces slivers, and the longer-axis rule is what keeps
//! rooms squarish), at a fraction drawn from the counter hash inside
//! `[0.5 − JITTER, 0.5 + JITTER]` and then clamped so both halves clear
//! `min_room`. Recursion stops at `max_room_area`, at `min_room`, or at
//! [`MAX_SPLIT_DEPTH`].
//!
//! # THE CONNECTIVITY PROOF
//!
//! A floor's rooms **tile** its rectangle, and the tiling's adjacency graph is
//! connected — provably, not hopefully:
//!
//! * Consider any split this algorithm made. Its two halves each end up tiled by
//!   leaves, and every leaf abutting the split line has extent ≥ `min_room`
//!   *along* that line (the clamp guarantees it).
//! * Both sides' leaves tile the **same** interval on that line. Take the first
//!   leaf on each side: they start at the same coordinate, so they overlap by at
//!   least `min(their extents) ≥ min_room`.
//! * `min_room ≥ 2 · door_width` is a palette invariant (asserted in
//!   `palettes::tests::palette_parameters_are_self_consistent`), and every
//!   palette's `door_width ≥ 0.85 > 2 · DOOR_JAMB`, so that overlap always
//!   clears `door_width + 2 · DOOR_JAMB` and can host a **full-width** door.
//! * By induction over the split tree, the door-capable adjacency graph of every
//!   subtree is connected, and every split adds an edge joining its two
//!   subtrees. ∎
//!
//! Note what the proof does **not** claim: that *every* pair of touching rooms
//! shares a door-width wall. Two leaves whose boundaries nearly align can share
//! a sliver, and [`connect`] simply does not treat a sliver as an edge — it
//! never narrows a door to make one fit. The connectivity is carried by the
//! wide edges the proof produces.
//!
//! [`connect`] therefore does not need to *repair* anything: it takes a
//! deterministic spanning walk of the adjacency graph and puts a door on each
//! tree edge. It returns whether it succeeded, and [`BuildingPlan::rooms_connected`]
//! re-checks the finished plan, because a proof that is never machine-checked is
//! a comment.
//!
//! # Determinism
//!
//! The split fraction, the room-type draw and the corridor's position are pure
//! functions of `(hash, index)`. The adjacency walk visits rooms in **index
//! order** and edges in `(a, b)` ascending order, so the spanning tree is a
//! function of the geometry alone.

use glam::DVec2;

use super::palettes::{BuildingArchetype, RoomWeight};
use super::{Opening, OpeningKind, Rect2, RoomType, Wall};
use crate::grammar::span::positive;
use crate::hash::Hash64;

/// The deepest a slab is split. A guard, not a design knob: `max_room_area`
/// stops the recursion long before this on every shipped palette, and a 14-deep
/// tree is 16 384 leaves.
pub const MAX_SPLIT_DEPTH: u32 = 14;

/// How far from the middle a split may wander, as a fraction of the axis.
const SPLIT_JITTER: f64 = 0.16;

/// Metres of solid wall an interior door keeps on each side.
///
/// A door flush against a corner is a modelling error you cannot un-see, but the
/// figure is set by something sharper: a module on the **perpendicular** wall
/// meeting this one at the corner straddles its own line and therefore reaches
/// `collider.x` metres *along* this run. If the jamb were narrower than that,
/// a corner post would stand inside the doorway. `0.4` is wider than the widest
/// cross half-extent any shipped palette declares (industrial's `Column`, 0.3),
/// and `palettes::tests::palette_parameters_are_self_consistent` asserts both
/// halves of that statement so a new palette cannot quietly break it.
pub const DOOR_JAMB: f64 = 0.4;

// Per-draw salts. Mixing a distinct constant into the hash **decorrelates**
// these three draw families — it does not *partition* the seed space, which no
// finite hash can do: two salts can still collide on a given index, just no more
// often than chance. That is exactly what is wanted (the split fraction, the room
// type and the core position must not move together) and it is worth stating
// precisely, because "carves out its own space" would promise a structural
// guarantee the counter-hash doctrine never made.
const SALT_SPLIT: u64 = 0x5350_4C54; // "SPLT"
const SALT_ROOM: u64 = 0x524F_4F4D; // "ROOM"
const SALT_CORE: u64 = 0x434F_5245; // "CORE"

/// `true` when two coordinates are the same wall line. The partition produces
/// shared boundaries by *construction* — a split writes one coordinate into both
/// halves — so this is an equality test with a hair of slack for the one place
/// arithmetic could differ (a rectangle rebuilt through [`Rect2::new`]'s
/// min/max), not a fuzzy geometric match.
fn same_line(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-9
}

/// A room-index pair that shares a wall, with the shared segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adjacency {
    pub a: usize,
    pub b: usize,
    /// The wall line: `true` when the shared boundary is a constant-X line
    /// (the rooms sit left/right of each other).
    pub vertical: bool,
    /// The constant coordinate of the shared line.
    pub line: f64,
    /// The shared interval on the other axis.
    pub from: f64,
    pub to: f64,
}

impl Adjacency {
    /// Length of the shared boundary in metres.
    #[inline]
    pub fn length(&self) -> f64 {
        self.to - self.from
    }
}

/// Cut one floor plate into rooms.
///
/// `core` is the stair-core rectangle — identical on every floor — and `hall` is
/// the corridor beside it; both are already carved out of `plate` by the caller
/// ([`super::plan`]), which is why this function only ever recurses on true
/// rectangles.
///
/// Returns rooms in a deterministic, depth-first order: a caller may rely on
/// index 0 being the first leaf of the first slab.
pub fn partition_floor(plate: Rect2, arch: &BuildingArchetype, hash: Hash64) -> Vec<Rect2> {
    let mut out = Vec::new();
    split(plate, arch, hash, 0, 1, &mut out);
    if out.is_empty() && plate.is_positive() {
        out.push(plate);
    }
    out
}

/// The recursive slab split. `path` is the node's position in the tree (a heap
/// index), which is what the counter hash is folded with — so a room's shape
/// depends on *where it is in the tree*, not on how many draws preceded it.
fn split(
    rect: Rect2,
    arch: &BuildingArchetype,
    hash: Hash64,
    depth: u32,
    path: u64,
    out: &mut Vec<Rect2>,
) {
    if !rect.is_positive() {
        return;
    }
    let (sx, sz) = (rect.size_x(), rect.size_z());
    let done = depth >= MAX_SPLIT_DEPTH
        || rect.area() <= arch.max_room_area
        || (sx < 2.0 * arch.min_room && sz < 2.0 * arch.min_room);
    if done {
        out.push(rect);
        return;
    }
    // The LONGER axis, always. A hashed axis choice makes slivers; this rule is
    // what keeps rooms squarish without any aspect-ratio term. A square region
    // ties to X (`>=`), the same stable tie-break `carve_core` uses — see there.
    let vertical = sx >= sz;
    let (lo, hi, len) = if vertical {
        (rect.min.x, rect.max.x, sx)
    } else {
        (rect.min.y, rect.max.y, sz)
    };
    if len < 2.0 * arch.min_room {
        out.push(rect);
        return;
    }
    let u = hash.mix_u64(SALT_SPLIT).mix_u64(path).unit();
    let frac = 0.5 + (u - 0.5) * 2.0 * SPLIT_JITTER;
    // Clamp so BOTH halves clear `min_room` — the step the connectivity proof
    // depends on.
    let min_frac = arch.min_room / len;
    let frac = frac.clamp(min_frac, 1.0 - min_frac);
    let cut = lo * (1.0 - frac) + hi * frac;
    let (a, b) = if vertical {
        (
            Rect2 {
                min: rect.min,
                max: DVec2::new(cut, rect.max.y),
            },
            Rect2 {
                min: DVec2::new(cut, rect.min.y),
                max: rect.max,
            },
        )
    } else {
        (
            Rect2 {
                min: rect.min,
                max: DVec2::new(rect.max.x, cut),
            },
            Rect2 {
                min: DVec2::new(rect.min.x, cut),
                max: rect.max,
            },
        )
    };
    split(a, arch, hash, depth + 1, path * 2, out);
    split(b, arch, hash, depth + 1, path * 2 + 1, out);
}

/// Draw a room type for room `index` on `floor` from the archetype's weighted
/// table.
pub fn room_type(arch: &BuildingArchetype, floor: u32, index: usize, hash: Hash64) -> RoomType {
    let table: &[RoomWeight] = arch.room_table(floor);
    if table.is_empty() {
        return RoomType::Storage;
    }
    let total: f64 = table.iter().map(|w| w.weight.max(0.0)).sum();
    if !positive(total) {
        return table[0].kind;
    }
    let u = hash
        .mix_u64(SALT_ROOM)
        .mix_u64(floor as u64)
        .mix_u64(index as u64)
        .unit()
        * total;
    let mut acc = 0.0;
    for w in table {
        acc += w.weight.max(0.0);
        if u < acc {
            return w.kind;
        }
    }
    table[table.len() - 1].kind
}

/// A hashed fraction in `[0, 1)` for the core strip's position — drawn from the
/// **building** hash with no floor index, so every storey gets the same answer.
pub fn core_fraction(hash: Hash64, which: u64) -> f64 {
    hash.mix_u64(SALT_CORE).mix_u64(which).unit()
}

/// Every pair of rooms on one floor that shares a wall of positive length.
///
/// `rooms` are indices into the plan's global room list; the returned
/// [`Adjacency`] carries those same indices, ascending in `(a, b)`, and the list
/// itself is in ascending `(a, b)` order — so everything downstream is a
/// function of the geometry alone.
pub fn adjacencies(rooms: &[(usize, Rect2)]) -> Vec<Adjacency> {
    let mut out = Vec::new();
    for (i, &(ia, ra)) in rooms.iter().enumerate() {
        for &(ib, rb) in rooms.iter().skip(i + 1) {
            // Vertical shared line: one's max.x is the other's min.x.
            let vert = if same_line(ra.max.x, rb.min.x) {
                Some(ra.max.x)
            } else if same_line(rb.max.x, ra.min.x) {
                Some(ra.min.x)
            } else {
                None
            };
            if let Some(line) = vert {
                let from = ra.min.y.max(rb.min.y);
                let to = ra.max.y.min(rb.max.y);
                if to - from > 0.0 {
                    out.push(Adjacency {
                        a: ia.min(ib),
                        b: ia.max(ib),
                        vertical: true,
                        line,
                        from,
                        to,
                    });
                }
                continue;
            }
            let horiz = if same_line(ra.max.y, rb.min.y) {
                Some(ra.max.y)
            } else if same_line(rb.max.y, ra.min.y) {
                Some(ra.min.y)
            } else {
                None
            };
            if let Some(line) = horiz {
                let from = ra.min.x.max(rb.min.x);
                let to = ra.max.x.min(rb.max.x);
                if to - from > 0.0 {
                    out.push(Adjacency {
                        a: ia.min(ib),
                        b: ia.max(ib),
                        vertical: false,
                        line,
                        from,
                        to,
                    });
                }
            }
        }
    }
    out.sort_by(|x, y| {
        (x.a, x.b).cmp(&(y.a, y.b)).then(
            x.line
                .partial_cmp(&y.line)
                .unwrap_or(std::cmp::Ordering::Equal),
        )
    });
    out
}

/// The walls of one floor: one run per interior adjacency plus one run per
/// stretch of a room's edge that lies on the plate's boundary.
///
/// A wall is canonically oriented — `a` is the end with the smaller `(x, z)` —
/// so a wall's direction, and therefore every offset measured along it, is a
/// pure function of its geometry rather than of which room was visited first.
pub fn walls_of(floor: u32, plate: Rect2, rooms: &[(usize, Rect2)]) -> Vec<Wall> {
    let mut out = Vec::new();
    // Interior: one wall per shared boundary.
    for adj in adjacencies(rooms) {
        let (a, b) = if adj.vertical {
            (DVec2::new(adj.line, adj.from), DVec2::new(adj.line, adj.to))
        } else {
            (DVec2::new(adj.from, adj.line), DVec2::new(adj.to, adj.line))
        };
        out.push(Wall {
            floor,
            a,
            b,
            inside: adj.a,
            outside: Some(adj.b),
        });
    }
    // Exterior: each room side that lies on the plate boundary.
    for &(idx, r) in rooms {
        let sides = [
            (
                same_line(r.min.x, plate.min.x),
                true,
                r.min.x,
                r.min.y,
                r.max.y,
            ),
            (
                same_line(r.max.x, plate.max.x),
                true,
                r.max.x,
                r.min.y,
                r.max.y,
            ),
            (
                same_line(r.min.y, plate.min.y),
                false,
                r.min.y,
                r.min.x,
                r.max.x,
            ),
            (
                same_line(r.max.y, plate.max.y),
                false,
                r.max.y,
                r.min.x,
                r.max.x,
            ),
        ];
        for (on_edge, vertical, line, from, to) in sides {
            if !on_edge || to - from <= 0.0 {
                continue;
            }
            let (a, b) = if vertical {
                (DVec2::new(line, from), DVec2::new(line, to))
            } else {
                (DVec2::new(from, line), DVec2::new(to, line))
            };
            out.push(Wall {
                floor,
                a,
                b,
                inside: idx,
                outside: None,
            });
        }
    }
    out
}

/// Choose the door openings that make one floor's rooms connected.
///
/// Walks the adjacency graph breadth-first from the lowest room index, adding a
/// door on each **tree** edge; then, for every room adjacent to a `Corridor`,
/// adds a door onto the corridor as well (the hotel/office idiom — a room opens
/// onto the spine even when the spanning tree reached it another way). Returns
/// the openings and whether every room was reached.
///
/// See the module docs for why the walk cannot fail on a plan this module
/// produced.
pub fn connect(
    floor_rooms: &[(usize, Rect2, RoomType)],
    walls: &[Wall],
    arch: &BuildingArchetype,
) -> (Vec<Opening>, bool) {
    let mut out = Vec::new();
    if floor_rooms.len() <= 1 {
        return (out, true);
    }
    // Wall index by (room, room) pair — the door has to land on a real run.
    let wall_of = |a: usize, b: usize| -> Option<usize> {
        walls.iter().position(|w| match w.outside {
            Some(y) => (w.inside == a && y == b) || (w.inside == b && y == a),
            None => false,
        })
    };

    let ids: Vec<usize> = floor_rooms.iter().map(|(i, _, _)| *i).collect();
    let rects: Vec<(usize, Rect2)> = floor_rooms.iter().map(|(i, r, _)| (*i, *r)).collect();
    // **Only door-capable adjacencies.** Two rooms may share a sliver of wall —
    // the proof guarantees that *some* pair across every split shares
    // `min_room`, not that every pair does — and squeezing a narrower door in
    // would silently shrink what the palette authored. So a short shared wall
    // is simply not an edge, and the spanning walk routes around it.
    let need = arch.door_width + 2.0 * DOOR_JAMB;
    let adj: Vec<Adjacency> = adjacencies(&rects)
        .into_iter()
        .filter(|e| e.length() >= need)
        .collect();

    // Breadth-first spanning walk, visiting in index order at every step so the
    // tree is a function of the geometry alone.
    let mut seen: Vec<usize> = vec![ids[0]];
    let mut queue: std::collections::VecDeque<usize> = std::collections::VecDeque::from([ids[0]]);
    let mut tree: Vec<(usize, usize)> = Vec::new();
    while let Some(cur) = queue.pop_front() {
        for e in &adj {
            let next = if e.a == cur {
                e.b
            } else if e.b == cur {
                e.a
            } else {
                continue;
            };
            if seen.contains(&next) {
                continue;
            }
            seen.push(next);
            queue.push_back(next);
            tree.push((cur.min(next), cur.max(next)));
        }
    }
    let connected = ids.iter().all(|i| seen.contains(i));

    // The corridor extras: every room touching a corridor opens onto it.
    let corridors: Vec<usize> = floor_rooms
        .iter()
        .filter(|(_, _, k)| *k == RoomType::Corridor)
        .map(|(i, _, _)| *i)
        .collect();
    let mut wanted: std::collections::BTreeSet<(usize, usize)> = tree.into_iter().collect();
    for e in &adj {
        if corridors.contains(&e.a) || corridors.contains(&e.b) {
            wanted.insert((e.a, e.b));
        }
    }

    for (a, b) in wanted {
        let Some(wi) = wall_of(a, b) else { continue };
        let Some(edge) = adj.iter().find(|e| e.a == a && e.b == b) else {
            continue;
        };
        // The edge was filtered to fit the AUTHORED width, so the door is never
        // narrowed to make it fit.
        let run = edge.length();
        let width = arch.door_width;
        // Centred on the shared run, so a door is never at a corner.
        let start = (run - width) * 0.5;
        out.push(Opening {
            kind: OpeningKind::Door,
            wall: wi,
            start,
            end: start + width,
            sill: 0.0,
            head: arch.door_height,
        });
    }
    out.sort_by_key(|o| o.wall);
    (out, connected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::building::palettes::{archetype, archetypes, ArchetypeId};

    fn plate(w: f64, h: f64) -> Rect2 {
        Rect2::new(DVec2::ZERO, DVec2::new(w, h))
    }

    /// The rooms **tile** the plate: their areas sum to it and no two overlap.
    /// This is the property everything else rests on — walls are derived from
    /// shared boundaries, and a gap or an overlap would invent or lose one.
    #[test]
    fn a_partition_tiles_its_plate_exactly() {
        for arch in archetypes() {
            for (w, h) in [(30.0, 20.0), (12.0, 40.0), (60.0, 60.0), (9.0, 7.0)] {
                let p = plate(w, h);
                let rooms = partition_floor(p, arch, Hash64::new(11));
                assert!(!rooms.is_empty(), "{} produced no rooms", arch.display);
                let sum: f64 = rooms.iter().map(|r| r.area()).sum();
                assert!(
                    (sum - p.area()).abs() < 1e-6,
                    "{}: {w}x{h} area {sum} != {}",
                    arch.display,
                    p.area()
                );
                for (i, a) in rooms.iter().enumerate() {
                    assert!(a.is_positive(), "{}: room {i} is degenerate", arch.display);
                    for b in rooms.iter().skip(i + 1) {
                        assert!(!a.overlaps(b), "{}: rooms overlap", arch.display);
                    }
                }
            }
        }
    }

    /// No room is ever cut below `min_room` on either axis — the clamp the
    /// connectivity proof depends on.
    #[test]
    fn no_room_falls_below_the_minimum() {
        for arch in archetypes() {
            let p = plate(80.0, 55.0);
            for seed in [1u64, 2, 3, 97] {
                for r in partition_floor(p, arch, Hash64::new(seed)) {
                    assert!(
                        r.size_x() >= arch.min_room - 1e-9 && r.size_z() >= arch.min_room - 1e-9,
                        "{}: {}x{} is under min_room {}",
                        arch.display,
                        r.size_x(),
                        r.size_z(),
                        arch.min_room
                    );
                }
            }
        }
    }

    /// A plate too small to split at all is one room, not zero and not a panic.
    #[test]
    fn a_tiny_or_degenerate_plate_degrades_gracefully() {
        let arch = archetype(ArchetypeId::House);
        assert_eq!(
            partition_floor(plate(3.0, 3.0), arch, Hash64::new(0)).len(),
            1
        );
        assert!(partition_floor(plate(0.0, 5.0), arch, Hash64::new(0)).is_empty());
        // `Rect2::new` normalizes, so a "negative size" is an ordinary
        // rectangle; only a directly-built inverted one is degenerate.
        assert!(partition_floor(
            Rect2 {
                min: DVec2::splat(3.0),
                max: DVec2::ZERO,
            },
            arch,
            Hash64::new(0)
        )
        .is_empty());
    }

    /// **The connectivity property**, machine-checked over every archetype,
    /// several plate shapes and several seeds: the adjacency graph of a
    /// partition is connected, which is what makes a door set exist at all.
    #[test]
    fn every_partition_is_adjacency_connected() {
        for arch in archetypes() {
            for (w, h) in [(30.0, 20.0), (14.0, 44.0), (55.0, 55.0)] {
                for seed in [5u64, 6, 7] {
                    let rooms = partition_floor(plate(w, h), arch, Hash64::new(seed));
                    let idx: Vec<(usize, Rect2)> = rooms.iter().copied().enumerate().collect();
                    let adj = adjacencies(&idx);
                    let mut seen = vec![0usize];
                    let mut stack = vec![0usize];
                    while let Some(cur) = stack.pop() {
                        for e in &adj {
                            let n = if e.a == cur {
                                e.b
                            } else if e.b == cur {
                                e.a
                            } else {
                                continue;
                            };
                            if !seen.contains(&n) {
                                seen.push(n);
                                stack.push(n);
                            }
                        }
                    }
                    assert_eq!(
                        seen.len(),
                        rooms.len(),
                        "{} {w}x{h} seed {seed}: {} of {} rooms reachable",
                        arch.display,
                        seen.len(),
                        rooms.len()
                    );
                }
            }
        }
    }

    /// **The second half of the proof**: the graph restricted to *door-capable*
    /// walls is still connected. Two leaves may share a sliver — that is
    /// expected — but the wide edges alone reach every room, which is what makes
    /// `connect` able to place full-width doors and never narrow one.
    #[test]
    fn the_door_capable_subgraph_is_still_connected() {
        for arch in archetypes() {
            let need = arch.door_width + 2.0 * DOOR_JAMB;
            for (w, h) in [(48.0, 36.0), (24.0, 60.0), (33.0, 21.0)] {
                for seed in [31u64, 32, 33] {
                    let rooms = partition_floor(plate(w, h), arch, Hash64::new(seed));
                    let idx: Vec<(usize, Rect2)> = rooms.iter().copied().enumerate().collect();
                    let adj: Vec<Adjacency> = adjacencies(&idx)
                        .into_iter()
                        .filter(|e| e.length() >= need)
                        .collect();
                    let mut seen = vec![0usize];
                    let mut stack = vec![0usize];
                    while let Some(cur) = stack.pop() {
                        for e in &adj {
                            let n = if e.a == cur {
                                e.b
                            } else if e.b == cur {
                                e.a
                            } else {
                                continue;
                            };
                            if !seen.contains(&n) {
                                seen.push(n);
                                stack.push(n);
                            }
                        }
                    }
                    assert_eq!(
                        seen.len(),
                        rooms.len(),
                        "{} {w}x{h} seed {seed}: only {} of {} rooms reachable through \
                         {need} m walls",
                        arch.display,
                        seen.len(),
                        rooms.len()
                    );
                }
            }
        }
    }

    /// **The depth cap is reachable and it holds.** `max_room_area` normally
    /// stops the recursion first, so an archetype tuned to split forever is the
    /// only way to reach [`MAX_SPLIT_DEPTH`] — and a guard nothing ever tests is
    /// a guard nobody knows works.
    ///
    /// Also the place to state the **O(n²) adjacency cliff**: `adjacencies` is a
    /// pairwise scan, so a plate that produced 2¹⁴ leaves would be ~1.3 × 10⁸
    /// comparisons per floor. It cannot happen with a shipped palette (the area
    /// cap bites at a few dozen rooms), but a much larger lot with a much smaller
    /// `max_room_area` would find it — recorded as a remainder rather than
    /// pre-optimized against a case nothing produces.
    #[test]
    fn the_split_depth_cap_bounds_a_pathological_palette() {
        // `max_room_area` at zero means "always split"; only the depth cap and
        // `min_room` can stop it.
        let mut greedy = *archetype(ArchetypeId::House);
        greedy.max_room_area = 0.0;
        greedy.min_room = 0.001;
        let rooms = partition_floor(plate(64.0, 64.0), &greedy, Hash64::new(1));
        assert_eq!(
            rooms.len(),
            1 << MAX_SPLIT_DEPTH,
            "a palette that never stops must stop at the cap"
        );
        // Still a tiling, still connected — the cap truncates depth, it does not
        // corrupt the partition.
        let sum: f64 = rooms.iter().map(|r| r.area()).sum();
        assert!(
            (sum - 64.0 * 64.0).abs() < 1e-6,
            "the capped partition lost area"
        );
    }

    #[test]
    fn partitioning_is_a_pure_function_of_its_inputs() {
        let arch = archetype(ArchetypeId::Office);
        let p = plate(42.0, 28.0);
        let a = partition_floor(p, arch, Hash64::new(4));
        let b = partition_floor(p, arch, Hash64::new(4));
        assert_eq!(a, b);
        let c = partition_floor(p, arch, Hash64::new(5));
        assert_ne!(a, c, "different seeds must build different plans");
    }

    #[test]
    fn adjacency_is_ordered_and_directionless() {
        let rooms = vec![
            (0usize, Rect2::new(DVec2::ZERO, DVec2::new(4.0, 6.0))),
            (1, Rect2::new(DVec2::new(4.0, 0.0), DVec2::new(9.0, 6.0))),
            (2, Rect2::new(DVec2::new(9.0, 0.0), DVec2::new(12.0, 6.0))),
        ];
        let adj = adjacencies(&rooms);
        assert_eq!(adj.len(), 2, "0-1 and 1-2, not 0-2");
        assert_eq!((adj[0].a, adj[0].b), (0, 1));
        assert_eq!((adj[1].a, adj[1].b), (1, 2));
        assert!(adj[0].vertical);
        assert_eq!(adj[0].line, 4.0);
        assert_eq!(adj[0].length(), 6.0);
        // Reversing the input order does not change the answer.
        let mut flipped = rooms.clone();
        flipped.reverse();
        assert_eq!(adjacencies(&flipped), adj);
    }

    /// Rooms that only touch at a corner are **not** adjacent — a doorway
    /// through a point is the classic partition bug.
    #[test]
    fn corner_contact_is_not_adjacency() {
        let rooms = vec![
            (0usize, Rect2::new(DVec2::ZERO, DVec2::new(4.0, 4.0))),
            (1, Rect2::new(DVec2::new(4.0, 4.0), DVec2::new(8.0, 8.0))),
        ];
        assert!(adjacencies(&rooms).is_empty());
    }

    #[test]
    fn walls_cover_interiors_and_the_plate_boundary() {
        let p = plate(10.0, 6.0);
        let rooms = vec![
            (0usize, Rect2::new(DVec2::ZERO, DVec2::new(4.0, 6.0))),
            (1, Rect2::new(DVec2::new(4.0, 0.0), DVec2::new(10.0, 6.0))),
        ];
        let walls = walls_of(2, p, &rooms);
        let interior: Vec<&Wall> = walls.iter().filter(|w| !w.is_exterior()).collect();
        assert_eq!(interior.len(), 1);
        assert_eq!(interior[0].length(), 6.0);
        assert_eq!(interior[0].floor, 2);
        // Six exterior runs: room 0 owns three plate sides, room 1 owns three.
        let ext: Vec<&Wall> = walls.iter().filter(|w| w.is_exterior()).collect();
        assert_eq!(ext.len(), 6);
        // Their total length is the plate's perimeter — nothing double-counted,
        // nothing missed.
        let total: f64 = ext.iter().map(|w| w.length()).sum();
        assert_eq!(total, 2.0 * (10.0 + 6.0));
    }

    #[test]
    fn room_types_are_drawn_from_the_right_table() {
        let arch = archetype(ArchetypeId::Hotel);
        let ground: Vec<RoomType> = (0..40)
            .map(|i| room_type(arch, 0, i, Hash64::new(3)))
            .collect();
        let upper: Vec<RoomType> = (0..40)
            .map(|i| room_type(arch, 3, i, Hash64::new(3)))
            .collect();
        assert!(ground
            .iter()
            .all(|k| arch.ground_rooms.iter().any(|w| w.kind == *k)));
        assert!(upper
            .iter()
            .all(|k| arch.upper_rooms.iter().any(|w| w.kind == *k)));
        // A hotel's upper floors are mostly guest rooms — the weighting works.
        let guests = upper.iter().filter(|k| **k == RoomType::Guest).count();
        assert!(guests > 25, "only {guests}/40 guest rooms");
        // Pure in its inputs.
        assert_eq!(room_type(arch, 3, 7, Hash64::new(3)), upper[7]);
    }

    /// The core position is drawn **without** a floor index, so every storey
    /// gets the same strip — the alignment guarantee, stated as a test.
    #[test]
    fn the_core_fraction_does_not_depend_on_a_floor() {
        let h = Hash64::new(1234);
        let a = core_fraction(h, 0);
        assert_eq!(a, core_fraction(h, 0));
        assert!((0.0..1.0).contains(&a));
        assert_ne!(a, core_fraction(h, 1), "different axes must decorrelate");
    }
}
