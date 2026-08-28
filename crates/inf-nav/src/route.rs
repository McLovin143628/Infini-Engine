//! The search: Dijkstra over a [`NavGraph`], answering a value.
//!
//! # Why Dijkstra and not A\*
//!
//! A\* needs an admissible heuristic, and the only one available here — the
//! straight-line distance — is admissible **only if every edge costs at least
//! its own chord**. [`NavGraph::link_with_cost`] exists precisely so a builder
//! can say a stair is dearer than its geometry or a highway cheaper, so the
//! heuristic would be admissible for the graphs this tree builds today and
//! silently wrong for the first one that is not. The island's whole road network
//! is a few hundred nodes and a settlement's grid is a few dozen; a search over
//! that is cheap, and the measurement is in this module's own arms rather than
//! in an argument.
//!
//! **What the arm measures is a count, not a clock.**
//! `a_town_sized_grid_is_searched_within_dijkstras_own_bounds` searches a
//! 1 600-node grid and pins the *shape* of the work —
//! `(settled, stale_pops, scanned, pushed)`, the four [`SearchStats`] — because
//! that is a function of the graph and the endpoints and of nothing else. The
//! microseconds are printed beside it and asserted nowhere: the first draft of
//! that arm asserted them and went red on two CI runners at 1 398 and 1 723 µs
//! for a search nobody had touched.
//!
//! The door to A\* is left open in the honest way: the frontier is ordered on
//! `(cost, id)` and nothing else, so adding a heuristic term is a change to one
//! expression on the day a graph is large enough to want one — and the arm's
//! pinned counts go red with it, which is the ledger being rewritten rather
//! than a gate being wrong.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use glam::DVec3;

use crate::graph::{NavGraph, NavKind, NavNodeId};
use crate::path::NavPath;
use crate::Ordered;

/// A found route: the nodes it visits, the chain it walks and what it cost.
#[derive(Clone, Debug, PartialEq)]
pub struct NavRoute {
    /// The node sequence, `from` first and `to` last. At least one element; a
    /// route from a node to itself is that node alone.
    pub nodes: Vec<NavNodeId>,
    /// The whole chain — every node position with every edge's `via` spliced in
    /// between — as an arc-length path.
    pub path: NavPath,
    /// The sum of the edge costs, metres. **Not** the path's length: a builder
    /// that made a stair dear is telling the search something the geometry does
    /// not say, and reporting the geometry back would hide it.
    pub cost_m: f64,
}

impl NavRoute {
    /// Whether this route goes nowhere — a start that is also the destination.
    pub fn is_stand(&self) -> bool {
        self.path.is_stand()
    }
}

/// **What a search answers.** Four outcomes, all values.
///
/// A refusal is a value here for the reason P21.4 made it a rule: a gameplay
/// question that *errors* takes its whole handler down with it, and "can this
/// NPC get to the shop" is a question with a legitimate no. Each refusal carries
/// the endpoints it was asked about, so a caller can log the actual pair rather
/// than re-deriving it.
#[derive(Clone, Debug, PartialEq)]
pub enum NavVerdict {
    /// A route, possibly a stand.
    Found(NavRoute),
    /// Both ends are nodes of the graph and no chain of edges joins them — a
    /// town the landslide cut off, an interior with a sealed room.
    Disconnected { from: NavNodeId, to: NavNodeId },
    /// One end is not a node of this graph at all. Distinct from
    /// [`Disconnected`](Self::Disconnected) on purpose: the first is a fact
    /// about the world and this one is a fact about the *caller*, and a level
    /// that produced it has a bug rather than a landslide.
    OffGraph { node: NavNodeId },
    /// The graph has no nodes. Its own case because "no route" and "no world" are
    /// different answers to a designer looking at a town with no NPCs in it.
    EmptyGraph,
}

impl NavVerdict {
    /// The route, or `None` for any of the three refusals.
    pub fn route(self) -> Option<NavRoute> {
        match self {
            NavVerdict::Found(r) => Some(r),
            _ => None,
        }
    }

    /// Whether a route was found.
    pub fn is_found(&self) -> bool {
        matches!(self, NavVerdict::Found(_))
    }

    /// A short reason, for a log line or a gate's message. `"found"` when there
    /// is a route, so a caller can print the verdict unconditionally.
    pub fn reason(&self) -> &'static str {
        match self {
            NavVerdict::Found(_) => "found",
            NavVerdict::Disconnected { .. } => "no chain of edges joins the two nodes",
            NavVerdict::OffGraph { .. } => "that node is not in this graph",
            NavVerdict::EmptyGraph => "the graph has no nodes",
        }
    }
}

/// **What a search DID**, counted rather than timed.
///
/// Four numbers, every one of them a function of the graph and the two
/// endpoints alone — no clock, no machine, no CI runner. That is the point of
/// them: the property worth defending about this search is its *shape* (a node
/// is decided once, an adjacency is read once), and a shape is countable, so it
/// never has to be inferred from a stopwatch. See
/// `a_town_sized_grid_is_searched_within_dijkstras_own_bounds`, which asserts
/// these and only prints the microseconds.
///
/// Returned by [`route_counted`]. Zero on all four for the three refusals and
/// for a stand, which do no search at all.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SearchStats {
    /// Frontier pops that were **final** — the pop that decided a node's
    /// shortest cost. Dijkstra decides a node at most once, so this is at most
    /// the node count.
    pub settled: usize,
    /// Frontier pops thrown away because the node had already been reached more
    /// cheaply after that entry was pushed: the tax this search pays for lazy
    /// deletion instead of a decrease-key.
    pub stale_pops: usize,
    /// Edge records read by the relaxation loop. A settled node's adjacency is
    /// read exactly once, and the degrees sum to the directed edge count, so
    /// this is at most [`NavGraph::edge_count`].
    pub scanned: usize,
    /// Frontier pushes, the source's included: one for the source and at most
    /// one per edge record that improved a node, so at most `1 + scanned`.
    pub pushed: usize,
}

/// **The shortest route from `from` to `to`.**
///
/// Deterministic in the two ways that matter and are easy to get wrong:
///
/// * the frontier is ordered on `(cost, node id)`, so two equally short routes —
///   which a street *grid* offers by the dozen — always resolve to the same one;
/// * relaxation walks `edges_from`, which is sorted by `(to, cost)`.
///
/// `O((V + E) log V)`. The island's whole road network is a few hundred nodes.
pub fn route(graph: &NavGraph, from: NavNodeId, to: NavNodeId) -> NavVerdict {
    route_counted(graph, from, to).0
}

/// [`route`], with the work the search did counted beside its answer.
///
/// The *same* search — [`route`] is this function with the counters dropped, so
/// there is no second copy of the loop to drift and an arm that drives this door
/// drives the shipped one. The counters exist because the shape of this search
/// is the thing worth defending and a wall clock cannot see it: a runner three
/// times slower moves a microsecond figure and moves none of these four.
pub fn route_counted(
    graph: &NavGraph,
    from: NavNodeId,
    to: NavNodeId,
) -> (NavVerdict, SearchStats) {
    let mut stats = SearchStats::default();
    if graph.is_empty() {
        return (NavVerdict::EmptyGraph, stats);
    }
    if !graph.contains(from) {
        return (NavVerdict::OffGraph { node: from }, stats);
    }
    if !graph.contains(to) {
        return (NavVerdict::OffGraph { node: to }, stats);
    }
    if from == to {
        let p = graph.node(from).map(|n| n.position).unwrap_or(DVec3::ZERO);
        return (
            NavVerdict::Found(NavRoute {
                nodes: vec![from],
                path: NavPath::single(p),
                cost_m: 0.0,
            }),
            stats,
        );
    }

    let mut dist: BTreeMap<NavNodeId, f64> = BTreeMap::new();
    let mut prev: BTreeMap<NavNodeId, NavNodeId> = BTreeMap::new();
    let mut heap: BinaryHeap<Reverse<(Ordered, NavNodeId)>> = BinaryHeap::new();
    dist.insert(from, 0.0);
    heap.push(Reverse((Ordered(0.0), from)));
    stats.pushed += 1;

    while let Some(Reverse((Ordered(d), node))) = heap.pop() {
        // A stale entry: this node was reached again more cheaply after it was
        // pushed. Cheaper than a decrease-key and the standard shape.
        if dist.get(&node).map(|best| d > *best).unwrap_or(true) {
            stats.stale_pops += 1;
            continue;
        }
        stats.settled += 1;
        if node == to {
            break;
        }
        for e in graph.edges_from(node) {
            stats.scanned += 1;
            let nd = d + e.cost_m;
            let better = match dist.get(&e.to) {
                None => true,
                Some(best) => nd < *best,
            };
            if better {
                dist.insert(e.to, nd);
                prev.insert(e.to, node);
                heap.push(Reverse((Ordered(nd), e.to)));
                stats.pushed += 1;
            }
        }
    }

    let Some(cost_m) = dist.get(&to).copied() else {
        return (NavVerdict::Disconnected { from, to }, stats);
    };

    // Unwind. `prev` is a tree rooted at `from`, so this terminates in at most
    // `V` steps; the bound is stated as a loop guard rather than trusted,
    // because a corrupted predecessor map would otherwise hang a fixed step.
    let mut nodes = vec![to];
    let mut cur = to;
    for _ in 0..graph.len() {
        if cur == from {
            break;
        }
        let Some(p) = prev.get(&cur).copied() else {
            return (NavVerdict::Disconnected { from, to }, stats);
        };
        nodes.push(p);
        cur = p;
    }
    if cur != from {
        return (NavVerdict::Disconnected { from, to }, stats);
    }
    nodes.reverse();

    (
        NavVerdict::Found(NavRoute {
            path: chain(graph, &nodes),
            nodes,
            cost_m,
        }),
        stats,
    )
}

/// **The route's geometry**: every node position with every edge's `via`
/// spliced in between, in travel order.
///
/// Public because the two consumers that build a route by hand — a settlement
/// walking its own grid, a building walking its own corridor — need the same
/// splice, and a second copy of it is the thing that would drift.
pub fn chain(graph: &NavGraph, nodes: &[NavNodeId]) -> NavPath {
    let mut pts: Vec<DVec3> = Vec::new();
    for (i, id) in nodes.iter().enumerate() {
        let Some(n) = graph.node(*id) else { continue };
        pts.push(n.position);
        if let Some(next) = nodes.get(i + 1) {
            if let Some(e) = graph.edges_from(*id).iter().find(|e| e.to == *next) {
                pts.extend_from_slice(&e.via);
            }
        }
    }
    NavPath::new(pts)
}

/// The kinds of edge a route walks, in order, deduplicated consecutively — what
/// a gate prints to say "street, doorway, room, stair, room".
pub fn kinds_of(graph: &NavGraph, nodes: &[NavNodeId]) -> Vec<NavKind> {
    let mut out: Vec<NavKind> = Vec::new();
    for w in nodes.windows(2) {
        if let Some(e) = graph.edges_from(w[0]).iter().find(|e| e.to == w[1]) {
            if out.last() != Some(&e.kind) {
                out.push(e.kind);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, z: f64) -> DVec3 {
        DVec3::new(x, 0.0, z)
    }

    /// A 3 × 3 lattice with unit spacing, ids `row * 3 + col`.
    fn lattice() -> NavGraph {
        let mut g = NavGraph::new();
        for r in 0..3u64 {
            for c in 0..3u64 {
                g.add_node(r * 3 + c, p(c as f64, r as f64), NavKind::Street);
            }
        }
        for r in 0..3u64 {
            for c in 0..3u64 {
                let id = r * 3 + c;
                if c + 1 < 3 {
                    g.link(id, id + 1, NavKind::Street, vec![]);
                }
                if r + 1 < 3 {
                    g.link(id, id + 3, NavKind::Street, vec![]);
                }
            }
        }
        g
    }

    #[test]
    fn the_shortest_route_across_a_grid_costs_its_manhattan_distance() {
        let g = lattice();
        let v = route(&g, 0, 8);
        let r = v.route().expect("a corner-to-corner route");
        assert_eq!(r.cost_m, 4.0);
        assert_eq!(r.nodes.first(), Some(&0));
        assert_eq!(r.nodes.last(), Some(&8));
        assert!((r.path.length_m() - 4.0).abs() < 1e-9);
    }

    /// **The tie-break is the whole determinism claim.** A grid offers six equally
    /// short corner-to-corner routes; the search must always pick one of them,
    /// and the same one, or two hosts walking the same town diverge.
    #[test]
    fn equally_short_routes_resolve_the_same_way_every_time() {
        let g = lattice();
        let first = route(&g, 0, 8).route().expect("a route").nodes;
        for _ in 0..64 {
            assert_eq!(route(&g, 0, 8).route().expect("a route").nodes, first);
        }
        // …and the tie-break is on the id, so the answer is the low-id chain.
        assert_eq!(first, vec![0, 1, 2, 5, 8]);
    }

    /// The three refusals, each as a value with its endpoints attached.
    #[test]
    fn every_refusal_is_a_value_that_names_what_went_wrong() {
        assert_eq!(route(&NavGraph::new(), 0, 1), NavVerdict::EmptyGraph);

        let g = lattice();
        assert_eq!(route(&g, 0, 99), NavVerdict::OffGraph { node: 99 });
        assert_eq!(route(&g, 99, 0), NavVerdict::OffGraph { node: 99 });

        let mut cut = NavGraph::new();
        cut.add_node(0, p(0.0, 0.0), NavKind::Road);
        cut.add_node(1, p(1000.0, 0.0), NavKind::Road);
        assert_eq!(
            route(&cut, 0, 1),
            NavVerdict::Disconnected { from: 0, to: 1 }
        );
        assert!(!route(&cut, 0, 1).is_found());
        assert_eq!(
            route(&cut, 0, 1).reason(),
            "no chain of edges joins the two nodes"
        );
    }

    #[test]
    fn a_route_to_itself_is_a_stand() {
        let g = lattice();
        let r = route(&g, 4, 4).route().expect("a stand");
        assert!(r.is_stand());
        assert_eq!(r.cost_m, 0.0);
        assert_eq!(r.path.position_at(9.0), p(1.0, 1.0));
    }

    /// The route walks the spine, not the chord — an agent handed a switchback
    /// climbs it.
    #[test]
    fn a_routes_chain_splices_every_edges_spine_in_travel_order() {
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Road);
        g.add_node(1, p(20.0, 0.0), NavKind::Road);
        g.add_node(2, p(40.0, 0.0), NavKind::Road);
        g.link(0, 1, NavKind::Road, vec![p(10.0, 10.0)]);
        g.link(1, 2, NavKind::Road, vec![p(30.0, -10.0)]);
        let r = route(&g, 0, 2).route().expect("a route");
        assert_eq!(
            r.path.points(),
            &[
                p(0.0, 0.0),
                p(10.0, 10.0),
                p(20.0, 0.0),
                p(30.0, -10.0),
                p(40.0, 0.0)
            ]
        );
        // …and backwards it is the same five points, reversed.
        let back = route(&g, 2, 0).route().expect("a route back");
        let mut want = r.path.points().to_vec();
        want.reverse();
        assert_eq!(back.path.points(), want.as_slice());
    }

    /// A dear edge is avoided even when it is short — which is what
    /// `link_with_cost` is for, and what makes the straight-line A\* heuristic
    /// inadmissible.
    #[test]
    fn a_stated_cost_beats_the_geometry() {
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Room);
        g.add_node(1, p(1.0, 0.0), NavKind::Room);
        g.add_node(2, p(0.0, 5.0), NavKind::Room);
        g.link_with_cost(0, 1, NavKind::Stair, vec![], 100.0);
        g.link(0, 2, NavKind::Room, vec![]);
        g.link(2, 1, NavKind::Room, vec![]);
        let r = route(&g, 0, 1).route().expect("a route");
        assert_eq!(r.nodes, vec![0, 2, 1]);
        // …and the cost reported is the SEARCH's, not the chain's.
        assert!(r.cost_m > 10.0);
        assert!(r.path.length_m() > 10.0);
    }

    #[test]
    fn the_kinds_a_route_walks_are_reported_in_order() {
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Street);
        g.add_node(1, p(5.0, 0.0), NavKind::Doorway);
        g.add_node(2, p(8.0, 0.0), NavKind::Room);
        g.add_node(3, p(8.0, 3.0), NavKind::Room);
        g.link(0, 1, NavKind::Street, vec![]);
        g.link(1, 2, NavKind::Doorway, vec![]);
        g.link(2, 3, NavKind::Stair, vec![]);
        let r = route(&g, 0, 3).route().expect("a route");
        assert_eq!(
            kinds_of(&g, &r.nodes),
            vec![NavKind::Street, NavKind::Doorway, NavKind::Stair]
        );
    }

    /// **THE MEASUREMENT THIS MODULE'S OWN DOC CLAIMS** (NPC1c audit).
    ///
    /// The header argues Dijkstra over A\* and closes with: *"a search over that
    /// is microseconds, and the measurement is in this module's own arms rather
    /// than in an argument."* It was in an argument. Every arm above runs on a
    /// nine-node lattice, and the two graphs the sentence is about — an island's
    /// whole road network and a town's street grid — are three orders of
    /// magnitude apart from it. A cited gate that does not exist is worse than no
    /// claim (the P20 law), so here is the gate.
    ///
    /// A 40 x 40 street lattice is **1 600 nodes / 6 240 directed edges**, which
    /// is comfortably past the 459 / 1 320 `Settlement::street_graph` builds over
    /// all seven of the island's settlements, and it is the worst shape for this
    /// search: a uniform grid has no dear edges to prune with and offers the
    /// frontier a fresh tie at every step.
    ///
    /// # Why the assertion is a COUNT and the microseconds are a `println!`
    ///
    /// The first draft of this arm asserted the clock — `best < 1000.0` µs — and
    /// went red on ubuntu **and** windows at **1 398.2 µs**, on a debug build on
    /// a shared runner, for a search nobody had touched. That is this tree's
    /// clock law met again (the I7 `step_profile` red, the I4b calibration
    /// incident, the VIS1b frozen-eye fixture), and its newest face: **an arm
    /// written during an audit is subject to it exactly as a wave's arm is.**
    /// A structural property must never be asserted through a wall clock. The
    /// failing message even named the property — *"the frontier or the
    /// adjacency changed shape"* — and a millisecond ceiling spanning a 30×
    /// machine spread cannot tell that from "the runner is busy", which is the
    /// one distinction it existed to draw.
    ///
    /// The shape is **countable**, so it is counted. [`route_counted`] is the
    /// same search with its own work tallied ([`route`] is it with the counters
    /// dropped, so there is no second loop to drift). The **ceilings** are
    /// Dijkstra's own law over any graph:
    ///
    /// * **`settled ≤ V = 40 × 40 = 1 600`.** The first pop whose key still
    ///   matches `dist` decides that node for good and it is never re-opened.
    ///   More than `V` means a node was decided **twice**.
    /// * **`scanned ≤ E = 4 × 40 × 39 = 6 240`.** A settled node's adjacency is
    ///   read exactly once and the out-degrees sum to `E`. More than `E` means
    ///   the relaxation re-scans: an adjacency degraded from "the slice stored
    ///   for this node" to "a filter over the edge set" reads `E` records per
    ///   settled node, i.e. `1 600 × 6 240 ≈ 10 000 000`.
    /// * **`pushed ≤ 1 + scanned`** — one push for the source and at most one per
    ///   edge record that improved something — and **`settled + stale_pops ≤
    ///   pushed`**, the identity that ties the three together so no counter can
    ///   be bypassed on one path.
    ///
    /// **And the ceilings alone are not a gate, which is the second thing this
    /// arm was taught the hard way.** They are all satisfied by a search doing
    /// *less* work, and a broken frontier does less: dropping the `Reverse` so
    /// the heap pops the **largest** key settles **79** nodes and reads **232**
    /// edge records here, passes every ceiling — and still answers the right
    /// route at the right cost, because on a uniform grid any monotone staircase
    /// costs the Manhattan distance. (The old wall clock did not catch it
    /// either; it got *faster*, 32.1 µs against 753.2.)
    ///
    /// So this fixture is pinned at its **floor**, which its own geometry
    /// forces: the destination is the **unique** node at the maximum cost
    /// `78 × 20 = 1 560` m, so a frontier that pops in non-decreasing key order
    /// must decide all 1 599 others before it. `(settled, stale_pops, scanned,
    /// pushed)` is exactly **`(1 600, 0, 6 238, 1 600)`** —
    ///
    /// * `settled = V`: forced above, both ways.
    /// * `scanned = E − deg(to) = 6 240 − 2`: every settled node but the
    ///   destination reads its whole adjacency, and the destination's pop
    ///   **breaks** before its own two edges are read.
    /// * `stale_pops = 0` and `pushed = V`: every edge costs the same 20 m, so a
    ///   node is never reached more cheaply after it is discovered.
    ///
    /// The equality is deliberate and it is the arm's teeth. It also means the
    /// day somebody adds the A\* term the header leaves a door for, this arm
    /// goes red — which is right: the header's Dijkstra-over-A\* argument is
    /// what it holds, and a heuristic rewrites the argument.
    ///
    /// **Mutation-verified**, both halves of the failing message:
    ///
    /// * *the adjacency changed shape* — relaxing over every node of the graph
    ///   instead of `graph.edges_from(node)` takes `scanned` from 6 238 to
    ///   **2 558 400**, failing the `E` ceiling by 410×, while the route, its
    ///   node count and its cost stay correct;
    /// * *the frontier changed shape* — the max-heap above fails the floor at
    ///   `(79, 0, 232, 155)` against `(1 600, 0, 6 238, 1 600)`, and nothing
    ///   else in this file fails with it.
    ///
    /// What a count cannot see is a constant-factor change — `Ordered` comparing
    /// through a `String`, an `edges_from` that allocates — and this arm says so
    /// rather than implying a clock covered it, because the red proves it did
    /// not. `edges_from`'s signature carries the second of those: it hands back
    /// a `&[NavEdge]` borrowed from `&self`, which an implementation that
    /// materialised a list per call could not do.
    #[test]
    fn a_town_sized_grid_is_searched_within_dijkstras_own_bounds() {
        const SIDE: u64 = 40;
        let mut g = NavGraph::new();
        for r in 0..SIDE {
            for c in 0..SIDE {
                g.add_node(
                    r * SIDE + c,
                    p(c as f64 * 20.0, r as f64 * 20.0),
                    NavKind::Street,
                );
            }
        }
        for r in 0..SIDE {
            for c in 0..SIDE {
                let id = r * SIDE + c;
                if c + 1 < SIDE {
                    g.link(id, id + 1, NavKind::Street, vec![]);
                }
                if r + 1 < SIDE {
                    g.link(id, id + SIDE, NavKind::Street, vec![]);
                }
            }
        }
        let v = g.len();
        let e = g.edge_count();
        assert_eq!(v as u64, SIDE * SIDE);
        assert_eq!(e as u64, 4 * SIDE * (SIDE - 1));

        let (from, to) = (0u64, SIDE * SIDE - 1);
        let (verdict, s) = route_counted(&g, from, to);
        let r = verdict.route().expect("a corner-to-corner route");

        // The world first: a search that answered nothing would settle the
        // fewest nodes of all.
        assert_eq!(
            r.nodes.len() as u64,
            2 * SIDE - 1,
            "the route is not a monotone staircase"
        );
        let manhattan_m = 2.0 * (SIDE - 1) as f64 * 20.0;
        assert!(
            (r.cost_m - manhattan_m).abs() < 1.0e-9,
            "the route costs {} m and the grid's Manhattan distance is {manhattan_m} m",
            r.cost_m
        );

        // THE BOUNDS. Dijkstra's own, as arithmetic over this fixture — see the
        // doc comment for what each one falsifies.
        assert!(
            s.settled <= v,
            "the search settled {} nodes over a {v}-node grid; Dijkstra decides a node once, so more than V means a node was decided twice and the frontier is not a min-priority queue",
            s.settled
        );
        assert!(
            s.scanned <= e,
            "the search read {} edge records over a graph with {e} of them (4 x {SIDE} x {} directed halves); a settled node's adjacency is read once and the out-degrees sum to E, so more than E means the relaxation is re-scanning the edge set",
            s.scanned,
            SIDE - 1
        );
        assert!(
            s.pushed <= 1 + s.scanned,
            "the frontier took {} pushes against {} edge records read; one push for the source plus at most one per improving record is {}",
            s.pushed,
            s.scanned,
            1 + s.scanned
        );
        assert!(
            s.settled + s.stale_pops <= s.pushed,
            "the frontier yielded {} pops ({} settled + {} stale) having been given {} entries",
            s.settled + s.stale_pops,
            s.settled,
            s.stale_pops,
            s.pushed
        );

        // THE FLOOR, which is the half with teeth — the ceilings above are all
        // satisfied by a search that does LESS work and answers this fixture
        // correctly anyway (measured: a max-heap frontier settles 79). This grid
        // forces every one of the four, so they are pinned rather than bounded.
        let deg_to = g.edges_from(to).len();
        assert_eq!(
            (s.settled, s.stale_pops, s.scanned, s.pushed),
            (v, 0, e - deg_to, v),
            "the search's shape is ({}, {}, {}, {}) and this fixture forces (settled, stale, scanned, pushed) = ({v}, 0, {}, {v}): the destination is the UNIQUE node at the maximum cost {manhattan_m} m, so a frontier popping in non-decreasing key order has to decide all {} others before it — settling fewer means the frontier is not a min-priority queue, and settling more means a node was decided twice; every settled node but the destination reads its whole adjacency, and the destination's own {deg_to} edges are never read because its pop BREAKS, so scanned is E - deg(to) = {e} - {deg_to}; and every edge costs the same 20 m, so a node is never reached more cheaply after it is discovered, which is why nothing is ever stale and pushed is V exactly",
            s.settled,
            s.stale_pops,
            s.scanned,
            s.pushed,
            e - deg_to,
            v - 1
        );

        // The clock, PRINTED and asserted nowhere: MIN of five rounds of twenty
        // searches, the shape every other instrument in this tree is read with.
        let mut best_us = f64::INFINITY;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                let _ = route(&g, from, to);
            }
            best_us = best_us.min(t.elapsed().as_secs_f64() * 1.0e6 / 20.0);
        }
        println!(
            "NPC1c audit / inf-nav: {v} nodes / {e} directed edges; corner to corner settles {} (<= {v}), reads {} edge records (<= {e}), pushes {} and discards {} stale; {} route nodes, {} m; {best_us:.1} us on this machine, REPORTED not asserted",
            s.settled,
            s.scanned,
            s.pushed,
            s.stale_pops,
            r.nodes.len(),
            r.cost_m,
        );
    }
}
