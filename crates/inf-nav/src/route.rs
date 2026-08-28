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
//! that is microseconds, and the measurement is in this module's own arms rather
//! than in an argument.
//!
//! The door is left open in the honest way: the frontier is ordered on
//! `(cost, id)` and nothing else, so adding a heuristic term is a change to one
//! expression on the day a graph is large enough to want one.

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
    if graph.is_empty() {
        return NavVerdict::EmptyGraph;
    }
    if !graph.contains(from) {
        return NavVerdict::OffGraph { node: from };
    }
    if !graph.contains(to) {
        return NavVerdict::OffGraph { node: to };
    }
    if from == to {
        let p = graph.node(from).map(|n| n.position).unwrap_or(DVec3::ZERO);
        return NavVerdict::Found(NavRoute {
            nodes: vec![from],
            path: NavPath::single(p),
            cost_m: 0.0,
        });
    }

    let mut dist: BTreeMap<NavNodeId, f64> = BTreeMap::new();
    let mut prev: BTreeMap<NavNodeId, NavNodeId> = BTreeMap::new();
    let mut heap: BinaryHeap<Reverse<(Ordered, NavNodeId)>> = BinaryHeap::new();
    dist.insert(from, 0.0);
    heap.push(Reverse((Ordered(0.0), from)));

    while let Some(Reverse((Ordered(d), node))) = heap.pop() {
        // A stale entry: this node was reached again more cheaply after it was
        // pushed. Cheaper than a decrease-key and the standard shape.
        if dist.get(&node).map(|best| d > *best).unwrap_or(true) {
            continue;
        }
        if node == to {
            break;
        }
        for e in graph.edges_from(node) {
            let nd = d + e.cost_m;
            let better = match dist.get(&e.to) {
                None => true,
                Some(best) => nd < *best,
            };
            if better {
                dist.insert(e.to, nd);
                prev.insert(e.to, node);
                heap.push(Reverse((Ordered(nd), e.to)));
            }
        }
    }

    let Some(cost_m) = dist.get(&to).copied() else {
        return NavVerdict::Disconnected { from, to };
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
            return NavVerdict::Disconnected { from, to };
        };
        nodes.push(p);
        cur = p;
    }
    if cur != from {
        return NavVerdict::Disconnected { from, to };
    }
    nodes.reverse();

    NavVerdict::Found(NavRoute {
        path: chain(graph, &nodes),
        nodes,
        cost_m,
    })
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
    /// The clock is **printed** and the bound asserted is deliberately loose —
    /// this file's own standing rule, and the margin is stated rather than
    /// implied: a corner-to-corner search over a 1 600-node grid is expected in
    /// the tens of microseconds and the ceiling is a **millisecond**, which is
    /// a factor of tens. What that bound is really for is the day somebody makes
    /// `edges_from` allocate or `Ordered` compare through a `String`: it catches
    /// a change of shape, not a change of machine.
    #[test]
    fn a_town_sized_grid_is_searched_in_microseconds() {
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
        assert_eq!(g.len() as u64, SIDE * SIDE);
        assert_eq!(g.edge_count() as u64, 4 * SIDE * (SIDE - 1));

        let (from, to) = (0u64, SIDE * SIDE - 1);
        // MIN of five rounds of twenty searches, the shape every other clock in
        // this tree is taken with.
        let mut best = f64::INFINITY;
        let mut nodes = 0usize;
        let mut cost = 0.0;
        for _ in 0..5 {
            let t = std::time::Instant::now();
            for _ in 0..20 {
                let r = route(&g, from, to)
                    .route()
                    .expect("a corner-to-corner route");
                nodes = r.nodes.len();
                cost = r.cost_m;
            }
            best = best.min(t.elapsed().as_secs_f64() * 1.0e6 / 20.0);
        }
        // The world first, then the clock: a search that answered nothing would
        // be the fastest of all.
        assert_eq!(
            nodes as u64,
            2 * SIDE - 1,
            "the route is not a monotone staircase"
        );
        assert!(
            (cost - 2.0 * (SIDE - 1) as f64 * 20.0).abs() < 1.0e-9,
            "the route costs {cost} m and the grid's Manhattan distance is {} m",
            2.0 * (SIDE - 1) as f64 * 20.0
        );
        println!(
            "NPC1c audit / inf-nav: {} nodes / {} directed edges; corner to corner in {best:.1} us ({nodes} nodes, {cost:.0} m)",
            g.len(),
            g.edge_count()
        );
        assert!(
            best > 0.0 && best < 1000.0,
            "a corner-to-corner search over {} nodes took {best:.1} us; the module's own claim is microseconds, and a millisecond means the frontier or the adjacency changed shape",
            g.len()
        );
    }
}
