//! The graph: nodes with world positions, edges with a cost and a spine.
//!
//! # An edge carries its own geometry
//!
//! A road segment between two junctions is a *polyline* — a surveyed centreline
//! that switchbacks up a face — and a route that kept only the two junctions
//! would cut the corner and walk an agent off a cliff. So [`NavEdge`] carries
//! the points **between** its two nodes ([`NavEdge::via`]), and the cost is the
//! length of the whole chain rather than the straight line between the ends.
//! A street grid's edges are straight and carry an empty `via`, which costs a
//! `Vec` header and no allocation.

use std::collections::BTreeMap;

use glam::{DVec2, DVec3};

/// A node's identity. Assigned by whoever builds the graph, and required to be a
/// function of the *source data* rather than of an insertion order — the rule
/// `RoadGraph` already follows when it numbers junctions in lattice order.
pub type NavNodeId = u64;

/// What a node or an edge **is**, so a caller can price a route by what it walks
/// on and a gate can say which half of a mixed graph it used.
///
/// This is a small closed enum on purpose. It is not persisted anywhere — a
/// graph is derived, never stored, exactly as `RoadGraph` is — so it costs no
/// schema ladder and it is not subject to the freeze-pinned wire-enum rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NavKind {
    /// An island road: a highway, an arterial or a lane between settlements.
    #[default]
    Road,
    /// A settlement's own street centreline.
    Street,
    /// The threshold of a doorway — the one point a wall lets a body through.
    Doorway,
    /// Standing room: a room's centre, a landing, a lot's frontage.
    Room,
    /// A flight between two storeys.
    Stair,
}

/// One node.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavNode {
    pub id: NavNodeId,
    /// World metres, Y up. For ground nodes this is the walking surface.
    pub position: DVec3,
    pub kind: NavKind,
}

/// One directed half of an edge. [`NavGraph::link`] adds both halves.
#[derive(Clone, Debug, PartialEq)]
pub struct NavEdge {
    /// The node this half leads to.
    pub to: NavNodeId,
    /// Traversal cost in metres — the chain length, times whatever multiplier
    /// the builder chose. Never negative and never NaN (see
    /// [`NavGraph::link_with_cost`]).
    pub cost_m: f64,
    pub kind: NavKind,
    /// The points strictly *between* the two nodes, in travel order. Empty for a
    /// straight edge.
    pub via: Vec<DVec3>,
}

/// A navigable graph.
///
/// `BTreeMap` for the nodes and for the adjacency, and each adjacency list is
/// kept sorted by `(to, cost)` — so every walk over this structure is a function
/// of what is in it. See the crate docs for why that is load-bearing rather than
/// a preference.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NavGraph {
    nodes: BTreeMap<NavNodeId, NavNode>,
    edges: BTreeMap<NavNodeId, Vec<NavEdge>>,
}

/// The length of a polyline through `points`, in metres.
///
/// Spelled out rather than `DVec3::length`-summed so the arithmetic a route's
/// cost is built from is visible in one place: three multiplies, two adds and a
/// `sqrt`, all IEEE-exact.
pub fn polyline_length(points: &[DVec3]) -> f64 {
    let mut total = 0.0;
    for w in points.windows(2) {
        let d = w[1] - w[0];
        total += (d.x * d.x + d.y * d.y + d.z * d.z).sqrt();
    }
    total
}

impl NavGraph {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// How many nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the graph has no nodes at all.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// How many directed edge halves. An undirected link counts twice.
    pub fn edge_count(&self) -> usize {
        self.edges.values().map(|v| v.len()).sum()
    }

    /// The nodes, in id order.
    pub fn nodes(&self) -> impl Iterator<Item = &NavNode> {
        self.nodes.values()
    }

    /// One node.
    pub fn node(&self, id: NavNodeId) -> Option<&NavNode> {
        self.nodes.get(&id)
    }

    /// Whether `id` is a node of this graph.
    pub fn contains(&self, id: NavNodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    /// The edges leaving `id`, in `(to, cost)` order. Empty for an unknown node.
    pub fn edges_from(&self, id: NavNodeId) -> &[NavEdge] {
        self.edges.get(&id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Add or replace a node.
    ///
    /// Replacing is legal and is what a builder joining two graphs does when the
    /// same junction is reached twice; the edges already leaving the node are
    /// untouched.
    pub fn add_node(&mut self, id: NavNodeId, position: DVec3, kind: NavKind) {
        self.nodes.insert(id, NavNode { id, position, kind });
    }

    /// Link `a` and `b` **both ways** along `via`, costed at the chain's length.
    ///
    /// `via` is given in `a → b` order; the `b → a` half reverses it, so an
    /// agent walking a switchback backwards walks the same switchback.
    ///
    /// A link naming a node this graph does not hold is **ignored**, silently
    /// and by design: a builder that filters its features (a road layer skips a
    /// degenerate segment, a settlement refuses a block off the pad) would
    /// otherwise have to filter its links a second time, and the two filters are
    /// exactly the pair that drifts apart.
    pub fn link(&mut self, a: NavNodeId, b: NavNodeId, kind: NavKind, via: Vec<DVec3>) {
        let (Some(pa), Some(pb)) = (
            self.node(a).map(|n| n.position),
            self.node(b).map(|n| n.position),
        ) else {
            return;
        };
        let mut chain = Vec::with_capacity(via.len() + 2);
        chain.push(pa);
        chain.extend_from_slice(&via);
        chain.push(pb);
        let cost = polyline_length(&chain);
        self.link_with_cost(a, b, kind, via, cost);
    }

    /// [`link`](Self::link) with the cost stated rather than measured — the door
    /// a builder uses when a metre of stair is not a metre of street.
    ///
    /// A non-finite or negative cost is clamped to zero rather than refused: a
    /// negative edge would make Dijkstra's "first pop is final" invariant false
    /// and turn a wrong answer into a *plausible* wrong answer, which is worse
    /// than a free edge.
    pub fn link_with_cost(
        &mut self,
        a: NavNodeId,
        b: NavNodeId,
        kind: NavKind,
        via: Vec<DVec3>,
        cost_m: f64,
    ) {
        if !self.contains(a) || !self.contains(b) || a == b {
            return;
        }
        let cost = if cost_m.is_finite() && cost_m > 0.0 {
            cost_m
        } else {
            0.0
        };
        let mut back: Vec<DVec3> = via.clone();
        back.reverse();
        self.push_edge(
            a,
            NavEdge {
                to: b,
                cost_m: cost,
                kind,
                via,
            },
        );
        self.push_edge(
            b,
            NavEdge {
                to: a,
                cost_m: cost,
                kind,
                via: back,
            },
        );
    }

    fn push_edge(&mut self, from: NavNodeId, edge: NavEdge) {
        let list = self.edges.entry(from).or_default();
        // Sorted by `(to, cost)` and deduplicated on the pair: two builders that
        // link the same junction twice — a settlement grid meeting the road that
        // arrives on it — must not double the frontier.
        match list.binary_search_by(|e| {
            e.to.cmp(&edge.to)
                .then(crate::Ordered(e.cost_m).cmp(&crate::Ordered(edge.cost_m)))
        }) {
            Ok(_) => {}
            Err(at) => list.insert(at, edge),
        }
    }

    /// The node nearest `p` in three dimensions, ties broken by id.
    ///
    /// Squared distance, so this costs no `sqrt` per node and answers the same
    /// node the un-squared comparison would.
    pub fn nearest(&self, p: DVec3) -> Option<NavNodeId> {
        self.nearest_where(p, |_| true)
    }

    /// [`nearest`](Self::nearest) over the nodes a predicate admits — how a
    /// caller asks for "the nearest *doorway*" rather than the nearest anything.
    pub fn nearest_where(
        &self,
        p: DVec3,
        mut keep: impl FnMut(&NavNode) -> bool,
    ) -> Option<NavNodeId> {
        let mut best: Option<(f64, NavNodeId)> = None;
        for n in self.nodes.values() {
            if !keep(n) {
                continue;
            }
            let d = n.position - p;
            let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
            let better = match best {
                None => true,
                // Ties break on the id, which the `BTreeMap` walk makes the
                // *first* one seen — so the `<` is enough and the comparison
                // never reads an id.
                Some((bd, _)) => d2 < bd,
            };
            if better {
                best = Some((d2, n.id));
            }
        }
        best.map(|(_, id)| id)
    }

    /// The node nearest `p` **on the ground plane** — the query an agent
    /// standing somewhere makes, where a two-storey building's upstairs node is
    /// not a candidate for a body in the street.
    pub fn nearest_planar(&self, p: DVec3, max_dy_m: f64) -> Option<NavNodeId> {
        let mut best: Option<(f64, NavNodeId)> = None;
        for n in self.nodes.values() {
            // A NaN height is not "within the bound", so the comparison is
            // written positively and the `continue` takes the else — spelling
            // it as a negation reads the NaN the other way AND trips
            // `neg_cmp_op_on_partial_ord`.
            let dy = (n.position.y - p.y).abs();
            if dy > max_dy_m || dy.is_nan() {
                continue;
            }
            let d = DVec2::new(n.position.x - p.x, n.position.z - p.z);
            let d2 = d.x * d.x + d.y * d.y;
            let better = match best {
                None => true,
                Some((bd, _)) => d2 < bd,
            };
            if better {
                best = Some((d2, n.id));
            }
        }
        best.map(|(_, id)| id)
    }

    /// **Join every pair of nodes closer than `tolerance_m` in plan**, with a
    /// zero-cost edge — how three producers' graphs become one network.
    ///
    /// The island's road planner routes **centre to centre**, so every highway
    /// that reaches a settlement terminates on the site's own `(x, z)`; the
    /// settlement's grid puts a street line through that same point. Two
    /// producers, two id domains, one place. The same is true of a lot's
    /// frontage and the street outside it, and of a building's entrance and its
    /// lot. Rather than teach any of the three about the other two, they are
    /// welded here, on the geometry they already agree about.
    ///
    /// **Plan distance, and a separate Y bound**, for `SNAP_TOLERANCE_M`'s own
    /// reason one level down: two points at the same plan position and different
    /// heights are an overpass, not a junction.
    ///
    /// `O(n²)` over the nodes and it says so: an island's whole network is a few
    /// hundred of them. It is deterministic because the walk is `BTreeMap` order
    /// twice over, and it never welds two nodes of the same domain — a producer
    /// that wanted them joined would have linked them.
    pub fn weld(&mut self, tolerance_m: f64, max_dy_m: f64) -> usize {
        let ids: Vec<(NavNodeId, DVec3)> =
            self.nodes.values().map(|n| (n.id, n.position)).collect();
        let t2 = tolerance_m * tolerance_m;
        let mut welded = 0;
        for (i, (a, pa)) in ids.iter().enumerate() {
            for (b, pb) in ids.iter().skip(i + 1) {
                if crate::domain::of(*a) == crate::domain::of(*b) {
                    continue;
                }
                if (pa.y - pb.y).abs() > max_dy_m {
                    continue;
                }
                let d = DVec2::new(pa.x - pb.x, pa.z - pb.z);
                if d.x * d.x + d.y * d.y <= t2 {
                    let before = self.edges_from(*a).len();
                    self.link_with_cost(*a, *b, NavKind::Street, vec![], 0.0);
                    if self.edges_from(*a).len() != before {
                        welded += 1;
                    }
                }
            }
        }
        welded
    }

    /// **Fold `other` into this graph**, node ids and all.
    ///
    /// The caller owns the id namespaces and this does not check them: two
    /// producers that hand out the same id mean one node, which is exactly how a
    /// settlement's central crossroads and the highway that terminates on it
    /// become one place. Every builder in this tree derives its ids from a
    /// namespace of its own for that reason — see [`crate::domain`].
    pub fn absorb(&mut self, other: &NavGraph) {
        for n in other.nodes.values() {
            self.nodes.entry(n.id).or_insert(*n);
        }
        for (from, list) in &other.edges {
            for e in list {
                if self.contains(*from) && self.contains(e.to) {
                    self.push_edge(*from, e.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, z: f64) -> DVec3 {
        DVec3::new(x, 0.0, z)
    }

    fn square() -> NavGraph {
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Street);
        g.add_node(1, p(10.0, 0.0), NavKind::Street);
        g.add_node(2, p(10.0, 10.0), NavKind::Street);
        g.add_node(3, p(0.0, 10.0), NavKind::Street);
        g.link(0, 1, NavKind::Street, vec![]);
        g.link(1, 2, NavKind::Street, vec![]);
        g.link(2, 3, NavKind::Street, vec![]);
        g.link(3, 0, NavKind::Street, vec![]);
        g
    }

    #[test]
    fn a_link_is_both_ways_and_costs_its_chain() {
        let g = square();
        assert_eq!(g.len(), 4);
        assert_eq!(g.edge_count(), 8);
        assert_eq!(g.edges_from(0).len(), 2);
        for e in g.edges_from(0) {
            assert_eq!(e.cost_m, 10.0);
        }
    }

    /// The whole reason an edge carries a spine: a switchback costs what it
    /// walks, not what it spans.
    #[test]
    fn an_edge_costs_the_chain_and_not_the_chord() {
        let mut g = NavGraph::new();
        g.add_node(0, p(0.0, 0.0), NavKind::Road);
        g.add_node(1, p(10.0, 0.0), NavKind::Road);
        g.link(0, 1, NavKind::Road, vec![p(5.0, 10.0)]);
        let chord = 10.0;
        let e = &g.edges_from(0)[0];
        assert!(e.cost_m > chord * 2.0, "{} is not a switchback", e.cost_m);
        // …and the way back walks the same bend, in reverse.
        assert_eq!(g.edges_from(1)[0].via, vec![p(5.0, 10.0)]);
    }

    /// A link naming a node that was filtered out is a no-op, not a panic and
    /// not a half-edge.
    #[test]
    fn a_link_to_a_missing_node_is_ignored() {
        let mut g = square();
        g.link(0, 99, NavKind::Street, vec![]);
        assert_eq!(g.edge_count(), 8);
        assert!(!g.contains(99));
    }

    /// Two builders linking the same pair — a settlement crossroads and the
    /// highway that lands on it — make one edge, not two.
    #[test]
    fn linking_the_same_pair_twice_makes_one_edge() {
        let mut g = square();
        g.link(0, 1, NavKind::Street, vec![]);
        assert_eq!(g.edge_count(), 8);
    }

    #[test]
    fn nearest_breaks_ties_on_the_lowest_id() {
        let mut g = NavGraph::new();
        g.add_node(7, p(1.0, 0.0), NavKind::Room);
        g.add_node(3, p(-1.0, 0.0), NavKind::Room);
        assert_eq!(g.nearest(p(0.0, 0.0)), Some(3));
    }

    /// A body in the street must not be handed the node above its head.
    #[test]
    fn nearest_planar_refuses_the_storey_above() {
        let mut g = NavGraph::new();
        g.add_node(0, DVec3::new(0.0, 3.2, 0.0), NavKind::Room);
        g.add_node(1, DVec3::new(6.0, 0.0, 0.0), NavKind::Room);
        assert_eq!(g.nearest(DVec3::ZERO), Some(0));
        assert_eq!(g.nearest_planar(DVec3::ZERO, 1.5), Some(1));
    }

    #[test]
    fn absorb_joins_two_graphs_on_a_shared_id() {
        let mut a = NavGraph::new();
        a.add_node(0, p(0.0, 0.0), NavKind::Road);
        a.add_node(1, p(100.0, 0.0), NavKind::Road);
        a.link(0, 1, NavKind::Road, vec![]);
        let mut b = NavGraph::new();
        b.add_node(1, p(100.0, 0.0), NavKind::Street);
        b.add_node(2, p(100.0, 50.0), NavKind::Street);
        b.link(1, 2, NavKind::Street, vec![]);
        a.absorb(&b);
        assert_eq!(a.len(), 3);
        assert_eq!(a.edges_from(1).len(), 2);
        // The shared node keeps the FIRST graph's own record — the road's — so
        // absorbing is not a way to overwrite a node behind a builder's back.
        assert_eq!(a.node(1).unwrap().kind, NavKind::Road);
    }

    /// Two producers' graphs meet where their geometry already agrees, and an
    /// overpass is not a junction.
    #[test]
    fn weld_joins_two_domains_where_they_touch_and_never_a_flyover() {
        let mut g = NavGraph::new();
        g.add_node(crate::domain::ROAD, p(0.0, 0.0), NavKind::Road);
        g.add_node(crate::domain::ROAD | 1, p(500.0, 0.0), NavKind::Road);
        g.link(
            crate::domain::ROAD,
            crate::domain::ROAD | 1,
            NavKind::Road,
            vec![],
        );
        // The settlement's crossroads, one metre off the road's terminus…
        g.add_node(crate::domain::STREET, p(500.5, 0.0), NavKind::Street);
        // …and a footbridge over it, at the same plan position, six metres up.
        g.add_node(
            crate::domain::STREET | 1,
            DVec3::new(500.5, 6.0, 0.0),
            NavKind::Street,
        );
        assert_eq!(g.weld(2.0, 1.0), 1);
        assert!(crate::route(&g, crate::domain::ROAD, crate::domain::STREET).is_found());
        assert!(!crate::route(&g, crate::domain::ROAD, crate::domain::STREET | 1).is_found());
        // Idempotent: welding twice adds no second edge.
        assert_eq!(g.weld(2.0, 1.0), 0);
    }

    /// Two nodes of ONE producer are never welded — a producer that wanted them
    /// joined would have linked them, and a street grid's parallel lines are
    /// metres apart on purpose.
    #[test]
    fn weld_never_joins_a_domain_to_itself() {
        let mut g = NavGraph::new();
        g.add_node(crate::domain::STREET, p(0.0, 0.0), NavKind::Street);
        g.add_node(crate::domain::STREET | 1, p(0.5, 0.0), NavKind::Street);
        assert_eq!(g.weld(2.0, 1.0), 0);
        assert_eq!(g.edge_count(), 0);
    }

    /// A negative cost would break Dijkstra's "first pop is final" invariant and
    /// hand back a wrong route that looks right.
    #[test]
    fn a_negative_or_nan_cost_is_clamped_to_zero() {
        let mut g = square();
        g.link_with_cost(0, 2, NavKind::Street, vec![], -5.0);
        let e = g
            .edges_from(0)
            .iter()
            .find(|e| e.to == 2)
            .expect("the diagonal");
        assert_eq!(e.cost_m, 0.0);
        let mut h = square();
        h.link_with_cost(1, 3, NavKind::Street, vec![], f64::NAN);
        assert_eq!(
            h.edges_from(1).iter().find(|e| e.to == 3).unwrap().cost_m,
            0.0
        );
    }
}
