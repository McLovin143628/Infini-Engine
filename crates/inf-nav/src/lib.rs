//! **inf-nav** (Ring 0) — the route substrate: a deterministic weighted graph,
//! a shortest path that is a **value**, and an arc-length polyline anything can
//! walk.
//!
//! # Why this crate holds no graph of its own
//!
//! This engine already had three navigable structures and no navigation. The
//! island's `RoadGraph` is a real adjacency graph — `BTreeMap`s, start/end
//! nodes, lattice-derived ids, `length_m` costs — built at bake time from the
//! committed road layer. Every settlement is an orthogonal street grid, planned
//! centreline by centreline and then *consumed* to place blocks. Every grammar
//! building knows its own door edges, its stair core and which rooms are
//! reachable from outside. What none of them had was a **search**, and what a
//! search needs is one shape to run on.
//!
//! So this crate contributes the shape and the search, and nothing else: it owns
//! no world, no entity, no asset, no file format and no notion of a settlement,
//! a road or a room. The three producers each expose their own structure *as* a
//! [`NavGraph`] — `inf_gis::RoadGraph::nav_graph`,
//! `inf_editor_core::settlement::Settlement::street_graph` and
//! `inf_pcg::BuildingPlan::interior_nav` — which keeps the knowledge of what a
//! street *is* in the crate that planned it, and keeps this one testable against
//! a five-node square.
//!
//! # A route is a value, and so is a refusal
//!
//! [`route()`] answers a [`NavVerdict`], never a `Result` and never a panic. "No
//! path" is a legitimate, common, *gameplay-visible* outcome — a town cut off by
//! a landslide, a door that is not there, an agent standing off the graph — and
//! the P21 ruling is explicit that a gameplay refusal must be a value: an
//! erroring node takes its whole handler down with it. The verdict names which
//! of the four things went wrong and carries the endpoints, so a caller can log
//! it, draw it or fall back on it without re-deriving anything.
//!
//! # Determinism, three ways
//!
//! * **`BTreeMap` order.** Nodes and adjacency lists are ordered by id, so a
//!   walk is a function of the graph's contents and not of a hash seed. This is
//!   the same rule `RoadGraph` states at length in its own module docs, applied
//!   one level up.
//! * **Ties break on the node id.** The frontier is ordered on
//!   `(cost, node id)`, so two nodes at the same cost are always relaxed in the
//!   same order. Without it, a symmetric graph — which a street *grid* is,
//!   exactly — hands back a different one of two equally short routes depending
//!   on heap internals.
//! * **Portable math.** Costs are built out of `+ - * / sqrt` alone, which IEEE
//!   754 specifies exactly, and a heading here is a *normalize* rather than an
//!   angle — there is no trigonometry in this crate at all, which is why it
//!   names no math crate. A route reaches an agent's `Transform` and therefore
//!   `state_bytes`, which two hosts and two machines compare, so the P14 law
//!   binds every line here; `crates/inf-anim/tests/portable_pose.rs` has all
//!   three files on the libm ban list from this crate's first commit, and that
//!   gate — not an import — is what enforces the claim.
//!
//!   *(The NPC1c audit's correction: this paragraph read "headings go through
//!   `inf_math::portable`" and `inf-nav` depended on `inf-math` for it. No line
//!   in the crate ever named that module. The dependency is gone and the
//!   sentence says what the code does.)*
//!
//! # Units
//!
//! SI, per the units doctrine: positions are world metres (`DVec3`, Y up), costs
//! are metres, arc length is metres. There is no unit-scale factor anywhere.

pub mod graph;
pub mod path;
pub mod route;

pub use graph::{NavEdge, NavGraph, NavKind, NavNode, NavNodeId};
pub use path::{NavPath, PathProjection};
pub use route::{route, route_counted, NavRoute, NavVerdict, SearchStats};

/// **The id namespace**, because [`NavGraph::absorb`] joins on id equality.
///
/// Four producers mint node ids independently and their graphs are folded into
/// one before an agent walks it, so two of them handing out the same number
/// would silently weld a road junction to a bedroom. The top four bits name who
/// minted an id; the low sixty are the producer's own. *(Three until NPC1d
/// added [`domain::PAVEMENT`].)*
///
/// This lives here rather than three times over for the reason this tree keeps
/// re-learning: a namespace agreed in three files is a namespace that drifts.
/// It is not a claim that `inf-nav` knows what a road is — it does not, and
/// nothing in this crate reads these constants. It knows that ids collide.
pub mod domain {
    /// `inf_gis::RoadGraph::nav_graph` — island roads.
    pub const ROAD: u64 = 1 << 60;
    /// `inf_editor_core::settlement::Settlement::street_graph` — town streets.
    pub const STREET: u64 = 2 << 60;
    /// `inf_pcg::BuildingPlan::interior_nav` — rooms, doorways and stairs.
    pub const BUILDING: u64 = 3 << 60;
    /// **A block's own pavement** (NPC1d) — the ring of nodes
    /// `inf_ecs::society` lays round a `PcgVolume`, and the crossings between
    /// two of them.
    ///
    /// A separate tag from [`STREET`] although it is a street, because the two
    /// mint their low sixty bits differently: a settlement's grid uses a
    /// structured (site, column, row) layout it can name a node by, and a
    /// pavement ring uses a hash of its own quantized position, which is the
    /// only key a level with no settlement plan in it has. Two id LAYOUTS in one
    /// tag is exactly the collision this module's domains exist to prevent, so
    /// they get a tag each.
    pub const PAVEMENT: u64 = 4 << 60;

    /// Reserved for a caller composing nodes of its own — a gate placing a
    /// destination, a level naming a spawn point.
    pub const CALLER: u64 = 15 << 60;
    /// The mask that recovers a producer's own id from a tagged one.
    pub const LOCAL_MASK: u64 = (1 << 60) - 1;

    /// Which domain minted `id`, as the tag alone.
    pub fn of(id: u64) -> u64 {
        id & !LOCAL_MASK
    }
}

/// An `f64` ordered by [`f64::total_cmp`], so a cost can key a `BinaryHeap`.
///
/// `total_cmp` rather than a partial compare with an `unwrap`: a NaN cost is a
/// bug in a caller's graph and must sort somewhere rather than abort the search
/// halfway through, and `total_cmp` is a total order over every bit pattern.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ordered(pub f64);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
