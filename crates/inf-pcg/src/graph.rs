//! The PCG editor node kit + `lower_graph` (P10.5b).
//!
//! The `.inf_pcg` editor authors a pure-data node DAG over the shared
//! [`inf_graph`] substrate — the same substrate the Blueprint and Material
//! editors use — with a PCG-specific [`pcg_registry`] and no exec pins. A single
//! `output.pcg` SINK collects one scatter chain; [`lower_graph`] walks the graph
//! backward from that sink into the stable, serialization-locked
//! [`PcgDocument`] runtime model (the way the material
//! emitter walks to WGSL and the blueprint lowerer targets `BlueprintFn`). The
//! runtime evaluates the lowered document, so **editor preview == runtime**.
//!
//! ## Wires
//!
//! Three `Named` port types carry the dataflow (the substrate has no domain
//! variants of its own):
//!
//! * [`DENSITY_KEY`] — a `[0,1]` density field, produced by sources
//!   (`const.density`, `noise.fbm`), terrain filters (`filter.slope`,
//!   `filter.altitude`), **terrain-layer masks** (`mask.*`, P19.3), and
//!   combinators (`combine.*`).
//! * [`SCATTER_KEY`] — a scatter pass. `scatter.scatter` consumes a density;
//!   `scatter.merge` joins two passes into one.
//! * [`LAYER_KEY`] — a named, toggleable group of passes (P19.3).
//!   `layer.layer` wraps a scatter chain; `layer.merge` joins two layers.
//! * [`SPAN_KEY`] / [`RULES_KEY`] — the P19.4 grammar's two feeds:
//!   `grammar.spline` / `grammar.footprint` produce a SPAN, `grammar.rules`
//!   produces a RULES, and `grammar.expand` consumes one of each.
//!
//! All five are `PortType::Named`, and P19.4 deliberately added **no new
//! `PortType` variant**. The substrate is domain-free on purpose — `Named` is
//! the mechanism it offers for exactly this — and a real variant would have to
//! be threaded through the blueprint, material and state-machine editors'
//! type tables, colour maps and TS mirrors to buy a grammar wire nothing else
//! can see.
//!
//! ## Where a grammar joins the population (P19.4)
//!
//! `grammar.expand` **outputs a SCATTER**, so it plugs into the `scatter.merge`
//! and `layer.layer` chains that already exist: no third merge node, no third
//! sink input, and a grammar inherits its layer's name and `enabled` flag for
//! free. What it lowers to is *not* a [`PcgRule`] — a grammar is a different
//! generator — so it goes into [`LoweredPcg::grammars`] beside the document
//! rather than into it. See [`crate::grammar`] for why the serialized
//! `PcgDocument` is deliberately left alone.
//!
//! ## Lowering semantics (P19.3: layers × rules)
//!
//! `lower_graph` walks backward from the single `output.pcg` sink and produces
//! the **full** [`PcgDocument`] the runtime model has always supported:
//!
//! ```text
//!   output.pcg .layers ◀── layer.merge ◀─┬─ layer.layer ◀── scatter.merge ◀─┬─ scatter.scatter
//!              .scatter ◀── (a bare scatter chain: one implicit layer)      └─ scatter.scatter
//! ```
//!
//! * A `scatter.merge` tree flattens **in order** (`a` before `b`, depth-first)
//!   into a `Vec<PcgRule>` — so a left-leaning chain reads top-to-bottom, and
//!   evaluation order is exactly what the canvas shows.
//! * A `layer.merge` tree flattens the same way into `Vec<PcgLayer>`; each
//!   `layer.layer` contributes its `name`/`enabled` params and its own flattened
//!   rule list.
//! * The sink's original `scatter` input is **kept**, and a graph that uses it
//!   lowers to exactly one layer named `layer` — so every `.inf_pcg` authored
//!   before P19.3 lowers byte-identically. Connecting both inputs is a
//!   node-anchored warning and `layers` wins.
//!
//! ### Why merge nodes rather than a variadic sink or a `rule` node
//!
//! The substrate enforces **one link per input pin** (`GraphEdit`'s door), so
//! several scatter chains cannot simply meet at one port. The two alternatives
//! were a variadic/indexed sink (`layer0…layerN` — an arbitrary cap, and a shape
//! no other domain in the engine uses) and an explicit `rule` node that would
//! duplicate what `scatter.scatter` already is. A **binary combinator that
//! associates left** is instead the convention this registry already has, twice
//! over: `combine.multiply` / `combine.max` / `combine.min` join two densities
//! exactly this way, and the lowerer already flattens *those* recursively. So
//! `scatter.merge` and `layer.merge` are the same idea one wire type up — no new
//! concept, no cap, and the sink keeps its single-pin shape.
//!
//! ## `mask.image` and the [`MaskSource`] seam
//!
//! `mask.image` names a texture asset by GUID, and the runtime
//! [`SamplerDef::Mask`] carries **pixels**, not a
//! reference — a lowered document is self-contained by design. The lowerer
//! therefore resolves the GUID through a [`MaskSource`] the caller supplies:
//! [`lower_graph_with`] takes one, and [`lower_graph`] passes [`NoMasks`], under
//! which an image mask lowers to an empty (`0 × 0`) mask that scores `0`
//! everywhere, with a node-anchored warning. Failing *closed* is the point: a
//! mask nobody could load must not silently become "place everywhere".

use uuid::Uuid;

use inf_graph::{
    Graph, NodeDef, NodeId, NodeRegistry, ParamDef, ParamValue, PortDef, PortType, SINK,
};

use crate::building::{ArchetypeId, BuildingPass};
use crate::grammar::{
    FootprintMode, Grammar, GrammarPass, Ground, RowAxis, SpanSource, DEFAULT_SPLINE_SAMPLES,
};
use crate::noise::ValueNoise;
use crate::rules::{PcgDocument, PcgKind, PcgLayer, PcgRule, SamplerDef};
use crate::scatter::{RotationMode, ScatterParams};

/// A `[0,1]` density-field wire.
pub const DENSITY_KEY: &str = "density";
/// A scatter-pass wire.
pub const SCATTER_KEY: &str = "scatter";
/// A layer wire (a named, toggleable group of scatter passes) — P19.3.
pub const LAYER_KEY: &str = "layer";
/// A 1-D grammar domain (spline arc length, footprint perimeter or rows) — P19.4.
pub const SPAN_KEY: &str = "span";
/// A parsed grammar (rule table + module palette) — P19.4.
pub const RULES_KEY: &str = "rules";
/// A building archetype (palette + plan parameters) — P19.5.
pub const BUILDING_KEY: &str = "building";

/// A density-field port.
fn density(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(DENSITY_KEY.into()))
}

/// A scatter-pass port.
fn scatter(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(SCATTER_KEY.into()))
}

/// A layer port.
fn layer(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(LAYER_KEY.into()))
}

/// A grammar-span port.
fn span(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(SPAN_KEY.into()))
}

/// A grammar rule-table port.
fn rules(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(RULES_KEY.into()))
}

/// A building-archetype port.
fn building(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(BUILDING_KEY.into()))
}

/// Resolves a texture asset GUID into a **grayscale bitmap** for the `mask.image`
/// node: `(width, height, row-major bytes)`, one byte per texel mapping
/// `0..=255` → `0.0..=1.0`.
///
/// The lowerer holds no asset database — it is a pure function of a graph — so
/// the pixels come in through this seam. The editor backs it with the live
/// content root; anything without one uses [`NoMasks`].
pub trait MaskSource {
    /// The grayscale bitmap for `texture`, or `None` when it cannot be resolved.
    fn mask(&self, texture: Uuid) -> Option<(u32, u32, Vec<u8>)>;
}

/// The empty mask source: nothing resolves.
///
/// An unresolved image mask lowers to a `0 × 0` mask, which
/// [`MaskImage`](crate::sampler::MaskImage) scores `0` everywhere — the mask
/// fails **closed**, so a graph whose texture is missing places nothing rather
/// than everything.
pub struct NoMasks;

impl MaskSource for NoMasks {
    fn mask(&self, _texture: Uuid) -> Option<(u32, u32, Vec<u8>)> {
        None
    }
}

impl<T: MaskSource + ?Sized> MaskSource for &T {
    fn mask(&self, texture: Uuid) -> Option<(u32, u32, Vec<u8>)> {
        (**self).mask(texture)
    }
}

/// The complete PCG node palette: density sources, terrain filters, terrain-layer
/// masks (P19.3), combinators, the scatter pass + its merge, the grammar kit
/// (P19.4), the building kit (P19.5), the layer wrapper + its merge, and one
/// output sink.
///
/// Registration order **is** palette order (the frontend groups by first
/// appearance), so the grammar sits beside scatter — its sibling generator —
/// rather than at the end.
pub fn pcg_registry() -> NodeRegistry {
    let mut reg = NodeRegistry::new();
    reg.register_all(source_nodes());
    reg.register_all(filter_nodes());
    reg.register_all(mask_nodes());
    reg.register_all(combine_nodes());
    reg.register(scatter_node());
    reg.register(scatter_merge_node());
    reg.register_all(grammar_nodes());
    reg.register_all(building_nodes());
    reg.register_all(layer_nodes());
    reg.register(output_node());
    reg
}

fn source_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("const.density", "Constant", "sources")
            .described("A uniform density everywhere (clamped to [0,1])")
            .with_outputs(vec![density("out")])
            .with_params(vec![ParamDef::number("value", 0.5).range(0.0, 1.0)]),
        NodeDef::new("noise.fbm", "fBm Noise", "sources")
            .described("Fractal value-noise density")
            .with_outputs(vec![density("out")])
            .with_params(vec![
                ParamDef::int("seed", 0),
                ParamDef::number("frequency", 0.02).range(0.0001, 4.0),
                ParamDef::int("octaves", 4).range(1.0, 12.0),
                ParamDef::number("lacunarity", 2.0).range(1.0, 4.0),
                ParamDef::number("gain", 0.5).range(0.0, 1.0),
            ]),
    ]
}

fn filter_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("filter.slope", "Slope Filter", "filters")
            .described("Accept a feathered terrain-slope band (degrees)")
            .with_outputs(vec![density("out")])
            .with_params(vec![
                ParamDef::number("min_deg", 0.0).range(0.0, 90.0),
                ParamDef::number("max_deg", 30.0).range(0.0, 90.0),
                ParamDef::number("feather_deg", 5.0).range(0.0, 45.0),
            ]),
        NodeDef::new("filter.altitude", "Altitude Filter", "filters")
            .described("Accept a feathered terrain-height band (world units)")
            .with_outputs(vec![density("out")])
            .with_params(vec![
                ParamDef::number("min", 0.0),
                ParamDef::number("max", 100.0),
                ParamDef::number("feather", 5.0).range(0.0, 100.0),
            ]),
    ]
}

/// The P19.3 **terrain-layer masks**: the erosion data maps (P19.1) and the
/// painted biome ids (P19.2), read as densities.
///
/// The data-map nodes each carry their own `min`/`max` **because the stored data
/// is raw** — flow in m³, deposition and wear in metres, all monotone
/// accumulators that a second bake adds to. Normalization is a view the reader
/// chooses (the P19.1 doctrine), so the window is a node param, not a terrain
/// property, and two masks over one terrain may legitimately differ.
fn mask_nodes() -> Vec<NodeDef> {
    let data_map = |id: &str, display: &str, unit: &str, what: &str, max: f64| {
        NodeDef::new(id, display, "masks")
            .described(format!(
                "{what} — normalized over [min, max] (raw values are in {unit})"
            ))
            .with_outputs(vec![density("out")])
            .with_params(vec![
                ParamDef::number("min", 0.0).described(format!("raw {unit} mapping to density 0")),
                ParamDef::number("max", max).described(format!("raw {unit} mapping to density 1")),
            ])
    };
    vec![
        NodeDef::new("mask.image", "Image Mask", "masks")
            .described("A grayscale texture stretched over a world rectangle")
            .with_outputs(vec![density("out")])
            .with_params(vec![
                ParamDef::text("texture", "").described("Texture asset GUID (blank → no mask)"),
                ParamDef::number("min_x", 0.0).described("World rect min X (metres)"),
                ParamDef::number("min_z", 0.0).described("World rect min Z (metres)"),
                ParamDef::number("max_x", 1000.0).described("World rect max X (metres)"),
                ParamDef::number("max_z", 1000.0).described("World rect max Z (metres)"),
            ]),
        data_map(
            "mask.flow",
            "Flow Mask",
            "m^3",
            "Where water ran: the P19.1 flow-accumulation map",
            1000.0,
        ),
        data_map(
            "mask.deposition",
            "Deposition Mask",
            "m",
            "Where sediment settled: the P19.1 deposition map",
            1.0,
        ),
        data_map(
            "mask.wear",
            "Wear Mask",
            "m",
            "Where material was stripped: the P19.1 wear map",
            1.0,
        ),
        NodeDef::new("mask.biome", "Biome Mask", "masks")
            .described("Where a painted biome id owns the terrain, feathered at its border")
            .with_outputs(vec![density("out")])
            .with_params(vec![
                ParamDef::int("id", 1)
                    .range(0.0, 255.0)
                    .described("Biome id (0 is the reserved 'unassigned')"),
                ParamDef::number("feather", 0.0)
                    .range(0.0, 256.0)
                    .described("Border blend width in metres (0 = hard edge)"),
            ]),
    ]
}

fn combine_nodes() -> Vec<NodeDef> {
    let binary = |id: &str, display: &str, desc: &str| {
        NodeDef::new(id, display, "combine")
            .described(desc)
            .with_inputs(vec![density("a"), density("b")])
            .with_outputs(vec![density("out")])
    };
    vec![
        binary("combine.multiply", "Multiply", "Product (intersection) a·b"),
        binary("combine.max", "Max", "Maximum (union) max(a,b)"),
        binary("combine.min", "Min", "Minimum min(a,b)"),
        NodeDef::new("combine.invert", "Invert", "combine")
            .described("1 − a")
            .with_inputs(vec![density("in")])
            .with_outputs(vec![density("out")]),
    ]
}

/// The scatter pass: consumes a density, emits a scatter into the sink.
fn scatter_node() -> NodeDef {
    NodeDef::new("scatter.scatter", "Scatter", "scatter")
        .described("Deterministically place instances by density over the terrain")
        .with_inputs(vec![density("density")])
        .with_outputs(vec![scatter("out")])
        .with_params(vec![
            ParamDef::text("name", "scatter")
                .described("Rule name — what the lowered document calls this pass"),
            ParamDef::number("cell_size", 32.0).range(1.0, 512.0),
            ParamDef::number("base_density", 0.1).range(0.0, 16.0),
            ParamDef::number("jitter", 1.0).range(0.0, 1.0),
            ParamDef::int("seed", 0),
            ParamDef::number("altitude_offset", 0.0),
            ParamDef::toggle("align_to_normal", false),
            ParamDef::number("scale_min", 1.0).range(0.0, 100.0),
            ParamDef::number("scale_max", 1.0).range(0.0, 100.0),
            ParamDef::choice(
                "rotation",
                vec!["RandomYaw".into(), "AlignNormal".into()],
                "RandomYaw",
            ),
            ParamDef::text("mesh", "").described("Mesh asset GUID (blank → debug marker)"),
            ParamDef::number("weight", 1.0).range(0.0, 100.0),
        ])
}

/// Joins two scatter passes into one — the multi-rule combinator (P19.3).
///
/// Binary and left-associating, exactly like `combine.*` one wire type down; the
/// lowerer flattens the tree in `a`-then-`b` depth-first order, so the rule list
/// reads the way the canvas does.
fn scatter_merge_node() -> NodeDef {
    NodeDef::new("scatter.merge", "Merge Scatters", "scatter")
        .described("Run both scatter passes (a first, then b) as separate rules")
        .with_inputs(vec![scatter("a"), scatter("b")])
        .with_outputs(vec![scatter("out")])
}

/// The default rule text a fresh `grammar.rules` node carries: a complete,
/// working fence, so the node teaches the DSL rather than starting blank.
const DEFAULT_RULES: &str = "\
# Modules are what a terminal places. `size` is the metres it consumes.
module Post  = size 0.2
module Panel = size 2

# The first rule is the default axiom.
Fence -> Post Panel* Post
";

/// The P19.4 **grammar kit**: a rule text, two span sources, and the expander
/// that joins them into the population.
fn grammar_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("grammar.rules", "Grammar Rules", "grammar")
            .described(
                "The rule text and module palette — a token grammar that rewrites \
                 along a span",
            )
            .with_outputs(vec![rules("out")])
            .with_params(vec![ParamDef::multiline("rules", DEFAULT_RULES).described(
                "`Symbol -> A B* C` rules and `module X = mesh <guid> size 2` \
                 palette entries, one per line",
            )]),
        NodeDef::new("grammar.spline", "Spline Span", "grammar")
            .described("A span following a scene entity's Spline, by arc length")
            .with_outputs(vec![span("out")])
            .with_params(vec![
                ParamDef::text("spline", "")
                    .described("Spline entity GUID (blank → the entity this graph evaluates on)"),
                ParamDef::int("samples_per_segment", DEFAULT_SPLINE_SAMPLES as i64)
                    .range(1.0, 256.0)
                    .described("Polyline samples per spline segment (arc-length accuracy)"),
            ]),
        NodeDef::new("grammar.footprint", "Footprint Span", "grammar")
            .described("A span set from a rectangle: its four walls, or parallel rows")
            .with_outputs(vec![span("out")])
            .with_params(vec![
                ParamDef::choice("mode", vec!["Perimeter".into(), "Rows".into()], "Perimeter"),
                ParamDef::number("size_x", 0.0)
                    .range(0.0, 100_000.0)
                    .described("Rectangle X size in metres (0 → the PCG volume's own extent)"),
                ParamDef::number("size_z", 0.0)
                    .range(0.0, 100_000.0)
                    .described("Rectangle Z size in metres (0 → the PCG volume's own extent)"),
                ParamDef::int("rows", 4)
                    .range(0.0, 4096.0)
                    .described("Rows mode: how many parallel spans"),
                ParamDef::choice("row_axis", vec!["X".into(), "Z".into()], "X")
                    .described("Rows mode: which world axis the rows run along"),
                ParamDef::text("corner", "")
                    .described("Perimeter mode: the module stamped on each corner (blank → none)"),
                ParamDef::number("corner_size", 0.0)
                    .range(0.0, 1000.0)
                    .described("Perimeter mode: metres each corner reserves (insets both edges)"),
            ]),
        // **Wave G** — the door a GIS road centreline, stream course, coastline
        // or parcel boundary reaches the grammar through. Before it, the only
        // area-shaped thing the grammar could take was an axis-aligned
        // rectangle, which is why an imported polygon collapsed to its bounding
        // box: `Span::from_points` was public the whole time and nothing could
        // call it.
        NodeDef::new("grammar.polyline", "Polyline Span", "grammar")
            .described("A span from explicit world-space points — a road centreline, a river, a parcel boundary")
            .with_outputs(vec![span("out")])
            .with_params(vec![
                ParamDef::multiline("points", "")
                    .described("One position per line, `x,z` or `x,y,z`, in world metres (blank y → on the ground)"),
                ParamDef::toggle("closed", false)
                    .described("Close the path back to its first point — a polygon ring rather than an open route"),
            ]),
        NodeDef::new("grammar.expand", "Expand Grammar", "grammar")
            .described("Rewrite the rules along the span and place the module instances")
            .with_inputs(vec![span("span"), rules("rules")])
            .with_outputs(vec![scatter("out")])
            .with_params(vec![
                ParamDef::text("name", "grammar")
                    .described("Pass name — what diagnostics call this expansion"),
                ParamDef::text("axiom", "")
                    .described("The symbol expansion starts from (blank → the first rule)"),
                ParamDef::int("seed", 0),
                ParamDef::choice("ground", vec!["Terrain".into(), "Span".into()], "Terrain")
                    .described("Take Y from the terrain under each slot, or from the span itself"),
                ParamDef::number("altitude_offset", 0.0),
            ]),
    ]
}

/// The P19.5 **building kit**: an archetype and the planner that stands one on a
/// lot.
///
/// The shape is the grammar kit's, one level up — a *definition* node and an
/// *expander* node, joined by one wire — and for the same reason: one archetype
/// can feed several planners (the same office block on three lots), and a
/// definition that is only ever a param on its consumer cannot be shared.
///
/// `building.plan` **outputs a SCATTER**, exactly as `grammar.expand` does, so
/// it joins the existing merge and layer chains with no third merge node and no
/// third sink input, and a disabled layer disables its buildings for free.
fn building_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("building.archetype", "Building Archetype", "building")
            .described(
                "One of the seven shipped building palettes: its module set, wall \
                 grammar, room table and furniture",
            )
            .with_outputs(vec![building("out")])
            .with_params(vec![
                ParamDef::choice(
                    "archetype",
                    ArchetypeId::ALL
                        .iter()
                        .map(|a| a.name().to_string())
                        .collect(),
                    ArchetypeId::Office.name(),
                ),
                ParamDef::int("floors", 0)
                    .range(0.0, crate::building::MAX_FLOORS as f64)
                    .described("Storey count (0 → drawn from the archetype's own range)"),
                ParamDef::toggle("furnish", true)
                    .described("Populate rooms with the archetype's furniture set"),
            ]),
        NodeDef::new("building.plan", "Plan Building", "building")
            .described(
                "Stand a building on a lot: floor stack, rooms, doors and windows, \
                 stairs and furniture",
            )
            .with_inputs(vec![building("archetype"), span("lot")])
            .with_outputs(vec![scatter("out")])
            .with_params(vec![
                ParamDef::text("name", "building")
                    .described("Pass name — what diagnostics call this building"),
                ParamDef::number("size_x", 0.0)
                    .range(0.0, 100_000.0)
                    .described("Lot X size in metres (0 → the PCG volume's own extent)"),
                ParamDef::number("size_z", 0.0)
                    .range(0.0, 100_000.0)
                    .described("Lot Z size in metres (0 → the PCG volume's own extent)"),
                ParamDef::int("seed", 0),
                ParamDef::choice("ground", vec!["Terrain".into(), "Span".into()], "Terrain")
                    .described(
                        "Take the building's datum from the terrain under its footprint \
                         centre, or from the volume itself",
                    ),
                ParamDef::number("altitude_offset", 0.0),
            ]),
    ]
}

/// The layer wrapper and its merge — the multi-**layer** half (P19.3).
fn layer_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("layer.layer", "Layer", "layer")
            .described("Group a scatter chain into a named, toggleable layer")
            .with_inputs(vec![scatter("scatter")])
            .with_outputs(vec![layer("out")])
            .with_params(vec![
                ParamDef::text("name", "layer"),
                ParamDef::toggle("enabled", true)
                    .described("A disabled layer lowers but evaluates to nothing"),
            ]),
        NodeDef::new("layer.merge", "Merge Layers", "layer")
            .described("Run both layers (a first, then b) in order")
            .with_inputs(vec![layer("a"), layer("b")])
            .with_outputs(vec![layer("out")]),
    ]
}

/// The PCG output sink.
///
/// Two inputs, and **which one is connected decides the document's shape**:
/// `layers` lowers the full layers × rules model; `scatter` — the pre-P19.3
/// input, kept so every existing `.inf_pcg` lowers unchanged — wraps its rule
/// list in one implicit layer.
fn output_node() -> NodeDef {
    NodeDef::new("output.pcg", "PCG Output", "output")
        .described("The final population this volume or biome evaluates")
        .with_inputs(vec![scatter("scatter"), layer("layers")])
        .with_flags(SINK)
}

/// Every mesh GUID a graph's grammar **module palettes** declare, sorted and
/// deduplicated — the cook's `.inf_pcg` → module edge (P19.4).
///
/// # Why this reads the nodes and not the lowered passes
///
/// A palette is declared by a `grammar.rules` node's own text, and that text
/// parses (or does not) entirely on its own. Lowering, by contrast, has five
/// ways to give up before a pass exists — an unconnected Span pin, an
/// unconnected Rules pin, a wrong node type on either, a rule text that does not
/// parse — and every one of them is an **ordinary mid-authoring state**. Driving
/// the cook's dependency edge off the lowered passes would mean a graph with a
/// wire not yet dragged ships without the meshes it plainly declares, and
/// without the advisory that would have said so.
///
/// So this walks the nodes. It is **over-inclusive by design**: a `grammar.rules`
/// node wired to nothing still contributes its meshes. That asymmetry is the
/// right one — packing a mesh an unwired node names costs bytes, missing one a
/// wired node names costs a hole in a wall.
pub fn grammar_mesh_refs(graph: &Graph, reg: &NodeRegistry) -> Vec<Uuid> {
    let mut out: std::collections::BTreeSet<Uuid> = std::collections::BTreeSet::new();
    for node in graph.nodes.values() {
        if node.type_id != "grammar.rules" {
            continue;
        }
        let text = ptext(&resolved(reg, node), "rules");
        if let Ok(g) = Grammar::parse(&text) {
            out.extend(g.mesh_refs());
        }
    }
    out.into_iter().collect()
}

/// Diagnostic severity for a lowering issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcgSeverity {
    Error,
    Warning,
}

/// A lowering diagnostic, optionally anchored to the offending node (mirrors the
/// material editor's `MatIssue`).
#[derive(Debug, Clone, PartialEq)]
pub struct PcgGraphIssue {
    /// The node id (`NodeId(0)` never occurs) this issue is anchored to, if any.
    pub node: Option<u32>,
    pub severity: PcgSeverity,
    pub message: String,
}

impl PcgGraphIssue {
    fn error(node: Option<NodeId>, message: impl Into<String>) -> Self {
        Self {
            node: node.map(|n| n.0),
            severity: PcgSeverity::Error,
            message: message.into(),
        }
    }

    fn warn(node: Option<NodeId>, message: impl Into<String>) -> Self {
        Self {
            node: node.map(|n| n.0),
            severity: PcgSeverity::Warning,
            message: message.into(),
        }
    }
}

/// The result of lowering a PCG graph: the runtime document, the P19.4 grammar
/// passes, and node-anchored diagnostics. `ok` is true when no error-severity
/// issue was raised.
///
/// **Two outputs, not one, and deliberately.** The document is the frozen,
/// serialization-locked scatter model a `.inf_pcg` stores a mirror of; the
/// grammar passes are a *different generator* whose lowered form has no place on
/// that bincode-positional wire. Since P19.3 the authored graph JSON is the
/// source of truth and every evaluation site re-lowers it, so a pass reaching
/// the runtime through this struct is exactly as available as a rule — see
/// [`crate::grammar`] for the argument in full.
#[derive(Debug, Clone)]
pub struct LoweredPcg {
    pub document: PcgDocument,
    /// The grammar passes, in canvas order, each tagged with the layer it was
    /// lowered under (so a disabled layer disables its grammars too).
    pub grammars: Vec<GrammarPass>,
    /// The P19.5 building passes, on the same terms as `grammars`.
    pub buildings: Vec<BuildingPass>,
    pub issues: Vec<PcgGraphIssue>,
    pub ok: bool,
}

impl LoweredPcg {
    /// `true` when the graph authors at least one grammar pass — the predicate
    /// the cook and the evaluation sites branch on.
    pub fn has_grammars(&self) -> bool {
        !self.grammars.is_empty()
    }

    /// `true` when the graph authors at least one building pass.
    pub fn has_buildings(&self) -> bool {
        !self.buildings.is_empty()
    }
}

/// Lower a PCG editor graph into the stable [`PcgDocument`] runtime model.
///
/// Walks backward from the single `output.pcg` sink through its scatter node and
/// density chain. See the module docs for the v1 single-rule semantics.
pub fn lower_graph(graph: &Graph, reg: &NodeRegistry) -> LoweredPcg {
    lower_graph_with(graph, reg, &NoMasks)
}

/// [`lower_graph`] with a [`MaskSource`] for the `mask.image` node's texture
/// GUIDs. Identical in every other respect.
pub fn lower_graph_with(graph: &Graph, reg: &NodeRegistry, masks: &dyn MaskSource) -> LoweredPcg {
    let mut cx = Ctx {
        graph,
        reg,
        masks,
        issues: Vec::new(),
        grammars: Vec::new(),
        buildings: Vec::new(),
        layer: "layer".into(),
        layer_enabled: true,
    };

    // Locate the sink(s). BTreeMap iteration → the lowest NodeId wins.
    let outputs: Vec<NodeId> = graph
        .nodes
        .values()
        .filter(|n| n.type_id == "output.pcg")
        .map(|n| n.id)
        .collect();
    let Some(&output) = outputs.first() else {
        cx.error(None, "graph has no PCG Output node — add one");
        return LoweredPcg::finish(PcgDocument::default(), cx.grammars, cx.buildings, cx.issues);
    };
    for &extra in outputs.iter().skip(1) {
        cx.warn(
            Some(extra),
            "multiple PCG Output nodes — only the first is evaluated",
        );
    }

    let via_layers = graph.link_into(output, "layers").map(|l| l.from);
    let via_scatter = graph.link_into(output, "scatter").map(|l| l.from);
    if via_layers.is_some() && via_scatter.is_some() {
        cx.warn(
            Some(output),
            "PCG Output has both Layers and Scatter connected — Layers wins",
        );
    }

    let layers = match (via_layers, via_scatter) {
        (Some(from), _) => cx.lower_layers(from, &mut Vec::new()),
        // The pre-P19.3 shape: one bare scatter chain becomes one implicit layer,
        // so every `.inf_pcg` authored before this batch lowers unchanged.
        (None, Some(from)) => vec![PcgLayer {
            name: "layer".into(),
            enabled: true,
            rules: cx.lower_rules(from, &mut Vec::new()),
        }],
        (None, None) => {
            cx.error(Some(output), "PCG Output has no Scatter connected");
            return LoweredPcg::finish(
                PcgDocument::default(),
                cx.grammars,
                cx.buildings,
                cx.issues,
            );
        }
    };

    LoweredPcg::finish(PcgDocument { layers }, cx.grammars, cx.buildings, cx.issues)
}

/// The lowering walk's shared state: the graph being read, the registry that
/// fills param defaults, the mask resolver, and the diagnostics accumulated so
/// far. Bundled so the three mutually-recursive walks (layers → rules → density)
/// do not each carry five arguments.
struct Ctx<'a> {
    graph: &'a Graph,
    reg: &'a NodeRegistry,
    masks: &'a dyn MaskSource,
    issues: Vec<PcgGraphIssue>,
    /// P19.4 grammar passes collected as the scatter walk meets them, in canvas
    /// order — they leave the walk as `Vec<PcgRule>`'s empty tail rather than as
    /// rules, because a grammar is not a scatter.
    grammars: Vec<GrammarPass>,
    /// P19.5 building passes, collected on exactly the same terms.
    buildings: Vec<BuildingPass>,
    /// The layer currently being lowered, stamped onto every pass so a layer's
    /// name and toggle govern its grammars exactly as they govern its rules.
    layer: String,
    layer_enabled: bool,
}

impl Ctx<'_> {
    fn error(&mut self, node: Option<NodeId>, message: impl Into<String>) {
        self.issues.push(PcgGraphIssue::error(node, message));
    }

    fn warn(&mut self, node: Option<NodeId>, message: impl Into<String>) {
        self.issues.push(PcgGraphIssue::warn(node, message));
    }

    /// Flatten a layer chain (`layer.merge` trees over `layer.layer` nodes) into
    /// an ordered `Vec<PcgLayer>`, `a` before `b`, depth-first.
    fn lower_layers(&mut self, node_id: NodeId, visiting: &mut Vec<NodeId>) -> Vec<PcgLayer> {
        if visiting.contains(&node_id) {
            self.error(Some(node_id), "cycle in layer chain");
            return Vec::new();
        }
        let Some(node) = self.graph.node(node_id) else {
            return Vec::new();
        };
        visiting.push(node_id);
        let type_id = node.type_id.clone();
        let out = match type_id.as_str() {
            "layer.merge" => {
                let mut a = self.lower_layer_input(node_id, "a", visiting);
                a.extend(self.lower_layer_input(node_id, "b", visiting));
                a
            }
            "layer.layer" => {
                let params = resolved(self.reg, node);
                let name = ptext(&params, "name");
                let name = if name.trim().is_empty() {
                    "layer".to_string()
                } else {
                    name
                };
                let enabled = pb_default(&params, "enabled", true);
                // Grammar passes lowered below this point belong to this layer.
                let outer = (self.layer.clone(), self.layer_enabled);
                self.layer = name.clone();
                self.layer_enabled = enabled;
                let rules = match self.graph.link_into(node_id, "scatter") {
                    Some(link) => self.lower_rules(link.from, visiting),
                    None => {
                        self.warn(
                            Some(node_id),
                            "Layer has no Scatter connected — it is empty",
                        );
                        Vec::new()
                    }
                };
                (self.layer, self.layer_enabled) = outer;
                vec![PcgLayer {
                    name,
                    enabled,
                    rules,
                }]
            }
            other => {
                self.error(Some(node_id), format!("`{other}` does not produce a layer"));
                Vec::new()
            }
        };
        visiting.pop();
        out
    }

    fn lower_layer_input(
        &mut self,
        node: NodeId,
        port: &str,
        visiting: &mut Vec<NodeId>,
    ) -> Vec<PcgLayer> {
        match self.graph.link_into(node, port) {
            Some(link) => self.lower_layers(link.from, visiting),
            None => {
                self.warn(
                    Some(node),
                    format!("layer input `{port}` is unconnected — contributing nothing"),
                );
                Vec::new()
            }
        }
    }

    /// Flatten a scatter chain (`scatter.merge` trees over `scatter.scatter`
    /// nodes) into an ordered `Vec<PcgRule>`, `a` before `b`, depth-first.
    fn lower_rules(&mut self, node_id: NodeId, visiting: &mut Vec<NodeId>) -> Vec<PcgRule> {
        if visiting.contains(&node_id) {
            self.error(Some(node_id), "cycle in scatter chain");
            return Vec::new();
        }
        let Some(node) = self.graph.node(node_id) else {
            return Vec::new();
        };
        visiting.push(node_id);
        let type_id = node.type_id.clone();
        let out = match type_id.as_str() {
            "scatter.merge" => {
                let mut a = self.lower_rule_input(node_id, "a", visiting);
                a.extend(self.lower_rule_input(node_id, "b", visiting));
                a
            }
            "scatter.scatter" => vec![self.lower_scatter(node_id, visiting)],
            // A grammar rides the SCATTER wire so it can join the same merge and
            // layer chains, but it lowers to a pass, not a rule — so it
            // contributes no `PcgRule` and appends to `self.grammars` instead.
            "grammar.expand" => {
                self.lower_grammar(node_id);
                Vec::new()
            }
            // Same shape, one level up: a building is a third generator on the
            // SCATTER wire, and lowers to a pass rather than to a rule.
            "building.plan" => {
                self.lower_building(node_id);
                Vec::new()
            }
            other => {
                self.error(
                    Some(node_id),
                    format!("`{other}` does not produce a scatter pass"),
                );
                Vec::new()
            }
        };
        visiting.pop();
        out
    }

    fn lower_rule_input(
        &mut self,
        node: NodeId,
        port: &str,
        visiting: &mut Vec<NodeId>,
    ) -> Vec<PcgRule> {
        match self.graph.link_into(node, port) {
            Some(link) => self.lower_rules(link.from, visiting),
            None => {
                self.warn(
                    Some(node),
                    format!("scatter input `{port}` is unconnected — contributing nothing"),
                );
                Vec::new()
            }
        }
    }

    /// One `scatter.scatter` node → one [`PcgRule`].
    fn lower_scatter(&mut self, node_id: NodeId, visiting: &mut Vec<NodeId>) -> PcgRule {
        let params = resolved(self.reg, self.graph.node(node_id).expect("checked"));
        let scatter = ScatterParams {
            seed: pi(&params, "seed") as u64,
            cell_size: pf(&params, "cell_size"),
            base_density: pf(&params, "base_density"),
            jitter: pf(&params, "jitter"),
            align_to_normal: pb(&params, "align_to_normal"),
            scale_range: (pf(&params, "scale_min"), pf(&params, "scale_max")),
            rotation: match penum(&params, "rotation").as_str() {
                "AlignNormal" => RotationMode::AlignNormal,
                _ => RotationMode::RandomYaw,
            },
            altitude_offset: pf(&params, "altitude_offset"),
        };

        // The density chain feeding the scatter. A density subgraph may fan into
        // several scatter nodes, so it gets its own `visiting` stack — sharing
        // the scatter walk's would read a legal diamond as a cycle.
        let sampler = match self.graph.link_into(node_id, "density") {
            Some(link) => self.lower_density(link.from, &mut Vec::new()),
            None => {
                self.warn(
                    Some(node_id),
                    "Scatter has no density input — placing at full density everywhere",
                );
                SamplerDef::Constant(1.0)
            }
        };
        let _ = visiting;

        let name = ptext(&params, "name");
        PcgRule {
            name: if name.trim().is_empty() {
                "scatter".into()
            } else {
                name
            },
            sampler,
            scatter,
            // The one-entry kind palette from the node's mesh/weight params.
            kinds: vec![PcgKind {
                mesh: parse_guid(&ptext(&params, "mesh")),
                weight: pf(&params, "weight"),
            }],
        }
    }
}

// ── P19.4: the grammar walk ─────────────────────────────────────────────────

impl Ctx<'_> {
    /// One `grammar.expand` node → one [`GrammarPass`], appended to
    /// [`Ctx::grammars`].
    ///
    /// Every failure is **node-anchored and fails closed**: a pass that cannot
    /// be built is simply not built, so the graph still lowers and the rest of
    /// the population is unaffected. Only a rule text that does not *parse* is
    /// an error — that is a mistake in authored source, and the message carries
    /// the DSL's own `line:col`, exactly like the WGSL emitter's.
    fn lower_grammar(&mut self, node_id: NodeId) {
        let params = resolved(self.reg, self.graph.node(node_id).expect("checked"));

        // ── the rule text ───────────────────────────────────────────────────
        let Some(rules_link) = self.graph.link_into(node_id, "rules") else {
            self.error(
                Some(node_id),
                "Expand Grammar has no Grammar Rules connected — it places nothing",
            );
            return;
        };
        let rules_node = rules_link.from;
        let Some(rn) = self.graph.node(rules_node) else {
            return;
        };
        if rn.type_id != "grammar.rules" {
            self.error(
                Some(rules_node),
                format!("`{}` does not produce a grammar rule table", rn.type_id),
            );
            return;
        }
        let text = ptext(&resolved(self.reg, rn), "rules");
        let grammar = match Grammar::parse(&text) {
            Ok(g) => g,
            Err(e) => {
                // Anchored on the RULES node — that is where the text lives.
                self.error(Some(rules_node), format!("grammar {e}"));
                return;
            }
        };
        if grammar.is_empty() && grammar.modules().is_empty() {
            self.warn(
                Some(rules_node),
                "Grammar Rules is empty — nothing to place",
            );
        }
        // A terminal with no module is a legal gap; say so once, listing them,
        // so a typo is visible rather than silently becoming empty space.
        let gaps = grammar.gaps();
        if !gaps.is_empty() {
            self.warn(
                Some(rules_node),
                format!(
                    "these symbols place nothing (no `module` declares them) and are \
                     treated as gaps: {}",
                    gaps.join(", ")
                ),
            );
        }

        // ── the span ────────────────────────────────────────────────────────
        let Some(span_link) = self.graph.link_into(node_id, "span") else {
            self.error(
                Some(node_id),
                "Expand Grammar has no Span connected — it places nothing",
            );
            return;
        };
        let span_node = span_link.from;
        let Some(sn) = self.graph.node(span_node) else {
            return;
        };
        let sn_type = sn.type_id.clone();
        let sp = resolved(self.reg, sn);
        let (span, corner_module) = match sn_type.as_str() {
            "grammar.spline" => {
                let raw = ptext(&sp, "spline");
                let entity = parse_guid(&raw);
                if entity.is_none() && !raw.trim().is_empty() {
                    self.warn(
                        Some(span_node),
                        format!(
                            "Spline Span entity `{raw}` is not a GUID — falling back to \
                             the entity this graph evaluates on"
                        ),
                    );
                }
                (
                    SpanSource::Spline {
                        entity,
                        samples_per_segment: pi(&sp, "samples_per_segment").clamp(1, 256) as usize,
                    },
                    String::new(),
                )
            }
            "grammar.footprint" => {
                let size = glam::DVec2::new(pf(&sp, "size_x").max(0.0), pf(&sp, "size_z").max(0.0));
                let corner = ptext(&sp, "corner");
                let mode = if penum(&sp, "mode") == "Rows" {
                    if !corner.trim().is_empty() {
                        self.warn(
                            Some(span_node),
                            "Rows mode has no corners — the `corner` module is ignored",
                        );
                    }
                    FootprintMode::Rows {
                        rows: pi(&sp, "rows").clamp(0, 4096) as u32,
                        axis: if penum(&sp, "row_axis") == "Z" {
                            RowAxis::Z
                        } else {
                            RowAxis::X
                        },
                    }
                } else {
                    FootprintMode::Perimeter {
                        corner_size: pf(&sp, "corner_size").max(0.0),
                    }
                };
                let corner = corner.trim().to_string();
                if !corner.is_empty() && grammar.module_index(&corner).is_none() {
                    self.warn(
                        Some(span_node),
                        format!(
                            "corner module `{corner}` is not declared in the grammar — \
                             no corner is placed"
                        ),
                    );
                }
                (SpanSource::Footprint { size, mode }, corner)
            }
            // **Wave G** — explicit world-space points, the door a GIS road
            // centreline or a polygon ring reaches the grammar through.
            //
            // The points ride the node's own param as text rather than as a
            // typed list because the graph JSON is the wire here (the player
            // re-lowers it), and a self-describing text field is additive where
            // a new param TYPE would not be. `x,z` or `x,y,z` per line; a `y`
            // omitted means "sit on the ground", which is what a published
            // 2-D centreline means and what the pass's own `Ground` mode then
            // resolves.
            "grammar.polyline" => {
                let closed = pb(&sp, "closed");
                let (points, bad) = parse_polyline_points(&ptext(&sp, "points"));
                if bad > 0 {
                    self.warn(
                        Some(span_node),
                        format!(
                            "{bad} line(s) of this polyline could not be read as a \
                             coordinate and were skipped — each line wants `x,z` or \
                             `x,y,z` in world metres"
                        ),
                    );
                }
                if points.len() < 2 {
                    self.warn(
                        Some(span_node),
                        format!(
                            "this polyline has {} usable position(s); a span needs at \
                             least 2, so this pass places nothing",
                            points.len()
                        ),
                    );
                }
                (SpanSource::Polyline { points, closed }, String::new())
            }
            other => {
                self.error(
                    Some(span_node),
                    format!("`{other}` does not produce a grammar span"),
                );
                return;
            }
        };

        // ── the pass ────────────────────────────────────────────────────────
        let axiom = {
            let authored = ptext(&params, "axiom");
            let authored = authored.trim();
            if authored.is_empty() {
                grammar.default_axiom().unwrap_or_default().to_string()
            } else {
                if grammar.rule(authored).is_none() && grammar.module_index(authored).is_none() {
                    self.warn(
                        Some(node_id),
                        format!(
                            "axiom `{authored}` names no rule and no module — this pass \
                             places nothing"
                        ),
                    );
                }
                authored.to_string()
            }
        };
        let name = ptext(&params, "name");
        self.grammars.push(GrammarPass {
            name: if name.trim().is_empty() {
                "grammar".into()
            } else {
                name
            },
            layer: self.layer.clone(),
            enabled: self.layer_enabled,
            seed: pi(&params, "seed") as u64,
            grammar,
            axiom,
            span,
            corner_module,
            ground: if penum(&params, "ground") == "Span" {
                Ground::Span
            } else {
                Ground::Terrain
            },
            altitude_offset: pf(&params, "altitude_offset"),
        });
    }
}

// ── P19.5: the building walk ────────────────────────────────────────────────

impl Ctx<'_> {
    /// One `building.plan` node → one [`BuildingPass`], appended to
    /// [`Ctx::buildings`].
    ///
    /// Fails closed and node-anchored, like the grammar walk. The one *error* is
    /// a missing archetype input — without it there is no palette and nothing to
    /// build; an unknown archetype **name** is a warning that falls back to the
    /// first palette, because a param whose choice list shifted is a migration
    /// artefact, not an authoring mistake.
    fn lower_building(&mut self, node_id: NodeId) {
        let params = resolved(self.reg, self.graph.node(node_id).expect("checked"));

        // ── the archetype ───────────────────────────────────────────────────
        let Some(arch_link) = self.graph.link_into(node_id, "archetype") else {
            self.error(
                Some(node_id),
                "Plan Building has no Building Archetype connected — it places nothing",
            );
            return;
        };
        let arch_node = arch_link.from;
        let Some(an) = self.graph.node(arch_node) else {
            return;
        };
        if an.type_id != "building.archetype" {
            self.error(
                Some(arch_node),
                format!("`{}` does not produce a building archetype", an.type_id),
            );
            return;
        }
        let ap = resolved(self.reg, an);
        let raw = penum(&ap, "archetype");
        let archetype = match ArchetypeId::parse(&raw) {
            Some(a) => a,
            None => {
                self.warn(
                    Some(arch_node),
                    format!(
                        "unknown building archetype `{raw}` — falling back to `{}`",
                        ArchetypeId::ALL[0].name()
                    ),
                );
                ArchetypeId::ALL[0]
            }
        };
        let floors = pi(&ap, "floors").clamp(0, crate::building::MAX_FLOORS as i64) as u32;
        let furnish = pb_default(&ap, "furnish", true);

        // ── the lot (optional) ──────────────────────────────────────────────
        let lot = match self.graph.link_into(node_id, "lot") {
            Some(link) => {
                let span_node = link.from;
                match self.lower_span(span_node) {
                    Some(source) => Some(source),
                    None => return,
                }
            }
            None => None,
        };

        let name = ptext(&params, "name");
        self.buildings.push(BuildingPass {
            name: if name.trim().is_empty() {
                "building".into()
            } else {
                name
            },
            layer: self.layer.clone(),
            enabled: self.layer_enabled,
            archetype,
            seed: pi(&params, "seed") as u64,
            floors,
            furnish,
            size: glam::DVec2::new(
                pf(&params, "size_x").max(0.0),
                pf(&params, "size_z").max(0.0),
            ),
            lot,
            ground: if penum(&params, "ground") == "Span" {
                Ground::Span
            } else {
                Ground::Terrain
            },
            altitude_offset: pf(&params, "altitude_offset"),
        });
    }

    /// The [`SpanSource`] a `grammar.spline` / `grammar.footprint` node
    /// describes, or `None` (with an anchored error) for anything else.
    ///
    /// Shared with [`Ctx::lower_grammar`]'s own span handling in *intent* but
    /// not in code: the grammar's version also reads a corner module and
    /// validates it against the grammar's palette, which a building has no use
    /// for. Extracting the common half would leave two callers passing flags to
    /// say which half they wanted.
    fn lower_span(&mut self, span_node: NodeId) -> Option<SpanSource> {
        let sn = self.graph.node(span_node)?;
        let sn_type = sn.type_id.clone();
        let sp = resolved(self.reg, sn);
        match sn_type.as_str() {
            "grammar.spline" => {
                let raw = ptext(&sp, "spline");
                let entity = parse_guid(&raw);
                if entity.is_none() && !raw.trim().is_empty() {
                    self.warn(
                        Some(span_node),
                        format!(
                            "Spline Span entity `{raw}` is not a GUID — falling back to \
                             the entity this graph evaluates on"
                        ),
                    );
                }
                Some(SpanSource::Spline {
                    entity,
                    samples_per_segment: pi(&sp, "samples_per_segment").clamp(1, 256) as usize,
                })
            }
            "grammar.footprint" => {
                let size = glam::DVec2::new(pf(&sp, "size_x").max(0.0), pf(&sp, "size_z").max(0.0));
                let mode = if penum(&sp, "mode") == "Rows" {
                    FootprintMode::Rows {
                        rows: pi(&sp, "rows").clamp(0, 4096) as u32,
                        axis: if penum(&sp, "row_axis") == "Z" {
                            RowAxis::Z
                        } else {
                            RowAxis::X
                        },
                    }
                } else {
                    FootprintMode::Perimeter {
                        corner_size: pf(&sp, "corner_size").max(0.0),
                    }
                };
                Some(SpanSource::Footprint { size, mode })
            }
            "grammar.polyline" => {
                let (points, _bad) = parse_polyline_points(&ptext(&sp, "points"));
                Some(SpanSource::Polyline {
                    points,
                    closed: pb(&sp, "closed"),
                })
            }
            other => {
                self.error(
                    Some(span_node),
                    format!("`{other}` does not produce a grammar span"),
                );
                None
            }
        }
    }
}

impl LoweredPcg {
    fn finish(
        document: PcgDocument,
        grammars: Vec<GrammarPass>,
        buildings: Vec<BuildingPass>,
        issues: Vec<PcgGraphIssue>,
    ) -> Self {
        let ok = !issues.iter().any(|i| i.severity == PcgSeverity::Error);
        Self {
            document,
            grammars,
            buildings,
            issues,
            ok,
        }
    }
}

impl Ctx<'_> {
    /// Recursively lower one density-chain node into a [`SamplerDef`]. A missing
    /// input defaults to `Constant(1.0)` (identity for the common `multiply`),
    /// with a node-anchored warning. Cycles (should not occur — the edit door
    /// forbids them) short-circuit to `Constant(0.0)`.
    fn lower_density(&mut self, node_id: NodeId, visiting: &mut Vec<NodeId>) -> SamplerDef {
        if visiting.contains(&node_id) {
            self.error(Some(node_id), "cycle in density chain");
            return SamplerDef::Constant(0.0);
        }
        let Some(node) = self.graph.node(node_id) else {
            return SamplerDef::Constant(1.0);
        };
        visiting.push(node_id);

        let params = resolved(self.reg, node);
        let type_id = node.type_id.clone();

        let out = match type_id.as_str() {
            "const.density" => SamplerDef::Constant(pf(&params, "value")),
            "noise.fbm" => SamplerDef::Noise(ValueNoise {
                seed: pi(&params, "seed") as u64,
                frequency: pf(&params, "frequency"),
                octaves: pi(&params, "octaves") as u32,
                lacunarity: pf(&params, "lacunarity"),
                gain: pf(&params, "gain"),
            }),
            "filter.slope" => SamplerDef::Slope {
                min_deg: pf(&params, "min_deg"),
                max_deg: pf(&params, "max_deg"),
                feather_deg: pf(&params, "feather_deg"),
            },
            "filter.altitude" => SamplerDef::Altitude {
                min: pf(&params, "min"),
                max: pf(&params, "max"),
                feather: pf(&params, "feather"),
            },
            "mask.image" => self.lower_mask_image(node_id, &params),
            "mask.flow" => SamplerDef::DataMap {
                kind: inf_terrain::DataMapKind::Flow,
                min: pf(&params, "min"),
                max: pf(&params, "max"),
            },
            "mask.deposition" => SamplerDef::DataMap {
                kind: inf_terrain::DataMapKind::Deposition,
                min: pf(&params, "min"),
                max: pf(&params, "max"),
            },
            "mask.wear" => SamplerDef::DataMap {
                kind: inf_terrain::DataMapKind::Wear,
                min: pf(&params, "min"),
                max: pf(&params, "max"),
            },
            "mask.biome" => SamplerDef::Biome {
                id: pi(&params, "id").clamp(0, 255) as u8,
                feather: pf(&params, "feather").max(0.0),
            },
            "combine.multiply" => {
                let a = self.lower_input(node_id, "a", visiting);
                let b = self.lower_input(node_id, "b", visiting);
                SamplerDef::Multiply(Box::new(a), Box::new(b))
            }
            "combine.max" => {
                let a = self.lower_input(node_id, "a", visiting);
                let b = self.lower_input(node_id, "b", visiting);
                SamplerDef::Max(Box::new(a), Box::new(b))
            }
            "combine.min" => {
                let a = self.lower_input(node_id, "a", visiting);
                let b = self.lower_input(node_id, "b", visiting);
                SamplerDef::Min(Box::new(a), Box::new(b))
            }
            "combine.invert" => {
                let a = self.lower_input(node_id, "in", visiting);
                SamplerDef::Invert(Box::new(a))
            }
            other => {
                self.error(
                    Some(node_id),
                    format!("`{other}` does not produce a density"),
                );
                SamplerDef::Constant(0.0)
            }
        };

        visiting.pop();
        out
    }

    /// `mask.image` → a [`SamplerDef::Mask`] with the texture's pixels resolved
    /// through the [`MaskSource`].
    ///
    /// Every failure — a blank GUID, an unparseable one, a texture the source
    /// cannot produce — lowers to an **empty** `0 × 0` mask, which scores `0`
    /// everywhere. Failing closed is the point: a mask nobody could load must not
    /// become "place everywhere". A blank GUID is a warning rather than an error
    /// because a freshly added node has one and that is not yet a mistake.
    fn lower_mask_image(&mut self, node_id: NodeId, params: &inf_graph::ParamMap) -> SamplerDef {
        let rect = [
            pf(params, "min_x"),
            pf(params, "min_z"),
            pf(params, "max_x"),
            pf(params, "max_z"),
        ];
        let raw = ptext(params, "texture");
        let empty = SamplerDef::Mask {
            rect,
            width: 0,
            height: 0,
            data: Vec::new(),
        };
        let Some(guid) = parse_guid(&raw) else {
            self.warn(
                Some(node_id),
                if raw.trim().is_empty() {
                    "Image Mask has no texture — it masks out everything".to_string()
                } else {
                    format!("Image Mask texture `{raw}` is not a GUID — it masks out everything")
                },
            );
            return empty;
        };
        match self.masks.mask(guid) {
            Some((width, height, data)) if width > 0 && height > 0 => SamplerDef::Mask {
                rect,
                width,
                height,
                data,
            },
            _ => {
                self.warn(
                    Some(node_id),
                    format!(
                        "Image Mask texture {guid} could not be loaded — it masks out everything"
                    ),
                );
                empty
            }
        }
    }

    /// Lower the density feeding `(node, port)`; an unconnected port warns and
    /// falls back to `Constant(1.0)`.
    fn lower_input(&mut self, node: NodeId, port: &str, visiting: &mut Vec<NodeId>) -> SamplerDef {
        match self.graph.link_into(node, port) {
            Some(link) => self.lower_density(link.from, visiting),
            None => {
                self.warn(
                    Some(node),
                    format!("input `{port}` is unconnected — using constant 1.0"),
                );
                SamplerDef::Constant(1.0)
            }
        }
    }
}

// ── param helpers ──────────────────────────────────────────────────────────

/// Registry defaults filled over the node's sparse overrides.
fn resolved(reg: &NodeRegistry, node: &inf_graph::Node) -> inf_graph::ParamMap {
    match reg.get(&node.type_id) {
        Some(def) => def.resolve(&node.params),
        None => node.params.clone(),
    }
}

/// A numeric param as `f64` (Float or Int; missing → 0.0).
#[cfg(test)]
mod polyline_tests {
    use super::parse_polyline_points;

    /// **The GIS coordinate-paste door.** Real coordinate lists arrive with
    /// blank lines, comments, mixed separators and the occasional junk row, and
    /// a parser that refuses the whole block on one bad line is a parser nobody
    /// can paste into.
    #[test]
    fn a_pasted_coordinate_list_parses_and_counts_what_it_could_not_read() {
        let (pts, bad) = parse_polyline_points(
            "\
# Main Street centreline, EPSG:32610 minus the anchor
0,0
100, 0
100,4.5,50   # this one carries a height

  200 0
not a coordinate
300,0,
",
        );
        // Only the junk line. A TRAILING SEPARATOR is tolerated on purpose:
        // empty segments are filtered before parsing, so `300,0,` is the pair it
        // obviously means. Coordinate lists are pasted, and pasted lists have
        // trailing commas.
        assert_eq!(bad, 1, "only the junk line should fail: {pts:?}");
        assert_eq!(pts.len(), 5);
        assert_eq!(pts[4], glam::DVec3::new(300.0, 0.0, 0.0));
        assert_eq!(pts[0], glam::DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(pts[1], glam::DVec3::new(100.0, 0.0, 0.0));
        // Three numbers means x,y,z — the middle one is the HEIGHT, not the
        // second planar axis. Getting that backwards would lay every road on its
        // side.
        assert_eq!(pts[2], glam::DVec3::new(100.0, 4.5, 50.0));
        // Two numbers means x,z with the height left for the ground rule.
        assert_eq!(pts[3], glam::DVec3::new(200.0, 0.0, 0.0));
    }

    /// Non-finite text never becomes a position. A `nan` here produces module
    /// transforms whose bounds still read healthy (`f32::min`/`max` ignore NaN),
    /// so it has to be stopped where it enters.
    #[test]
    fn non_finite_coordinates_are_dropped_and_counted() {
        let (pts, bad) = parse_polyline_points("0,0\nnan,5\n1,inf\n2,2\n-1e400,0");
        assert_eq!(
            pts,
            vec![glam::DVec3::ZERO, glam::DVec3::new(2.0, 0.0, 2.0)]
        );
        assert_eq!(
            bad, 3,
            "each non-finite line is REPORTED, not silently dropped"
        );
    }

    /// Empty and comment-only input yields nothing and reports no failure —
    /// a blank node is not an error, it just places nothing.
    #[test]
    fn blank_input_is_not_an_error() {
        assert_eq!(parse_polyline_points(""), (vec![], 0));
        assert_eq!(parse_polyline_points("\n\n  \n"), (vec![], 0));
        assert_eq!(parse_polyline_points("# just a note\n"), (vec![], 0));
    }
}

/// Parse a `grammar.polyline` node's `points` text into world positions.
///
/// One coordinate per line, `x,z` or `x,y,z`, in world metres. Returns the
/// positions and **the count of lines that could not be read**, because a
/// silently-dropped coordinate is a road that ends early for no visible reason.
///
/// Blank lines and `#` comments are skipped without counting as failures — an
/// author pasting a coordinate list will have both.
///
/// Non-finite values are dropped and counted: this text can be pasted, and a
/// `nan` reaching a span produces module transforms whose bounds still look
/// healthy (the `f32::min`/`max` mechanism).
fn parse_polyline_points(text: &str) -> (Vec<glam::DVec3>, usize) {
    let mut out = Vec::new();
    let mut bad = 0usize;
    for line in text.lines() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let nums: Vec<Option<f64>> = line
            .split([',', ';', ' ', '\t'])
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<f64>().ok().filter(|v| v.is_finite()))
            .collect();
        match nums.as_slice() {
            [Some(x), Some(z)] => out.push(glam::DVec3::new(*x, 0.0, *z)),
            [Some(x), Some(y), Some(z)] => out.push(glam::DVec3::new(*x, *y, *z)),
            _ => bad += 1,
        }
    }
    (out, bad)
}

fn pf(params: &inf_graph::ParamMap, key: &str) -> f64 {
    match params.get(key) {
        Some(ParamValue::Float(f)) => *f,
        Some(ParamValue::Int(i)) => *i as f64,
        _ => 0.0,
    }
}

/// An integer param as `i64` (Int or rounded Float; missing → 0).
fn pi(params: &inf_graph::ParamMap, key: &str) -> i64 {
    match params.get(key) {
        Some(ParamValue::Int(i)) => *i,
        Some(ParamValue::Float(f)) => f.round() as i64,
        _ => 0,
    }
}

fn pb(params: &inf_graph::ParamMap, key: &str) -> bool {
    matches!(params.get(key), Some(ParamValue::Bool(true)))
}

/// A boolean param whose **absence** means something other than `false` — the
/// `enabled` toggle, where a node with no override is enabled, not disabled.
fn pb_default(params: &inf_graph::ParamMap, key: &str, default: bool) -> bool {
    match params.get(key) {
        Some(ParamValue::Bool(b)) => *b,
        _ => default,
    }
}

fn ptext(params: &inf_graph::ParamMap, key: &str) -> String {
    match params.get(key) {
        Some(ParamValue::Text(s)) => s.clone(),
        _ => String::new(),
    }
}

fn penum(params: &inf_graph::ParamMap, key: &str) -> String {
    match params.get(key) {
        Some(ParamValue::Enum(s)) => s.clone(),
        _ => String::new(),
    }
}

/// Parse a mesh GUID param; blank/invalid → `None` (a bare debug marker).
fn parse_guid(s: &str) -> Option<uuid::Uuid> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    uuid::Uuid::parse_str(s).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use inf_graph::{apply_edits, GraphEdit, Link, NodeUi};

    #[test]
    fn registry_has_output_sink_and_the_kit() {
        let reg = pcg_registry();
        let out = reg.get("output.pcg").expect("output node");
        assert!(out.has(SINK));
        // Two inputs since P19.3: the pre-existing `scatter` (kept, so old graphs
        // lower unchanged) and `layers` (the layers × rules shape).
        assert_eq!(out.inputs.len(), 2);
        assert_eq!(
            out.input("scatter").unwrap().ty,
            PortType::Named(SCATTER_KEY.into())
        );
        assert_eq!(
            out.input("layers").unwrap().ty,
            PortType::Named(LAYER_KEY.into())
        );
        for id in [
            "const.density",
            "noise.fbm",
            "filter.slope",
            "filter.altitude",
            "mask.image",
            "mask.flow",
            "mask.deposition",
            "mask.wear",
            "mask.biome",
            "combine.multiply",
            "combine.max",
            "combine.min",
            "combine.invert",
            "scatter.scatter",
            "scatter.merge",
            "layer.layer",
            "layer.merge",
            "output.pcg",
        ] {
            assert!(reg.contains(id), "missing {id}");
        }
        // The merge nodes mirror the `combine.*` convention exactly: binary, one
        // output, same wire type in and out.
        for (id, key) in [("scatter.merge", SCATTER_KEY), ("layer.merge", LAYER_KEY)] {
            let def = reg.get(id).unwrap();
            assert_eq!(def.inputs.len(), 2, "{id}");
            assert_eq!(def.outputs.len(), 1, "{id}");
            for p in def.inputs.iter().chain(&def.outputs) {
                assert_eq!(p.ty, PortType::Named(key.into()), "{id}.{}", p.name);
            }
        }
        // Every mask node produces a density, and the data-map ones carry the
        // normalization window (the stored accumulators are raw).
        for id in [
            "mask.image",
            "mask.flow",
            "mask.deposition",
            "mask.wear",
            "mask.biome",
        ] {
            let def = reg.get(id).unwrap();
            assert!(def.inputs.is_empty(), "{id} is a source");
            assert_eq!(
                def.outputs[0].ty,
                PortType::Named(DENSITY_KEY.into()),
                "{id}"
            );
            assert_eq!(def.category, "masks", "{id}");
        }
        for id in ["mask.flow", "mask.deposition", "mask.wear"] {
            let def = reg.get(id).unwrap();
            assert!(
                def.param("min").is_some() && def.param("max").is_some(),
                "{id}"
            );
        }
    }

    /// Build noise × slope → scatter → output and assert the lowered document.
    #[test]
    fn lowers_a_known_graph_to_the_expected_document() {
        let reg = pcg_registry();
        let mut g = Graph::empty();
        // Allocate ids up front so the links can reference them.
        let noise = NodeId(1);
        let slope = NodeId(2);
        let mul = NodeId(3);
        let scat = NodeId(4);
        let out = NodeId(5);
        let params = |pairs: &[(&str, ParamValue)]| {
            let mut m = inf_graph::ParamMap::new();
            for (k, v) in pairs {
                m.insert((*k).to_string(), v.clone());
            }
            m
        };
        apply_edits(
            &mut g,
            &reg,
            &[
                GraphEdit::AddNode {
                    id: noise,
                    type_id: "noise.fbm".into(),
                    x: 0.0,
                    y: 0.0,
                    params: params(&[
                        ("seed", ParamValue::Int(7)),
                        ("frequency", ParamValue::Float(0.02)),
                        ("octaves", ParamValue::Int(3)),
                    ]),
                },
                GraphEdit::AddNode {
                    id: slope,
                    type_id: "filter.slope".into(),
                    x: 0.0,
                    y: 80.0,
                    params: params(&[
                        ("min_deg", ParamValue::Float(0.0)),
                        ("max_deg", ParamValue::Float(30.0)),
                        ("feather_deg", ParamValue::Float(5.0)),
                    ]),
                },
                GraphEdit::AddNode {
                    id: mul,
                    type_id: "combine.multiply".into(),
                    x: 160.0,
                    y: 40.0,
                    params: Default::default(),
                },
                GraphEdit::AddNode {
                    id: scat,
                    type_id: "scatter.scatter".into(),
                    x: 320.0,
                    y: 40.0,
                    params: params(&[
                        ("cell_size", ParamValue::Float(16.0)),
                        ("base_density", ParamValue::Float(0.5)),
                        ("seed", ParamValue::Int(2024)),
                        (
                            "mesh",
                            ParamValue::Text(uuid::Uuid::from_u128(1).to_string()),
                        ),
                        ("weight", ParamValue::Float(3.0)),
                    ]),
                },
                GraphEdit::AddNode {
                    id: out,
                    type_id: "output.pcg".into(),
                    x: 480.0,
                    y: 40.0,
                    params: Default::default(),
                },
                GraphEdit::Connect {
                    link: Link {
                        from: noise,
                        from_port: "out".into(),
                        to: mul,
                        to_port: "a".into(),
                    },
                },
                GraphEdit::Connect {
                    link: Link {
                        from: slope,
                        from_port: "out".into(),
                        to: mul,
                        to_port: "b".into(),
                    },
                },
                GraphEdit::Connect {
                    link: Link {
                        from: mul,
                        from_port: "out".into(),
                        to: scat,
                        to_port: "density".into(),
                    },
                },
                GraphEdit::Connect {
                    link: Link {
                        from: scat,
                        from_port: "out".into(),
                        to: out,
                        to_port: "scatter".into(),
                    },
                },
            ],
        );

        let lowered = lower_graph(&g, &reg);
        assert!(lowered.ok, "issues: {:?}", lowered.issues);
        assert_eq!(lowered.document.layers.len(), 1);
        let rule = &lowered.document.layers[0].rules[0];
        assert_eq!(rule.scatter.cell_size, 16.0);
        assert_eq!(rule.scatter.base_density, 0.5);
        assert_eq!(rule.scatter.seed, 2024);
        assert_eq!(rule.kinds.len(), 1);
        assert_eq!(rule.kinds[0].mesh, Some(uuid::Uuid::from_u128(1)));
        assert_eq!(rule.kinds[0].weight, 3.0);
        // sampler = Multiply(Noise, Slope) with the authored params.
        match &rule.sampler {
            SamplerDef::Multiply(a, b) => {
                assert!(matches!(**a, SamplerDef::Noise(ValueNoise { seed: 7, .. })));
                assert!(matches!(
                    **b,
                    SamplerDef::Slope {
                        max_deg,
                        ..
                    } if max_deg == 30.0
                ));
            }
            other => panic!("expected Multiply, got {other:?}"),
        }
    }

    #[test]
    fn missing_output_is_an_anchored_error() {
        let reg = pcg_registry();
        let g = Graph::empty();
        let lowered = lower_graph(&g, &reg);
        assert!(!lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error && i.node.is_none()));
    }

    #[test]
    fn output_without_scatter_errors_on_the_output_node() {
        let reg = pcg_registry();
        let mut g = Graph::empty();
        let out = g.insert("output.pcg", NodeUi::default());
        let lowered = lower_graph(&g, &reg);
        assert!(!lowered.ok);
        let err = lowered
            .issues
            .iter()
            .find(|i| i.severity == PcgSeverity::Error)
            .unwrap();
        assert_eq!(err.node, Some(out.0));
    }

    #[test]
    fn scatter_without_density_warns_and_still_lowers() {
        let reg = pcg_registry();
        let mut g = Graph::empty();
        let scat = g.insert("scatter.scatter", NodeUi::default());
        let out = g.insert("output.pcg", NodeUi::default());
        g.links.push(Link {
            from: scat,
            from_port: "out".into(),
            to: out,
            to_port: "scatter".into(),
        });
        let lowered = lower_graph(&g, &reg);
        assert!(lowered.ok, "a missing density is only a warning");
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Warning && i.node == Some(scat.0)));
        // Fell back to a full-density constant.
        let rule = &lowered.document.layers[0].rules[0];
        assert_eq!(rule.sampler, SamplerDef::Constant(1.0));
    }

    // ── P19.3: the mask nodes, and layers × rules ───────────────────────────

    /// A test builder: nodes with params, links, then lower. Keeps the
    /// multi-node cases readable.
    struct Builder {
        reg: NodeRegistry,
        g: Graph,
        next: u32,
    }

    impl Builder {
        fn new() -> Self {
            Self {
                reg: pcg_registry(),
                g: Graph::empty(),
                next: 1,
            }
        }

        fn node(&mut self, type_id: &str, params: &[(&str, ParamValue)]) -> NodeId {
            let id = NodeId(self.next);
            self.next += 1;
            let mut m = inf_graph::ParamMap::new();
            for (k, v) in params {
                m.insert((*k).to_string(), v.clone());
            }
            apply_edits(
                &mut self.g,
                &self.reg,
                &[GraphEdit::AddNode {
                    id,
                    type_id: type_id.into(),
                    x: 0.0,
                    y: 0.0,
                    params: m,
                }],
            );
            id
        }

        fn link(&mut self, from: NodeId, from_port: &str, to: NodeId, to_port: &str) {
            apply_edits(
                &mut self.g,
                &self.reg,
                &[GraphEdit::Connect {
                    link: Link {
                        from,
                        from_port: from_port.into(),
                        to,
                        to_port: to_port.into(),
                    },
                }],
            );
        }

        fn lower(&self) -> LoweredPcg {
            lower_graph(&self.g, &self.reg)
        }

        fn lower_with(&self, masks: &dyn MaskSource) -> LoweredPcg {
            lower_graph_with(&self.g, &self.reg, masks)
        }

        /// A `scatter.scatter` fed by `density`, named `name`.
        fn scatter_named(&mut self, name: &str, density: Option<NodeId>) -> NodeId {
            let s = self.node(
                "scatter.scatter",
                &[("name", ParamValue::Text(name.into()))],
            );
            if let Some(d) = density {
                self.link(d, "out", s, "density");
            }
            s
        }
    }

    /// Every terrain-layer mask lowers to its `SamplerDef` **with its params**,
    /// and each round-trips through serde unchanged.
    #[test]
    fn the_mask_nodes_lower_to_their_sampler_defs_and_round_trip() {
        /// `(node type, authored params, the SamplerDef it must lower to)`.
        type Case = (&'static str, Vec<(&'static str, ParamValue)>, SamplerDef);
        let cases: Vec<Case> = vec![
            (
                "mask.flow",
                vec![
                    ("min", ParamValue::Float(100.0)),
                    ("max", ParamValue::Float(900.0)),
                ],
                SamplerDef::DataMap {
                    kind: inf_terrain::DataMapKind::Flow,
                    min: 100.0,
                    max: 900.0,
                },
            ),
            (
                "mask.deposition",
                vec![
                    ("min", ParamValue::Float(0.0)),
                    ("max", ParamValue::Float(2.5)),
                ],
                SamplerDef::DataMap {
                    kind: inf_terrain::DataMapKind::Deposition,
                    min: 0.0,
                    max: 2.5,
                },
            ),
            (
                "mask.wear",
                vec![
                    ("min", ParamValue::Float(0.25)),
                    ("max", ParamValue::Float(4.0)),
                ],
                SamplerDef::DataMap {
                    kind: inf_terrain::DataMapKind::Wear,
                    min: 0.25,
                    max: 4.0,
                },
            ),
            (
                "mask.biome",
                vec![
                    ("id", ParamValue::Int(3)),
                    ("feather", ParamValue::Float(6.0)),
                ],
                SamplerDef::Biome {
                    id: 3,
                    feather: 6.0,
                },
            ),
        ];
        for (type_id, params, expect) in cases {
            let mut b = Builder::new();
            let m = b.node(type_id, &params);
            let s = b.scatter_named("r", Some(m));
            let o = b.node("output.pcg", &[]);
            b.link(s, "out", o, "scatter");
            let lowered = b.lower();
            assert!(lowered.ok, "{type_id}: {:?}", lowered.issues);
            let got = &lowered.document.layers[0].rules[0].sampler;
            assert_eq!(got, &expect, "{type_id} lowered wrong");
            // The permanent serde discipline: every new variant round-trips.
            let json = serde_json::to_string(&lowered.document).unwrap();
            let back: PcgDocument = serde_json::from_str(&json).unwrap();
            assert_eq!(back, lowered.document, "{type_id} did not round-trip");
        }
        // Out-of-range params are clamped rather than wrapped: an id of 300 is a
        // typo, not biome 44.
        let mut b = Builder::new();
        let m = b.node(
            "mask.biome",
            &[
                ("id", ParamValue::Int(300)),
                ("feather", ParamValue::Float(-5.0)),
            ],
        );
        let s = b.scatter_named("r", Some(m));
        let o = b.node("output.pcg", &[]);
        b.link(s, "out", o, "scatter");
        assert_eq!(
            b.lower().document.layers[0].rules[0].sampler,
            SamplerDef::Biome {
                id: 255,
                feather: 0.0
            }
        );
    }

    /// `mask.image` resolves its texture through the [`MaskSource`], carries the
    /// authored world rect, and **fails closed** when nothing resolves.
    #[test]
    fn mask_image_resolves_through_the_source_and_round_trips() {
        struct One(Uuid);
        impl MaskSource for One {
            fn mask(&self, texture: Uuid) -> Option<(u32, u32, Vec<u8>)> {
                (texture == self.0).then(|| (2, 2, vec![0, 255, 128, 64]))
            }
        }
        let guid = Uuid::from_u128(0x5A5A);
        let build = |texture: String| {
            let mut b = Builder::new();
            let m = b.node(
                "mask.image",
                &[
                    ("texture", ParamValue::Text(texture)),
                    ("min_x", ParamValue::Float(-10.0)),
                    ("min_z", ParamValue::Float(-20.0)),
                    ("max_x", ParamValue::Float(30.0)),
                    ("max_z", ParamValue::Float(40.0)),
                ],
            );
            let s = b.scatter_named("r", Some(m));
            let o = b.node("output.pcg", &[]);
            b.link(s, "out", o, "scatter");
            (b, m)
        };

        // Resolved: real pixels, authored rect.
        let (b, _) = build(guid.to_string());
        let lowered = b.lower_with(&One(guid));
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert_eq!(
            lowered.document.layers[0].rules[0].sampler,
            SamplerDef::Mask {
                rect: [-10.0, -20.0, 30.0, 40.0],
                width: 2,
                height: 2,
                data: vec![0, 255, 128, 64],
            }
        );
        // …and it round-trips (the mask bytes ride the document).
        let json = serde_json::to_string(&lowered.document).unwrap();
        assert_eq!(
            serde_json::from_str::<PcgDocument>(&json).unwrap(),
            lowered.document
        );

        // Unresolvable, blank and malformed GUIDs all fail CLOSED — an empty
        // mask that scores 0 — with a node-anchored warning, never an error.
        for texture in [
            Uuid::from_u128(0xDEAD).to_string(),
            String::new(),
            "not-a-guid".into(),
        ] {
            let (b, node) = build(texture.clone());
            let lowered = b.lower_with(&One(guid));
            assert!(lowered.ok, "an unloadable mask is not a hard error");
            assert!(
                lowered
                    .issues
                    .iter()
                    .any(|i| i.severity == PcgSeverity::Warning && i.node == Some(node.0)),
                "no anchored warning for `{texture}`"
            );
            match &lowered.document.layers[0].rules[0].sampler {
                SamplerDef::Mask { width, height, .. } => {
                    assert_eq!((*width, *height), (0, 0), "`{texture}` did not fail closed")
                }
                other => panic!("expected an empty Mask, got {other:?}"),
            }
        }
    }

    /// **Multi-rule lowering**: a `scatter.merge` tree flattens into an ordered
    /// rule list, `a` before `b`, depth-first — and the rules keep their names.
    #[test]
    fn a_scatter_merge_tree_lowers_to_ordered_rules() {
        let mut b = Builder::new();
        let s1 = b.scatter_named("trees", None);
        let s2 = b.scatter_named("rocks", None);
        let s3 = b.scatter_named("grass", None);
        // ((trees ⊕ rocks) ⊕ grass) — the left-leaning chain a canvas builds.
        let m1 = b.node("scatter.merge", &[]);
        b.link(s1, "out", m1, "a");
        b.link(s2, "out", m1, "b");
        let m2 = b.node("scatter.merge", &[]);
        b.link(m1, "out", m2, "a");
        b.link(s3, "out", m2, "b");
        let o = b.node("output.pcg", &[]);
        b.link(m2, "out", o, "scatter");

        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert_eq!(lowered.document.layers.len(), 1, "one implicit layer");
        assert_eq!(
            lowered.document.layers[0]
                .rules
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>(),
            vec!["trees", "rocks", "grass"],
            "merge order must be a-then-b, depth-first"
        );
    }

    /// **Layers × rules**: N rules across M layers lower to the full
    /// `PcgDocument`, with each layer's name and `enabled` flag preserved.
    #[test]
    fn layers_times_rules_lower_to_the_full_document() {
        let mut b = Builder::new();
        // Layer "ground": 2 rules. Layer "canopy": 1 rule, disabled.
        let g1 = b.scatter_named("grass", None);
        let g2 = b.scatter_named("pebbles", None);
        let gm = b.node("scatter.merge", &[]);
        b.link(g1, "out", gm, "a");
        b.link(g2, "out", gm, "b");
        let ground = b.node(
            "layer.layer",
            &[("name", ParamValue::Text("ground".into()))],
        );
        b.link(gm, "out", ground, "scatter");

        let c1 = b.scatter_named("trees", None);
        let canopy = b.node(
            "layer.layer",
            &[
                ("name", ParamValue::Text("canopy".into())),
                ("enabled", ParamValue::Bool(false)),
            ],
        );
        b.link(c1, "out", canopy, "scatter");

        let lm = b.node("layer.merge", &[]);
        b.link(ground, "out", lm, "a");
        b.link(canopy, "out", lm, "b");
        let o = b.node("output.pcg", &[]);
        b.link(lm, "out", o, "layers");

        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        let doc = &lowered.document;
        assert_eq!(doc.layers.len(), 2);
        assert_eq!(doc.layers[0].name, "ground");
        assert!(doc.layers[0].enabled);
        assert_eq!(
            doc.layers[0]
                .rules
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>(),
            vec!["grass", "pebbles"]
        );
        assert_eq!(doc.layers[1].name, "canopy");
        assert!(!doc.layers[1].enabled, "the toggle must survive lowering");
        assert_eq!(doc.layers[1].rules.len(), 1);
        // The whole thing round-trips through the payload codec.
        let payload = crate::PcgAssetPayload::from_graph(&b.g, doc.clone());
        let back = crate::PcgAssetPayload::decode(&payload.encode().unwrap()).unwrap();
        assert_eq!(&back.document, doc);
        assert_eq!(back.graph().as_ref(), Some(&b.g));
        // …and re-lowering the round-tripped graph reproduces the document — the
        // property the player leans on when it re-lowers instead of trusting the
        // stored mirror.
        assert_eq!(
            lower_graph(&back.graph().unwrap(), &pcg_registry()).document,
            *doc
        );
    }

    /// The pre-P19.3 sink shape still lowers to exactly one layer named `layer`,
    /// and connecting **both** sink inputs warns on the output node with `layers`
    /// winning.
    #[test]
    fn the_legacy_scatter_input_still_lowers_and_layers_wins_a_tie() {
        let mut b = Builder::new();
        let s = b.scatter_named("only", None);
        let o = b.node("output.pcg", &[]);
        b.link(s, "out", o, "scatter");
        let doc = b.lower().document;
        assert_eq!(doc.layers.len(), 1);
        assert_eq!(doc.layers[0].name, "layer");
        assert!(doc.layers[0].enabled);
        assert_eq!(doc.layers[0].rules[0].name, "only");

        // Now also wire a layer chain: it wins, and the tie is announced.
        let s2 = b.scatter_named("winner", None);
        let l = b.node("layer.layer", &[("name", ParamValue::Text("L".into()))]);
        b.link(s2, "out", l, "scatter");
        b.link(l, "out", o, "layers");
        let lowered = b.lower();
        assert!(lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Warning && i.node == Some(o.0)));
        assert_eq!(lowered.document.layers[0].name, "L");
        assert_eq!(lowered.document.layers[0].rules[0].name, "winner");
    }

    /// Diagnostics stay **node-anchored** through the new walks: a wrong node
    /// type on a layer or scatter port errors on that node, and an unconnected
    /// merge input warns on the merge.
    #[test]
    fn the_new_walks_keep_their_diagnostics_anchored() {
        // A density node wired where a scatter belongs.
        let mut b = Builder::new();
        let c = b.node("const.density", &[]);
        let o = b.node("output.pcg", &[]);
        // Bypass the type check the edit door would apply — a hand-edited or
        // migrated document can hold this, and the lowerer must not panic.
        b.g.links.push(Link {
            from: c,
            from_port: "out".into(),
            to: o,
            to_port: "scatter".into(),
        });
        let lowered = b.lower();
        assert!(!lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error && i.node == Some(c.0)));

        // A scatter wired where a layer belongs.
        let mut b = Builder::new();
        let s = b.scatter_named("s", None);
        let o = b.node("output.pcg", &[]);
        b.g.links.push(Link {
            from: s,
            from_port: "out".into(),
            to: o,
            to_port: "layers".into(),
        });
        let lowered = b.lower();
        assert!(!lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error && i.node == Some(s.0)));

        // A half-wired merge warns on the merge and still lowers the other side.
        let mut b = Builder::new();
        let s = b.scatter_named("half", None);
        let m = b.node("scatter.merge", &[]);
        b.link(s, "out", m, "a");
        let o = b.node("output.pcg", &[]);
        b.link(m, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "a half-wired merge is not fatal");
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Warning && i.node == Some(m.0)));
        assert_eq!(lowered.document.layers[0].rules.len(), 1);

        // An empty layer warns on the layer node.
        let mut b = Builder::new();
        let l = b.node("layer.layer", &[]);
        let o = b.node("output.pcg", &[]);
        b.link(l, "out", o, "layers");
        let lowered = b.lower();
        assert!(lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Warning && i.node == Some(l.0)));
        assert!(lowered.document.layers[0].rules.is_empty());
    }

    // ── P19.4: the grammar kit ──────────────────────────────────────────────

    const RULE_TEXT: &str = "\
module Post = mesh 6f9619ff-8b86-d011-b42d-00c04fc964ff size 0.2
module Panel = mesh 6f9619ff-8b86-d011-b42d-00c04fc964f0 size 2
Fence -> Post Panel* Post
";

    impl Builder {
        /// `grammar.rules → grammar.expand ← span_node`, returning
        /// `(expand, rules, span)`.
        fn grammar(
            &mut self,
            span_type: &str,
            span_params: &[(&str, ParamValue)],
            expand_params: &[(&str, ParamValue)],
            text: &str,
        ) -> (NodeId, NodeId, NodeId) {
            let rules = self.node("grammar.rules", &[("rules", ParamValue::Text(text.into()))]);
            let span = self.node(span_type, span_params);
            let expand = self.node("grammar.expand", expand_params);
            self.link(rules, "out", expand, "rules");
            self.link(span, "out", expand, "span");
            (expand, rules, span)
        }
    }

    /// The four nodes exist, wear the right wire types, and the rule text is a
    /// **multiline** param — the node IS the editor for it.
    #[test]
    fn the_grammar_kit_is_registered_with_its_own_wires() {
        let reg = pcg_registry();
        for id in [
            "grammar.rules",
            "grammar.spline",
            "grammar.footprint",
            "grammar.expand",
        ] {
            let def = reg.get(id).unwrap_or_else(|| panic!("missing {id}"));
            assert_eq!(def.category, "grammar", "{id}");
        }
        let rules_def = reg.get("grammar.rules").unwrap();
        assert!(rules_def.inputs.is_empty());
        assert_eq!(rules_def.outputs[0].ty, PortType::Named(RULES_KEY.into()));
        assert_eq!(
            rules_def.param("rules").unwrap().ui,
            inf_graph::UiHint::Multiline,
            "the rule text must render as a text area, not a one-line input"
        );
        // The shipped default text is itself a valid grammar — a fresh node
        // teaches the DSL rather than erroring.
        let default = match &rules_def.param("rules").unwrap().default {
            ParamValue::Text(t) => t.clone(),
            other => panic!("default is {other:?}"),
        };
        let parsed = crate::grammar::Grammar::parse(&default).expect("default rules must parse");
        assert_eq!(parsed.default_axiom(), Some("Fence"));

        for id in ["grammar.spline", "grammar.footprint", "grammar.polyline"] {
            let def = reg.get(id).unwrap();
            assert!(def.inputs.is_empty(), "{id} is a source");
            assert_eq!(def.outputs[0].ty, PortType::Named(SPAN_KEY.into()), "{id}");
        }
        let ex = reg.get("grammar.expand").unwrap();
        assert_eq!(
            ex.input("span").unwrap().ty,
            PortType::Named(SPAN_KEY.into())
        );
        assert_eq!(
            ex.input("rules").unwrap().ty,
            PortType::Named(RULES_KEY.into())
        );
        // …and it emits a SCATTER, which is what lets it join the P19.3 merge
        // and layer chains with no new combinator and no new sink input.
        assert_eq!(
            ex.outputs[0].ty,
            PortType::Named(SCATTER_KEY.into()),
            "a grammar must ride the scatter wire"
        );
        // No new PortType variant was introduced for any of this.
        for def in reg.ordered() {
            for p in def.inputs.iter().chain(&def.outputs) {
                assert!(
                    matches!(p.ty, PortType::Named(_)),
                    "{}.{} is not a Named wire",
                    def.type_id,
                    p.name
                );
            }
        }
    }

    /// A grammar graph lowers to a pass carrying every authored param, and the
    /// **document is untouched** — a grammar is not a rule.
    #[test]
    fn a_grammar_graph_lowers_to_a_pass_beside_an_empty_document() {
        let mut b = Builder::new();
        let (expand, _, _) = b.grammar(
            "grammar.footprint",
            &[
                ("mode", ParamValue::Enum("Perimeter".into())),
                ("size_x", ParamValue::Float(20.0)),
                ("size_z", ParamValue::Float(12.0)),
                ("corner", ParamValue::Text("Post".into())),
                ("corner_size", ParamValue::Float(0.2)),
            ],
            &[
                ("name", ParamValue::Text("walls".into())),
                ("seed", ParamValue::Int(4242)),
                ("altitude_offset", ParamValue::Float(0.25)),
                ("ground", ParamValue::Enum("Span".into())),
            ],
            RULE_TEXT,
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");

        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert!(lowered.has_grammars());
        assert_eq!(lowered.grammars.len(), 1);
        let p = &lowered.grammars[0];
        assert_eq!(p.name, "walls");
        assert_eq!(p.seed, 4242);
        assert_eq!(p.altitude_offset, 0.25);
        assert_eq!(p.ground, crate::grammar::Ground::Span);
        assert_eq!(p.corner_module, "Post");
        assert_eq!(p.axiom, "Fence", "a blank axiom takes the first rule");
        assert_eq!(p.layer, "layer");
        assert!(p.enabled);
        assert_eq!(
            p.span,
            SpanSource::Footprint {
                size: glam::DVec2::new(20.0, 12.0),
                mode: FootprintMode::Perimeter { corner_size: 0.2 },
            }
        );
        assert_eq!(p.grammar.modules().len(), 2);
        // The scatter document is empty — a grammar contributes NO rule, and the
        // frozen `PcgDocument` wire therefore never sees it.
        assert_eq!(lowered.document.layers.len(), 1);
        assert!(lowered.document.layers[0].rules.is_empty());
    }

    #[test]
    fn a_spline_span_lowers_its_entity_ref_and_sample_density() {
        let guid = uuid::Uuid::from_u128(0xFEED);
        let mut b = Builder::new();
        let (expand, _, _) = b.grammar(
            "grammar.spline",
            &[
                ("spline", ParamValue::Text(guid.to_string())),
                ("samples_per_segment", ParamValue::Int(32)),
            ],
            &[],
            RULE_TEXT,
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert_eq!(
            lowered.grammars[0].span,
            SpanSource::Spline {
                entity: Some(guid),
                samples_per_segment: 32
            }
        );

        // A blank ref means "this entity's own spline" — the zero-config case.
        let mut b = Builder::new();
        let (expand, _, _) = b.grammar("grammar.spline", &[], &[], RULE_TEXT);
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok);
        assert!(matches!(
            lowered.grammars[0].span,
            SpanSource::Spline { entity: None, .. }
        ));

        // A malformed ref warns on the span node and falls back to self.
        let mut b = Builder::new();
        let (expand, _, span) = b.grammar(
            "grammar.spline",
            &[("spline", ParamValue::Text("not-a-guid".into()))],
            &[],
            RULE_TEXT,
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "a bad GUID is not fatal");
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Warning && i.node == Some(span.0)));
    }

    #[test]
    fn rows_mode_lowers_its_count_and_axis() {
        let mut b = Builder::new();
        let (expand, _, span) = b.grammar(
            "grammar.footprint",
            &[
                ("mode", ParamValue::Enum("Rows".into())),
                ("rows", ParamValue::Int(7)),
                ("row_axis", ParamValue::Enum("Z".into())),
                ("corner", ParamValue::Text("Post".into())),
            ],
            &[],
            RULE_TEXT,
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok);
        assert_eq!(
            lowered.grammars[0].span,
            SpanSource::Footprint {
                size: glam::DVec2::ZERO,
                mode: FootprintMode::Rows {
                    rows: 7,
                    axis: RowAxis::Z
                },
            }
        );
        // Rows have no corners; the ignored param says so on its own node.
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.node == Some(span.0) && i.message.contains("no corners")));
    }

    /// **The parse error is anchored on the rules node and carries the DSL's own
    /// `line:col`** — the diagnostics contract the WGSL emitter and the density
    /// walk already keep.
    #[test]
    fn grammar_diagnostics_are_anchored() {
        // A rule text that does not parse.
        let mut b = Builder::new();
        let (expand, rules, _) = b.grammar(
            "grammar.footprint",
            &[],
            &[],
            "module P = size 2\nWall -> P\nBad -> Missing\n",
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(!lowered.ok);
        let err = lowered
            .issues
            .iter()
            .find(|i| i.severity == PcgSeverity::Error)
            .unwrap();
        assert_eq!(err.node, Some(rules.0));
        assert!(err.message.contains("line 3"), "{}", err.message);
        assert!(err.message.contains("has no size"), "{}", err.message);
        assert!(lowered.grammars.is_empty(), "a broken pass is not built");

        // No rules connected → anchored error on the expand node.
        let mut b = Builder::new();
        let span = b.node("grammar.footprint", &[]);
        let expand = b.node("grammar.expand", &[]);
        b.link(span, "out", expand, "span");
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(!lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error && i.node == Some(expand.0)));

        // No span connected → likewise.
        let mut b = Builder::new();
        let rules = b.node(
            "grammar.rules",
            &[("rules", ParamValue::Text(RULE_TEXT.into()))],
        );
        let expand = b.node("grammar.expand", &[]);
        b.link(rules, "out", expand, "rules");
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(!lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error && i.node == Some(expand.0)));

        // Wrong node types on either input error on the OFFENDING node.
        for (bad_type, port) in [("const.density", "rules"), ("const.density", "span")] {
            let mut b = Builder::new();
            let bad = b.node(bad_type, &[]);
            let rules = b.node(
                "grammar.rules",
                &[("rules", ParamValue::Text(RULE_TEXT.into()))],
            );
            let span = b.node("grammar.footprint", &[]);
            let expand = b.node("grammar.expand", &[]);
            b.link(rules, "out", expand, "rules");
            b.link(span, "out", expand, "span");
            let o = b.node("output.pcg", &[]);
            b.link(expand, "out", o, "scatter");
            // The edit door forbids the mismatch; a hand-edited document can
            // still hold it, and the lowerer must diagnose rather than panic.
            b.g.links.retain(|l| !(l.to == expand && l.to_port == port));
            b.g.links.push(Link {
                from: bad,
                from_port: "out".into(),
                to: expand,
                to_port: port.into(),
            });
            let lowered = b.lower();
            assert!(!lowered.ok, "{port}");
            assert!(
                lowered
                    .issues
                    .iter()
                    .any(|i| i.severity == PcgSeverity::Error && i.node == Some(bad.0)),
                "{port} did not anchor on the offending node"
            );
        }

        // Gap symbols are a WARNING listing them — legal, and visible.
        let mut b = Builder::new();
        let (expand, rules, _) = b.grammar(
            "grammar.footprint",
            &[],
            &[],
            "module P = size 2\nWall -> P Gap[1] Spacer[2]\n",
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok);
        let warn = lowered
            .issues
            .iter()
            .find(|i| i.node == Some(rules.0) && i.severity == PcgSeverity::Warning)
            .unwrap();
        assert!(warn.message.contains("Gap"), "{}", warn.message);
        assert!(warn.message.contains("Spacer"), "{}", warn.message);

        // An undeclared corner module warns on the footprint node.
        let mut b = Builder::new();
        let (expand, _, span) = b.grammar(
            "grammar.footprint",
            &[("corner", ParamValue::Text("Turret".into()))],
            &[],
            RULE_TEXT,
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.node == Some(span.0) && i.message.contains("Turret")));

        // An axiom naming nothing warns on the expand node.
        let mut b = Builder::new();
        let (expand, _, _) = b.grammar(
            "grammar.footprint",
            &[],
            &[("axiom", ParamValue::Text("Nope".into()))],
            RULE_TEXT,
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.node == Some(expand.0) && i.message.contains("axiom")));
    }

    /// **A grammar mixes into the P19.3 chains untouched**: merged beside a
    /// scatter, wrapped in a layer, and taking that layer's name and toggle.
    #[test]
    fn a_grammar_merges_into_layers_and_rules_like_any_scatter() {
        let mut b = Builder::new();
        let s = b.scatter_named("grass", None);
        let (expand, _, _) = b.grammar("grammar.footprint", &[], &[], RULE_TEXT);
        let m = b.node("scatter.merge", &[]);
        b.link(s, "out", m, "a");
        b.link(expand, "out", m, "b");
        let l = b.node(
            "layer.layer",
            &[
                ("name", ParamValue::Text("village".into())),
                ("enabled", ParamValue::Bool(false)),
            ],
        );
        b.link(m, "out", l, "scatter");
        let o = b.node("output.pcg", &[]);
        b.link(l, "out", o, "layers");

        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        // The scatter half is one rule in one layer …
        assert_eq!(lowered.document.layers.len(), 1);
        assert_eq!(lowered.document.layers[0].name, "village");
        assert_eq!(
            lowered.document.layers[0]
                .rules
                .iter()
                .map(|r| r.name.as_str())
                .collect::<Vec<_>>(),
            vec!["grass"]
        );
        // … and the grammar half carries the SAME layer identity.
        assert_eq!(lowered.grammars.len(), 1);
        assert_eq!(lowered.grammars[0].layer, "village");
        assert!(
            !lowered.grammars[0].enabled,
            "a disabled layer must disable its grammars too"
        );
    }

    /// Two grammars merge in canvas order, `a` before `b` — the same
    /// depth-first flattening the rule list gets.
    #[test]
    fn grammar_passes_keep_their_canvas_order() {
        let mut b = Builder::new();
        let (e1, _, _) = b.grammar(
            "grammar.footprint",
            &[],
            &[("name", ParamValue::Text("one".into()))],
            RULE_TEXT,
        );
        let (e2, _, _) = b.grammar(
            "grammar.footprint",
            &[],
            &[("name", ParamValue::Text("two".into()))],
            RULE_TEXT,
        );
        let (e3, _, _) = b.grammar(
            "grammar.footprint",
            &[],
            &[("name", ParamValue::Text("three".into()))],
            RULE_TEXT,
        );
        let m1 = b.node("scatter.merge", &[]);
        b.link(e1, "out", m1, "a");
        b.link(e2, "out", m1, "b");
        let m2 = b.node("scatter.merge", &[]);
        b.link(m1, "out", m2, "a");
        b.link(e3, "out", m2, "b");
        let o = b.node("output.pcg", &[]);
        b.link(m2, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert_eq!(
            lowered
                .grammars
                .iter()
                .map(|g| g.name.as_str())
                .collect::<Vec<_>>(),
            vec!["one", "two", "three"]
        );
    }

    /// **The permanent transpile discipline, and the no-schema-movement claim.**
    /// A grammar graph survives the `.inf_pcg` payload codec and re-lowers to
    /// the identical passes — the property the player leans on, since it
    /// re-lowers the stored graph rather than trusting the document mirror. And
    /// the encoded payload of a grammar-only graph is byte-identical to that of
    /// the same graph with the grammar chain removed *plus its graph JSON*, i.e.
    /// nothing about the grammar reached the frozen document wire.
    #[test]
    fn a_grammar_graph_round_trips_through_the_payload_and_re_lowers() {
        let mut b = Builder::new();
        let (expand, _, _) = b.grammar(
            "grammar.spline",
            &[("samples_per_segment", ParamValue::Int(24))],
            &[
                ("name", ParamValue::Text("fence".into())),
                ("seed", ParamValue::Int(31337)),
            ],
            RULE_TEXT,
        );
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);

        let payload = crate::PcgAssetPayload::from_graph(&b.g, lowered.document.clone());
        let bytes = payload.encode().unwrap();
        let back = crate::PcgAssetPayload::decode(&bytes).unwrap();
        assert_eq!(back.schema_version, 2, "no schema bump was taken");
        let re = lower_graph(&back.graph().unwrap(), &pcg_registry());
        assert_eq!(re.document, lowered.document);
        assert_eq!(re.grammars, lowered.grammars);
        assert_eq!(re.grammars[0].name, "fence");
        assert_eq!(re.grammars[0].seed, 31337);
        // Re-encoding is byte-identical, so a `.inf_pcg`'s content hash is stable.
        assert_eq!(bytes, back.encode().unwrap());

        // The document mirror of a grammar-only graph is EXACTLY an empty
        // one-layer document — the frozen wire never learned a new field.
        let empty = crate::PcgAssetPayload::new(lowered.document.clone());
        assert_eq!(
            empty.encode().unwrap(),
            crate::PcgAssetPayload::new(PcgDocument::single_layer("layer", Vec::new()))
                .encode()
                .unwrap()
        );
    }

    /// **The cook's module edge survives a graph that does not lower.**
    ///
    /// `grammar_mesh_refs` reads the `grammar.rules` nodes, not the lowered
    /// passes — because a Span pin nobody has dragged yet is an ordinary
    /// mid-authoring state, and a cook that dropped the meshes for it would ship
    /// a wall with pieces missing and no advisory. Deduplicated and sorted, so
    /// the closure and the advisory are both deterministic.
    #[test]
    fn grammar_mesh_refs_reads_the_palette_not_the_lowered_passes() {
        let post = uuid::Uuid::parse_str("6f9619ff-8b86-d011-b42d-00c04fc964ff").unwrap();
        let panel = uuid::Uuid::parse_str("6f9619ff-8b86-d011-b42d-00c04fc964f0").unwrap();

        // A fully wired graph declares both.
        let mut b = Builder::new();
        let (expand, _, _) = b.grammar("grammar.footprint", &[], &[], RULE_TEXT);
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        assert!(b.lower().ok);
        assert_eq!(grammar_mesh_refs(&b.g, &b.reg), vec![panel, post]);

        // …and so does a graph whose Span pin was never connected, which does
        // NOT lower. This is the finding: the edge must not depend on wiring.
        let mut b = Builder::new();
        let rules = b.node(
            "grammar.rules",
            &[("rules", ParamValue::Text(RULE_TEXT.into()))],
        );
        let expand = b.node("grammar.expand", &[]);
        b.link(rules, "out", expand, "rules");
        let o = b.node("output.pcg", &[]);
        b.link(expand, "out", o, "scatter");
        let lowered = b.lower();
        assert!(!lowered.ok, "an unconnected Span must still fail to lower");
        assert!(lowered.grammars.is_empty(), "…and produce no pass");
        assert_eq!(
            grammar_mesh_refs(&b.g, &b.reg),
            vec![panel, post],
            "the cook edge must survive a graph that does not lower"
        );

        // A rules node wired to NOTHING at all still declares its meshes —
        // over-inclusive on purpose (bytes, not a hole in a wall).
        let mut b = Builder::new();
        b.node(
            "grammar.rules",
            &[("rules", ParamValue::Text(RULE_TEXT.into()))],
        );
        assert_eq!(grammar_mesh_refs(&b.g, &b.reg), vec![panel, post]);

        // **Deduplicated across nodes, and sorted.** Two rules nodes naming the
        // same mesh contribute one edge, in a stable order.
        let mut b = Builder::new();
        for _ in 0..3 {
            b.node(
                "grammar.rules",
                &[("rules", ParamValue::Text(RULE_TEXT.into()))],
            );
        }
        let refs = grammar_mesh_refs(&b.g, &b.reg);
        assert_eq!(refs, vec![panel, post], "not deduplicated or not sorted");
        assert!(refs.windows(2).all(|w| w[0] < w[1]));

        // A rule text that does not parse contributes nothing rather than
        // panicking …
        let mut b = Builder::new();
        b.node(
            "grammar.rules",
            &[("rules", ParamValue::Text("A -> ) (".into()))],
        );
        assert!(grammar_mesh_refs(&b.g, &b.reg).is_empty());
        // … and a graph with no grammar at all contributes nothing, so the
        // advisory built on this stays silent for every pre-P19.4 `.inf_pcg`.
        let mut b = Builder::new();
        let s = b.scatter_named("grass", None);
        let o = b.node("output.pcg", &[]);
        b.link(s, "out", o, "scatter");
        assert!(grammar_mesh_refs(&b.g, &b.reg).is_empty());
        // The default (unmodified) rules node names no mesh, so dropping one on
        // the canvas does not invent a dependency.
        let mut b = Builder::new();
        b.node("grammar.rules", &[]);
        assert!(grammar_mesh_refs(&b.g, &b.reg).is_empty());
    }

    /// One density subgraph feeding **two** scatter nodes is a legal diamond, not
    /// a cycle — the walk must not read the shared node as one.
    #[test]
    fn a_density_shared_by_two_scatters_is_not_a_cycle() {
        let mut b = Builder::new();
        let n = b.node("noise.fbm", &[("seed", ParamValue::Int(3))]);
        let s1 = b.scatter_named("a", Some(n));
        let s2 = b.scatter_named("b", Some(n));
        let m = b.node("scatter.merge", &[]);
        b.link(s1, "out", m, "a");
        b.link(s2, "out", m, "b");
        let o = b.node("output.pcg", &[]);
        b.link(m, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        let rules = &lowered.document.layers[0].rules;
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].sampler, rules[1].sampler);
        assert!(matches!(rules[0].sampler, SamplerDef::Noise(_)));
    }

    // ── P19.5: the building kit ─────────────────────────────────────────────

    impl Builder {
        /// `building.archetype → building.plan`, returning `(plan, archetype)`.
        fn building(
            &mut self,
            arch_params: &[(&str, ParamValue)],
            plan_params: &[(&str, ParamValue)],
        ) -> (NodeId, NodeId) {
            let arch = self.node("building.archetype", arch_params);
            let plan = self.node("building.plan", plan_params);
            self.link(arch, "out", plan, "archetype");
            (plan, arch)
        }
    }

    /// The two nodes exist, wear the right wires, and — the load-bearing one —
    /// `building.plan` outputs a **SCATTER**, so it joins the merge and layer
    /// chains that already exist rather than needing a third sink input.
    #[test]
    fn the_building_kit_is_registered_on_the_scatter_wire() {
        let reg = pcg_registry();
        let arch = reg
            .get("building.archetype")
            .expect("missing archetype node");
        let plan = reg.get("building.plan").expect("missing plan node");
        assert_eq!(arch.category, "building");
        assert_eq!(plan.category, "building");
        assert!(arch.inputs.is_empty());
        assert_eq!(arch.outputs[0].ty, PortType::Named(BUILDING_KEY.into()));
        assert_eq!(plan.outputs[0].ty, PortType::Named(SCATTER_KEY.into()));
        // The lot pin takes the SAME span wire the grammar kit produces — which
        // is what makes a spline-derived lot free rather than a second concept.
        let lot = plan
            .inputs
            .iter()
            .find(|p| p.name == "lot")
            .expect("lot pin");
        assert_eq!(lot.ty, PortType::Named(SPAN_KEY.into()));
        let a = plan
            .inputs
            .iter()
            .find(|p| p.name == "archetype")
            .expect("archetype pin");
        assert_eq!(a.ty, PortType::Named(BUILDING_KEY.into()));
        // Every shipped palette is offered, in the canonical order.
        assert_eq!(
            arch.param("archetype").unwrap().options,
            ArchetypeId::ALL
                .iter()
                .map(|a| a.name().to_string())
                .collect::<Vec<_>>()
        );
    }

    /// A building lowers to a pass carrying every authored param, contributes
    /// **no rule**, and leaves the document exactly as an empty graph would.
    #[test]
    fn a_building_lowers_to_a_pass_beside_an_empty_document() {
        let mut b = Builder::new();
        let (plan, _) = b.building(
            &[
                ("archetype", ParamValue::Enum("Hotel".into())),
                ("floors", ParamValue::Int(5)),
                ("furnish", ParamValue::Bool(false)),
            ],
            &[
                ("name", ParamValue::Text("tower".into())),
                ("seed", ParamValue::Int(19)),
                ("size_x", ParamValue::Float(40.0)),
                ("size_z", ParamValue::Float(22.0)),
                ("altitude_offset", ParamValue::Float(0.25)),
                ("ground", ParamValue::Enum("Span".into())),
            ],
        );
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert!(lowered.has_buildings());
        assert_eq!(lowered.buildings.len(), 1);
        let p = &lowered.buildings[0];
        assert_eq!(p.name, "tower");
        assert_eq!(p.archetype, ArchetypeId::Hotel);
        assert_eq!(p.floors, 5);
        assert!(!p.furnish);
        assert_eq!(p.seed, 19);
        assert_eq!(p.size, glam::DVec2::new(40.0, 22.0));
        assert_eq!(p.ground, Ground::Span);
        assert_eq!(p.altitude_offset, 0.25);
        assert!(p.lot.is_none());
        assert_eq!(p.layer, "layer");
        assert!(p.enabled);
        // The document is what an empty one-layer graph lowers to: a building is
        // not a scatter rule.
        assert_eq!(lowered.document.layers.len(), 1);
        assert!(lowered.document.layers[0].rules.is_empty());
    }

    /// A connected span becomes the lot, and it is the same `SpanSource` the
    /// grammar kit produces — no second span concept.
    #[test]
    fn a_span_on_the_lot_pin_becomes_the_footprint() {
        let mut b = Builder::new();
        let (plan, _) = b.building(&[], &[]);
        let span = b.node(
            "grammar.footprint",
            &[
                ("size_x", ParamValue::Float(18.0)),
                ("size_z", ParamValue::Float(12.0)),
            ],
        );
        b.link(span, "out", plan, "lot");
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert!(matches!(
            lowered.buildings[0].lot,
            Some(SpanSource::Footprint { size, .. }) if size == glam::DVec2::new(18.0, 12.0)
        ));

        // A spline works the same way — the closure P19.4's remainder named.
        let mut b = Builder::new();
        let (plan, _) = b.building(&[], &[]);
        let span = b.node("grammar.spline", &[]);
        b.link(span, "out", plan, "lot");
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        assert!(matches!(
            lowered.buildings[0].lot,
            Some(SpanSource::Spline { entity: None, .. })
        ));
    }

    /// Buildings merge and layer exactly like scatters and grammars: a disabled
    /// layer disables them, and the canvas order is the pass order.
    #[test]
    fn buildings_join_the_merge_and_layer_chains() {
        let mut b = Builder::new();
        let (b1, _) = b.building(&[], &[("name", ParamValue::Text("first".into()))]);
        let (b2, _) = b.building(
            &[("archetype", ParamValue::Enum("Shop".into()))],
            &[("name", ParamValue::Text("second".into()))],
        );
        let s = b.scatter_named("trees", None);
        let m1 = b.node("scatter.merge", &[]);
        b.link(b1, "out", m1, "a");
        b.link(b2, "out", m1, "b");
        let m2 = b.node("scatter.merge", &[]);
        b.link(m1, "out", m2, "a");
        b.link(s, "out", m2, "b");
        let l = b.node(
            "layer.layer",
            &[
                ("name", ParamValue::Text("town".into())),
                ("enabled", ParamValue::Bool(false)),
            ],
        );
        b.link(m2, "out", l, "scatter");
        let o = b.node("output.pcg", &[]);
        b.link(l, "out", o, "layers");
        let lowered = b.lower();
        assert!(lowered.ok, "{:?}", lowered.issues);
        let names: Vec<&str> = lowered.buildings.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, vec!["first", "second"], "canvas order");
        assert_eq!(lowered.buildings[1].archetype, ArchetypeId::Shop);
        for p in &lowered.buildings {
            assert_eq!(p.layer, "town");
            assert!(!p.enabled, "a disabled layer disables its buildings");
        }
        // The scatter beside them is untouched.
        assert_eq!(lowered.document.layers[0].rules.len(), 1);
        assert_eq!(lowered.document.layers[0].rules[0].name, "trees");
    }

    /// Every failure is node-anchored, and only the missing archetype is fatal:
    /// an unknown archetype NAME is a migration artefact, not an authoring
    /// mistake, so it warns and falls back.
    #[test]
    fn building_diagnostics_are_anchored_and_fail_closed() {
        // No archetype connected: an error on the plan node, and no pass.
        let mut b = Builder::new();
        let plan = b.node("building.plan", &[]);
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        assert!(!lowered.ok);
        assert!(lowered.buildings.is_empty());
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error
                && i.node == Some(plan.0)
                && i.message.contains("Building Archetype")));

        // A wrong node type on the archetype pin errors on THAT node.
        let mut b = Builder::new();
        let plan = b.node("building.plan", &[]);
        let rules = b.node("grammar.rules", &[]);
        b.g.links.push(Link {
            from: rules,
            from_port: "out".into(),
            to: plan,
            to_port: "archetype".into(),
        });
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        assert!(!lowered.ok);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error && i.node == Some(rules.0)));

        // A wrong node type on the LOT pin errors on that node too.
        let mut b = Builder::new();
        let (plan, _) = b.building(&[], &[]);
        let c = b.node("const.density", &[]);
        b.g.links.push(Link {
            from: c,
            from_port: "out".into(),
            to: plan,
            to_port: "lot".into(),
        });
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        assert!(!lowered.ok);
        assert!(lowered.buildings.is_empty());
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Error && i.node == Some(c.0)));

        // An unknown archetype name WARNS on the archetype node and falls back.
        // The edit door's `sanitize` resets an out-of-set choice, so the only
        // way to hold one is a hand-edited or migrated document — which is
        // exactly the case this branch exists for, and how it is driven here.
        let mut b = Builder::new();
        let (plan, arch) = b.building(&[], &[]);
        b.g.nodes
            .get_mut(&arch)
            .expect("archetype node")
            .params
            .insert("archetype".into(), ParamValue::Enum("Castle".into()));
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        assert!(lowered.ok, "an unknown palette is not fatal");
        assert_eq!(lowered.buildings[0].archetype, ArchetypeId::ALL[0]);
        assert!(lowered
            .issues
            .iter()
            .any(|i| i.severity == PcgSeverity::Warning
                && i.node == Some(arch.0)
                && i.message.contains("Castle")));
    }

    /// The payload round trip: a building-only graph re-lowers to identical
    /// passes, and its stored `PcgDocument` is what an empty one-layer graph
    /// stores — the P19.4 statement, extended to the third generator.
    #[test]
    fn a_building_graph_re_lowers_identically_and_stores_an_empty_document() {
        let mut b = Builder::new();
        let (plan, _) = b.building(
            &[("archetype", ParamValue::Enum("Estate".into()))],
            &[("seed", ParamValue::Int(88))],
        );
        let o = b.node("output.pcg", &[]);
        b.link(plan, "out", o, "scatter");
        let lowered = b.lower();
        let payload = crate::asset::PcgAssetPayload::from_graph(&b.g, lowered.document.clone());
        let bytes = payload.encode().expect("encode");
        let back = crate::asset::PcgAssetPayload::decode(&bytes).expect("decode");
        let graph = back.graph().expect("graph");
        let again = lower_graph(&graph, &pcg_registry());
        assert_eq!(again.buildings, lowered.buildings);
        assert_eq!(again.document, lowered.document);
        assert!(lowered.document.layers[0].rules.is_empty());
    }
}
