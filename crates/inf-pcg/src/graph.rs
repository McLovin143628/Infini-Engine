//! The PCG editor node kit + `lower_graph` (P10.5b).
//!
//! The `.inf_pcg` editor authors a pure-data node DAG over the shared
//! [`inf_graph`] substrate — the same substrate the Blueprint and Material
//! editors use — with a PCG-specific [`pcg_registry`] and no exec pins. A single
//! `output.pcg` SINK collects one scatter chain; [`lower_graph`] walks the graph
//! backward from that sink into the stable, serialization-locked
//! [`PcgDocument`](crate::rules::PcgDocument) runtime model (the way the material
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
//! [`SamplerDef::Mask`](crate::rules::SamplerDef::Mask) carries **pixels**, not a
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

use crate::noise::ValueNoise;
use crate::rules::{PcgDocument, PcgKind, PcgLayer, PcgRule, SamplerDef};
use crate::scatter::{RotationMode, ScatterParams};

/// A `[0,1]` density-field wire.
pub const DENSITY_KEY: &str = "density";
/// A scatter-pass wire.
pub const SCATTER_KEY: &str = "scatter";
/// A layer wire (a named, toggleable group of scatter passes) — P19.3.
pub const LAYER_KEY: &str = "layer";

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
/// masks (P19.3), combinators, the scatter pass + its merge, the layer wrapper +
/// its merge, and one output sink.
pub fn pcg_registry() -> NodeRegistry {
    let mut reg = NodeRegistry::new();
    reg.register_all(source_nodes());
    reg.register_all(filter_nodes());
    reg.register_all(mask_nodes());
    reg.register_all(combine_nodes());
    reg.register(scatter_node());
    reg.register(scatter_merge_node());
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

/// The result of lowering a PCG graph: the runtime document plus node-anchored
/// diagnostics. `ok` is true when no error-severity issue was raised.
#[derive(Debug, Clone)]
pub struct LoweredPcg {
    pub document: PcgDocument,
    pub issues: Vec<PcgGraphIssue>,
    pub ok: bool,
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
        return LoweredPcg::finish(PcgDocument::default(), cx.issues);
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
            return LoweredPcg::finish(PcgDocument::default(), cx.issues);
        }
    };

    LoweredPcg::finish(PcgDocument { layers }, cx.issues)
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
                vec![PcgLayer {
                    name: if name.trim().is_empty() {
                        "layer".into()
                    } else {
                        name
                    },
                    enabled: pb_default(&params, "enabled", true),
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

impl LoweredPcg {
    fn finish(document: PcgDocument, issues: Vec<PcgGraphIssue>) -> Self {
        let ok = !issues.iter().any(|i| i.severity == PcgSeverity::Error);
        Self {
            document,
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
}
