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
//! Two `Named` port types carry the dataflow (the substrate has no domain
//! variants of its own):
//!
//! * [`DENSITY`] — a `[0,1]` density field, produced by sources
//!   (`const.density`, `noise.fbm`), terrain filters (`filter.slope`,
//!   `filter.altitude`), and combinators (`combine.*`).
//! * [`SCATTER`] — a scatter pass; the single [`scatter.scatter`](scatter) node
//!   consumes a density and emits it into the `output.pcg` sink.
//!
//! ## Lowering semantics (v1)
//!
//! `lower_graph` produces a **single-layer, single-rule** [`PcgDocument`]: the
//! one `scatter.scatter` node feeding `output.pcg` becomes the rule, its density
//! input becomes the [`SamplerDef`](crate::rules::SamplerDef) tree, and its
//! `mesh`/`weight` params become the one-entry [`PcgKind`](crate::rules::PcgKind)
//! palette. **Multi-rule / multi-layer** authoring (several scatter nodes, or an
//! explicit layer node) is the documented next step — the runtime model already
//! supports arbitrarily many `PcgLayer`/`PcgRule`s, so it is purely an editor +
//! lowerer extension (walk every scatter node into its own rule).
//!
//! ## Deferred nodes
//!
//! A **mask** node (`crate::sampler::MaskImage`) needs an image/texture input,
//! which the pure param model can't carry yet — it is intentionally omitted from
//! the kit until a texture-input pin lands (mirroring the material editor's
//! texture pin). The runtime `SamplerDef::Mask` variant stays available for code.

use inf_graph::{
    Graph, NodeDef, NodeId, NodeRegistry, ParamDef, ParamValue, PortDef, PortType, SINK,
};

use crate::noise::ValueNoise;
use crate::rules::{PcgDocument, PcgKind, PcgLayer, PcgRule, SamplerDef};
use crate::scatter::{RotationMode, ScatterParams};

/// A `[0,1]` density-field wire.
pub const DENSITY_KEY: &str = "density";
/// A scatter-pass wire (feeds the sink).
pub const SCATTER_KEY: &str = "scatter";

/// A density-field port.
fn density(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(DENSITY_KEY.into()))
}

/// A scatter-pass port.
fn scatter(name: &str) -> PortDef {
    PortDef::new(name, PortType::Named(SCATTER_KEY.into()))
}

/// The complete PCG node palette (density sources, terrain filters,
/// combinators, one scatter node, one output sink).
pub fn pcg_registry() -> NodeRegistry {
    let mut reg = NodeRegistry::new();
    reg.register_all(source_nodes());
    reg.register_all(filter_nodes());
    reg.register_all(combine_nodes());
    reg.register(scatter_node());
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

/// The PCG output sink: collects one scatter pass.
fn output_node() -> NodeDef {
    NodeDef::new("output.pcg", "PCG Output", "output")
        .described("The final scatter this volume evaluates")
        .with_inputs(vec![scatter("scatter")])
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
    let mut issues = Vec::new();

    // Locate the sink(s). BTreeMap iteration → the lowest NodeId wins.
    let outputs: Vec<NodeId> = graph
        .nodes
        .values()
        .filter(|n| n.type_id == "output.pcg")
        .map(|n| n.id)
        .collect();
    let Some(&output) = outputs.first() else {
        issues.push(PcgGraphIssue::error(
            None,
            "graph has no PCG Output node — add one",
        ));
        return LoweredPcg::finish(PcgDocument::default(), issues);
    };
    for &extra in outputs.iter().skip(1) {
        issues.push(PcgGraphIssue::warn(
            Some(extra),
            "multiple PCG Output nodes — only the first is evaluated",
        ));
    }

    // The scatter feeding the sink.
    let Some(scatter_link) = graph.link_into(output, "scatter") else {
        issues.push(PcgGraphIssue::error(
            Some(output),
            "PCG Output has no Scatter connected",
        ));
        return LoweredPcg::finish(PcgDocument::default(), issues);
    };
    let scatter_id = scatter_link.from;
    let Some(scatter_node) = graph.node(scatter_id) else {
        return LoweredPcg::finish(PcgDocument::default(), issues);
    };
    if scatter_node.type_id != "scatter.scatter" {
        issues.push(PcgGraphIssue::error(
            Some(scatter_id),
            "PCG Output must be fed by a Scatter node",
        ));
        return LoweredPcg::finish(PcgDocument::default(), issues);
    }

    let params = resolved(reg, scatter_node);
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

    // The density chain feeding the scatter.
    let sampler = match graph.link_into(scatter_id, "density") {
        Some(link) => lower_density(graph, reg, link.from, &mut issues, &mut Vec::new()),
        None => {
            issues.push(PcgGraphIssue::warn(
                Some(scatter_id),
                "Scatter has no density input — placing at full density everywhere",
            ));
            SamplerDef::Constant(1.0)
        }
    };

    // The one-entry kind palette from the scatter node's mesh/weight params.
    let mesh = parse_guid(&ptext(&params, "mesh"));
    let weight = pf(&params, "weight");
    let kinds = vec![PcgKind { mesh, weight }];

    let rule = PcgRule {
        name: "scatter".into(),
        sampler,
        scatter,
        kinds,
    };
    let document = PcgDocument {
        layers: vec![PcgLayer {
            name: "layer".into(),
            enabled: true,
            rules: vec![rule],
        }],
    };
    LoweredPcg::finish(document, issues)
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

/// Recursively lower one density-chain node into a [`SamplerDef`]. A missing
/// input defaults to `Constant(1.0)` (identity for the common `multiply`), with
/// a node-anchored warning. Cycles (should not occur — the edit door forbids
/// them) short-circuit to `Constant(0.0)`.
fn lower_density(
    graph: &Graph,
    reg: &NodeRegistry,
    node_id: NodeId,
    issues: &mut Vec<PcgGraphIssue>,
    visiting: &mut Vec<NodeId>,
) -> SamplerDef {
    if visiting.contains(&node_id) {
        issues.push(PcgGraphIssue::error(
            Some(node_id),
            "cycle in density chain",
        ));
        return SamplerDef::Constant(0.0);
    }
    let Some(node) = graph.node(node_id) else {
        return SamplerDef::Constant(1.0);
    };
    visiting.push(node_id);

    let params = resolved(reg, node);
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
        "combine.multiply" => {
            let a = lower_input(graph, reg, node_id, "a", issues, visiting);
            let b = lower_input(graph, reg, node_id, "b", issues, visiting);
            SamplerDef::Multiply(Box::new(a), Box::new(b))
        }
        "combine.max" => {
            let a = lower_input(graph, reg, node_id, "a", issues, visiting);
            let b = lower_input(graph, reg, node_id, "b", issues, visiting);
            SamplerDef::Max(Box::new(a), Box::new(b))
        }
        "combine.min" => {
            let a = lower_input(graph, reg, node_id, "a", issues, visiting);
            let b = lower_input(graph, reg, node_id, "b", issues, visiting);
            SamplerDef::Min(Box::new(a), Box::new(b))
        }
        "combine.invert" => {
            let a = lower_input(graph, reg, node_id, "in", issues, visiting);
            SamplerDef::Invert(Box::new(a))
        }
        other => {
            issues.push(PcgGraphIssue::error(
                Some(node_id),
                format!("`{other}` does not produce a density"),
            ));
            SamplerDef::Constant(0.0)
        }
    };

    visiting.pop();
    out
}

/// Lower the density feeding `(node, port)`; an unconnected port warns and
/// falls back to `Constant(1.0)`.
fn lower_input(
    graph: &Graph,
    reg: &NodeRegistry,
    node: NodeId,
    port: &str,
    issues: &mut Vec<PcgGraphIssue>,
    visiting: &mut Vec<NodeId>,
) -> SamplerDef {
    match graph.link_into(node, port) {
        Some(link) => lower_density(graph, reg, link.from, issues, visiting),
        None => {
            issues.push(PcgGraphIssue::warn(
                Some(node),
                format!("input `{port}` is unconnected — using constant 1.0"),
            ));
            SamplerDef::Constant(1.0)
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
        assert_eq!(out.inputs.len(), 1);
        for id in [
            "const.density",
            "noise.fbm",
            "filter.slope",
            "filter.altitude",
            "combine.multiply",
            "combine.max",
            "combine.min",
            "combine.invert",
            "scatter.scatter",
            "output.pcg",
        ] {
            assert!(reg.contains(id), "missing {id}");
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
}
