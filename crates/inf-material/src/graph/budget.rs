//! Material complexity-budget analysis (P13.2.3).
//!
//! A lightweight, GPU-free pass over a compiled material graph that estimates
//! shader cost and compares it against advisory budgets. The report is surfaced
//! in [`MaterialCompile`](super::emit::MaterialCompile) and shown in the editor's
//! stat strip. Budgets are **advisory in v1** — exceeding one is a warning to
//! the author, never a hard compile error; per-project configurable budgets and
//! reachability-pruned counting are documented follow-ups. Counting is over all
//! authored nodes in the graph (deterministic, order-independent).

use inf_graph::Graph;

/// Advisory ceilings for a single material. The defaults are a first-pass
/// calibration for the P7 node kit + P13.2 slabs; they are intentionally
/// generous and meant to flag runaway graphs, not to gate normal authoring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MaterialBudget {
    /// Total authored nodes.
    pub max_nodes: u32,
    /// `slab.*` layer nodes.
    pub max_slabs: u32,
    /// `tex.sample` nodes (texture reads).
    pub max_textures: u32,
    /// Estimated fragment ALU ops (sum of per-node cost weights).
    pub max_alu_ops: u32,
}

impl Default for MaterialBudget {
    fn default() -> Self {
        Self {
            max_nodes: 64,
            max_slabs: 8,
            max_textures: 16,
            max_alu_ops: 512,
        }
    }
}

/// The result of a complexity analysis: the measured counts plus the budget they
/// were compared against. Over-budget queries are derived, so the report stays a
/// plain value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComplexityReport {
    pub nodes: u32,
    pub slabs: u32,
    pub textures: u32,
    pub est_alu_ops: u32,
    pub budget: MaterialBudget,
}

impl ComplexityReport {
    pub fn nodes_over(&self) -> bool {
        self.nodes > self.budget.max_nodes
    }
    pub fn slabs_over(&self) -> bool {
        self.slabs > self.budget.max_slabs
    }
    pub fn textures_over(&self) -> bool {
        self.textures > self.budget.max_textures
    }
    pub fn alu_over(&self) -> bool {
        self.est_alu_ops > self.budget.max_alu_ops
    }
    /// True when any measured metric exceeds its budget.
    pub fn over_budget(&self) -> bool {
        self.nodes_over() || self.slabs_over() || self.textures_over() || self.alu_over()
    }
}

/// Estimated fragment-ALU cost weight for one node kind. Cheap arithmetic is 1;
/// transcendentals, procedural generators, and texture reads are weighted up;
/// inputs/constants/output are free (loads / immediates / the final store).
fn node_cost(type_id: &str) -> u32 {
    match type_id {
        t if t.starts_with("input.") || t.starts_with("const.") => 0,
        "output.surface" => 0,
        // cheap component-wise arithmetic
        "math.add" | "math.sub" | "math.mul" | "math.div" | "math.min" | "math.max"
        | "math.abs" | "math.saturate" | "math.oneminus" | "math.frac" => 1,
        "math.dot" | "math.lerp" | "math.clamp" | "math.normalize" => 2,
        // transcendentals
        "math.sin" | "math.pow" => 4,
        // vector assembly / swizzle
        t if t.starts_with("vec.") => 1,
        // procedural generators
        "proc.gradient" => 1,
        "proc.radial" => 3,
        "proc.checker" => 4,
        "proc.noise" => 12,
        // texture read (sample + filter)
        "tex.sample" => 8,
        // slabs: packing is cheap; a blend touches all four channels
        "slab.surface" => 1,
        "slab.blend" => 4,
        "slab.mask_blend" => 6,
        // unknown / future kinds: assume a modest op
        _ => 1,
    }
}

/// Analyze a material graph against the [default](MaterialBudget::default)
/// budgets.
pub fn analyze_complexity(graph: &Graph) -> ComplexityReport {
    analyze_with_budget(graph, MaterialBudget::default())
}

/// Analyze a material graph against explicit budgets.
pub fn analyze_with_budget(graph: &Graph, budget: MaterialBudget) -> ComplexityReport {
    let mut nodes = 0u32;
    let mut slabs = 0u32;
    let mut textures = 0u32;
    let mut est_alu_ops = 0u32;
    for n in graph.nodes.values() {
        nodes += 1;
        if n.type_id.starts_with("slab.") {
            slabs += 1;
        }
        if n.type_id == "tex.sample" {
            textures += 1;
        }
        est_alu_ops += node_cost(&n.type_id);
    }
    ComplexityReport {
        nodes,
        slabs,
        textures,
        est_alu_ops,
        budget,
    }
}

#[cfg(test)]
mod tests {
    use super::super::nodekit::material_registry;
    use super::*;
    use inf_graph::{apply_edits, GraphEdit, Link, NodeId};

    fn add(id: u32, type_id: &str) -> GraphEdit {
        GraphEdit::AddNode {
            id: NodeId(id),
            type_id: type_id.into(),
            x: 0.0,
            y: 0.0,
            params: Default::default(),
        }
    }
    fn wire(from: u32, fp: &str, to: u32, tp: &str) -> GraphEdit {
        GraphEdit::Connect {
            link: Link {
                from: NodeId(from),
                from_port: fp.into(),
                to: NodeId(to),
                to_port: tp.into(),
            },
        }
    }

    #[test]
    fn counts_nodes_slabs_and_textures() {
        let reg = material_registry();
        let mut g = Graph::empty();
        let edits = vec![
            add(1, "output.surface"),
            add(2, "slab.surface"),
            add(3, "slab.surface"),
            add(4, "slab.blend"),
            add(5, "tex.sample"),
            add(6, "const.color"),
            wire(6, "out", 2, "base_color"),
            wire(5, "rgb", 3, "base_color"),
            wire(2, "out", 4, "a"),
            wire(3, "out", 4, "b"),
            wire(4, "out", 1, "surface"),
        ];
        assert_eq!(apply_edits(&mut g, &reg, &edits), edits.len());
        let r = analyze_complexity(&g);
        assert_eq!(r.nodes, 6);
        assert_eq!(r.slabs, 3); // 2× slab.surface + 1× slab.blend
        assert_eq!(r.textures, 1);
        // const(0) + output(0) + 2×slab.surface(1) + slab.blend(4) + tex(8) = 14
        assert_eq!(r.est_alu_ops, 14);
        assert!(!r.over_budget(), "small graph is within budget");
    }

    #[test]
    fn flags_over_budget_at_tight_limits() {
        let reg = material_registry();
        let mut g = Graph::empty();
        let edits = vec![
            add(1, "output.surface"),
            add(2, "slab.surface"),
            add(3, "slab.surface"),
            add(4, "slab.blend"),
        ];
        assert_eq!(apply_edits(&mut g, &reg, &edits), edits.len());
        let tight = MaterialBudget {
            max_nodes: 100,
            max_slabs: 2, // 3 slabs present → over
            max_textures: 100,
            max_alu_ops: 100,
        };
        let r = analyze_with_budget(&g, tight);
        assert!(r.slabs_over());
        assert!(r.over_budget());
        assert!(!r.nodes_over());
    }

    #[test]
    fn empty_graph_is_zero_and_within_budget() {
        let r = analyze_complexity(&Graph::empty());
        assert_eq!(r.nodes, 0);
        assert_eq!(r.est_alu_ops, 0);
        assert!(!r.over_budget());
    }
}
