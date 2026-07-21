//! Material graph → WGSL codegen + naga validation (P7.2.2).
//!
//! Walks the graph backward from the `output.surface` SINK (mirroring the
//! Blueprint lowerer's data half): each node resolves to a WGSL expression; a
//! value that fans out to ≥2 consumers is materialized into a `let` once so it
//! is evaluated a single time. The result is a `material_surface(MatIn) ->
//! Surface` function plus any helper functions / texture bindings the graph
//! used. naga parses + validates the module; errors are mapped back onto the
//! node whose emitted line they fall on.

use std::collections::{BTreeSet, HashMap};

use inf_graph::{Graph, NodeId, NodeRegistry, ParamValue};

use super::nodekit::MatType;

/// A texture asset referenced by a `tex.sample` node → a preview binding slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextureBinding {
    pub node: u32,
    pub slot: u32,
    /// The texture asset GUID (may be empty → bind a white 1×1).
    pub asset: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatSeverity {
    Error,
    Warning,
}

/// A compile diagnostic, optionally anchored to the node that produced the
/// offending WGSL line.
#[derive(Debug, Clone, PartialEq)]
pub struct MatIssue {
    pub node: Option<u32>,
    pub severity: MatSeverity,
    pub message: String,
}

/// The result of compiling a material graph.
#[derive(Debug, Clone)]
pub struct MaterialCompile {
    /// The `material_surface` fn + helpers + texture globals (embed in a
    /// preview/scene shader).
    pub surface_wgsl: String,
    /// A complete, self-contained module (surface + a validation entry point) —
    /// what "View Generated WGSL" shows and what naga validated.
    pub module_wgsl: String,
    pub issues: Vec<MatIssue>,
    pub textures: Vec<TextureBinding>,
    /// True when naga accepted the module (no error-severity issues).
    pub ok: bool,
}

/// Compile a material graph to WGSL and validate it.
pub fn emit_wgsl(graph: &Graph, reg: &NodeRegistry) -> MaterialCompile {
    let mut em = Emitter::new(graph, reg);
    let (base, metallic, roughness, emissive) = em.emit_surface();

    let surface_wgsl = em.assemble_surface(&base, &metallic, &roughness, &emissive);
    let module_wgsl = format!("{surface_wgsl}\n{VALIDATION_ENTRY}");

    let mut issues = std::mem::take(&mut em.issues);
    match validate_wgsl(&module_wgsl) {
        Ok(()) => {}
        Err(naga_issues) => {
            for (loc_line, message) in naga_issues {
                let node = loc_line.and_then(|l| em.line_owner(l));
                issues.push(MatIssue {
                    node,
                    severity: MatSeverity::Error,
                    message,
                });
            }
        }
    }

    let ok = !issues.iter().any(|i| i.severity == MatSeverity::Error);
    MaterialCompile {
        surface_wgsl,
        module_wgsl,
        issues,
        textures: em.textures,
        ok,
    }
}

/// The result of compiling a material graph to a **bake compute shader**
/// (P7.3): each texel evaluates the graph and its `base_color` is written to a
/// storage texture.
#[derive(Debug, Clone)]
pub struct TextureCompile {
    /// The complete, naga-validated compute module (entry `cs_bake`).
    pub compute_wgsl: String,
    pub issues: Vec<MatIssue>,
    /// `tex.sample` slots the graph references (bound white at bake time).
    pub texture_count: u32,
    pub ok: bool,
}

/// The compute wrapper that bakes a material graph's `base_color` to a texture.
const BAKE_COMPUTE: &str = "\
@group(0) @binding(0) var bake_out: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn cs_bake(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(bake_out);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(dims.x), f32(dims.y));
    var mi: MatIn;
    mi.uv = uv;
    mi.normal = vec3<f32>(0.0, 0.0, 1.0);
    mi.world_pos = vec3<f32>(uv, 0.0);
    mi.time = 0.0;
    let s = material_surface(mi);
    textureStore(bake_out, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(s.base_color, 1.0));
}";

/// Compile a material/texture graph to a bake **compute** shader (P7.3). Reuses
/// the surface codegen and appends a compute entry that writes `base_color` per
/// texel; the whole module is naga-validated.
pub fn emit_texture_compute(graph: &Graph, reg: &NodeRegistry) -> TextureCompile {
    let surface = emit_wgsl(graph, reg);
    let compute_wgsl = format!("{}\n{BAKE_COMPUTE}", surface.surface_wgsl);

    let mut issues = surface.issues;
    if let Err(list) = validate_wgsl(&compute_wgsl) {
        for (_, message) in list {
            issues.push(MatIssue {
                node: None,
                severity: MatSeverity::Error,
                message,
            });
        }
    }
    let ok = !issues.iter().any(|i| i.severity == MatSeverity::Error);
    TextureCompile {
        compute_wgsl,
        issues,
        texture_count: surface.textures.len() as u32,
        ok,
    }
}

/// A validation entry point so naga sees every global/fn as reachable.
const VALIDATION_ENTRY: &str = "\
@fragment
fn fs_validate() -> @location(0) vec4<f32> {
    var mi: MatIn;
    mi.uv = vec2<f32>(0.5, 0.5);
    mi.normal = vec3<f32>(0.0, 1.0, 0.0);
    mi.world_pos = vec3<f32>(0.0, 0.0, 0.0);
    mi.time = 0.0;
    let s = material_surface(mi);
    return vec4<f32>(s.base_color + s.emissive, s.metallic + s.roughness);
}";

struct Emitter<'a> {
    graph: &'a Graph,
    reg: &'a NodeRegistry,
    /// Emitted `let` lines (the function body prelude).
    body: String,
    /// Number of newlines emitted into `body` so far (for line→node mapping).
    body_lines: u32,
    /// Which body line each node's materialized `let` starts on.
    line_nodes: Vec<(u32, u32)>,
    /// Module lines before the first body `let` (set during assembly), so naga
    /// line numbers can be translated back to body lines.
    header_lines: u32,
    /// Materialized outputs: (node, port) → (wgsl expr, type).
    memo: HashMap<(NodeId, String), (String, MatType)>,
    textures: Vec<TextureBinding>,
    tex_slots: HashMap<NodeId, u32>,
    helpers: BTreeSet<&'static str>,
    issues: Vec<MatIssue>,
    next_local: u32,
    visiting: Vec<NodeId>,
}

impl<'a> Emitter<'a> {
    fn new(graph: &'a Graph, reg: &'a NodeRegistry) -> Self {
        Self {
            graph,
            reg,
            body: String::new(),
            body_lines: 0,
            line_nodes: Vec::new(),
            header_lines: 0,
            memo: HashMap::new(),
            textures: Vec::new(),
            tex_slots: HashMap::new(),
            helpers: BTreeSet::new(),
            issues: Vec::new(),
            next_local: 0,
            visiting: Vec::new(),
        }
    }

    /// Body line (0-based within the function body) → owning node id.
    fn line_owner(&self, module_line: u32) -> Option<u32> {
        // The body sits after the fixed function-header lines in `assemble`.
        let header = self.surface_header_lines();
        if module_line < header {
            return None;
        }
        let body_line = module_line - header;
        let mut owner = None;
        for &(line, node) in &self.line_nodes {
            if line <= body_line {
                owner = Some(node);
            } else {
                break;
            }
        }
        owner
    }

    /// Resolve the four PBR channels from the output node (or defaults).
    fn emit_surface(&mut self) -> (String, String, String, String) {
        let Some(out_id) = self.find_output() else {
            self.issues.push(MatIssue {
                node: None,
                severity: MatSeverity::Warning,
                message: "no Material Output node — using a default gray material".into(),
            });
            return (
                "vec3<f32>(0.8, 0.8, 0.8)".into(),
                "0.0".into(),
                "0.5".into(),
                "vec3<f32>(0.0)".into(),
            );
        };
        let base = self.resolve_input_typed(out_id, "base_color", MatType::Vec3, "vec3<f32>(0.8)");
        let metallic = self.resolve_input_typed(out_id, "metallic", MatType::Float, "0.0");
        let roughness = self.resolve_input_typed(out_id, "roughness", MatType::Float, "0.5");
        let emissive =
            self.resolve_input_typed(out_id, "emissive", MatType::Vec3, "vec3<f32>(0.0)");
        (base.0, metallic.0, roughness.0, emissive.0)
    }

    fn find_output(&self) -> Option<NodeId> {
        self.graph
            .nodes
            .values()
            .find(|n| n.type_id == "output.surface")
            .map(|n| n.id)
    }

    /// Resolve an output-node channel, coercing to the expected type; falls back
    /// to `default` when unconnected.
    fn resolve_input_typed(
        &mut self,
        node: NodeId,
        port: &str,
        want: MatType,
        default: &str,
    ) -> (String, MatType) {
        match self.graph.link_into(node, port) {
            Some(link) => {
                let (expr, ty) = self.resolve_output(link.from, &link.from_port);
                (self.coerce(&expr, ty, want), want)
            }
            None => (default.to_string(), want),
        }
    }

    /// Resolve a general input port to (expr, type), or a zero literal of the
    /// port's declared type when unconnected.
    fn resolve_input(&mut self, node: NodeId, port: &str) -> (String, MatType) {
        if let Some(link) = self.graph.link_into(node, port) {
            return self.resolve_output(link.from, &link.from_port);
        }
        let ty = self.declared_input_type(node, port);
        (zero_literal(ty), ty)
    }

    fn declared_input_type(&self, node: NodeId, port: &str) -> MatType {
        self.graph
            .node(node)
            .and_then(|n| self.reg.get(&n.type_id))
            .and_then(|d| d.input(port))
            .map(|p| MatType::from_port(&p.ty))
            .unwrap_or(MatType::Float)
    }

    /// Resolve a node's output port to (expr, type), materializing shared
    /// results into a `let`.
    fn resolve_output(&mut self, node: NodeId, port: &str) -> (String, MatType) {
        let key = (node, port.to_string());
        if let Some(v) = self.memo.get(&key) {
            return v.clone();
        }
        if self.visiting.contains(&node) {
            // The edit door forbids cycles; this is a defensive guard.
            return ("0.0".into(), MatType::Float);
        }
        self.visiting.push(node);
        let (expr, ty) = self.emit_node(node, port);
        self.visiting.pop();

        // Materialize when this exact output feeds ≥2 consumers and isn't a
        // trivial reference already.
        let shared = self.graph.links_from(node, port).count() >= 2;
        if shared && !is_trivial(&expr) {
            let name = self.alloc_local();
            self.push_line(node.0, &format!("let {name}: {} = {expr};", ty.wgsl()));
            let v = (name, ty);
            self.memo.insert(key, v.clone());
            return v;
        }
        let v = (expr, ty);
        self.memo.insert(key, v.clone());
        v
    }

    /// The heart: emit the WGSL expression for `node`'s `port`.
    fn emit_node(&mut self, node: NodeId, port: &str) -> (String, MatType) {
        let Some(n) = self.graph.node(node) else {
            return ("0.0".into(), MatType::Float);
        };
        let type_id = n.type_id.clone();
        let params = self
            .reg
            .get(&type_id)
            .map(|d| d.resolve(&n.params))
            .unwrap_or_default();

        match type_id.as_str() {
            // ── surface interpolants ──
            "input.uv" => ("mi.uv".into(), MatType::Vec2),
            "input.normal" => ("mi.normal".into(), MatType::Vec3),
            "input.position" => ("mi.world_pos".into(), MatType::Vec3),
            "input.time" => ("mi.time".into(), MatType::Float),

            // ── constants ──
            "const.float" => (fmt_f(pf(&params, "value")), MatType::Float),
            "const.vec2" => (
                format!(
                    "vec2<f32>({}, {})",
                    fmt_f(pf(&params, "x")),
                    fmt_f(pf(&params, "y"))
                ),
                MatType::Vec2,
            ),
            "const.color" => (
                format!(
                    "vec3<f32>({}, {}, {})",
                    fmt_f(pf(&params, "r")),
                    fmt_f(pf(&params, "g")),
                    fmt_f(pf(&params, "b"))
                ),
                MatType::Vec3,
            ),

            // ── binary math ──
            "math.add" => self.binary(node, "+"),
            "math.sub" => self.binary(node, "-"),
            "math.mul" => self.binary(node, "*"),
            "math.div" => self.binary(node, "/"),
            "math.min" => self.binary_fn(node, "min"),
            "math.max" => self.binary_fn(node, "max"),
            "math.pow" => self.binary_fn(node, "pow"),
            "math.dot" => {
                let (a, ta) = self.resolve_input(node, "a");
                let (b, tb) = self.resolve_input(node, "b");
                // dot needs vectors; if both are scalar, lift to vec2.
                let w = wider(ta, tb);
                let t = if w == MatType::Float {
                    MatType::Vec2
                } else {
                    w
                };
                (
                    format!(
                        "dot({}, {})",
                        self.coerce(&a, ta, t),
                        self.coerce(&b, tb, t)
                    ),
                    MatType::Float,
                )
            }

            // ── unary math ──
            "math.sin" => self.unary_fn(node, "sin"),
            "math.frac" => self.unary_fn(node, "fract"),
            "math.abs" => self.unary_fn(node, "abs"),
            "math.normalize" => self.unary_fn(node, "normalize"),
            "math.saturate" => {
                let (a, t) = self.resolve_input(node, "in");
                (
                    format!("clamp({a}, {}, {})", zero_literal(t), one_literal(t)),
                    t,
                )
            }
            "math.oneminus" => {
                let (a, t) = self.resolve_input(node, "in");
                (format!("({} - {a})", one_literal(t)), t)
            }
            "math.lerp" => {
                let (a, ta) = self.resolve_input(node, "a");
                let (b, tb) = self.resolve_input(node, "b");
                let (t_expr, _tt) = self.resolve_input(node, "t");
                let out = wider(ta, tb);
                (
                    format!(
                        "mix({}, {}, {})",
                        self.coerce(&a, ta, out),
                        self.coerce(&b, tb, out),
                        t_expr
                    ),
                    out,
                )
            }
            "math.clamp" => {
                let (a, t) = self.resolve_input(node, "in");
                let (lo, tlo) = self.resolve_input(node, "min");
                let (hi, thi) = self.resolve_input(node, "max");
                (
                    format!(
                        "clamp({a}, {}, {})",
                        self.coerce(&lo, tlo, t),
                        self.coerce(&hi, thi, t)
                    ),
                    t,
                )
            }

            // ── vectors ──
            "vec.make2" => {
                let x = self.resolve_input_scalar(node, "x");
                let y = self.resolve_input_scalar(node, "y");
                (format!("vec2<f32>({x}, {y})"), MatType::Vec2)
            }
            "vec.make3" => {
                let x = self.resolve_input_scalar(node, "x");
                let y = self.resolve_input_scalar(node, "y");
                let z = self.resolve_input_scalar(node, "z");
                (format!("vec3<f32>({x}, {y}, {z})"), MatType::Vec3)
            }
            "vec.make4" => {
                let (xyz, txyz) = self.resolve_input(node, "xyz");
                let w = self.resolve_input_scalar(node, "w");
                (
                    format!("vec4<f32>({}, {w})", self.coerce(&xyz, txyz, MatType::Vec3)),
                    MatType::Vec4,
                )
            }
            "vec.x" => self.swizzle(node, "x"),
            "vec.y" => self.swizzle(node, "y"),
            "vec.z" => self.swizzle(node, "z"),

            // ── procedural ──
            "proc.checker" => {
                self.helpers.insert("checker");
                let (uv, tuv) = self.resolve_input(node, "uv");
                let scale = fmt_f(pf(&params, "scale"));
                (
                    format!("checker({}, {scale})", self.coerce(&uv, tuv, MatType::Vec2)),
                    MatType::Float,
                )
            }
            "proc.noise" => {
                self.helpers.insert("noise");
                let (uv, tuv) = self.resolve_input(node, "uv");
                let scale = fmt_f(pf(&params, "scale"));
                (
                    format!(
                        "value_noise({} * {scale})",
                        self.coerce(&uv, tuv, MatType::Vec2)
                    ),
                    MatType::Float,
                )
            }
            "proc.gradient" => {
                let (uv, tuv) = self.resolve_input(node, "uv");
                let uv2 = self.coerce(&uv, tuv, MatType::Vec2);
                let comp = if matches!(params.get("vertical"), Some(ParamValue::Bool(true))) {
                    "y"
                } else {
                    "x"
                };
                (format!("({uv2}).{comp}"), MatType::Float)
            }
            "proc.radial" => {
                let (uv, tuv) = self.resolve_input(node, "uv");
                let uv2 = self.coerce(&uv, tuv, MatType::Vec2);
                (
                    format!("clamp(length({uv2} - vec2<f32>(0.5)) * 2.0, 0.0, 1.0)"),
                    MatType::Float,
                )
            }

            // ── texture ──
            "tex.sample" => {
                let slot = self.tex_slot(node, &params);
                let (uv, tuv) = self.resolve_input(node, "uv");
                let sample = format!(
                    "textureSampleLevel(mat_tex_{slot}, mat_samp_{slot}, {}, 0.0)",
                    self.coerce(&uv, tuv, MatType::Vec2)
                );
                if port == "rgb" {
                    (format!("({sample}).rgb"), MatType::Vec3)
                } else {
                    (sample, MatType::Vec4)
                }
            }

            other => {
                self.issues.push(MatIssue {
                    node: Some(node.0),
                    severity: MatSeverity::Error,
                    message: format!("unknown material node `{other}`"),
                });
                ("0.0".into(), MatType::Float)
            }
        }
    }

    fn binary(&mut self, node: NodeId, op: &str) -> (String, MatType) {
        let (a, ta) = self.resolve_input(node, "a");
        let (b, tb) = self.resolve_input(node, "b");
        let out = wider(ta, tb);
        (
            format!(
                "({} {op} {})",
                self.coerce(&a, ta, out),
                self.coerce(&b, tb, out)
            ),
            out,
        )
    }

    fn binary_fn(&mut self, node: NodeId, func: &str) -> (String, MatType) {
        let (a, ta) = self.resolve_input(node, "a");
        let (b, tb) = self.resolve_input(node, "b");
        let out = wider(ta, tb);
        (
            format!(
                "{func}({}, {})",
                self.coerce(&a, ta, out),
                self.coerce(&b, tb, out)
            ),
            out,
        )
    }

    fn unary_fn(&mut self, node: NodeId, func: &str) -> (String, MatType) {
        let (a, t) = self.resolve_input(node, "in");
        (format!("{func}({a})"), t)
    }

    fn swizzle(&mut self, node: NodeId, comp: &str) -> (String, MatType) {
        let (a, t) = self.resolve_input(node, "in");
        if t == MatType::Float {
            return (a, MatType::Float);
        }
        (format!("({a}).{comp}"), MatType::Float)
    }

    /// Resolve an input, forcing it to a scalar (`.x` if it's a vector).
    fn resolve_input_scalar(&mut self, node: NodeId, port: &str) -> String {
        let (a, t) = self.resolve_input(node, port);
        self.coerce(&a, t, MatType::Float)
    }

    fn tex_slot(&mut self, node: NodeId, params: &inf_graph::ParamMap) -> u32 {
        if let Some(slot) = self.tex_slots.get(&node) {
            return *slot;
        }
        let slot = self.textures.len() as u32;
        let asset = params
            .get("texture")
            .and_then(|v| match v {
                ParamValue::Text(s) => Some(s.clone()),
                _ => None,
            })
            .unwrap_or_default();
        self.textures.push(TextureBinding {
            node: node.0,
            slot,
            asset,
        });
        self.tex_slots.insert(node, slot);
        slot
    }

    /// Coerce `expr` (of type `from`) to `to`. Handles scalar↔vector broadcast
    /// and vector truncation; otherwise passes through (naga validates).
    fn coerce(&self, expr: &str, from: MatType, to: MatType) -> String {
        if from == to {
            return expr.to_string();
        }
        match (from, to) {
            // scalar → vector broadcast
            (MatType::Float, MatType::Vec2) => format!("vec2<f32>({expr})"),
            (MatType::Float, MatType::Vec3) => format!("vec3<f32>({expr})"),
            (MatType::Float, MatType::Vec4) => format!("vec4<f32>({expr})"),
            // vector → scalar (take x)
            (MatType::Vec2 | MatType::Vec3 | MatType::Vec4, MatType::Float) => {
                format!("({expr}).x")
            }
            // vector → smaller vector (swizzle prefix)
            (MatType::Vec3, MatType::Vec2) | (MatType::Vec4, MatType::Vec2) => {
                format!("({expr}).xy")
            }
            (MatType::Vec4, MatType::Vec3) => format!("({expr}).xyz"),
            // vector → larger vector: pad with zeros / one
            (MatType::Vec2, MatType::Vec3) => format!("vec3<f32>({expr}, 0.0)"),
            (MatType::Vec2, MatType::Vec4) => format!("vec4<f32>({expr}, 0.0, 1.0)"),
            (MatType::Vec3, MatType::Vec4) => format!("vec4<f32>({expr}, 1.0)"),
            _ => expr.to_string(),
        }
    }

    fn alloc_local(&mut self) -> String {
        let n = self.next_local;
        self.next_local += 1;
        format!("v{n}")
    }

    fn push_line(&mut self, node: u32, line: &str) {
        self.line_nodes.push((self.body_lines, node));
        self.body.push_str("    ");
        self.body.push_str(line);
        self.body.push('\n');
        self.body_lines += 1;
    }

    /// Number of module lines before the function body's first `let` (used to
    /// translate naga line numbers back to body lines).
    fn surface_header_lines(&self) -> u32 {
        // Set by `assemble_surface`; recomputed identically there.
        self.header_lines
    }

    fn assemble_surface(
        &mut self,
        base: &str,
        metallic: &str,
        roughness: &str,
        emissive: &str,
    ) -> String {
        let mut out = String::new();
        out.push_str(PREAMBLE);
        for h in &self.helpers {
            out.push_str(helper_src(h));
            out.push('\n');
        }
        // Texture globals.
        for tex in &self.textures {
            out.push_str(&format!(
                "@group(2) @binding({}) var mat_tex_{}: texture_2d<f32>;\n@group(2) @binding({}) var mat_samp_{}: sampler;\n",
                tex.slot * 2,
                tex.slot,
                tex.slot * 2 + 1,
                tex.slot
            ));
        }
        // Count header lines up to the first body `let`.
        let header = out.lines().count() as u32 + 1 /* fn signature line */;
        self.header_lines = header;

        out.push_str("fn material_surface(mi: MatIn) -> Surface {\n");
        out.push_str(&self.body);
        out.push_str("    var surf: Surface;\n");
        out.push_str(&format!("    surf.base_color = {base};\n"));
        out.push_str(&format!("    surf.metallic = {metallic};\n"));
        out.push_str(&format!("    surf.roughness = {roughness};\n"));
        out.push_str(&format!("    surf.emissive = {emissive};\n"));
        out.push_str("    return surf;\n}\n");
        out
    }
}

const PREAMBLE: &str = "\
// Generated by inf-material (P7.2). Do not edit by hand.
struct MatIn {
    uv: vec2<f32>,
    normal: vec3<f32>,
    world_pos: vec3<f32>,
    time: f32,
};
struct Surface {
    base_color: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>,
};
";

fn helper_src(name: &str) -> &'static str {
    match name {
        "checker" => {
            "fn checker(uv: vec2<f32>, scale: f32) -> f32 {\n    let c = floor(uv * scale);\n    return abs(fract((c.x + c.y) * 0.5) * 2.0);\n}\n"
        }
        "noise" => {
            "fn value_hash(p: vec2<f32>) -> f32 {\n    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);\n}\nfn value_noise(p: vec2<f32>) -> f32 {\n    let i = floor(p);\n    let f = fract(p);\n    let a = value_hash(i);\n    let b = value_hash(i + vec2<f32>(1.0, 0.0));\n    let c = value_hash(i + vec2<f32>(0.0, 1.0));\n    let d = value_hash(i + vec2<f32>(1.0, 1.0));\n    let u = f * f * (3.0 - 2.0 * f);\n    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);\n}\n"
        }
        _ => "",
    }
}

/// The wider of two types (vector beats scalar; larger vector wins).
fn wider(a: MatType, b: MatType) -> MatType {
    if a.arity() >= b.arity() {
        a
    } else {
        b
    }
}

fn zero_literal(t: MatType) -> String {
    match t {
        MatType::Float => "0.0".into(),
        MatType::Vec2 => "vec2<f32>(0.0)".into(),
        MatType::Vec3 => "vec3<f32>(0.0)".into(),
        MatType::Vec4 => "vec4<f32>(0.0)".into(),
    }
}

fn one_literal(t: MatType) -> String {
    match t {
        MatType::Float => "1.0".into(),
        MatType::Vec2 => "vec2<f32>(1.0)".into(),
        MatType::Vec3 => "vec3<f32>(1.0)".into(),
        MatType::Vec4 => "vec4<f32>(1.0)".into(),
    }
}

/// A "trivial" expression is a bare reference/literal — no need to hoist into a
/// `let` even if shared.
fn is_trivial(expr: &str) -> bool {
    expr.starts_with("mi.")
        || expr
            .chars()
            .all(|c| c.is_ascii_digit() || c == '.' || c == '-')
}

fn pf(params: &inf_graph::ParamMap, key: &str) -> f64 {
    match params.get(key) {
        Some(ParamValue::Float(f)) => *f,
        Some(ParamValue::Int(i)) => *i as f64,
        _ => 0.0,
    }
}

/// Format a float as a WGSL f32 literal (always with a decimal point).
fn fmt_f(v: f64) -> String {
    if v.is_finite() {
        let s = format!("{v:?}");
        if s.contains('.') || s.contains('e') {
            s
        } else {
            format!("{s}.0")
        }
    } else {
        "0.0".into()
    }
}

/// Parse + validate the module with naga. Returns per-error (optional
/// module-line, message).
fn validate_wgsl(src: &str) -> Result<(), Vec<(Option<u32>, String)>> {
    let module = match naga::front::wgsl::parse_str(src) {
        Ok(m) => m,
        Err(e) => {
            let line = e.location(src).map(|loc| loc.line_number.saturating_sub(1));
            return Err(vec![(line, format!("WGSL parse error: {e}"))]);
        }
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    match validator.validate(&module) {
        Ok(_) => Ok(()),
        Err(e) => {
            let line = e
                .spans()
                .next()
                .map(|(span, _)| span.location(src).line_number.saturating_sub(1));
            let inner = e.as_inner().to_string();
            Err(vec![(line, format!("WGSL validation error: {inner}"))])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::nodekit::material_registry;
    use super::*;
    use inf_graph::{apply_edits, GraphEdit, Link};

    fn add(id: u32, type_id: &str) -> GraphEdit {
        GraphEdit::AddNode {
            id: NodeId(id),
            type_id: type_id.into(),
            x: 0.0,
            y: 0.0,
            params: Default::default(),
        }
    }
    fn set(id: u32, name: &str, v: f64) -> GraphEdit {
        GraphEdit::SetParam {
            id: NodeId(id),
            name: name.into(),
            value: ParamValue::Float(v),
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
    fn empty_graph_yields_default_gray_and_validates() {
        let reg = material_registry();
        let g = Graph::empty();
        let c = emit_wgsl(&g, &reg);
        assert!(c.ok, "default material should validate: {:?}", c.issues);
        assert!(c.module_wgsl.contains("material_surface"));
        // A no-output graph warns but does not error.
        assert!(c.issues.iter().all(|i| i.severity == MatSeverity::Warning));
    }

    #[test]
    fn color_and_scalars_compile_and_validate() {
        let reg = material_registry();
        let mut g = Graph::empty();
        let edits = vec![
            add(1, "output.surface"),
            add(2, "const.color"),
            set(2, "r", 0.9),
            set(2, "g", 0.1),
            set(2, "b", 0.2),
            add(3, "const.float"),
            set(3, "value", 0.7),
            wire(2, "out", 1, "base_color"),
            wire(3, "out", 1, "roughness"),
        ];
        assert_eq!(apply_edits(&mut g, &reg, &edits), edits.len());
        let c = emit_wgsl(&g, &reg);
        assert!(c.ok, "issues: {:?}", c.issues);
        assert!(c.surface_wgsl.contains("surf.base_color = vec3<f32>(0.9"));
        assert!(c.surface_wgsl.contains("surf.roughness = 0.7"));
    }

    #[test]
    fn shared_node_is_materialized_into_one_let() {
        // uv → checker feeding BOTH base_color (via make3) and roughness would
        // fan out; simpler: a checker feeding roughness twice through two output
        // ports isn't possible, so wire checker to roughness and metallic.
        let reg = material_registry();
        let mut g = Graph::empty();
        let edits = vec![
            add(1, "output.surface"),
            add(2, "input.uv"),
            add(3, "proc.checker"),
            wire(2, "out", 3, "uv"),
            wire(3, "out", 1, "metallic"),
            wire(3, "out", 1, "roughness"),
        ];
        assert_eq!(apply_edits(&mut g, &reg, &edits), edits.len());
        let c = emit_wgsl(&g, &reg);
        assert!(c.ok, "issues: {:?}", c.issues);
        // The shared checker result is hoisted to a single `let` (the call site
        // `= checker(...)`; the `fn checker(` definition doesn't match).
        let calls = c.surface_wgsl.matches("= checker(").count();
        assert_eq!(calls, 1, "checker evaluated once:\n{}", c.surface_wgsl);
    }

    #[test]
    fn texture_node_emits_binding_and_records_reference() {
        let reg = material_registry();
        let mut g = Graph::empty();
        let edits = vec![
            add(1, "output.surface"),
            add(2, "tex.sample"),
            GraphEdit::SetParam {
                id: NodeId(2),
                name: "texture".into(),
                value: ParamValue::Text("abc-123".into()),
            },
            wire(2, "rgb", 1, "base_color"),
        ];
        assert_eq!(apply_edits(&mut g, &reg, &edits), edits.len());
        let c = emit_wgsl(&g, &reg);
        assert!(c.ok, "issues: {:?}", c.issues);
        assert!(c.surface_wgsl.contains("mat_tex_0"));
        assert_eq!(c.textures.len(), 1);
        assert_eq!(c.textures[0].asset, "abc-123");
    }

    #[test]
    fn texture_compute_bakes_gradient_and_validates() {
        let reg = material_registry();
        let mut g = Graph::empty();
        let edits = vec![
            add(1, "output.surface"),
            add(2, "input.uv"),
            add(3, "proc.gradient"),
            add(4, "proc.radial"),
            add(5, "vec.make3"),
            wire(2, "out", 3, "uv"),
            wire(2, "out", 4, "uv"),
            wire(3, "out", 5, "x"),
            wire(4, "out", 5, "y"),
            wire(3, "out", 5, "z"),
            wire(5, "out", 1, "base_color"),
        ];
        assert_eq!(apply_edits(&mut g, &reg, &edits), edits.len());
        let c = emit_texture_compute(&g, &reg);
        assert!(c.ok, "issues: {:?}", c.issues);
        assert!(c.compute_wgsl.contains("@compute"));
        assert!(c.compute_wgsl.contains("textureStore(bake_out"));
    }

    #[test]
    fn noise_math_chain_validates() {
        let reg = material_registry();
        let mut g = Graph::empty();
        let edits = vec![
            add(1, "output.surface"),
            add(2, "input.uv"),
            add(3, "proc.noise"),
            add(4, "const.color"),
            add(5, "math.mul"),
            wire(2, "out", 3, "uv"),
            wire(4, "out", 5, "a"),
            wire(3, "out", 5, "b"), // vec3 * f32 → coerced
            wire(5, "out", 1, "base_color"),
        ];
        assert_eq!(apply_edits(&mut g, &reg, &edits), edits.len());
        let c = emit_wgsl(&g, &reg);
        assert!(c.ok, "issues: {:?}", c.issues);
    }
}
