//! The material node kit: the palette of `NodeDef`s a `.inf_mat` graph is built
//! from, plus the [`MatType`] ↔ WGSL/`PortType` mapping the emitter reads.
//!
//! All material nodes are pure (no exec pins); a single `output.surface` SINK
//! collects the PBR channels. `type_id`s follow a `namespace.name` convention
//! the emitter switches on: `input.*`, `const.*`, `math.*`, `vec.*`, `proc.*`,
//! `tex.*`, `output.*`.

use inf_graph::{NodeDef, NodeRegistry, ParamDef, PortDef, PortType, SINK};

/// The WGSL value type carried on a material wire.
///
/// `Slab` (P13.2) is the Substrate-class layered-material value: a full surface
/// parameter set (albedo/metallic/roughness/emissive) travelling on one wire as
/// the WGSL `MatSlab` struct. Slab wires are produced by `slab.*` nodes and
/// consumed by `output.surface` (either a Slab **or** the legacy scalar inputs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatType {
    Float,
    Vec2,
    Vec3,
    Vec4,
    Slab,
}

impl MatType {
    /// The WGSL type spelling.
    pub fn wgsl(self) -> &'static str {
        match self {
            MatType::Float => "f32",
            MatType::Vec2 => "vec2<f32>",
            MatType::Vec3 => "vec3<f32>",
            MatType::Vec4 => "vec4<f32>",
            MatType::Slab => "MatSlab",
        }
    }

    /// The graph port type (vectors and slabs are `Named` — the substrate has no
    /// vector/struct variant; `compatible_with` matches `Named` by name).
    pub fn port(self) -> PortType {
        match self {
            MatType::Float => PortType::Float,
            MatType::Vec2 => PortType::Named("vec2f".into()),
            MatType::Vec3 => PortType::Named("vec3f".into()),
            MatType::Vec4 => PortType::Named("vec4f".into()),
            MatType::Slab => PortType::Named("slab".into()),
        }
    }

    /// Number of scalar components (a `Slab` is a struct, not a vector → 0, so it
    /// never wins the `wider` numeric-coercion pick).
    pub fn arity(self) -> u32 {
        match self {
            MatType::Float => 1,
            MatType::Vec2 => 2,
            MatType::Vec3 => 3,
            MatType::Vec4 => 4,
            MatType::Slab => 0,
        }
    }

    /// Recover a [`MatType`] from a wire [`PortType`] (defaults to `Float`).
    pub fn from_port(ty: &PortType) -> MatType {
        match ty {
            PortType::Named(n) if n == "vec2f" => MatType::Vec2,
            PortType::Named(n) if n == "vec3f" => MatType::Vec3,
            PortType::Named(n) if n == "vec4f" => MatType::Vec4,
            PortType::Named(n) if n == "slab" => MatType::Slab,
            _ => MatType::Float,
        }
    }
}

fn port(name: &str, ty: MatType) -> PortDef {
    PortDef::new(name, ty.port())
}

/// A wildcard-typed input (accepts float or any vector; naga type-checks the
/// generated WGSL). Used by the polymorphic math nodes.
fn any_in(name: &str) -> PortDef {
    PortDef::new(name, PortType::Wildcard)
}

/// The complete material node palette.
pub fn material_registry() -> NodeRegistry {
    let mut reg = NodeRegistry::new();
    reg.register_all(input_nodes());
    reg.register_all(const_nodes());
    reg.register_all(math_nodes());
    reg.register_all(vector_nodes());
    reg.register_all(proc_nodes());
    reg.register_all(tex_nodes());
    reg.register_all(slab_nodes());
    reg.register(output_node());
    reg
}

fn input_nodes() -> Vec<NodeDef> {
    let input = |id: &str, display: &str, ty: MatType| {
        NodeDef::new(id, display, "input")
            .described("Surface interpolant")
            .with_outputs(vec![port("out", ty)])
    };
    vec![
        input("input.uv", "UV", MatType::Vec2),
        input("input.normal", "Normal", MatType::Vec3),
        input("input.position", "World Position", MatType::Vec3),
        input("input.time", "Time", MatType::Float),
    ]
}

fn const_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("const.float", "Float", "constants")
            .with_outputs(vec![port("out", MatType::Float)])
            .with_params(vec![ParamDef::number("value", 0.0)]),
        NodeDef::new("const.vec2", "Vector2", "constants")
            .with_outputs(vec![port("out", MatType::Vec2)])
            .with_params(vec![ParamDef::number("x", 0.0), ParamDef::number("y", 0.0)]),
        NodeDef::new("const.color", "Color", "constants")
            .described("An RGB color constant")
            .with_outputs(vec![port("out", MatType::Vec3)])
            .with_params(vec![
                ParamDef::number("r", 0.8).range(0.0, 1.0),
                ParamDef::number("g", 0.8).range(0.0, 1.0),
                ParamDef::number("b", 0.8).range(0.0, 1.0),
            ]),
    ]
}

fn math_nodes() -> Vec<NodeDef> {
    let binary = |id: &str, display: &str| {
        NodeDef::new(id, display, "math")
            .with_inputs(vec![any_in("a"), any_in("b")])
            .with_outputs(vec![port("out", MatType::Float)])
    };
    let unary = |id: &str, display: &str| {
        NodeDef::new(id, display, "math")
            .with_inputs(vec![any_in("in")])
            .with_outputs(vec![port("out", MatType::Float)])
    };
    vec![
        binary("math.add", "Add (+)"),
        binary("math.sub", "Subtract (−)"),
        binary("math.mul", "Multiply (×)"),
        binary("math.div", "Divide (÷)"),
        binary("math.min", "Min"),
        binary("math.max", "Max"),
        binary("math.pow", "Power"),
        binary("math.dot", "Dot").with_outputs(vec![port("out", MatType::Float)]),
        unary("math.sin", "Sine"),
        unary("math.frac", "Fraction"),
        unary("math.abs", "Absolute"),
        unary("math.saturate", "Saturate"),
        unary("math.oneminus", "One Minus (1−x)"),
        unary("math.normalize", "Normalize"),
        NodeDef::new("math.lerp", "Lerp", "math")
            .described("Linear blend a→b by t")
            .with_inputs(vec![any_in("a"), any_in("b"), any_in("t")])
            .with_outputs(vec![any_wild("out")]),
        NodeDef::new("math.clamp", "Clamp", "math")
            .with_inputs(vec![any_in("in"), any_in("min"), any_in("max")])
            .with_outputs(vec![any_wild("out")]),
    ]
}

fn any_wild(name: &str) -> PortDef {
    PortDef::new(name, PortType::Wildcard)
}

fn vector_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("vec.make3", "Make Vector3", "vector")
            .with_inputs(vec![
                port("x", MatType::Float),
                port("y", MatType::Float),
                port("z", MatType::Float),
            ])
            .with_outputs(vec![port("out", MatType::Vec3)]),
        NodeDef::new("vec.make2", "Make Vector2", "vector")
            .with_inputs(vec![port("x", MatType::Float), port("y", MatType::Float)])
            .with_outputs(vec![port("out", MatType::Vec2)]),
        NodeDef::new("vec.make4", "Make Vector4", "vector")
            .with_inputs(vec![port("xyz", MatType::Vec3), port("w", MatType::Float)])
            .with_outputs(vec![port("out", MatType::Vec4)]),
        NodeDef::new("vec.x", "X", "vector")
            .with_inputs(vec![any_in("in")])
            .with_outputs(vec![port("out", MatType::Float)]),
        NodeDef::new("vec.y", "Y", "vector")
            .with_inputs(vec![any_in("in")])
            .with_outputs(vec![port("out", MatType::Float)]),
        NodeDef::new("vec.z", "Z", "vector")
            .with_inputs(vec![any_in("in")])
            .with_outputs(vec![port("out", MatType::Float)]),
    ]
}

fn proc_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("proc.checker", "Checker", "procedural")
            .described("Black/white checker over UV")
            .with_inputs(vec![port("uv", MatType::Vec2)])
            .with_outputs(vec![port("out", MatType::Float)])
            .with_params(vec![ParamDef::number("scale", 8.0).range(1.0, 128.0)]),
        NodeDef::new("proc.noise", "Value Noise", "procedural")
            .described("Hash value noise over UV")
            .with_inputs(vec![port("uv", MatType::Vec2)])
            .with_outputs(vec![port("out", MatType::Float)])
            .with_params(vec![ParamDef::number("scale", 4.0).range(0.1, 128.0)]),
        NodeDef::new("proc.gradient", "Linear Gradient", "procedural")
            .described("A 0→1 gradient along U (or V)")
            .with_inputs(vec![port("uv", MatType::Vec2)])
            .with_outputs(vec![port("out", MatType::Float)])
            .with_params(vec![ParamDef::toggle("vertical", false)]),
        NodeDef::new("proc.radial", "Radial Gradient", "procedural")
            .described("0 at UV center → 1 at the edges")
            .with_inputs(vec![port("uv", MatType::Vec2)])
            .with_outputs(vec![port("out", MatType::Float)]),
    ]
}

fn tex_nodes() -> Vec<NodeDef> {
    vec![NodeDef::new("tex.sample", "Texture Sample", "texture")
        .described("Sample a texture asset at UV (RGBA)")
        .with_inputs(vec![port("uv", MatType::Vec2)])
        .with_outputs(vec![
            port("rgba", MatType::Vec4),
            port("rgb", MatType::Vec3),
        ])
        .with_params(vec![
            ParamDef::text("texture", "").described("Texture asset GUID (leave blank for white)")
        ])]
}

/// The Substrate-class layered-material node kit (P13.2). A `slab.surface`
/// packs the four PBR channels into a `Slab` wire value; `slab.blend` /
/// `slab.mask_blend` combine two slabs channel-wise. `output.surface` consumes
/// either a single Slab or the legacy scalar channels (back-compat).
fn slab_nodes() -> Vec<NodeDef> {
    vec![
        NodeDef::new("slab.surface", "Surface Slab", "slab")
            .described("Pack base color + metallic-roughness + emissive into a layer slab")
            .with_inputs(vec![
                port("base_color", MatType::Vec3).labeled("Base Color"),
                port("metallic", MatType::Float),
                port("roughness", MatType::Float),
                port("emissive", MatType::Vec3),
            ])
            .with_outputs(vec![port("out", MatType::Slab).labeled("Slab")]),
        NodeDef::new("slab.blend", "Blend Slabs", "slab")
            .described("Linearly blend two slabs channel-wise by a factor (0 = A, 1 = B)")
            .with_inputs(vec![
                port("a", MatType::Slab).labeled("A"),
                port("b", MatType::Slab).labeled("B"),
                port("factor", MatType::Float).param_pin(),
            ])
            .with_outputs(vec![port("out", MatType::Slab).labeled("Slab")])
            .with_params(vec![ParamDef::number("factor", 0.5)
                .range(0.0, 1.0)
                .described(
                    "Blend weight A→B (overridden by the Factor pin when wired)",
                )]),
        NodeDef::new("slab.mask_blend", "Mask Blend Slabs", "slab")
            .described("Blend two slabs by a mask, feathered through the 0.5 threshold")
            .with_inputs(vec![
                port("a", MatType::Slab).labeled("A"),
                port("b", MatType::Slab).labeled("B"),
                port("mask", MatType::Float),
            ])
            .with_outputs(vec![port("out", MatType::Slab).labeled("Slab")])
            .with_params(vec![ParamDef::number("feather", 0.1)
                .range(0.0, 0.5)
                .described("Half-width of the smooth transition around mask = 0.5")]),
    ]
}

/// The material output sink: collects the PBR channels. Accepts EITHER a single
/// `Slab` (P13.2) OR the legacy scalar channels — wiring both is a compile
/// error (see [`super::emit`]).
fn output_node() -> NodeDef {
    NodeDef::new("output.surface", "Material Output", "output")
        .described("The final surface: a Slab, or base color + metallic-roughness + emissive")
        .with_inputs(vec![
            port("base_color", MatType::Vec3).labeled("Base Color"),
            port("metallic", MatType::Float),
            port("roughness", MatType::Float),
            port("emissive", MatType::Vec3),
            port("surface", MatType::Slab).labeled("Surface (Slab)"),
        ])
        .with_flags(SINK)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_has_output_sink() {
        let reg = material_registry();
        let out = reg.get("output.surface").expect("output node");
        assert!(out.has(SINK));
        // P13.2: the four legacy scalar channels + the new Slab input.
        assert_eq!(out.inputs.len(), 5);
        assert!(out.input("surface").is_some());
        // The legacy scalar channels are unchanged (back-compat).
        for ch in ["base_color", "metallic", "roughness", "emissive"] {
            assert!(out.input(ch).is_some(), "missing legacy channel {ch}");
        }
    }

    #[test]
    fn registry_covers_the_kit() {
        let reg = material_registry();
        for id in [
            "input.uv",
            "const.color",
            "math.mul",
            "math.lerp",
            "vec.make3",
            "vec.x",
            "proc.checker",
            "tex.sample",
            "slab.surface",
            "slab.blend",
            "slab.mask_blend",
            "output.surface",
        ] {
            assert!(reg.contains(id), "missing {id}");
        }
    }

    #[test]
    fn slab_nodes_carry_slab_wires() {
        let reg = material_registry();
        let s = reg.get("slab.surface").expect("slab.surface");
        assert_eq!(s.output("out").map(|p| &p.ty), Some(&MatType::Slab.port()));
        let blend = reg.get("slab.blend").expect("slab.blend");
        assert_eq!(blend.input("a").map(|p| &p.ty), Some(&MatType::Slab.port()));
        assert_eq!(blend.input("b").map(|p| &p.ty), Some(&MatType::Slab.port()));
    }

    #[test]
    fn mat_type_roundtrips_through_port() {
        for t in [
            MatType::Float,
            MatType::Vec2,
            MatType::Vec3,
            MatType::Vec4,
            MatType::Slab,
        ] {
            assert_eq!(MatType::from_port(&t.port()), t);
        }
    }
}
