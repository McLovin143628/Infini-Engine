//! The material node kit: the palette of `NodeDef`s a `.inf_mat` graph is built
//! from, plus the [`MatType`] ↔ WGSL/`PortType` mapping the emitter reads.
//!
//! All material nodes are pure (no exec pins); a single `output.surface` SINK
//! collects the PBR channels. `type_id`s follow a `namespace.name` convention
//! the emitter switches on: `input.*`, `const.*`, `math.*`, `vec.*`, `proc.*`,
//! `tex.*`, `output.*`.

use inf_graph::{NodeDef, NodeRegistry, ParamDef, PortDef, PortType, SINK};

/// The WGSL value type carried on a material wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatType {
    Float,
    Vec2,
    Vec3,
    Vec4,
}

impl MatType {
    /// The WGSL type spelling.
    pub fn wgsl(self) -> &'static str {
        match self {
            MatType::Float => "f32",
            MatType::Vec2 => "vec2<f32>",
            MatType::Vec3 => "vec3<f32>",
            MatType::Vec4 => "vec4<f32>",
        }
    }

    /// The graph port type (vectors are `Named` — the substrate has no vector
    /// variant; `compatible_with` matches `Named` by name).
    pub fn port(self) -> PortType {
        match self {
            MatType::Float => PortType::Float,
            MatType::Vec2 => PortType::Named("vec2f".into()),
            MatType::Vec3 => PortType::Named("vec3f".into()),
            MatType::Vec4 => PortType::Named("vec4f".into()),
        }
    }

    /// Number of scalar components.
    pub fn arity(self) -> u32 {
        match self {
            MatType::Float => 1,
            MatType::Vec2 => 2,
            MatType::Vec3 => 3,
            MatType::Vec4 => 4,
        }
    }

    /// Recover a [`MatType`] from a wire [`PortType`] (defaults to `Float`).
    pub fn from_port(ty: &PortType) -> MatType {
        match ty {
            PortType::Named(n) if n == "vec2f" => MatType::Vec2,
            PortType::Named(n) if n == "vec3f" => MatType::Vec3,
            PortType::Named(n) if n == "vec4f" => MatType::Vec4,
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

/// The material output sink: collects the PBR channels.
fn output_node() -> NodeDef {
    NodeDef::new("output.surface", "Material Output", "output")
        .described("The final surface: base color + metallic-roughness + emissive")
        .with_inputs(vec![
            port("base_color", MatType::Vec3).labeled("Base Color"),
            port("metallic", MatType::Float),
            port("roughness", MatType::Float),
            port("emissive", MatType::Vec3),
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
        assert_eq!(out.inputs.len(), 4);
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
            "output.surface",
        ] {
            assert!(reg.contains(id), "missing {id}");
        }
    }

    #[test]
    fn mat_type_roundtrips_through_port() {
        for t in [MatType::Float, MatType::Vec2, MatType::Vec3, MatType::Vec4] {
            assert_eq!(MatType::from_port(&t.port()), t);
        }
    }
}
