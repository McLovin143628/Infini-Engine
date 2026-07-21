//! Render-graph nodes: one module per pass. Scene passes share `group(0)`
//! (view uniforms) and render into the MSAA scene targets; the composite node
//! resolves to the output.

pub mod composite;
pub mod debug;
pub mod grid;
pub mod mask;
pub mod mesh;
pub mod resolve;
pub mod sky;
pub mod sprite;
pub mod terrain;

/// Scene shaders share the `View` uniform block + helpers.
pub(crate) fn scene_shader(source: &str) -> String {
    format!(
        "{}\n{}",
        include_str!("../shaders/common_view.wgsl"),
        source
    )
}
