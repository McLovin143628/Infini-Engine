// Shared view uniforms — prepended to every scene shader by pipeline.rs's
// include mechanism (string concat at compile time via include_str!).
// Layout must match camera.rs::ViewUniforms.

struct View {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz = eye position (render-local), w unused.
    eye: vec4<f32>,
    // xyz = unit direction toward the sun.
    sun_dir: vec4<f32>,
    // xy = render-local position of the world X/Z axes (-origin.xz),
    // zw = viewport size in physical px.
    grid_axis_viewport: vec4<f32>,
    // x = 1.0 in orthographic (2D) mode else 0.0; y = -origin.y (render-local
    // world Y axis, for the 2D XY grid); zw reserved.
    mode_axis: vec4<f32>,
    // Camera right / up basis (render-local), xyz; w unused. Only the sprite
    // pass reads these (to orient billboards). Appended last — other passes that
    // declare the shorter View struct are unaffected.
    cam_right: vec4<f32>,
    cam_up: vec4<f32>,
    // View-mode flags (R-P2): x = unlit (1.0 ⇒ the lit passes return albedo+emissive
    // and skip lighting — drives Unlit, Wireframe and Biomes); y = biome overlay
    // (P19.2: 1.0 ⇒ the terrain pass tints by biome id; set only by Biomes, which
    // sets x as well); zw reserved. Appended last; both are 0 for the default Lit
    // mode so every pre-R-P2 golden stays byte-identical.
    flags: vec4<f32>,
};
@group(0) @binding(0) var<uniform> view: View;

// Fullscreen triangle covering NDC.
fn fullscreen_ndc(i: u32) -> vec2<f32> {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(3.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    return p[i];
}

// Unproject an NDC point at a given depth to render-local space.
// NOTE: reverse-infinite projection — depth 0.0 is at infinity, never
// unproject there (w hits 0). Use two finite depths to build ray directions.
fn unproject(ndc: vec2<f32>, depth: f32) -> vec3<f32> {
    let p = view.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    return p.xyz / p.w;
}

// Unit view ray through an NDC point.
fn view_ray(ndc: vec2<f32>) -> vec3<f32> {
    let a = unproject(ndc, 0.8);
    let b = unproject(ndc, 0.4);
    return normalize(b - a);
}
