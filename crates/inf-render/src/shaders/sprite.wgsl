// 2D sprite pass: instanced textured quads expanded in the vertex shader from
// per-instance data (no CPU vertex buffers per sprite). One instance = one quad
// drawn as a 4-vertex triangle strip; the corner comes from @builtin(vertex_index).
//
// Draws after the 3D meshes: depth-tested against the scene depth buffer
// (reverse-Z, Greater) but NOT depth-writing, so sprite-vs-sprite order is the
// painter order the batcher produced. Straight (non-premultiplied) alpha.
//
// The quad / UV / pivot math here mirrors inf_render_2d::{corner_offset,
// corner_uv} 1:1 — keep them in sync.

struct VsIn {
    @location(0) center: vec4<f32>,   // xyz = render-local pivot position
    @location(1) size_rot: vec4<f32>, // x,y = size; z = rotation (radians)
    @location(2) pivot: vec4<f32>,    // x,y = pivot in [0,1]
    @location(3) uv: vec4<f32>,       // xy = uv_min, zw = uv_max
    @location(4) color: vec4<f32>,    // linear straight-alpha tint
    @location(5) flags: vec4<u32>,    // x = flip_x, y = flip_y, z = billboard mode
};

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    // Render-local world XY of this fragment (for 2D-light distance).
    @location(2) world_xy: vec2<f32>,
};

@vertex
fn vs(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {
    // Unit-quad corner (triangle strip): 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1).
    let cx = f32(vi & 1u);
    let cy = f32((vi >> 1u) & 1u);

    let size = in.size_rot.xy;
    let rot = in.size_rot.z;
    let pivot = in.pivot.xy;

    // Pivot-relative, size-scaled local plane offset, rotated about the quad
    // normal (billboard modes keep rotation in the oriented plane).
    let local = vec2<f32>((cx - pivot.x) * size.x, (cy - pivot.y) * size.y);
    let s = sin(rot);
    let c = cos(rot);
    let r = vec2<f32>(local.x * c - local.y * s, local.x * s + local.y * c);

    // Billboard mode (flags.z): 0 = planar (world XY plane, facing +Z — the
    // pre-2.5D path, byte-identical to before), 1 = spherical (full camera
    // basis), 2 = cylindrical (world-up, camera-facing right). Mirrors
    // inf_render_2d::billboard_basis / corner_offset_billboard.
    let mode = in.flags.z;
    var world: vec3<f32>;
    if (mode == 0u) {
        // Sprite lies in the world XY plane facing +Z; offset along world X/Y.
        world = in.center.xyz + vec3<f32>(r.x, r.y, 0.0);
    } else {
        var right = view.cam_right.xyz;
        var up = view.cam_up.xyz;
        if (mode == 2u) {
            // Cylindrical: upright about world +Y, right flattened to horizontal.
            let flat = vec3<f32>(right.x, 0.0, right.z);
            if (dot(flat, flat) > 1e-12) {
                right = normalize(flat);
            } else {
                right = vec3<f32>(1.0, 0.0, 0.0);
            }
            up = vec3<f32>(0.0, 1.0, 0.0);
        }
        world = in.center.xyz + right * r.x + up * r.y;
    }

    var out: VsOut;
    out.pos = view.view_proj * vec4<f32>(world, 1.0);
    out.world_xy = world.xy;

    // UVs: +Y (top, cy==1) maps to uv_min.y (texture top). Flips mirror.
    var u = cx;
    var v = 1.0 - cy;
    if (in.flags.x != 0u) { u = 1.0 - u; }
    if (in.flags.y != 0u) { v = 1.0 - v; }
    let uv_min = in.uv.xy;
    let uv_max = in.uv.zw;
    out.uv = vec2<f32>(
        uv_min.x + (uv_max.x - uv_min.x) * u,
        uv_min.y + (uv_max.y - uv_min.y) * v,
    );
    out.color = in.color;
    return out;
}

@group(1) @binding(0) var sprite_tex: texture_2d<f32>;
@group(1) @binding(1) var sprite_samp: sampler;

// ── 2D lights (must match Lights2DUniform / MAX_2D_LIGHTS in passes/sprite.rs) ──
const MAX_2D_LIGHTS: u32 = 16u;

struct GpuLight2D {
    color: vec4<f32>, // rgb = color, a = intensity
    pos:   vec4<f32>, // xy = render-local position, z = radius
};
struct Lights2D {
    count:   vec4<u32>, // x = active count
    ambient: vec4<f32>, // rgb = scene ambient (default 1,1,1)
    items:   array<GpuLight2D, MAX_2D_LIGHTS>,
};
@group(2) @binding(0) var<uniform> lights2d: Lights2D;

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(sprite_tex, sprite_samp, in.uv);
    // Straight alpha: texel and tint multiply; the pipeline does the over-blend.
    let base = texel * in.color;

    // Accumulate lighting: ambient + Σ light·intensity·smoothstep(radius,0,dist).
    // With the default white ambient and no lights, factor == (1,1,1), so
    // `base.rgb * factor == base.rgb` exactly — pre-P8.1c sprites are unchanged.
    var factor = lights2d.ambient.rgb;
    let count = lights2d.count.x;
    for (var i = 0u; i < count && i < MAX_2D_LIGHTS; i = i + 1u) {
        let lt = lights2d.items[i];
        let radius = lt.pos.z;
        if (radius > 0.0) {
            let d = distance(in.world_xy, lt.pos.xy);
            let atten = smoothstep(radius, 0.0, d);
            factor += lt.color.rgb * lt.color.a * atten;
        }
    }
    return vec4<f32>(base.rgb * factor, base.a);
}
