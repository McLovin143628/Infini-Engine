// Infinite editor grid: fullscreen pass, ray-vs-ground-plane (y = 0) with
// anti-aliased 1 m / 10 m lines and world X/Z axes. Writes real depth for the
// plane point so meshes occlude the grid correctly (reverse-Z, compare
// Greater, early-Z is off because of the frag_depth output).
//
// Precision note: the floating origin snaps to 10 m multiples (inf-math
// ORIGIN_SNAP), so render-local coordinates are already grid-aligned — the
// line pattern needs no world offset. Only the world axes (world x=0 / z=0)
// need the origin, passed as view.grid_axis_viewport.xy.

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var out: VsOut;
    let p = fullscreen_ndc(i);
    out.pos = vec4<f32>(p, 0.5, 1.0);
    out.ndc = p;
    return out;
}

struct FsOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color = vec4<f32>(0.0);
    out.depth = 0.0; // infinity under reverse-Z

    let ro = view.eye.xyz;
    let rd = view_ray(in.ndc);
    let t = -ro.y / rd.y;
    if (t <= 0.0 || rd.y == 0.0) {
        return out;
    }
    let wp = ro + rd * t;

    // Real depth of the ground point → correct occlusion against meshes.
    let clip = view.view_proj * vec4<f32>(wp, 1.0);
    out.depth = clip.z / clip.w;

    // Anti-aliased line mask at 1 m and 10 m spacing (UE-style dual grid).
    let coord = wp.xz;
    let d = max(fwidth(coord), vec2<f32>(1e-6));
    let g1 = abs(fract(coord - 0.5) - 0.5) / d;
    let l1 = 1.0 - min(min(g1.x, g1.y), 1.0);
    let g10 = abs(fract(coord / 10.0 - 0.5) - 0.5) / (d / 10.0);
    let l10 = 1.0 - min(min(g10.x, g10.y), 1.0);

    let dist = length(wp - ro);
    let fade = exp(-dist * 0.018);

    var col = vec3<f32>(0.30, 0.33, 0.38);
    var a = max(l1 * 0.30, l10 * 0.75) * fade;

    // World axes: X in red, Z in blue, at world x=0 / z=0 (render-local
    // position comes from the floating origin).
    let axis = view.grid_axis_viewport.xy;
    if (abs(wp.z - axis.y) < d.y * 1.5) {
        col = vec3<f32>(0.95, 0.28, 0.30);
        a = max(a, 0.85 * fade);
    }
    if (abs(wp.x - axis.x) < d.x * 1.5) {
        col = vec3<f32>(0.25, 0.45, 1.0);
        a = max(a, 0.85 * fade);
    }

    // Premultiplied alpha over the sky/meshes.
    out.color = vec4<f32>(col * a, a);
    return out;
}
