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

// Orthographic 2D grid: lines/axes on the world XY plane (z = 0). Kept in a
// separate branch so the perspective path below stays byte-identical.
fn grid_xy(in: VsOut) -> FsOut {
    var out: FsOut;
    out.color = vec4<f32>(0.0);
    out.depth = 0.0;

    // Parallel rays: origin varies per pixel (near plane), direction is forward.
    let ro = unproject(in.ndc, 1.0);
    let rd = view_ray(in.ndc);
    if (rd.z == 0.0) {
        return out;
    }
    let t = -ro.z / rd.z;
    let wp = ro + rd * t;

    // Real depth of the plane point → correct occlusion against meshes/sprites.
    let clip = view.view_proj * vec4<f32>(wp, 1.0);
    out.depth = clip.z / clip.w;

    // Anti-aliased 1 m / 10 m line mask in the XY plane. fwidth is constant
    // across the screen in ortho, giving uniform line width at any zoom.
    let coord = wp.xy;
    let d = max(fwidth(coord), vec2<f32>(1e-6));
    let g1 = abs(fract(coord - 0.5) - 0.5) / d;
    let l1 = 1.0 - min(min(g1.x, g1.y), 1.0);
    let g10 = abs(fract(coord / 10.0 - 0.5) - 0.5) / (d / 10.0);
    let l10 = 1.0 - min(min(g10.x, g10.y), 1.0);

    var col = vec3<f32>(0.30, 0.33, 0.38);
    var a = max(l1 * 0.30, l10 * 0.75);

    // World axes in the XY plane: the horizontal X axis (world y = 0) in red,
    // the vertical Y axis (world x = 0) in green. Render-local positions come
    // from the floating origin (x = -origin.x, y = -origin.y).
    let ax = view.grid_axis_viewport.x; // render-local world x = 0
    let ay = view.mode_axis.y;          // render-local world y = 0
    if (abs(wp.y - ay) < d.y * 1.5) {
        col = vec3<f32>(0.95, 0.28, 0.30);
        a = max(a, 0.85);
    }
    if (abs(wp.x - ax) < d.x * 1.5) {
        col = vec3<f32>(0.45, 0.85, 0.30);
        a = max(a, 0.85);
    }

    out.color = vec4<f32>(col * a, a);
    return out;
}

@fragment
fn fs(in: VsOut) -> FsOut {
    // 2D editor mode: grid on the XY plane (parallel to the sprite plane).
    if (view.mode_axis.x > 0.5) {
        return grid_xy(in);
    }

    var out: FsOut;
    out.color = vec4<f32>(0.0);
    out.depth = 0.0; // infinity under reverse-Z

    let ro = view.eye.xyz;
    let rd = view_ray(in.ndc);
    // PAR1b.2: the plane is WORLD y = 0 — render-local `-origin.y`
    // (`mode_axis.y`). It used to be render-local 0, which is wherever the
    // floating origin snapped the camera's height to: on the island that put
    // the grid a few metres over the streets, drawn over every road and
    // pavement. At world zero the terrain, the roads and the buildings above
    // sea level occlude it through the depth test like any surface.
    let t = (view.mode_axis.y - ro.y) / rd.y;
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

    // PAR1b.2: NIGHT-FADED. The grid is drawn into the HDR scene before the
    // eye's exposure, so at night (the island's eye opens x40 - x180) its
    // 0.3-grey lines tonemapped to a white lattice. Below the horizon the
    // grid dims to 1/64 (six stops — the eye's night opening), back to full
    // over the civil-twilight band; a day sun (every golden) is untouched.
    let day = clamp((view.sun_dir.y + 0.10) / 0.20, 0.0, 1.0);
    let dim = mix(1.0 / 64.0, 1.0, day);
    a = a * dim;

    // Premultiplied alpha over the sky/meshes.
    out.color = vec4<f32>(col * a, a);
    return out;
}
